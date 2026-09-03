use crate::hash::{Hasher, MerkleHasher};
use anyhow::{Result, anyhow};
use primitive_types::{H256, U256, U512};
use sha2::Digest;
use starkom_ff::{Field, Field256};
use std::any::TypeId;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::{LazyLock, Mutex};

fn make_leaf_dst(modulus: U512) -> U256 {
    static DST_STRING: &'static [u8] = b"starkom/merkle/leaf";
    let mut hasher = sha3::Sha3_512::new();
    hasher.update(DST_STRING);
    let hash: U512 = U512::from_little_endian(hasher.finalize().as_slice());
    let value = hash % modulus;
    U256::from_little_endian(&value.to_little_endian()[0..32])
}

fn get_leaf_dst<F: Field>() -> F {
    // TODO: this is a global mutex on a hot path. This map is add-only, so we should really switch
    // to a lock-free map.
    static DST_CACHE: LazyLock<Mutex<BTreeMap<TypeId, U256>>> = LazyLock::new(|| Mutex::default());
    let value = {
        let mut cache = DST_CACHE.lock().unwrap();
        *cache
            .entry(TypeId::of::<F>())
            .or_insert_with(|| make_leaf_dst(F::MODULUS.parse().unwrap()))
    };
    F::try_from_le_bytes(&value.to_little_endian()[0..(F::LEN)]).unwrap()
}

/// Hashes a leaf of a Merkle tree.
///
/// Our Merkle trees have vectors of values as leaves (there's one element for every committed
/// polynomial so that we can commit multiple polynomials into the same tree), so the input `values`
/// parameter is a slice of scalar values.
fn hash_leaf<F: Field256, H: Hasher<F>>(
    values: impl IntoIterator<Item = F, IntoIter: ExactSizeIterator>,
) -> H256 {
    let values = values.into_iter();
    let count = F::try_from(values.len()).unwrap();
    H::hash(
        std::iter::once(get_leaf_dst::<F>())
            .chain(std::iter::once(count))
            .chain(values),
    )
}

/// Computes the internal nodes of a binary Merkle tree up to the root.
///
/// `n` is the number of leaves and must be a power of two. The specified array already contains `n`
/// leaf hashes plus `n-1` empty slots; `merklify` fills the latter with the internal node hashes.
///
/// The full Merkle tree is stored inline in the `hashes` array as follows:
///
///   * the first `n` elements are the caller-provided leaf hashes,
///   * the next `n / 2` elements are the hashes of the second-last layer of the tree,
///   * the next `n / 4` elements are the hashes of the third-last layer of the tree,
///   * ...
///   * the last stored element is the Merkle root.
///
/// It's the caller's responsibility to ensure that `hashes` has at least `n * 2 - 1` slots so that
/// the full tree can be stored.
///
/// Note that the Merkle root will be at index `(n - 1) * 2`.
pub(crate) fn merklify<H: MerkleHasher>(mut hashes: &mut [H256], mut n: usize) {
    assert!(n.is_power_of_two());
    while n > 1 {
        let m = n / 2;
        for j in 0..m {
            hashes[n + j] = H::hash_binary(hashes[j * 2], hashes[j * 2 + 1]);
        }
        hashes = &mut hashes[n..];
        n = m;
    }
}

/// A Merkle proof.
///
/// A FRI `Query` uses several of these: two from the main Merkle tree and two for each folding
/// round.
///
/// NOTE: this object only stores the sister hashes of the Merkle path and the opened leaf values,
/// it doesn't store the lookup key and the root hash anywhere because those pieces of information
/// are reconstructed separately during the verification of a whole `Query`. In particular, all root
/// hashes are stored in the `Commitment`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Proof<F: Field256, H: Hasher<F>> {
    leaf: Vec<F>,
    path: Vec<H256>,
    _data: PhantomData<H>,
}

impl<F: Field256, H: Hasher<F>> Proof<F, H> {
    /// Returns a reference to the leaf values (one for every committed polynomial).
    pub(crate) fn leaf(&self) -> &[F] {
        self.leaf.as_slice()
    }

    /// Checks the leaf of this proof against the provided slice.
    ///
    /// The two must match or an error is returned.
    pub(crate) fn check_leaf(&self, expected: &[F]) -> Result<()> {
        if expected.len() != self.leaf.len() || self.leaf.as_slice() != expected {
            return Err(anyhow!("leaf value mismatch"));
        }
        Ok(())
    }

    /// Returns the length of the Merkle path, corresponding to the height of the tree minus 1 (the
    /// root hash is not included in this count).
    pub(crate) fn len(&self) -> usize {
        self.path.len()
    }

    /// Verifies the proof against the given root hash.
    pub(crate) fn verify(&self, mut index: usize, root_hash: H256) -> Result<()> {
        let mut hash = hash_leaf::<F, H>(self.leaf.iter().copied());
        for &sibling in &self.path {
            hash = if index & 1 != 0 {
                H::hash_binary(sibling, hash)
            } else {
                H::hash_binary(hash, sibling)
            };
            index >>= 1;
        }
        if index != 0 {
            return Err(anyhow!("invalid index"));
        }
        if hash != root_hash {
            return Err(anyhow!(
                "root hash mismatch (got {}, want {})",
                hash,
                root_hash
            ));
        }
        Ok(())
    }

    /// Returns a boolean indicating whether or not the committed polynomials are constant.
    ///
    /// This function is used in low degree testing to check when the folding process collapses to
    /// degree-0 polynomials.
    ///
    /// Note that some polynomials may collapse earlier than others, and this function returns false
    /// if one or more haven't collapsed yet. So it returns true if and only if all have collapsed.
    pub(crate) fn is_constant(&self) -> bool {
        let mut hash = hash_leaf::<F, H>(self.leaf.iter().copied());
        for &sibling in &self.path {
            if sibling != hash {
                return false;
            }
            hash = H::hash_binary(hash, hash);
        }
        true
    }
}

/// A Merkle tree whose leaves are multiple polynomial evaluations.
///
/// The tree has N leaves in total, with N being the size of the extended domain, and each leaf has
/// K polynomial evaluations, with K being the number of committed polynomials.
///
/// The internal nodes are single hashes.
#[derive(Debug, Clone)]
pub(crate) struct Tree<F: Field256, H: Hasher<F>> {
    /// The polynomial evaluations committed in the tree. The outer array has K entries, one for
    /// every committed polynomial, and the inner array has N entries, one for every evaluation of a
    /// polynomial.
    leaves: Vec<Vec<F>>,
    /// The internal nodes of the tree. There are 2*N-1 nodes in this array, with N = number of
    /// leaves. The nodes of the bottom layer are the hashes of the corresponding leaves.
    hashes: Vec<H256>,
    _data: PhantomData<H>,
}

impl<F: Field256, H: Hasher<F>> Tree<F, H> {
    /// Constructs a Merkle tree from a matrix of polynomial evaluations.
    ///
    /// The outer array of `polynomials` contains one entry per committed polynomial, and each of
    /// the inner arrays represents the evaluations of a polynomial.
    ///
    /// The outer array must have at least 1 element (at least 1 polynomial must be committed) and
    /// all inner arrays must have the same size N which must be the size of the (extended)
    /// evaluation domain (always a power of 2).
    ///
    /// Neither the outer array nor the inner arrays can be empty.
    pub(crate) fn new(polynomials: Vec<Vec<F>>) -> Self {
        let num_polys = polynomials.len();
        assert!(num_polys > 0);
        let n = polynomials[0].len();
        assert!(n.is_power_of_two());
        assert!(polynomials.iter().all(|polynomial| polynomial.len() == n));
        let mut hashes = vec![H256::default(); n * 2 - 1];
        for i in 0..n {
            hashes[i] = hash_leaf::<F, H>(polynomials.iter().map(|polynomial| polynomial[i]));
        }
        merklify::<H>(hashes.as_mut_slice(), n);
        Self {
            leaves: polynomials,
            hashes,
            _data: Default::default(),
        }
    }

    /// Returns the number of polynomials stored in the tree.
    ///
    /// Each leaf of the tree has this number of values.
    pub(crate) fn num_polys(&self) -> usize {
        self.leaves.len()
    }

    /// Returns the number of leaves in the tree, corresponding to the size of the evaluation domain
    /// (always a power of 2).
    pub(crate) fn num_leaves(&self) -> usize {
        self.leaves[0].len()
    }

    /// Returns the root hash of the Merkle tree.
    pub(crate) fn root_hash(&self) -> H256 {
        let n = self.num_leaves();
        self.hashes[(n - 1) * 2]
    }

    /// Returns a vector representing the i-th leaf.
    ///
    /// Note that the leaf contains k elements, one for every committed polynomial.
    pub(crate) fn leaf(&self, index: usize) -> Vec<F> {
        self.leaves.iter().map(|values| values[index]).collect()
    }

    /// Returns the value at a given leaf for a specific polynomial.
    ///
    /// `polynomial_index` must be less than [`Self::num_polys()`] and `leaf_index` must be less
    /// than [`Self::num_leaves()`].
    pub(crate) fn leaf_value(&self, polynomial_index: usize, leaf_index: usize) -> F {
        self.leaves[polynomial_index][leaf_index]
    }

    /// Returns a Merkle proof for the leaf at `index`.
    pub(crate) fn query(&self, mut index: usize) -> Proof<F, H> {
        let mut n = self.num_leaves();
        assert!(n.is_power_of_two());
        assert!(index < n);
        let leaf = self.leaf(index);
        let mut path = Vec::with_capacity(n.trailing_zeros() as usize);
        let mut hashes = self.hashes.as_slice();
        while n > 1 {
            path.push(hashes[index ^ 1]);
            hashes = &hashes[n..];
            n /= 2;
            index >>= 1;
        }
        Proof {
            leaf,
            path,
            _data: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::{Keccak256Hash, Sha2Hash};
    use starkom_bluesky::Scalar as BS;
    use starkom_goldilocks::{GL, GL4};
    use std::fmt::Debug;
    use std::str::FromStr;

    fn from_const<F: Field>(value: u16) -> F {
        F::from(value)
    }

    fn parse<V: FromStr<Err: Debug>>(s: &'static str) -> V {
        s.parse().unwrap()
    }

    #[test]
    fn test_dsts() {
        assert_eq!(
            get_leaf_dst::<BS>(),
            parse("0x08cb2652a56289bd316cfcb356f5d2be485538e04a601fb14fc2c98f03077fcb")
        );
        assert_eq!(get_leaf_dst::<GL>(), parse("0x9e8c852f6e39922a"));
    }

    #[test]
    fn test_hash_leaf_sha2_bluesky() {
        assert_eq!(
            hash_leaf::<BS, Sha2Hash<BS>>([from_const(12)]),
            parse("0x0bd187bc3deea1ef6c2a9ae254cf4e493f1dbbda32c79a662fc1d8437ab7e7c6")
        );
        assert_eq!(
            hash_leaf::<BS, Sha2Hash<BS>>([from_const(34), from_const(56)]),
            parse("0xbcd7235ceb553ca6682fd2df813650cad056a20fd6c76e1e04ae9f6a248c6c02")
        );
        assert_eq!(
            hash_leaf::<BS, Sha2Hash<BS>>([from_const(78), from_const(90), from_const(12)]),
            parse("0xeb8b9e0099332552744b9111d5a478dc61b231407d6c776586a50a3fc4513ca4")
        );
    }

    #[test]
    fn test_hash_leaf_sha2_goldilocks() {
        assert_eq!(
            hash_leaf::<GL4, Sha2Hash<GL4>>([from_const(12)]),
            parse("0x35b8c46b2ddc91d8b6bf3a5f5c53be0fb8856ea801986a278693d6c5f0233c59")
        );
        assert_eq!(
            hash_leaf::<GL4, Sha2Hash<GL4>>([from_const(34), from_const(56)]),
            parse("0x16b937b4393988c76386bb0fbda2ade97c28ec6b6f4d4aa49ce61fece35a48ec")
        );
        assert_eq!(
            hash_leaf::<GL4, Sha2Hash<GL4>>([from_const(78), from_const(90), from_const(12)]),
            parse("0x36b7284c5d4b93f9a4d75343ca68866fbb39c773ff956a6ae9d7f7552a7a7051")
        );
    }

    #[test]
    fn test_hash_leaf_keccak256_bluesky() {
        assert_eq!(
            hash_leaf::<BS, Keccak256Hash<BS>>([from_const(12)]),
            parse("0x5e70242e081756f445b5f3048611464568002e68800238dcfef718504de01782")
        );
        assert_eq!(
            hash_leaf::<BS, Keccak256Hash<BS>>([from_const(34), from_const(56)]),
            parse("0x6cbf9a3af9793caffe8c509dfaccf0198e833cafa8eb83a0a7d57ce35a97dbd1")
        );
        assert_eq!(
            hash_leaf::<BS, Keccak256Hash<BS>>([from_const(78), from_const(90), from_const(12)]),
            parse("0x05ce2ee03a570540c8a67a9b5a419dc7813922da6a39dff3c293806ed0f88fbb")
        );
    }

    #[test]
    fn test_hash_leaf_keccak256_goldilocks() {
        assert_eq!(
            hash_leaf::<GL4, Keccak256Hash<GL4>>([from_const(12)]),
            parse("0x0582d9f0e3652d6cef9c301b16fbff366dd1e8910befbc7f991cf7a4862574d3")
        );
        assert_eq!(
            hash_leaf::<GL4, Keccak256Hash<GL4>>([from_const(34), from_const(56)]),
            parse("0x0c8448a67efe90928470801e2b75ece4fb838f232ae4bb79e361b1bb97813dbd")
        );
        assert_eq!(
            hash_leaf::<GL4, Keccak256Hash<GL4>>([from_const(78), from_const(90), from_const(12)]),
            parse("0x48769fec704d28d0747b4bd8cd4fdef4025dfbfc9afb301f0bec283e9c0a6e6d")
        );
    }

    #[test]
    fn test_merklify_one_sha2() {
        let mut hashes = vec![parse(
            "0x1fe9e33b7ff790473e13eb6384d61d6abb32d2e50b30a382c9767b5347f5846b",
        )];
        merklify::<Sha2Hash<BS>>(&mut hashes, 1);
        assert_eq!(
            hashes,
            vec![parse(
                "0x1fe9e33b7ff790473e13eb6384d61d6abb32d2e50b30a382c9767b5347f5846b"
            )]
        );
    }

    #[test]
    fn test_merklify_one_keccak256() {
        let mut hashes = vec![parse(
            "0x1fe9e33b7ff790473e13eb6384d61d6abb32d2e50b30a382c9767b5347f5846b",
        )];
        merklify::<Keccak256Hash<BS>>(&mut hashes, 1);
        assert_eq!(
            hashes,
            vec![parse(
                "0x1fe9e33b7ff790473e13eb6384d61d6abb32d2e50b30a382c9767b5347f5846b"
            )]
        );
    }

    #[test]
    fn test_merklify_two_sha2() {
        let mut hashes = vec![
            parse("0x0cbb1061e55efef40fae3c8e34d301c9940889146f816edd475052fc45caf060"),
            parse("0x2d8257d72c4b6e6bfa3a21a22c053d02d1d52b4b5e6097acfd0ed45f827d6ba4"),
        ];
        hashes.resize(3, H256::default());
        merklify::<Sha2Hash<BS>>(&mut hashes, 2);
        assert_eq!(
            hashes,
            vec![
                parse("0x0cbb1061e55efef40fae3c8e34d301c9940889146f816edd475052fc45caf060"),
                parse("0x2d8257d72c4b6e6bfa3a21a22c053d02d1d52b4b5e6097acfd0ed45f827d6ba4"),
                parse("0x3af448b6612ac9931b7413ecca5d209187cd808ae2d3647099aa99fada16c955")
            ]
        );
    }

    #[test]
    fn test_merklify_two_keccak256() {
        let mut hashes = vec![
            parse("0x0cbb1061e55efef40fae3c8e34d301c9940889146f816edd475052fc45caf060"),
            parse("0x2d8257d72c4b6e6bfa3a21a22c053d02d1d52b4b5e6097acfd0ed45f827d6ba4"),
        ];
        hashes.resize(3, H256::default());
        merklify::<Keccak256Hash<BS>>(&mut hashes, 2);
        assert_eq!(
            hashes,
            vec![
                parse("0x0cbb1061e55efef40fae3c8e34d301c9940889146f816edd475052fc45caf060"),
                parse("0x2d8257d72c4b6e6bfa3a21a22c053d02d1d52b4b5e6097acfd0ed45f827d6ba4"),
                parse("0x6e70b19bd160e181bb367ab730eababc6e3673977f7fdce38934ee197bbcb568")
            ]
        );
    }

    #[test]
    fn test_merklify_four_sha2() {
        let mut hashes = vec![
            parse("0x034acf2cced8e9744784ec9c3c626fa83f6f8ddd83faed9d5caa5ad72c91eb1c"),
            parse("0x39dce95a2271a57f999981eef6917f3df8ad25d116c98d809a3b7da5d54805a3"),
            parse("0x435246f701f1483adcb7037fa64b3fa027c41e13a52d1e6c502e9b71dea3ca81"),
            parse("0x7952ed41c4062f594b53b8c548874b41a7a7f05593d7a5a313ed82c2fe1c62d7"),
        ];
        hashes.resize(7, H256::default());
        merklify::<Sha2Hash<BS>>(&mut hashes, 4);
        assert_eq!(
            hashes,
            vec![
                parse("0x034acf2cced8e9744784ec9c3c626fa83f6f8ddd83faed9d5caa5ad72c91eb1c"),
                parse("0x39dce95a2271a57f999981eef6917f3df8ad25d116c98d809a3b7da5d54805a3"),
                parse("0x435246f701f1483adcb7037fa64b3fa027c41e13a52d1e6c502e9b71dea3ca81"),
                parse("0x7952ed41c4062f594b53b8c548874b41a7a7f05593d7a5a313ed82c2fe1c62d7"),
                parse("0x3d139c65859969e86db444ccc0d36ffd0456b3f92d3dfb34fa10c8c07f5ada06"),
                parse("0xdc3dd313db4b53f2720fcfe6373d286a9c40911546e1c434fa24ab650a75b586"),
                parse("0xed171c6bd6876b93f4fc8914e0b2a8db53d7204048ad9fdff6e7127c077ca072"),
            ]
        );
    }

    #[test]
    fn test_merklify_four_keccak256() {
        let mut hashes = vec![
            parse("0x034acf2cced8e9744784ec9c3c626fa83f6f8ddd83faed9d5caa5ad72c91eb1c"),
            parse("0x39dce95a2271a57f999981eef6917f3df8ad25d116c98d809a3b7da5d54805a3"),
            parse("0x435246f701f1483adcb7037fa64b3fa027c41e13a52d1e6c502e9b71dea3ca81"),
            parse("0x7952ed41c4062f594b53b8c548874b41a7a7f05593d7a5a313ed82c2fe1c62d7"),
        ];
        hashes.resize(7, H256::default());
        merklify::<Keccak256Hash<BS>>(&mut hashes, 4);
        assert_eq!(
            hashes,
            vec![
                parse("0x034acf2cced8e9744784ec9c3c626fa83f6f8ddd83faed9d5caa5ad72c91eb1c"),
                parse("0x39dce95a2271a57f999981eef6917f3df8ad25d116c98d809a3b7da5d54805a3"),
                parse("0x435246f701f1483adcb7037fa64b3fa027c41e13a52d1e6c502e9b71dea3ca81"),
                parse("0x7952ed41c4062f594b53b8c548874b41a7a7f05593d7a5a313ed82c2fe1c62d7"),
                parse("0x8a2dbe0367d89a172a0eb470026f5bb85ddd542e0d93885318c5d6b8a3764f28"),
                parse("0x31354bf6d6ca07dbde80395609890a558ff35587ddc2b5c72ccde2b4e15d114f"),
                parse("0xbd82b7f4f83de57b73e00aa0924a5fc4dfaf959bd252aa3fda8abdaf87713dc0"),
            ]
        );
    }

    fn test_merkle_tree<F: Field256, H: Hasher<F>>(
        evaluations: Vec<Vec<F>>,
        expected_root_hash: H256,
    ) {
        let k = evaluations.len();
        let n = evaluations[0].len();
        let tree = Tree::<F, H>::new(evaluations.clone());
        assert_eq!(tree.num_polys(), k);
        assert_eq!(tree.num_leaves(), n);
        assert_eq!(tree.root_hash(), expected_root_hash);
        for i in 0..n {
            let proof = tree.query(i);
            assert!(proof.verify(i, expected_root_hash).is_ok());
            assert_eq!(proof.leaf().len(), k);
            assert!(
                proof
                    .leaf()
                    .iter()
                    .zip(evaluations.iter())
                    .all(|(&lhs, values)| lhs == values[i])
            );
            for j in 0..k {
                assert_eq!(tree.leaf_value(j, i), evaluations[j][i]);
            }
        }
    }

    #[test]
    fn test_merkle_tree_one_leaf_1() {
        test_merkle_tree::<BS, Sha2Hash<BS>>(
            vec![vec![from_const(12)]],
            parse("0x0bd187bc3deea1ef6c2a9ae254cf4e493f1dbbda32c79a662fc1d8437ab7e7c6"),
        );
        test_merkle_tree::<GL4, Sha2Hash<GL4>>(
            vec![vec![from_const(12)]],
            parse("0x35b8c46b2ddc91d8b6bf3a5f5c53be0fb8856ea801986a278693d6c5f0233c59"),
        );
        test_merkle_tree::<BS, Keccak256Hash<BS>>(
            vec![vec![from_const(12)]],
            parse("0x5e70242e081756f445b5f3048611464568002e68800238dcfef718504de01782"),
        );
        test_merkle_tree::<GL4, Keccak256Hash<GL4>>(
            vec![vec![from_const(12)]],
            parse("0x0582d9f0e3652d6cef9c301b16fbff366dd1e8910befbc7f991cf7a4862574d3"),
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_2() {
        test_merkle_tree::<BS, Sha2Hash<BS>>(
            vec![vec![from_const(34)]],
            parse("0x825f71a1d38bedb88129450457f7943f988ee940aa1755aa89982e139e67047a"),
        );
        test_merkle_tree::<GL4, Sha2Hash<GL4>>(
            vec![vec![from_const(34)]],
            parse("0xfef88d77ef0fd0acf4b65461542e8103d2161112bd6e60ed1c5c1b89c18e27fd"),
        );
        test_merkle_tree::<BS, Keccak256Hash<BS>>(
            vec![vec![from_const(34)]],
            parse("0xe6b7d9bb7fab250037d1ede2453bf936d350c03d028048fa491663a8d4a070ae"),
        );
        test_merkle_tree::<GL4, Keccak256Hash<GL4>>(
            vec![vec![from_const(34)]],
            parse("0x3c54733b9ccb4772ed9a7d7c1ad6d3f6b34a9644f3aa1aa5f09c0fae24b45500"),
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_two_polynomials_1() {
        test_merkle_tree::<BS, Sha2Hash<BS>>(
            vec![vec![from_const(12)], vec![from_const(34)]],
            parse("0x41c90ef8e7fa7e79e54b14cf8395f9707e9768a02aa79ce7f0f4e68668837c26"),
        );
        test_merkle_tree::<GL4, Sha2Hash<GL4>>(
            vec![vec![from_const(12)], vec![from_const(34)]],
            parse("0xdeb1534e334b8008495afddf7146a6ebbc9f7d7a1dcff2c1b21c05ae612aae25"),
        );
        test_merkle_tree::<BS, Keccak256Hash<BS>>(
            vec![vec![from_const(12)], vec![from_const(34)]],
            parse("0x8a39979b1cf9df3dd02d420608cb73a92423a9ef2e44afb1118f5d62a1e35b59"),
        );
        test_merkle_tree::<GL4, Keccak256Hash<GL4>>(
            vec![vec![from_const(12)], vec![from_const(34)]],
            parse("0x1c2534e0c33573bb8e8fab98b0972a32cc1fe3effab73ff8603b8186c55906d1"),
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_two_polynomials_2() {
        test_merkle_tree::<BS, Sha2Hash<BS>>(
            vec![vec![from_const(34)], vec![from_const(12)]],
            parse("0xc3dbf8cc67db17f01dd3527ab16da2120af998c7ebe3d06a2d9dff445e1adaee"),
        );
        test_merkle_tree::<GL4, Sha2Hash<GL4>>(
            vec![vec![from_const(34)], vec![from_const(12)]],
            parse("0x0a0d5b00cb05493d56cab5077a39a12f6fe4c0f7c6a53fdc268d9a3be1190591"),
        );
        test_merkle_tree::<BS, Keccak256Hash<BS>>(
            vec![vec![from_const(34)], vec![from_const(12)]],
            parse("0xa8f2b055d24c330298c32a0ed891a91378af0a495403fd5f95eca566e34a1c39"),
        );
        test_merkle_tree::<GL4, Keccak256Hash<GL4>>(
            vec![vec![from_const(34)], vec![from_const(12)]],
            parse("0xb1e2a185742ede9bc1a085cedec939ee6b24658996b068555cf8f45bc738e3e2"),
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_three_polynomials_1() {
        test_merkle_tree::<BS, Sha2Hash<BS>>(
            vec![
                vec![from_const(12)],
                vec![from_const(34)],
                vec![from_const(56)],
            ],
            parse("0x9bca77d625d9a50f1807d27c273cae980f706ef734a014c00e6a97bbb77472e3"),
        );
        test_merkle_tree::<GL4, Sha2Hash<GL4>>(
            vec![
                vec![from_const(12)],
                vec![from_const(34)],
                vec![from_const(56)],
            ],
            parse("0x03e100a3e1b7dc7047f362676d6d44704eeda9dfac6714417330c4f232b5ae92"),
        );
        test_merkle_tree::<BS, Keccak256Hash<BS>>(
            vec![
                vec![from_const(12)],
                vec![from_const(34)],
                vec![from_const(56)],
            ],
            parse("0x11e4b3b9246b3678b751e40aac1ceb433a51be72a6c3560a49e0954d04689017"),
        );
        test_merkle_tree::<GL4, Keccak256Hash<GL4>>(
            vec![
                vec![from_const(12)],
                vec![from_const(34)],
                vec![from_const(56)],
            ],
            parse("0x1795b3459e1e257a198dbaf5ffbb31274db87b7cf785d96fc97762d0a144f417"),
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_three_polynomials_2() {
        test_merkle_tree::<BS, Sha2Hash<BS>>(
            vec![
                vec![from_const(34)],
                vec![from_const(12)],
                vec![from_const(78)],
            ],
            parse("0xf48317b0be7caae4a15185cc8d0795c15c2bc98e18b8cbdc906270ada921fd25"),
        );
        test_merkle_tree::<GL4, Sha2Hash<GL4>>(
            vec![
                vec![from_const(34)],
                vec![from_const(12)],
                vec![from_const(78)],
            ],
            parse("0x3fb57228b65495ac10bd7c27c04934f652d740f36ccc4308dfc663b99e9a7476"),
        );
        test_merkle_tree::<BS, Keccak256Hash<BS>>(
            vec![
                vec![from_const(34)],
                vec![from_const(12)],
                vec![from_const(78)],
            ],
            parse("0xf3f5da8d03ec645ab1624876b7515f440d6dbbb5366e9e2da5218f9c603e990d"),
        );
        test_merkle_tree::<GL4, Keccak256Hash<GL4>>(
            vec![
                vec![from_const(34)],
                vec![from_const(12)],
                vec![from_const(78)],
            ],
            parse("0x19285d2130c6c6b5827034e3a48fe4a108901d098456aeb43e07dfd5ffc73775"),
        );
    }

    #[test]
    fn test_merkle_tree_two_leaves_1() {
        test_merkle_tree::<BS, Sha2Hash<BS>>(
            vec![vec![from_const(12), from_const(34)]],
            parse("0x2624006228d517eeda393d1440f25ed1c20887664f2444021849345167aadaf4"),
        );
        test_merkle_tree::<GL4, Sha2Hash<GL4>>(
            vec![vec![from_const(12), from_const(34)]],
            parse("0x46e71c8cf94bae7f04ab0ec21c6341bce3b5293da7be6b9b2ed75fcf526b9a86"),
        );
        test_merkle_tree::<BS, Keccak256Hash<BS>>(
            vec![vec![from_const(12), from_const(34)]],
            parse("0xc781424ee41551ea0e56431b57519317a836b94c193ee898b39d058507217ee4"),
        );
        test_merkle_tree::<GL4, Keccak256Hash<GL4>>(
            vec![vec![from_const(12), from_const(34)]],
            parse("0x7086d87f5a1f7187c16a0b3a18e1ec6454adfba533a9bb32b5d3f0f28c4e88b5"),
        );
    }

    #[test]
    fn test_merkle_tree_two_leaves_2() {
        test_merkle_tree::<BS, Sha2Hash<BS>>(
            vec![vec![from_const(34), from_const(56)]],
            parse("0x85d6170f1dcec468c3a42f35d24b7689733abf26ff0278a9bc1edc7a9a0a7333"),
        );
        test_merkle_tree::<GL4, Sha2Hash<GL4>>(
            vec![vec![from_const(34), from_const(56)]],
            parse("0x3ddde531f8532974c5911f7107567f0f216feb1856a63d06eec9859957aafa96"),
        );
        test_merkle_tree::<BS, Keccak256Hash<BS>>(
            vec![vec![from_const(34), from_const(56)]],
            parse("0x00526855bdd13b55968dd71bddfca2409e77ee8c75cdc17fcdb641673558ae51"),
        );
        test_merkle_tree::<GL4, Keccak256Hash<GL4>>(
            vec![vec![from_const(34), from_const(56)]],
            parse("0x57dee6e134693b1ebe8536a46f20fd22ba7fee68ad3af4156427dee220ef951e"),
        );
    }

    #[test]
    fn test_merkle_tree_two_leaves_two_polynomials_1() {
        test_merkle_tree::<BS, Sha2Hash<BS>>(
            vec![
                vec![from_const(12), from_const(56)],
                vec![from_const(34), from_const(78)],
            ],
            parse("0xdc144d55dd7a9c48f00b495d30172b38db7d4dc71ee4a0feab99177e30a10d21"),
        );
        test_merkle_tree::<GL4, Sha2Hash<GL4>>(
            vec![
                vec![from_const(12), from_const(56)],
                vec![from_const(34), from_const(78)],
            ],
            parse("0x8533f65a8cc96ace4b1c9c676d12ddd4dbc8f1a4fc7d161fa274985615d332f6"),
        );
        test_merkle_tree::<BS, Keccak256Hash<BS>>(
            vec![
                vec![from_const(12), from_const(56)],
                vec![from_const(34), from_const(78)],
            ],
            parse("0x086ff3e1257cbff588f5d01ce891ea152e15863c5e0d4407d4c3564d354b0614"),
        );
        test_merkle_tree::<GL4, Keccak256Hash<GL4>>(
            vec![
                vec![from_const(12), from_const(56)],
                vec![from_const(34), from_const(78)],
            ],
            parse("0xcbe8abdd08b5a47254beb0e73ea2191f120c40b310268985a6c96306e73395b3"),
        );
    }

    #[test]
    fn test_merkle_tree_two_leaves_two_polynomials_2() {
        test_merkle_tree::<BS, Sha2Hash<BS>>(
            vec![
                vec![from_const(78), from_const(34)],
                vec![from_const(56), from_const(12)],
            ],
            parse("0xfb5a002c9ef7dad6d9b4edc690f323f4c302c665926c085dd5d8681496c937c6"),
        );
        test_merkle_tree::<GL4, Sha2Hash<GL4>>(
            vec![
                vec![from_const(78), from_const(34)],
                vec![from_const(56), from_const(12)],
            ],
            parse("0xb330dd303935029654afd13ea17eda755842eca21b3152bd570550d3bdaed46f"),
        );
        test_merkle_tree::<BS, Keccak256Hash<BS>>(
            vec![
                vec![from_const(78), from_const(34)],
                vec![from_const(56), from_const(12)],
            ],
            parse("0xb18e697b08221e2f411826d3c17c80bff6e06e90a55bcd5cc191ad67f1eb657a"),
        );
        test_merkle_tree::<GL4, Keccak256Hash<GL4>>(
            vec![
                vec![from_const(78), from_const(34)],
                vec![from_const(56), from_const(12)],
            ],
            parse("0x763ae7db144ceade048b6f96efb13975cefa9cd9874fe57469ff8e1f6cff2512"),
        );
    }
}
