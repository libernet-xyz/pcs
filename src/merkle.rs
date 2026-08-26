use crate::dst;
use crate::hash::{Hasher, MerkleHasher};
use anyhow::{Result, anyhow};
use primitive_types::H256;
use starkom_ff::{Field, Field256};
use std::marker::PhantomData;

/// Hashes a leaf of a Merkle tree.
///
/// Our Merkle trees have vectors of values as leaves (there's one element for every committed
/// polynomial so that we can commit multiple polynomials into the same tree), so the input `values`
/// parameter is a slice of scalar values.
fn hash_leaf<F: Field, G: Field256 + From<F>, H: Hasher<G>>(values: &[F]) -> H256 {
    H::hash(
        std::iter::once(dst::get_dst::<F>(b"starkom/merkle/leaf"))
            .chain(std::iter::once(F::try_from(values.len()).unwrap()))
            .chain(values.iter().cloned())
            .map(Into::<G>::into),
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
pub(crate) struct Proof<F: Field, G: Field256 + From<F>, H: Hasher<G>> {
    leaf: Vec<F>,
    path: Vec<H256>,
    _data: PhantomData<(G, H)>,
}

impl<F: Field, G: Field256 + From<F>, H: Hasher<G>> Proof<F, G, H> {
    /// Returns a reference to the leaf values (one for every committed polynomial).
    pub(crate) fn leaf(&self) -> &[F] {
        self.leaf.as_slice()
    }

    /// Checks the leaf of this proof against the provided slice.
    ///
    /// The two must match or an error is returned.
    pub(crate) fn check_leaf(&self, expected: &[F]) -> Result<()> {
        if expected.len() != self.leaf.len()
            || self
                .leaf
                .iter()
                .zip(expected.iter())
                .any(|(&value1, &value2)| value1 != value2)
        {
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
        let mut hash = hash_leaf::<F, G, H>(self.leaf.as_slice());
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
    /// An error is returned if one or more hashes in the proof are out of range; see [`Hash<H256>`]
    /// for details.
    ///
    /// This function is used in low degree testing to check when the folding process collapses to
    /// degree-0 polynomials.
    ///
    /// Note that some polynomials may collapse earlier than others, and this function returns false
    /// if one or more haven't collapsed yet. So it returns true if and only if all have collapsed.
    pub(crate) fn is_constant(&self) -> Result<bool> {
        let mut hash = hash_leaf::<F, G, H>(self.leaf.as_slice());
        for &sibling in &self.path {
            if sibling != hash {
                return Ok(false);
            }
            hash = H::hash_binary(hash, hash);
        }
        Ok(true)
    }
}

/// A Merkle tree whose leaves are multiple polynomial evaluations.
///
/// The tree has N leaves in total, with N being the size of the extended domain, and each leaf has
/// K polynomial evaluations, with K being the number of committed polynomials.
///
/// The internal nodes are single hashes.
#[derive(Debug, Clone)]
pub(crate) struct Tree<F: Field, G: Field256 + From<F>, H: Hasher<G>> {
    /// The polynomial evaluations committed in the tree. The outer array has K entries, one for
    /// every committed polynomial, and the inner array has N entries, one for every evaluation of a
    /// polynomial.
    leaves: Vec<Vec<F>>,
    /// The internal nodes of the tree. There are 2*N-1 nodes in this array, with N = number of
    /// leaves. The nodes of the bottom layer are the hashes of the corresponding leaves.
    hashes: Vec<H256>,
    _data: PhantomData<(G, H)>,
}

impl<F: Field, G: Field256 + From<F>, H: Hasher<G>> Tree<F, G, H> {
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
            hashes[i] = hash_leaf::<F, G, H>(
                polynomials
                    .iter()
                    .map(|polynomial| polynomial[i])
                    .collect::<Vec<F>>()
                    .as_slice(),
            );
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
    pub(crate) fn query(&self, mut index: usize) -> Proof<F, G, H> {
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
    fn test_hash_leaf_sha2_bluesky() {
        assert_eq!(
            hash_leaf::<BS, BS, Sha2Hash<BS>>(&[from_const(12)]),
            parse("0x0bd187bc3deea1ef6c2a9ae254cf4e493f1dbbda32c79a662fc1d8437ab7e7c6")
        );
        assert_eq!(
            hash_leaf::<BS, BS, Sha2Hash<BS>>(&[from_const(34), from_const(56)]),
            parse("0xbcd7235ceb553ca6682fd2df813650cad056a20fd6c76e1e04ae9f6a248c6c02")
        );
        assert_eq!(
            hash_leaf::<BS, BS, Sha2Hash<BS>>(&[from_const(78), from_const(90), from_const(12)]),
            parse("0xeb8b9e0099332552744b9111d5a478dc61b231407d6c776586a50a3fc4513ca4")
        );
    }

    #[test]
    fn test_hash_leaf_sha2_goldilocks() {
        assert_eq!(
            hash_leaf::<GL, GL4, Sha2Hash<GL4>>(&[from_const(12)]),
            parse("0xa76cc2edc0687213f2f6ae0352cdae9bb437a5589d51d863458e3f94fdcf1a1f")
        );
        assert_eq!(
            hash_leaf::<GL, GL4, Sha2Hash<GL4>>(&[from_const(34), from_const(56)]),
            parse("0x2197f79dd32a7b9e974f6724b585178b7350e6fc916db209bc0f8a4d68f2e362")
        );
        assert_eq!(
            hash_leaf::<GL, GL4, Sha2Hash<GL4>>(&[from_const(78), from_const(90), from_const(12)]),
            parse("0xa7045f9cf7303a4cd91fd168624cddc5aa063a26d37c8c554d3afc9400669661")
        );
    }

    #[test]
    fn test_hash_leaf_keccak256_bluesky() {
        assert_eq!(
            hash_leaf::<BS, BS, Keccak256Hash<BS>>(&[from_const(12)]),
            parse("0x5e70242e081756f445b5f3048611464568002e68800238dcfef718504de01782")
        );
        assert_eq!(
            hash_leaf::<BS, BS, Keccak256Hash<BS>>(&[from_const(34), from_const(56)]),
            parse("0x6cbf9a3af9793caffe8c509dfaccf0198e833cafa8eb83a0a7d57ce35a97dbd1")
        );
        assert_eq!(
            hash_leaf::<BS, BS, Keccak256Hash<BS>>(&[
                from_const(78),
                from_const(90),
                from_const(12)
            ]),
            parse("0x05ce2ee03a570540c8a67a9b5a419dc7813922da6a39dff3c293806ed0f88fbb")
        );
    }

    #[test]
    fn test_hash_leaf_keccak256_goldilocks() {
        assert_eq!(
            hash_leaf::<GL, GL4, Keccak256Hash<GL4>>(&[from_const(12)]),
            parse("0xf8766a3ad2ed6d5bd215b3ed0531bde06e7a31e218a17d494a99fa8ba255ba2c")
        );
        assert_eq!(
            hash_leaf::<GL, GL4, Keccak256Hash<GL4>>(&[from_const(34), from_const(56)]),
            parse("0x8b5b64c41c3dd48c62fbf06e5f20641aba3b0b69b97b4a6e82215303a033aa77")
        );
        assert_eq!(
            hash_leaf::<GL, GL4, Keccak256Hash<GL4>>(&[
                from_const(78),
                from_const(90),
                from_const(12)
            ]),
            parse("0x51c62c7131bfcd24c7ab2a1a3d1dda4fcec722cd78a833205370db18f8bd60d3")
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

    fn test_merkle_tree<F: Field, G: Field256 + From<F>, H: Hasher<G>>(
        evaluations: Vec<Vec<F>>,
        expected_root_hash: H256,
    ) {
        let k = evaluations.len();
        let n = evaluations[0].len();
        let tree = Tree::<F, G, H>::new(evaluations.clone());
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
        test_merkle_tree::<BS, BS, Sha2Hash<BS>>(
            vec![vec![from_const(12)]],
            parse("0x0bd187bc3deea1ef6c2a9ae254cf4e493f1dbbda32c79a662fc1d8437ab7e7c6"),
        );
        test_merkle_tree::<GL, GL4, Sha2Hash<GL4>>(
            vec![vec![from_const(12)]],
            parse("0xa76cc2edc0687213f2f6ae0352cdae9bb437a5589d51d863458e3f94fdcf1a1f"),
        );
        test_merkle_tree::<BS, BS, Keccak256Hash<BS>>(
            vec![vec![from_const(12)]],
            parse("0x5e70242e081756f445b5f3048611464568002e68800238dcfef718504de01782"),
        );
        test_merkle_tree::<GL, GL4, Keccak256Hash<GL4>>(
            vec![vec![from_const(12)]],
            parse("0xf8766a3ad2ed6d5bd215b3ed0531bde06e7a31e218a17d494a99fa8ba255ba2c"),
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_2() {
        test_merkle_tree::<BS, BS, Sha2Hash<BS>>(
            vec![vec![from_const(34)]],
            parse("0x825f71a1d38bedb88129450457f7943f988ee940aa1755aa89982e139e67047a"),
        );
        test_merkle_tree::<GL, GL4, Sha2Hash<GL4>>(
            vec![vec![from_const(34)]],
            parse("0x538de22f49a0422afebfcb7768d1ee33f5a2fbb21b694fb3c521390e8334b6df"),
        );
        test_merkle_tree::<BS, BS, Keccak256Hash<BS>>(
            vec![vec![from_const(34)]],
            parse("0xe6b7d9bb7fab250037d1ede2453bf936d350c03d028048fa491663a8d4a070ae"),
        );
        test_merkle_tree::<GL, GL4, Keccak256Hash<GL4>>(
            vec![vec![from_const(34)]],
            parse("0xf778d7b3ce8b27a68b7d5749eb2460ef7b3d0a478ee275558fd811afea4b7397"),
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_two_polynomials_1() {
        test_merkle_tree::<BS, BS, Sha2Hash<BS>>(
            vec![vec![from_const(12)], vec![from_const(34)]],
            parse("0x41c90ef8e7fa7e79e54b14cf8395f9707e9768a02aa79ce7f0f4e68668837c26"),
        );
        test_merkle_tree::<GL, GL4, Sha2Hash<GL4>>(
            vec![vec![from_const(12)], vec![from_const(34)]],
            parse("0x6825e5d46e04f4177259dd4d64883ae16ae2f74d3d30ac2deadd4f9e33776847"),
        );
        test_merkle_tree::<BS, BS, Keccak256Hash<BS>>(
            vec![vec![from_const(12)], vec![from_const(34)]],
            parse("0x8a39979b1cf9df3dd02d420608cb73a92423a9ef2e44afb1118f5d62a1e35b59"),
        );
        test_merkle_tree::<GL, GL4, Keccak256Hash<GL4>>(
            vec![vec![from_const(12)], vec![from_const(34)]],
            parse("0xb310dbe74714244344ee38a8f4c937578719cff53cc084e4a40147310244b377"),
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_two_polynomials_2() {
        test_merkle_tree::<BS, BS, Sha2Hash<BS>>(
            vec![vec![from_const(34)], vec![from_const(12)]],
            parse("0xc3dbf8cc67db17f01dd3527ab16da2120af998c7ebe3d06a2d9dff445e1adaee"),
        );
        test_merkle_tree::<GL, GL4, Sha2Hash<GL4>>(
            vec![vec![from_const(34)], vec![from_const(12)]],
            parse("0xf53c731e056686bea4d706dd4db1a713710dac4ece17bb548704901915d5f94d"),
        );
        test_merkle_tree::<BS, BS, Keccak256Hash<BS>>(
            vec![vec![from_const(34)], vec![from_const(12)]],
            parse("0xa8f2b055d24c330298c32a0ed891a91378af0a495403fd5f95eca566e34a1c39"),
        );
        test_merkle_tree::<GL, GL4, Keccak256Hash<GL4>>(
            vec![vec![from_const(34)], vec![from_const(12)]],
            parse("0x5cb70e1b45235d85eff0628e3642fb30b36867d38b2410f263fd7bc01598bdea"),
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_three_polynomials_1() {
        test_merkle_tree::<BS, BS, Sha2Hash<BS>>(
            vec![
                vec![from_const(12)],
                vec![from_const(34)],
                vec![from_const(56)],
            ],
            parse("0x9bca77d625d9a50f1807d27c273cae980f706ef734a014c00e6a97bbb77472e3"),
        );
        test_merkle_tree::<GL, GL4, Sha2Hash<GL4>>(
            vec![
                vec![from_const(12)],
                vec![from_const(34)],
                vec![from_const(56)],
            ],
            parse("0x6cbeda0560df214b07ce7d13c3602659ff7d74f621e2c68623c0aad94ede5d46"),
        );
        test_merkle_tree::<BS, BS, Keccak256Hash<BS>>(
            vec![
                vec![from_const(12)],
                vec![from_const(34)],
                vec![from_const(56)],
            ],
            parse("0x11e4b3b9246b3678b751e40aac1ceb433a51be72a6c3560a49e0954d04689017"),
        );
        test_merkle_tree::<GL, GL4, Keccak256Hash<GL4>>(
            vec![
                vec![from_const(12)],
                vec![from_const(34)],
                vec![from_const(56)],
            ],
            parse("0x20986d98839fac785d61cbf2121807e408e565c64ff3f033a0b752b0bedd3cc3"),
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_three_polynomials_2() {
        test_merkle_tree::<BS, BS, Sha2Hash<BS>>(
            vec![
                vec![from_const(34)],
                vec![from_const(12)],
                vec![from_const(78)],
            ],
            parse("0xf48317b0be7caae4a15185cc8d0795c15c2bc98e18b8cbdc906270ada921fd25"),
        );
        test_merkle_tree::<GL, GL4, Sha2Hash<GL4>>(
            vec![
                vec![from_const(34)],
                vec![from_const(12)],
                vec![from_const(78)],
            ],
            parse("0x9ddecf7894d55cf1509ce47a6532e610366362bea31727429fe8538054893bfe"),
        );
        test_merkle_tree::<BS, BS, Keccak256Hash<BS>>(
            vec![
                vec![from_const(34)],
                vec![from_const(12)],
                vec![from_const(78)],
            ],
            parse("0xf3f5da8d03ec645ab1624876b7515f440d6dbbb5366e9e2da5218f9c603e990d"),
        );
        test_merkle_tree::<GL, GL4, Keccak256Hash<GL4>>(
            vec![
                vec![from_const(34)],
                vec![from_const(12)],
                vec![from_const(78)],
            ],
            parse("0xc13c7933afc11a827a307cbbfff174eca8ce0e9abc79a130f5c586c1e73a4404"),
        );
    }

    #[test]
    fn test_merkle_tree_two_leaves_1() {
        test_merkle_tree::<BS, BS, Sha2Hash<BS>>(
            vec![vec![from_const(12), from_const(34)]],
            parse("0x2624006228d517eeda393d1440f25ed1c20887664f2444021849345167aadaf4"),
        );
        test_merkle_tree::<GL, GL4, Sha2Hash<GL4>>(
            vec![vec![from_const(12), from_const(34)]],
            parse("0x8f8d5f408b0d875c2e0e913c0129f973262b00bbdaefee701af33f368f434e07"),
        );
        test_merkle_tree::<BS, BS, Keccak256Hash<BS>>(
            vec![vec![from_const(12), from_const(34)]],
            parse("0xc781424ee41551ea0e56431b57519317a836b94c193ee898b39d058507217ee4"),
        );
        test_merkle_tree::<GL, GL4, Keccak256Hash<GL4>>(
            vec![vec![from_const(12), from_const(34)]],
            parse("0x379024f592268f303e1222ff41084f5c6782f59efd15b58be49dcc1cae6d4a95"),
        );
    }

    #[test]
    fn test_merkle_tree_two_leaves_2() {
        test_merkle_tree::<BS, BS, Sha2Hash<BS>>(
            vec![vec![from_const(34), from_const(56)]],
            parse("0x85d6170f1dcec468c3a42f35d24b7689733abf26ff0278a9bc1edc7a9a0a7333"),
        );
        test_merkle_tree::<GL, GL4, Sha2Hash<GL4>>(
            vec![vec![from_const(34), from_const(56)]],
            parse("0xed7bf3e4c440f10f7d3de6c0e41e0d325e0f3ffe24e4187fb782b2714e580968"),
        );
        test_merkle_tree::<BS, BS, Keccak256Hash<BS>>(
            vec![vec![from_const(34), from_const(56)]],
            parse("0x00526855bdd13b55968dd71bddfca2409e77ee8c75cdc17fcdb641673558ae51"),
        );
        test_merkle_tree::<GL, GL4, Keccak256Hash<GL4>>(
            vec![vec![from_const(34), from_const(56)]],
            parse("0x97371a612a466e02663b2eef2800a09342003fbc0fdcc9a37111a3285b60c2a8"),
        );
    }

    #[test]
    fn test_merkle_tree_two_leaves_two_polynomials_1() {
        test_merkle_tree::<BS, BS, Sha2Hash<BS>>(
            vec![
                vec![from_const(12), from_const(56)],
                vec![from_const(34), from_const(78)],
            ],
            parse("0xdc144d55dd7a9c48f00b495d30172b38db7d4dc71ee4a0feab99177e30a10d21"),
        );
        test_merkle_tree::<GL, GL4, Sha2Hash<GL4>>(
            vec![
                vec![from_const(12), from_const(56)],
                vec![from_const(34), from_const(78)],
            ],
            parse("0xde24d40a6507555666c04470640ff49396094a617478c8bd94688eb1996d4209"),
        );
        test_merkle_tree::<BS, BS, Keccak256Hash<BS>>(
            vec![
                vec![from_const(12), from_const(56)],
                vec![from_const(34), from_const(78)],
            ],
            parse("0x086ff3e1257cbff588f5d01ce891ea152e15863c5e0d4407d4c3564d354b0614"),
        );
        test_merkle_tree::<GL, GL4, Keccak256Hash<GL4>>(
            vec![
                vec![from_const(12), from_const(56)],
                vec![from_const(34), from_const(78)],
            ],
            parse("0xe862ac873453969fd46121d7fcc9e119190c2e2c4d2b6d23fb86764c94752532"),
        );
    }

    #[test]
    fn test_merkle_tree_two_leaves_two_polynomials_2() {
        test_merkle_tree::<BS, BS, Sha2Hash<BS>>(
            vec![
                vec![from_const(78), from_const(34)],
                vec![from_const(56), from_const(12)],
            ],
            parse("0xfb5a002c9ef7dad6d9b4edc690f323f4c302c665926c085dd5d8681496c937c6"),
        );
        test_merkle_tree::<GL, GL4, Sha2Hash<GL4>>(
            vec![
                vec![from_const(78), from_const(34)],
                vec![from_const(56), from_const(12)],
            ],
            parse("0x3674569ea14d34d8615c8e0ba390af4f72e5510b50ad933bad3139ecaf9a8e1f"),
        );
        test_merkle_tree::<BS, BS, Keccak256Hash<BS>>(
            vec![
                vec![from_const(78), from_const(34)],
                vec![from_const(56), from_const(12)],
            ],
            parse("0xb18e697b08221e2f411826d3c17c80bff6e06e90a55bcd5cc191ad67f1eb657a"),
        );
        test_merkle_tree::<GL, GL4, Keccak256Hash<GL4>>(
            vec![
                vec![from_const(78), from_const(34)],
                vec![from_const(56), from_const(12)],
            ],
            parse("0xd3f639689d31146178166f434b1b6a0dd6b139a371b310946bee5b8fa0c8b0c0"),
        );
    }
}
