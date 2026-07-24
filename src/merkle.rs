use crate::hash::{Hash, HashBackend};
use crate::utils;
use anyhow::{Result, anyhow};
use primitive_types::H256;
use starkom_bluesky::Scalar;
use std::marker::PhantomData;
use std::sync::LazyLock;

/// Domain separator tag used when hashing the leaves of a Merkle tree.
static LEAF_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/merkle/leaf"));

/// Hashes a leaf of a Merkle tree.
///
/// Our Merkle trees have vectors of values as leaves (there's one element for every committed
/// polynomial so that we can commit multiple polynomials into the same tree), so the input `values`
/// parameter is a slice of scalar values.
fn hash_leaf<H: Hash<Scalar>>(values: &[Scalar]) -> H256 {
    H::hash_many(
        std::iter::once(*LEAF_DST)
            .chain(std::iter::once(Scalar::from(values.len() as u64)))
            .chain(values.iter().cloned()),
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
pub(crate) fn merklify<H: Hash<H256>>(mut hashes: &mut [H256], mut n: usize) {
    assert!(n.is_power_of_two());
    while n > 1 {
        let m = n / 2;
        for j in 0..m {
            hashes[n + j] = H::hash_two(hashes[j * 2], hashes[j * 2 + 1]);
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
pub(crate) struct Proof<H: HashBackend<Scalar>> {
    leaf: Vec<Scalar>,
    path: Vec<H256>,
    _data: PhantomData<H>,
}

impl<H: HashBackend<Scalar>> Proof<H> {
    /// Returns a reference to the leaf values (one for every committed polynomial).
    pub(crate) fn leaf(&self) -> &[Scalar] {
        self.leaf.as_slice()
    }

    /// Checks the leaf of this proof against the provided slice.
    ///
    /// The two must match or an error is returned.
    pub(crate) fn check_leaf(&self, expected: &[Scalar]) -> Result<()> {
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
        let mut hash = hash_leaf::<H>(self.leaf.as_slice());
        for &sibling in &self.path {
            hash = if index & 1 != 0 {
                H::hash_two(sibling, hash)
            } else {
                H::hash_two(hash, sibling)
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
        let mut hash = hash_leaf::<H>(self.leaf.as_slice());
        for &sibling in &self.path {
            if sibling != hash {
                return Ok(false);
            }
            hash = H::hash_two(hash, hash);
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
pub(crate) struct Tree<H: HashBackend<Scalar>> {
    /// The polynomial evaluations committed in the tree. The outer array has K entries, one for
    /// every committed polynomial, and the inner array has N entries, one for every evaluation of a
    /// polynomial.
    leaves: Vec<Vec<Scalar>>,
    /// The internal nodes of the tree. There are 2*N-1 nodes in this array, with N = number of
    /// leaves. The nodes of the bottom layer are the hashes of the corresponding leaves.
    hashes: Vec<H256>,
    _data: PhantomData<H>,
}

impl<H: HashBackend<Scalar>> Tree<H> {
    /// Constructs a Merkle tree from a matrix of polynomial evaluations.
    ///
    /// The outer array of `polynomials` contains one entry per committed polynomial, and each of
    /// the inner arrays represents the evaluations of a polynomial.
    ///
    /// The outer array must have at least 1 element (at least 1 polynomial must be committed) and
    /// all inner arrays must have the same size N which must be the size of the (extended)
    /// evaluation domain (always a power of 2).
    ///
    /// Neither the outer array nor the inner array can be empty.
    pub(crate) fn new(polynomials: Vec<Vec<Scalar>>) -> Self {
        let num_polys = polynomials.len();
        assert!(num_polys > 0);
        let n = polynomials[0].len();
        assert!(n.is_power_of_two());
        let mut hashes = vec![H256::default(); n * 2 - 1];
        for i in 0..n {
            hashes[i] = hash_leaf::<H>(
                polynomials
                    .iter()
                    .map(|polynomial| polynomial[i])
                    .collect::<Vec<Scalar>>()
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
    pub(crate) fn leaf(&self, index: usize) -> Vec<Scalar> {
        self.leaves.iter().map(|values| values[index]).collect()
    }

    /// Returns the value at a given leaf for a specific polynomial.
    ///
    /// `polynomial_index` must be less than [`Self::num_polys()`] and `leaf_index` must be less
    /// than [`Self::num_leaves()`].
    pub(crate) fn leaf_value(&self, polynomial_index: usize, leaf_index: usize) -> Scalar {
        self.leaves[polynomial_index][leaf_index]
    }

    /// Returns a Merkle proof for the leaf at `index`.
    pub(crate) fn query(&self, mut index: usize) -> Proof<H> {
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
    use crate::hash;
    use starkom_bluesky::from_const;

    type Poseidon2Hash = hash::Poseidon2Hash<Scalar>;
    type Sha2Hash = hash::Sha2Hash<Scalar>;

    fn parse_hash(s: &'static str) -> H256 {
        s.parse().unwrap()
    }

    #[test]
    fn test_merklify_one_sha2() {
        let mut hashes = vec![parse_hash(
            "0x1fe9e33b7ff790473e13eb6384d61d6abb32d2e50b30a382c9767b5347f5846b",
        )];
        merklify::<Sha2Hash>(&mut hashes, 1);
        assert_eq!(
            hashes,
            vec![parse_hash(
                "0x1fe9e33b7ff790473e13eb6384d61d6abb32d2e50b30a382c9767b5347f5846b"
            )]
        );
    }

    #[test]
    fn test_merklify_one_poseidon2() {
        let mut hashes = vec![parse_hash(
            "0x1fe9e33b7ff790473e13eb6384d61d6abb32d2e50b30a382c9767b5347f5846b",
        )];
        merklify::<Poseidon2Hash>(&mut hashes, 1);
        assert_eq!(
            hashes,
            vec![parse_hash(
                "0x1fe9e33b7ff790473e13eb6384d61d6abb32d2e50b30a382c9767b5347f5846b"
            )]
        );
    }

    #[test]
    fn test_merklify_two_sha2() {
        let mut hashes = vec![
            parse_hash("0x0cbb1061e55efef40fae3c8e34d301c9940889146f816edd475052fc45caf060"),
            parse_hash("0x2d8257d72c4b6e6bfa3a21a22c053d02d1d52b4b5e6097acfd0ed45f827d6ba4"),
        ];
        hashes.resize(3, H256::default());
        merklify::<Sha2Hash>(&mut hashes, 2);
        assert_eq!(
            hashes,
            vec![
                parse_hash("0x0cbb1061e55efef40fae3c8e34d301c9940889146f816edd475052fc45caf060"),
                parse_hash("0x2d8257d72c4b6e6bfa3a21a22c053d02d1d52b4b5e6097acfd0ed45f827d6ba4"),
                parse_hash("0x3af448b6612ac9931b7413ecca5d209187cd808ae2d3647099aa99fada16c955")
            ]
        );
    }

    #[test]
    fn test_merklify_two_poseidon2() {
        let mut hashes = vec![
            parse_hash("0x0cbb1061e55efef40fae3c8e34d301c9940889146f816edd475052fc45caf060"),
            parse_hash("0x2d8257d72c4b6e6bfa3a21a22c053d02d1d52b4b5e6097acfd0ed45f827d6ba4"),
        ];
        hashes.resize(3, H256::default());
        merklify::<Poseidon2Hash>(&mut hashes, 2);
        assert_eq!(
            hashes,
            vec![
                parse_hash("0x0cbb1061e55efef40fae3c8e34d301c9940889146f816edd475052fc45caf060"),
                parse_hash("0x2d8257d72c4b6e6bfa3a21a22c053d02d1d52b4b5e6097acfd0ed45f827d6ba4"),
                parse_hash("0x3ff95fb9aa3d97ddb2483415494484499bf4f06b1dfdb5e74e0475c2f6c0c6c8")
            ]
        );
    }

    #[test]
    fn test_merklify_four_sha2() {
        let mut hashes = vec![
            parse_hash("0x034acf2cced8e9744784ec9c3c626fa83f6f8ddd83faed9d5caa5ad72c91eb1c"),
            parse_hash("0x39dce95a2271a57f999981eef6917f3df8ad25d116c98d809a3b7da5d54805a3"),
            parse_hash("0x435246f701f1483adcb7037fa64b3fa027c41e13a52d1e6c502e9b71dea3ca81"),
            parse_hash("0x7952ed41c4062f594b53b8c548874b41a7a7f05593d7a5a313ed82c2fe1c62d7"),
        ];
        hashes.resize(7, H256::default());
        merklify::<Sha2Hash>(&mut hashes, 4);
        assert_eq!(
            hashes,
            vec![
                parse_hash("0x034acf2cced8e9744784ec9c3c626fa83f6f8ddd83faed9d5caa5ad72c91eb1c"),
                parse_hash("0x39dce95a2271a57f999981eef6917f3df8ad25d116c98d809a3b7da5d54805a3"),
                parse_hash("0x435246f701f1483adcb7037fa64b3fa027c41e13a52d1e6c502e9b71dea3ca81"),
                parse_hash("0x7952ed41c4062f594b53b8c548874b41a7a7f05593d7a5a313ed82c2fe1c62d7"),
                parse_hash("0x3d139c65859969e86db444ccc0d36ffd0456b3f92d3dfb34fa10c8c07f5ada06"),
                parse_hash("0xdc3dd313db4b53f2720fcfe6373d286a9c40911546e1c434fa24ab650a75b586"),
                parse_hash("0xed171c6bd6876b93f4fc8914e0b2a8db53d7204048ad9fdff6e7127c077ca072"),
            ]
        );
    }

    #[test]
    fn test_merklify_four_poseidon2() {
        let mut hashes = vec![
            parse_hash("0x034acf2cced8e9744784ec9c3c626fa83f6f8ddd83faed9d5caa5ad72c91eb1c"),
            parse_hash("0x39dce95a2271a57f999981eef6917f3df8ad25d116c98d809a3b7da5d54805a3"),
            parse_hash("0x435246f701f1483adcb7037fa64b3fa027c41e13a52d1e6c502e9b71dea3ca81"),
            parse_hash("0x7952ed41c4062f594b53b8c548874b41a7a7f05593d7a5a313ed82c2fe1c62d7"),
        ];
        hashes.resize(7, H256::default());
        merklify::<Poseidon2Hash>(&mut hashes, 4);
        assert_eq!(
            hashes,
            vec![
                parse_hash("0x034acf2cced8e9744784ec9c3c626fa83f6f8ddd83faed9d5caa5ad72c91eb1c"),
                parse_hash("0x39dce95a2271a57f999981eef6917f3df8ad25d116c98d809a3b7da5d54805a3"),
                parse_hash("0x435246f701f1483adcb7037fa64b3fa027c41e13a52d1e6c502e9b71dea3ca81"),
                parse_hash("0x7952ed41c4062f594b53b8c548874b41a7a7f05593d7a5a313ed82c2fe1c62d7"),
                parse_hash("0x5d405b8c4875e7509085d43c8130b10a3b034b88e9f3c413a0a4460be077858b"),
                parse_hash("0x0346e1fa000ed1aa3dbde3cef3996f9a532e4f4486dd72bfbfd836a9c6079d90"),
                parse_hash("0x1681bb07015f5c6413ee62486871b6c95feb5ad89f62638998f6e599fa11193d"),
            ]
        );
    }

    fn test_merkle_tree<H: HashBackend<Scalar>>(
        evaluations: Vec<Vec<Scalar>>,
        expected_root_hash: H256,
    ) {
        let k = evaluations.len();
        let n = evaluations[0].len();
        let tree = Tree::<H>::new(evaluations.clone());
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
        test_merkle_tree::<Sha2Hash>(
            vec![vec![from_const(12)]],
            parse_hash("0x0bd187bc3deea1ef6c2a9ae254cf4e493f1dbbda32c79a662fc1d8437ab7e7c6"),
        );
        test_merkle_tree::<Poseidon2Hash>(
            vec![vec![from_const(12)]],
            parse_hash("0x7e1fffed4d53ef893858a7345de37e9b366f6a52961fd9722f2104217758aa2a"),
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_2() {
        test_merkle_tree::<Sha2Hash>(
            vec![vec![from_const(34)]],
            parse_hash("0x825f71a1d38bedb88129450457f7943f988ee940aa1755aa89982e139e67047a"),
        );
        test_merkle_tree::<Poseidon2Hash>(
            vec![vec![from_const(34)]],
            parse_hash("0x1fa5228af05e75ef806f5772be53106e0d91f54950d60a4759421f77a0b39095"),
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_two_polynomials_1() {
        test_merkle_tree::<Sha2Hash>(
            vec![vec![from_const(12)], vec![from_const(34)]],
            parse_hash("0x41c90ef8e7fa7e79e54b14cf8395f9707e9768a02aa79ce7f0f4e68668837c26"),
        );
        test_merkle_tree::<Poseidon2Hash>(
            vec![vec![from_const(12)], vec![from_const(34)]],
            parse_hash("0x1c6b73bec8f42fe634f5f77d8741ace13ff5ff49105b4c3e69d54c55ddc56dc7"),
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_two_polynomials_2() {
        test_merkle_tree::<Sha2Hash>(
            vec![vec![from_const(34)], vec![from_const(12)]],
            parse_hash("0xc3dbf8cc67db17f01dd3527ab16da2120af998c7ebe3d06a2d9dff445e1adaee"),
        );
        test_merkle_tree::<Poseidon2Hash>(
            vec![vec![from_const(34)], vec![from_const(12)]],
            parse_hash("0x62f2c10ac158f290a9178feec675a0cce5c8c9610c393f4bf17c7838760ebe19"),
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_three_polynomials_1() {
        test_merkle_tree::<Sha2Hash>(
            vec![
                vec![from_const(12)],
                vec![from_const(34)],
                vec![from_const(56)],
            ],
            parse_hash("0x9bca77d625d9a50f1807d27c273cae980f706ef734a014c00e6a97bbb77472e3"),
        );
        test_merkle_tree::<Poseidon2Hash>(
            vec![
                vec![from_const(12)],
                vec![from_const(34)],
                vec![from_const(56)],
            ],
            parse_hash("0x363f810134655fa478c1fbc55495b9b810f0a33ba3aa87b8d43ac6f1c3f2ed75"),
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_three_polynomials_2() {
        test_merkle_tree::<Sha2Hash>(
            vec![
                vec![from_const(34)],
                vec![from_const(12)],
                vec![from_const(78)],
            ],
            parse_hash("0xf48317b0be7caae4a15185cc8d0795c15c2bc98e18b8cbdc906270ada921fd25"),
        );
        test_merkle_tree::<Poseidon2Hash>(
            vec![
                vec![from_const(34)],
                vec![from_const(12)],
                vec![from_const(78)],
            ],
            parse_hash("0x6906a253c3b386890be44e5fbd6c6b613e73c8960a04ef390f6fc6a584cfcaa1"),
        );
    }

    #[test]
    fn test_merkle_tree_two_leaves_1() {
        test_merkle_tree::<Sha2Hash>(
            vec![vec![from_const(12), from_const(34)]],
            parse_hash("0x2624006228d517eeda393d1440f25ed1c20887664f2444021849345167aadaf4"),
        );
        test_merkle_tree::<Poseidon2Hash>(
            vec![vec![from_const(12), from_const(34)]],
            parse_hash("0x04df34feebaccfe8d6611b5d308726e8c27e94f2d3f3dfc9e68d39f917125a49"),
        );
    }

    #[test]
    fn test_merkle_tree_two_leaves_2() {
        test_merkle_tree::<Sha2Hash>(
            vec![vec![from_const(34), from_const(56)]],
            parse_hash("0x85d6170f1dcec468c3a42f35d24b7689733abf26ff0278a9bc1edc7a9a0a7333"),
        );
        test_merkle_tree::<Poseidon2Hash>(
            vec![vec![from_const(34), from_const(56)]],
            parse_hash("0x623c0f527bde2b6c6a152c224df77a82bd1957188908965c42d167554dc75318"),
        );
    }

    #[test]
    fn test_merkle_tree_two_leaves_two_polynomials_1() {
        test_merkle_tree::<Sha2Hash>(
            vec![
                vec![from_const(12), from_const(56)],
                vec![from_const(34), from_const(78)],
            ],
            parse_hash("0xdc144d55dd7a9c48f00b495d30172b38db7d4dc71ee4a0feab99177e30a10d21"),
        );
        test_merkle_tree::<Poseidon2Hash>(
            vec![
                vec![from_const(12), from_const(56)],
                vec![from_const(34), from_const(78)],
            ],
            parse_hash("0x290f8ab7fabbbf6bf79e435e2900f0eb619cfb1c3933c3dca13f89f3abd4b9de"),
        );
    }

    #[test]
    fn test_merkle_tree_two_leaves_two_polynomials_2() {
        test_merkle_tree::<Sha2Hash>(
            vec![
                vec![from_const(78), from_const(34)],
                vec![from_const(56), from_const(12)],
            ],
            parse_hash("0xfb5a002c9ef7dad6d9b4edc690f323f4c302c665926c085dd5d8681496c937c6"),
        );
        test_merkle_tree::<Poseidon2Hash>(
            vec![
                vec![from_const(78), from_const(34)],
                vec![from_const(56), from_const(12)],
            ],
            parse_hash("0x494da394aee5068da56b511142feb6ca060b4f9dddbeab2c7d74d2b5d58c6813"),
        );
    }
}
