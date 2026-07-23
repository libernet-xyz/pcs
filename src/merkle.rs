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

    /// Indicates whether or not the committed polynomials are constant.
    ///
    /// This is used in low degree testing to check when the folding process collapses to degree-0
    /// polynomials.
    ///
    /// Note that some polynomials may collapse earlier than others, and this function returns false
    /// if one or more haven't collapsed yet. So it returns true if and only if all have collapsed.
    pub(crate) fn is_constant(&self) -> bool {
        let mut hash = hash_leaf::<H>(self.leaf.as_slice());
        for &sibling in &self.path {
            if sibling != hash {
                return false;
            }
            hash = H::hash_two(hash, hash);
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
    use starkom_bluesky::{from_const, parse_scalar};
    use starkom_ff::Field;

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

    // fn test_merkle_tree<H: Hash<Scalar>>(
    //     evaluations: Vec<Vec<Scalar>>,
    //     expected_root_hash: Scalar,
    // ) {
    //     let k = evaluations.len();
    //     let n = evaluations[0].len();
    //     let tree = Tree::<H>::new(evaluations.clone());
    //     assert_eq!(tree.num_polys(), k);
    //     assert_eq!(tree.num_leaves(), n);
    //     assert_eq!(tree.root_hash(), expected_root_hash);
    //     for i in 0..n {
    //         let proof = tree.query(i);
    //         assert!(proof.verify(i, expected_root_hash).is_ok());
    //         assert_eq!(proof.leaf().len(), k);
    //         assert!(
    //             proof
    //                 .leaf()
    //                 .iter()
    //                 .zip(evaluations.iter())
    //                 .all(|(&lhs, values)| lhs == values[i])
    //         );
    //         for j in 0..k {
    //             assert_eq!(tree.leaf_value(j, i), evaluations[j][i]);
    //         }
    //     }
    // }

    // #[test]
    // fn test_merkle_tree_one_leaf_1() {
    //     test_merkle_tree::<Sha2Hash>(
    //         vec![vec![from_const(12)]],
    //         parse_scalar("0x71169269b911f4d1c8edbd4ba3e4107b1f1017f53d21a202c6d9865d4a95cdf6"),
    //     );
    //     test_merkle_tree::<Poseidon2Hash>(
    //         vec![vec![from_const(12)]],
    //         parse_scalar("0x7e1fffed4d53ef893858a7345de37e9b366f6a52961fd9722f2104217758aa2a"),
    //     );
    // }

    // #[test]
    // fn test_merkle_tree_one_leaf_2() {
    //     test_merkle_tree::<Sha2Hash>(
    //         vec![vec![from_const(34)]],
    //         parse_scalar("0x7c4c892249a33dccdc42c2b7002a98c54420f272f63fa886192c94944aae3374"),
    //     );
    //     test_merkle_tree::<Poseidon2Hash>(
    //         vec![vec![from_const(34)]],
    //         parse_scalar("0x1fa5228af05e75ef806f5772be53106e0d91f54950d60a4759421f77a0b39095"),
    //     );
    // }

    // #[test]
    // fn test_merkle_tree_one_leaf_two_polynomials_1() {
    //     test_merkle_tree::<Sha2Hash>(
    //         vec![vec![from_const(12)], vec![from_const(34)]],
    //         parse_scalar("0x2abfa190025b3afd41194aa282634e633b5ffed43d522fc2d87d269dd0eb60a8"),
    //     );
    //     test_merkle_tree::<Poseidon2Hash>(
    //         vec![vec![from_const(12)], vec![from_const(34)]],
    //         parse_scalar("0x1c6b73bec8f42fe634f5f77d8741ace13ff5ff49105b4c3e69d54c55ddc56dc7"),
    //     );
    // }

    // #[test]
    // fn test_merkle_tree_one_leaf_two_polynomials_2() {
    //     test_merkle_tree::<Sha2Hash>(
    //         vec![vec![from_const(34)], vec![from_const(12)]],
    //         parse_scalar("0x48147ad111986cc8bb753677fb9c9864db281bd55443ea93be2533e22b7d0112"),
    //     );
    //     test_merkle_tree::<Poseidon2Hash>(
    //         vec![vec![from_const(34)], vec![from_const(12)]],
    //         parse_scalar("0x62f2c10ac158f290a9178feec675a0cce5c8c9610c393f4bf17c7838760ebe19"),
    //     );
    // }

    // #[test]
    // fn test_merkle_tree_one_leaf_three_polynomials_1() {
    //     test_merkle_tree::<Sha2Hash>(
    //         vec![
    //             vec![from_const(12)],
    //             vec![from_const(34)],
    //             vec![from_const(56)],
    //         ],
    //         parse_scalar("0x12ce7e1aefd29256e5dfb6c1dcfed5c959cf4346e64d3b4a9c90671fb3bf7e3f"),
    //     );
    //     test_merkle_tree::<Poseidon2Hash>(
    //         vec![
    //             vec![from_const(12)],
    //             vec![from_const(34)],
    //             vec![from_const(56)],
    //         ],
    //         parse_scalar("0x363f810134655fa478c1fbc55495b9b810f0a33ba3aa87b8d43ac6f1c3f2ed75"),
    //     );
    // }

    // #[test]
    // fn test_merkle_tree_one_leaf_three_polynomials_2() {
    //     test_merkle_tree::<Sha2Hash>(
    //         vec![
    //             vec![from_const(34)],
    //             vec![from_const(12)],
    //             vec![from_const(78)],
    //         ],
    //         parse_scalar("0x0d19f881e2cf998e3d1fc3095b741b2b2086833550530c4e394578c94eeae3f2"),
    //     );
    //     test_merkle_tree::<Poseidon2Hash>(
    //         vec![
    //             vec![from_const(34)],
    //             vec![from_const(12)],
    //             vec![from_const(78)],
    //         ],
    //         parse_scalar("0x6906a253c3b386890be44e5fbd6c6b613e73c8960a04ef390f6fc6a584cfcaa1"),
    //     );
    // }

    // #[test]
    // fn test_merkle_tree_two_leaves_1() {
    //     test_merkle_tree::<Sha2Hash>(
    //         vec![vec![from_const(12), from_const(34)]],
    //         parse_scalar("0x661f00deed4778c0f35e3b46a50a56d52cfc628534ab4bf7526fa2b7eb64fb2d"),
    //     );
    //     test_merkle_tree::<Poseidon2Hash>(
    //         vec![vec![from_const(12), from_const(34)]],
    //         parse_scalar("0x76aafc5a8995c9d435039d454a33a488898f6ef0b2622ee5b003ec8c5aa3332b"),
    //     );
    // }

    // #[test]
    // fn test_merkle_tree_two_leaves_2() {
    //     test_merkle_tree::<Sha2Hash>(
    //         vec![vec![from_const(34), from_const(56)]],
    //         parse_scalar("0x6299bd932dd8ada9a15540b13b5ce037d9ec4c015e4182fbc05a6e3d4242b621"),
    //     );
    //     test_merkle_tree::<Poseidon2Hash>(
    //         vec![vec![from_const(34), from_const(56)]],
    //         parse_scalar("0x39635888b223cf2fa7fd3584790e86e5380da23281f53710669cc72345d0a935"),
    //     );
    // }

    // #[test]
    // fn test_merkle_tree_two_leaves_two_polynomials_1() {
    //     test_merkle_tree::<Sha2Hash>(
    //         vec![
    //             vec![from_const(12), from_const(56)],
    //             vec![from_const(34), from_const(78)],
    //         ],
    //         parse_scalar("0x38ff40459f948d6f52e0df0e1cb2f4e4216e21765339d0c2abece6dd86b4e606"),
    //     );
    //     test_merkle_tree::<Poseidon2Hash>(
    //         vec![
    //             vec![from_const(12), from_const(56)],
    //             vec![from_const(34), from_const(78)],
    //         ],
    //         parse_scalar("0x5d494ed90f85da6cfef7581f8134f0bb454b423a69a1e5b79aa1dd077ef918f1"),
    //     );
    // }

    // #[test]
    // fn test_merkle_tree_two_leaves_two_polynomials_2() {
    //     test_merkle_tree::<Sha2Hash>(
    //         vec![
    //             vec![from_const(78), from_const(34)],
    //             vec![from_const(56), from_const(12)],
    //         ],
    //         parse_scalar("0x7af2c224e5681096e6e26311a34657e10dcd6d0fb2343ceca285da0f5d445150"),
    //     );
    //     test_merkle_tree::<Poseidon2Hash>(
    //         vec![
    //             vec![from_const(78), from_const(34)],
    //             vec![from_const(56), from_const(12)],
    //         ],
    //         parse_scalar("0x3f01eb2c0ebff44e6ab56b513563247916785675f145580bfae842ddbe01c2fa"),
    //     );
    // }
}
