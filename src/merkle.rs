use crate::hash::{Hasher, MerkleHasher};
use crate::starkom::proto::pcs::v10 as proto;
use crate::utils;
use anyhow::{Result, anyhow};
use primitive_types::{H256, U256, U512};
use sha2::Digest;
use starkom_ff::{Field, Field256};
use std::any::TypeId;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::{LazyLock, Mutex};

/// String from which the DST of the leaves of polynomials over the extension field is derived.
const LEAF_DST_ID: &[u8] = b"starkom/merkle/leaf";

/// String from which the DST of the leaves of polynomials over the base field is derived.
const BASE_LEAF_DST_ID: &[u8] = b"starkom/merkle/leaf/base";

/// Derives a DST from `id`: the SHA3-512 hash of the string, reduced modulo `modulus`.
///
/// `id` is either [`LEAF_DST_ID`] or [`BASE_LEAF_DST_ID`].
fn make_leaf_dst(id: &[u8], modulus: U512) -> U256 {
    let mut hasher = sha3::Sha3_512::new();
    hasher.update(id);
    let hash: U512 = U512::from_little_endian(hasher.finalize().as_slice());
    let value = hash % modulus;
    U256::from_little_endian(&value.to_little_endian()[0..32])
}

/// Returns the DST derived from `id` (see [`make_leaf_dst`]) as an element of `F`, expressed as its
/// coordinates over `F::BaseField`.
fn get_leaf_dst<F: Field256>(id: &'static [u8]) -> Vec<F::BaseField> {
    // TODO: this is a global mutex on a hot path. This map is add-only, so we should really switch
    // to a lock-free map.
    static DST_CACHE: LazyLock<Mutex<BTreeMap<(TypeId, &'static [u8]), U256>>> =
        LazyLock::new(|| Mutex::default());
    let value = {
        let mut cache = DST_CACHE.lock().unwrap();
        *cache
            .entry((TypeId::of::<F>(), id))
            .or_insert_with(|| make_leaf_dst(id, F::MODULUS.parse().unwrap()))
    };
    let dst = F::try_from_le_bytes(&value.to_little_endian()[0..(F::LEN)]).unwrap();
    utils::flatten(&[dst])
}

/// Hashes the leaves of a Merkle tree, whose values are always base field elements (see [`Tree`]).
///
/// There is one implementation for every kind of tree: [`WideLeafHasher`] for trees committing
/// polynomials over `F` and [`BaseLeafHasher`] for trees committing polynomials over
/// `F::BaseField`. They use different DSTs, so the root of a tree also determines the field of its
/// polynomials.
trait LeafHasher<F: Field256, H: Hasher<F>> {
    /// Hashes the values of a leaf.
    fn hash(values: &[F::BaseField]) -> H256;
}

/// Hashes a leaf of a Merkle tree committing polynomials over the ~256-bit field `F`.
///
/// Our Merkle trees have vectors of values as leaves, so that we can commit multiple polynomials
/// into the same tree. The leaves are always made of base field values, so `coordinates` holds the
/// evaluations of the committed polynomials [flattened](`utils::flatten`) into their coordinates
/// over `F::BaseField`: `T` values for every polynomial, `T` being the extension degree of `F`.
///
/// The hashed message is made of the leaf DST derived from [`LEAF_DST_ID`], the number of
/// polynomials, and the coordinates. Leaves of polynomials over `F::BaseField` are hashed by
/// [`BaseLeafHasher`] with a different DST, so the root of a tree also determines the field of its
/// polynomials.
///
/// REQUIRES: the number of coordinates must be a multiple of the extension degree of `F`.
#[derive(Debug, Default, Copy, Clone)]
struct WideLeafHasher<F: Field256, H: Hasher<F>> {
    _data: PhantomData<(F, H)>,
}

impl<F: Field256, H: Hasher<F>> LeafHasher<F, H> for WideLeafHasher<F, H> {
    fn hash(coordinates: &[F::BaseField]) -> H256 {
        assert!(coordinates.len().is_multiple_of(F::EXTENSION_DEGREE));
        let dst = get_leaf_dst::<F>(LEAF_DST_ID);
        let count = F::BaseField::try_from(coordinates.len() / F::EXTENSION_DEGREE).unwrap();
        H::hash(
            dst.into_iter()
                .chain(std::iter::once(count))
                .chain(coordinates.iter().copied()),
        )
    }
}

/// Hashes a leaf of a Merkle tree committing polynomials over the base field `F::BaseField`.
///
/// Same as [`WideLeafHasher`], but the values are elements of `F::BaseField` and the DST is derived
/// from [`BASE_LEAF_DST_ID`].
#[derive(Debug, Default, Copy, Clone)]
struct BaseLeafHasher<F: Field256, H: Hasher<F>> {
    _data: PhantomData<(F, H)>,
}

impl<F: Field256, H: Hasher<F>> LeafHasher<F, H> for BaseLeafHasher<F, H> {
    fn hash(values: &[F::BaseField]) -> H256 {
        let dst = get_leaf_dst::<F>(BASE_LEAF_DST_ID);
        let count = F::BaseField::try_from(values.len()).unwrap();
        H::hash(
            dst.into_iter()
                .chain(std::iter::once(count))
                .chain(values.iter().copied()),
        )
    }
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

/// Computes the Merkle root of a set of values on the fly, without allocating space for the
/// intermediate nodes.
///
/// The computed root is compatible with the one resulting from a call to [`merklify`] for the same
/// values.
///
/// REQUIRES: `values.len()` must be a power of two.
pub(crate) fn merkle_root<H: MerkleHasher>(values: &[H256]) -> H256 {
    assert!(values.len().is_power_of_two());
    match values.len() {
        1 => values[0],
        n => {
            let m = n >> 1;
            let left = merkle_root::<H>(&values[0..m]);
            let right = merkle_root::<H>(&values[m..]);
            H::hash_binary(left, right)
        }
    }
}

/// Returns the Merkle path of the leaf at `index`, that is the sibling hashes from the leaf up to,
/// but excluding, the root.
///
/// `tree` must contain a full Merkle tree with `n` leaves, stored as described in [`merklify`]. `n`
/// must be a power of two and `index` must be less than `n`.
pub(crate) fn merkle_path(mut tree: &[H256], mut n: usize, mut index: usize) -> Vec<H256> {
    assert!(n.is_power_of_two());
    assert!(index < n);
    let mut path = Vec::with_capacity(n.trailing_zeros() as usize);
    while n > 1 {
        path.push(tree[index ^ 1]);
        tree = &tree[n..];
        n /= 2;
        index >>= 1;
    }
    path
}

/// Recomputes a Merkle root from a leaf and its Merkle path, and checks it against `root_hash`.
///
/// `leaf` is the 32-byte leaf at position `index`: the hash of the leaf values for vector leaves
/// (see [`Proof`]), or the leaf itself for trees whose leaves are single 32-byte values, like the
/// FRI ones.
pub(crate) fn verify_path<H: MerkleHasher>(
    leaf: H256,
    path: &[H256],
    mut index: usize,
    root_hash: H256,
) -> Result<()> {
    let mut hash = leaf;
    for &sibling in path {
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

/// A Merkle proof.
///
/// The DEEP layer uses these to open the committed polynomial batches at the queried positions.
///
/// The leaf values are always base field elements (see [`Tree`]).
///
/// NOTE: this object only stores the sister hashes of the Merkle path and the opened leaf values,
/// it doesn't store the lookup key and the root hash anywhere because those pieces of information
/// are reconstructed separately during the verification of a whole [`Proof`](`crate::deep::Proof`).
/// In particular, all root hashes are stored in the [`Commitment`](`crate::deep::Commitment`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Proof<F: Field256, H: Hasher<F>> {
    leaf: Vec<F::BaseField>,
    path: Vec<H256>,
    _data: PhantomData<(F, H)>,
}

impl<F: Field256, H: Hasher<F>> Proof<F, H> {
    /// Returns a reference to the leaf values.
    pub(crate) fn leaf(&self) -> &[F::BaseField] {
        self.leaf.as_slice()
    }

    /// Returns the length of the Merkle path, corresponding to the height of the tree minus 1 (the
    /// root hash is not included in this count).
    pub(crate) fn len(&self) -> usize {
        self.path.len()
    }

    /// Verifies the proof against the root hash of a tree of polynomials over `F` (see
    /// [`Tree::new`]).
    ///
    /// Returns an error if the number of leaf values is not a multiple of the extension degree of
    /// `F`, i.e. if the leaf doesn't hold the coordinates of a whole number of evaluations.
    pub(crate) fn verify(&self, index: usize, root_hash: H256) -> Result<()> {
        if !self.leaf.len().is_multiple_of(F::EXTENSION_DEGREE) {
            return Err(anyhow!(
                "invalid leaf size {} (not a multiple of {})",
                self.leaf.len(),
                F::EXTENSION_DEGREE
            ));
        }
        let leaf_hash = WideLeafHasher::<F, H>::hash(self.leaf.as_slice());
        verify_path::<H>(leaf_hash, self.path.as_slice(), index, root_hash)
    }

    /// Verifies the proof against the root hash of a tree of polynomials over `F::BaseField` (see
    /// [`Tree::new_from_base`]).
    pub(crate) fn verify_base(&self, index: usize, root_hash: H256) -> Result<()> {
        let leaf_hash = BaseLeafHasher::<F, H>::hash(self.leaf.as_slice());
        verify_path::<H>(leaf_hash, self.path.as_slice(), index, root_hash)
    }

    /// Serializes this Merkle proof to a [`MerkleProof`](`proto::MerkleProof`) protobuf.
    pub(crate) fn to_proto(&self) -> proto::MerkleProof {
        proto::MerkleProof {
            leaf_values: self
                .leaf
                .iter()
                .map(|&value| utils::store_scalar(value))
                .collect(),
            merkle_path: self
                .path
                .iter()
                .map(|&hash| hash.as_bytes().to_vec())
                .collect(),
        }
    }

    /// Deserializes a Merkle proof from a [`MerkleProof`](`proto::MerkleProof`) protobuf.
    ///
    /// NOTE: this method does not validate the proof, it only deserializes it. The caller must
    /// invoke [`Self::verify`] separately.
    pub(crate) fn from_proto(proto: &proto::MerkleProof) -> Result<Self> {
        Ok(Self {
            leaf: proto
                .leaf_values
                .iter()
                .map(|value| utils::load_scalar(value.as_slice()))
                .collect::<Result<_>>()?,
            path: proto
                .merkle_path
                .iter()
                .map(|hash| H::load_hash(hash.as_slice()))
                .collect::<Result<_>>()?,
            _data: PhantomData,
        })
    }
}

/// A Merkle tree whose leaves are multiple polynomial evaluations.
///
/// The tree has N leaves in total, with N being the size of the extended domain, and its leaves
/// are always made of base field values: every committed polynomial over `F::BaseField` contributes
/// one column of evaluations, while every committed polynomial over `F` contributes `T` columns,
/// one for every coordinate of its evaluations over `F::BaseField` (`T` being the extension degree
/// of `F`). Trees of the two kinds are built by [`Self::new_from_base`] and [`Self::new`]
/// respectively, and hash their leaves with different DSTs (see [`BaseLeafHasher`] and
/// [`WideLeafHasher`]).
///
/// The internal nodes are single hashes.
#[derive(Debug, Clone)]
pub(crate) struct Tree<F: Field256, H: Hasher<F>> {
    /// The base field values committed in the tree. The outer array has one entry for every column,
    /// and every inner array has N entries.
    columns: Vec<Vec<F::BaseField>>,
    /// The internal nodes of the tree. There are 2*N-1 nodes in this array, with N = number of
    /// leaves. The nodes of the bottom layer are the hashes of the corresponding leaves.
    hashes: Vec<H256>,
    _data: PhantomData<(F, H)>,
}

impl<F: Field256, H: Hasher<F>> Tree<F, H> {
    /// Constructs a Merkle tree from a matrix of base field values, hashing every leaf with `K`.
    ///
    /// The outer array of `columns` contains one entry per column, and each of the inner arrays
    /// represents the values of a column at all points of the (extended) evaluation domain.
    ///
    /// The outer array must have at least 1 element and all inner arrays must have the same size N,
    /// which must be the size of the (extended) evaluation domain (always a power of 2).
    fn with_leaf_hasher<K: LeafHasher<F, H>>(columns: Vec<Vec<F::BaseField>>) -> Self {
        let num_columns = columns.len();
        assert!(num_columns > 0);
        let n = columns[0].len();
        assert!(n.is_power_of_two());
        assert!(columns.iter().all(|column| column.len() == n));
        let mut hashes = vec![H256::default(); n * 2 - 1];
        let mut leaf = Vec::with_capacity(num_columns);
        for i in 0..n {
            leaf.clear();
            leaf.extend(columns.iter().map(|column| column[i]));
            hashes[i] = K::hash(leaf.as_slice());
        }
        merklify::<H>(hashes.as_mut_slice(), n);
        Self {
            columns,
            hashes,
            _data: Default::default(),
        }
    }

    /// Constructs a Merkle tree committing polynomials over `F`.
    ///
    /// The outer array of `evaluations` contains one entry per committed polynomial, and each of
    /// the inner arrays represents the evaluations of a polynomial over the (extended) evaluation
    /// domain. Every polynomial contributes `T` columns to the tree, one for every coordinate of
    /// its evaluations over `F::BaseField`.
    ///
    /// The outer array must have at least 1 element (at least 1 polynomial must be committed) and
    /// all inner arrays must have the same size N, which must be the size of the (extended)
    /// evaluation domain (always a power of 2).
    pub(crate) fn new(evaluations: Vec<Vec<F>>) -> Self {
        let mut columns = Vec::with_capacity(evaluations.len() * F::EXTENSION_DEGREE);
        let mut coordinates = vec![F::BaseField::ZERO; F::EXTENSION_DEGREE];
        for values in evaluations {
            let start = columns.len();
            columns.resize(
                start + F::EXTENSION_DEGREE,
                vec![F::BaseField::ZERO; values.len()],
            );
            for (i, value) in values.iter().enumerate() {
                value.to_base_coordinates(coordinates.as_mut_slice());
                for (column, &coordinate) in columns[start..].iter_mut().zip(coordinates.iter()) {
                    column[i] = coordinate;
                }
            }
        }
        Self::with_leaf_hasher::<WideLeafHasher<F, H>>(columns)
    }

    /// Constructs a Merkle tree committing polynomials over `F::BaseField`.
    ///
    /// Same as [`Self::new`], but every polynomial contributes a single column of evaluations.
    pub(crate) fn new_from_base(evaluations: Vec<Vec<F::BaseField>>) -> Self {
        Self::with_leaf_hasher::<BaseLeafHasher<F, H>>(evaluations)
    }

    /// Returns the number of leaves in the tree, corresponding to the size of the evaluation domain
    /// (always a power of 2).
    pub(crate) fn num_leaves(&self) -> usize {
        self.columns[0].len()
    }

    /// Returns the root hash of the Merkle tree.
    pub(crate) fn root_hash(&self) -> H256 {
        let n = self.num_leaves();
        self.hashes[(n - 1) * 2]
    }

    /// Returns a vector representing the i-th leaf, with one value for every column.
    pub(crate) fn leaf(&self, index: usize) -> Vec<F::BaseField> {
        self.columns.iter().map(|column| column[index]).collect()
    }

    /// Returns a Merkle proof for the leaf at `index`.
    pub(crate) fn query(&self, index: usize) -> Proof<F, H> {
        Proof {
            leaf: self.leaf(index),
            path: merkle_path(self.hashes.as_slice(), self.num_leaves(), index),
            _data: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::{Keccak256Hash, Sha2Hash};
    use starkom_bluesky::Scalar as BS;
    use starkom_ff::Field;
    use starkom_goldilocks::{GL, GL4};
    use starkom_koalabear::{KB, KB8};
    use std::fmt::Debug;
    use std::str::FromStr;

    fn from_const<F: Field>(value: u16) -> F {
        F::from(value)
    }

    fn parse<V: FromStr<Err: Debug>>(s: &'static str) -> V {
        s.parse().unwrap()
    }

    /// Converts small integers into field elements.
    fn make_values<F: Field>(values: &[u16]) -> Vec<F> {
        values.iter().copied().map(from_const).collect()
    }

    /// Converts a matrix of small integers, one row for every polynomial, into field elements.
    fn make_columns<F: Field>(columns: &[&[u16]]) -> Vec<Vec<F>> {
        columns.iter().map(|column| make_values(column)).collect()
    }

    /// Builds a Merkle tree on the evaluations of polynomials over `F`, given as small integers.
    fn make_tree<F: Field256, H: Hasher<F>>(columns: &[&[u16]]) -> Tree<F, H> {
        Tree::new(make_columns(columns))
    }

    /// Builds a Merkle tree on the evaluations of polynomials over the base field of `F`.
    fn make_base_tree<F: Field256, H: Hasher<F>>(columns: &[&[u16]]) -> Tree<F, H> {
        Tree::new_from_base(make_columns(columns))
    }

    /// Eight distinct leaf hashes.
    fn sample_leaves() -> Vec<H256> {
        (1..=8).map(H256::from_low_u64_be).collect()
    }

    /// Builds a Merkle tree on the given leaf hashes with [`merklify`] and returns all its nodes.
    fn make_hashes(leaves: &[H256]) -> Vec<H256> {
        let mut hashes = leaves.to_vec();
        hashes.resize(leaves.len() * 2 - 1, H256::zero());
        merklify::<Sha2Hash<BS>>(hashes.as_mut_slice(), leaves.len());
        hashes
    }

    /// The evaluations of three polynomials over eight points.
    const SAMPLE_COLUMNS: [&[u16]; 3] = [
        &[12, 34, 56, 78, 90, 12, 34, 56],
        &[42, 43, 44, 45, 46, 47, 48, 49],
        &[78, 56, 34, 12, 90, 78, 56, 34],
    ];

    #[test]
    fn test_leaf_dst_bluesky() {
        assert_eq!(
            get_leaf_dst::<BS>(LEAF_DST_ID),
            [parse::<BS>(
                "0x08cb2652a56289bd316cfcb356f5d2be485538e04a601fb14fc2c98f03077fcb"
            )]
        );
    }

    #[test]
    fn test_leaf_dst_goldilocks() {
        assert_eq!(
            get_leaf_dst::<GL4>(LEAF_DST_ID),
            [
                parse::<GL>("0x9e8c852f6e39922a"),
                parse("0x854aa11e65b19103"),
                parse("0xf8b726cf174f0587"),
                parse("0x5c2a4bd05cc84e0a"),
            ]
        );
    }

    #[test]
    fn test_leaf_dst_koalabear() {
        assert_eq!(
            get_leaf_dst::<KB8>(LEAF_DST_ID),
            [
                parse::<KB>("0x1ce29ce4"),
                parse("0x083f4fb2"),
                parse("0x3831650c"),
                parse("0x1101aae2"),
                parse("0x2c778024"),
                parse("0x52908f48"),
                parse("0x225c25b5"),
                parse("0x11f8363a"),
            ]
        );
    }

    #[test]
    fn test_base_leaf_dst_bluesky() {
        assert_eq!(
            get_leaf_dst::<BS>(BASE_LEAF_DST_ID),
            [parse::<BS>(
                "0x2e704cc1e66fb23985c37d2b9530ed50c0e48ae8fa82f63fd25fe7f78bb23a89"
            )]
        );
    }

    #[test]
    fn test_base_leaf_dst_goldilocks() {
        assert_eq!(
            get_leaf_dst::<GL4>(BASE_LEAF_DST_ID),
            [
                parse::<GL>("0x67878f2864889a1b"),
                parse("0xc80c685a654cbf92"),
                parse("0xed6ab0d1fa155343"),
                parse("0x35fcd3c62358fd13"),
            ]
        );
    }

    #[test]
    fn test_base_leaf_dst_koalabear() {
        assert_eq!(
            get_leaf_dst::<KB8>(BASE_LEAF_DST_ID),
            [
                parse::<KB>("0x350fc794"),
                parse("0x778448b1"),
                parse("0x148beddc"),
                parse("0x4b1462fd"),
                parse("0x228868d2"),
                parse("0x58127a5a"),
                parse("0x5973bbe1"),
                parse("0x63d51f3b"),
            ]
        );
    }

    fn test_hash_leaf_impl<F: Field256, H: Hasher<F>>(values: &[u16], expected: &'static str) {
        let hash = WideLeafHasher::<F, H>::hash(
            utils::flatten(make_values::<F>(values).as_slice()).as_slice(),
        );
        assert_eq!(hash, parse::<H256>(expected));
    }

    #[test]
    fn test_hash_leaf_sha2_bluesky() {
        test_hash_leaf_impl::<BS, Sha2Hash<BS>>(
            &[12],
            "0x0bd187bc3deea1ef6c2a9ae254cf4e493f1dbbda32c79a662fc1d8437ab7e7c6",
        );
        test_hash_leaf_impl::<BS, Sha2Hash<BS>>(
            &[34, 56],
            "0xbcd7235ceb553ca6682fd2df813650cad056a20fd6c76e1e04ae9f6a248c6c02",
        );
        test_hash_leaf_impl::<BS, Sha2Hash<BS>>(
            &[78, 90, 12],
            "0xeb8b9e0099332552744b9111d5a478dc61b231407d6c776586a50a3fc4513ca4",
        );
    }

    #[test]
    fn test_hash_leaf_sha2_goldilocks() {
        test_hash_leaf_impl::<GL4, Sha2Hash<GL4>>(
            &[12],
            "0xe76b4fc031cdd6d43d4098d336c45f5bb5a83f739d7d404f59280c4449b3d2e4",
        );
        test_hash_leaf_impl::<GL4, Sha2Hash<GL4>>(
            &[34, 56],
            "0x51e312e8a9446539b1b09b6b6d25f68f5c6a177762cb15b8562881b3434299de",
        );
        test_hash_leaf_impl::<GL4, Sha2Hash<GL4>>(
            &[78, 90, 12],
            "0xa391abff2e848b8f03fb7262d112c9845760ee0ab638a8f976d766c1cad2a837",
        );
    }

    #[test]
    fn test_hash_leaf_keccak256_bluesky() {
        test_hash_leaf_impl::<BS, Keccak256Hash<BS>>(
            &[12],
            "0x5e70242e081756f445b5f3048611464568002e68800238dcfef718504de01782",
        );
        test_hash_leaf_impl::<BS, Keccak256Hash<BS>>(
            &[34, 56],
            "0x6cbf9a3af9793caffe8c509dfaccf0198e833cafa8eb83a0a7d57ce35a97dbd1",
        );
        test_hash_leaf_impl::<BS, Keccak256Hash<BS>>(
            &[78, 90, 12],
            "0x05ce2ee03a570540c8a67a9b5a419dc7813922da6a39dff3c293806ed0f88fbb",
        );
    }

    #[test]
    fn test_hash_leaf_keccak256_goldilocks() {
        test_hash_leaf_impl::<GL4, Keccak256Hash<GL4>>(
            &[12],
            "0xfa15cca7ea7fff7366dd498ef239d82384f949e9f7861602fda4630c9a5765ce",
        );
        test_hash_leaf_impl::<GL4, Keccak256Hash<GL4>>(
            &[34, 56],
            "0x9511cb37e43259b557f7ecd20708bf184d35581b3428eb2626ba393b191f1b80",
        );
        test_hash_leaf_impl::<GL4, Keccak256Hash<GL4>>(
            &[78, 90, 12],
            "0xd51cf0cbfd048cd6cd2dd2bc4796869477ac5095a8bf6da97cc590191645e2c7",
        );
    }

    #[test]
    fn test_hash_leaf_with_all_coordinates() {
        let values = [
            GL4::from_base_coordinates(&[12u32, 34, 56, 78].map(GL::from)),
            GL4::from_base_coordinates(&[90u32, 12, 34, 56].map(GL::from)),
        ];
        let hash = WideLeafHasher::<GL4, Sha2Hash<GL4>>::hash(utils::flatten(&values).as_slice());
        assert_eq!(
            hash,
            parse::<H256>("0x78e0c20c431bd9f3ce8ff41dbe56034997063af9d0f3e0e12e359bb2f9dc8e6e")
        );
    }

    fn test_hash_leaf_base_impl<F: Field256, H: Hasher<F>>(values: &[u16], expected: &'static str) {
        let hash = BaseLeafHasher::<F, H>::hash(make_values(values).as_slice());
        assert_eq!(hash, parse::<H256>(expected));
    }

    #[test]
    fn test_hash_leaf_base_sha2_bluesky() {
        test_hash_leaf_base_impl::<BS, Sha2Hash<BS>>(
            &[12],
            "0xe170d928a5e6c38fbd80d58fb3a1eb5f051dddee37f4e4a3ef8b70f889b5100d",
        );
        test_hash_leaf_base_impl::<BS, Sha2Hash<BS>>(
            &[34, 56],
            "0x44b0c207defcf75d718b53720593cafdb46eba7a7b2a50c118bf4f419827fff3",
        );
        test_hash_leaf_base_impl::<BS, Sha2Hash<BS>>(
            &[78, 90, 12],
            "0x539cb1e01e609de3aef1f9776089174a545a34a83255c11576a0bf0b085d3342",
        );
    }

    #[test]
    fn test_hash_leaf_base_sha2_goldilocks() {
        test_hash_leaf_base_impl::<GL4, Sha2Hash<GL4>>(
            &[12],
            "0x0f878699196975569fc3a7974a1611eeca46acd83050c8fb5a7f8b1f53956118",
        );
        test_hash_leaf_base_impl::<GL4, Sha2Hash<GL4>>(
            &[34, 56],
            "0x65af2f1091844f52f6c15885c7186af130ecf97ca54675bdb0f198575e9a3487",
        );
        test_hash_leaf_base_impl::<GL4, Sha2Hash<GL4>>(
            &[78, 90, 12],
            "0xbddb7cb1a913d934753489612acb2a70a498a947d4431d171b3ec25cd87b3326",
        );
    }

    #[test]
    fn test_hash_leaf_base_sha2_koalabear() {
        test_hash_leaf_base_impl::<KB8, Sha2Hash<KB8>>(
            &[12],
            "0x847d11f0d678f1d7dedba79be190a0719cfcdb063ce8d1221aa583465650ff41",
        );
        test_hash_leaf_base_impl::<KB8, Sha2Hash<KB8>>(
            &[34, 56],
            "0x877bf1f6fc9b8b630897246f81261bd977a80dddfa1185d9b35e265f4d529b66",
        );
        test_hash_leaf_base_impl::<KB8, Sha2Hash<KB8>>(
            &[78, 90, 12],
            "0x5cda4a9deea64d42275fc841ddb8d4497d63130454dc3b9785d84890d9b50ec9",
        );
    }

    #[test]
    fn test_hash_leaf_base_keccak256_bluesky() {
        test_hash_leaf_base_impl::<BS, Keccak256Hash<BS>>(
            &[12],
            "0x7d288e2b2c9f5c4b911c5aca935d67e7d2bd5751e043d3b67d95f3b17e39fcf6",
        );
        test_hash_leaf_base_impl::<BS, Keccak256Hash<BS>>(
            &[34, 56],
            "0xd2764d0d6a75bd75b93e17960cf2e5028933c132484e869ad16a9d97c1535c15",
        );
        test_hash_leaf_base_impl::<BS, Keccak256Hash<BS>>(
            &[78, 90, 12],
            "0x84a5099e10a4f47788ba1d537dc6386127a0d270673e7a1df30dd1b18954e572",
        );
    }

    #[test]
    fn test_hash_leaf_base_keccak256_goldilocks() {
        test_hash_leaf_base_impl::<GL4, Keccak256Hash<GL4>>(
            &[12],
            "0x5690fa153fbf2eb9a1a7a01b6a63fe539d957c10edb62995eccba2a50ce017c5",
        );
        test_hash_leaf_base_impl::<GL4, Keccak256Hash<GL4>>(
            &[34, 56],
            "0x92eab4b89b20cb2943572aac9847829a2fe66bb18a0ce7be1ba0d59c42566c70",
        );
        test_hash_leaf_base_impl::<GL4, Keccak256Hash<GL4>>(
            &[78, 90, 12],
            "0x8656ab3c6c2f109ebe0503ba2fbfdc5351a731252b8fb69de4431d2096353b53",
        );
    }

    /// The base and extension leaf hashes differ even when the two fields coincide.
    #[test]
    fn test_base_and_extension_leaf_hashes_differ() {
        let values = make_values::<BS>(&[12, 34]);
        assert_ne!(
            WideLeafHasher::<BS, Sha2Hash<BS>>::hash(values.as_slice()),
            BaseLeafHasher::<BS, Sha2Hash<BS>>::hash(values.as_slice())
        );
    }

    fn test_merklify_impl<H: MerkleHasher>(
        leaves: &[&'static str],
        expected_internal_nodes: &[&'static str],
    ) {
        let n = leaves.len();
        let leaves: Vec<H256> = leaves.iter().map(|&leaf| parse(leaf)).collect();
        let expected_internal_nodes: Vec<H256> = expected_internal_nodes
            .iter()
            .map(|&node| parse(node))
            .collect();
        let mut hashes = leaves.clone();
        hashes.resize(n * 2 - 1, H256::zero());
        merklify::<H>(hashes.as_mut_slice(), n);
        assert_eq!(hashes[0..n], leaves);
        assert_eq!(hashes[n..], expected_internal_nodes);
    }

    #[test]
    fn test_merklify_one_leaf() {
        test_merklify_impl::<Sha2Hash<BS>>(
            &["0x1fe9e33b7ff790473e13eb6384d61d6abb32d2e50b30a382c9767b5347f5846b"],
            &[],
        );
        test_merklify_impl::<Keccak256Hash<BS>>(
            &["0x1fe9e33b7ff790473e13eb6384d61d6abb32d2e50b30a382c9767b5347f5846b"],
            &[],
        );
    }

    #[test]
    fn test_merklify_two_leaves() {
        test_merklify_impl::<Sha2Hash<BS>>(
            &[
                "0x0cbb1061e55efef40fae3c8e34d301c9940889146f816edd475052fc45caf060",
                "0x2d8257d72c4b6e6bfa3a21a22c053d02d1d52b4b5e6097acfd0ed45f827d6ba4",
            ],
            &["0x3af448b6612ac9931b7413ecca5d209187cd808ae2d3647099aa99fada16c955"],
        );
        test_merklify_impl::<Keccak256Hash<BS>>(
            &[
                "0x0cbb1061e55efef40fae3c8e34d301c9940889146f816edd475052fc45caf060",
                "0x2d8257d72c4b6e6bfa3a21a22c053d02d1d52b4b5e6097acfd0ed45f827d6ba4",
            ],
            &["0x6e70b19bd160e181bb367ab730eababc6e3673977f7fdce38934ee197bbcb568"],
        );
    }

    #[test]
    fn test_merklify_four_leaves() {
        test_merklify_impl::<Sha2Hash<BS>>(
            &[
                "0x034acf2cced8e9744784ec9c3c626fa83f6f8ddd83faed9d5caa5ad72c91eb1c",
                "0x39dce95a2271a57f999981eef6917f3df8ad25d116c98d809a3b7da5d54805a3",
                "0x435246f701f1483adcb7037fa64b3fa027c41e13a52d1e6c502e9b71dea3ca81",
                "0x7952ed41c4062f594b53b8c548874b41a7a7f05593d7a5a313ed82c2fe1c62d7",
            ],
            &[
                "0x3d139c65859969e86db444ccc0d36ffd0456b3f92d3dfb34fa10c8c07f5ada06",
                "0xdc3dd313db4b53f2720fcfe6373d286a9c40911546e1c434fa24ab650a75b586",
                "0xed171c6bd6876b93f4fc8914e0b2a8db53d7204048ad9fdff6e7127c077ca072",
            ],
        );
        test_merklify_impl::<Keccak256Hash<BS>>(
            &[
                "0x034acf2cced8e9744784ec9c3c626fa83f6f8ddd83faed9d5caa5ad72c91eb1c",
                "0x39dce95a2271a57f999981eef6917f3df8ad25d116c98d809a3b7da5d54805a3",
                "0x435246f701f1483adcb7037fa64b3fa027c41e13a52d1e6c502e9b71dea3ca81",
                "0x7952ed41c4062f594b53b8c548874b41a7a7f05593d7a5a313ed82c2fe1c62d7",
            ],
            &[
                "0x8a2dbe0367d89a172a0eb470026f5bb85ddd542e0d93885318c5d6b8a3764f28",
                "0x31354bf6d6ca07dbde80395609890a558ff35587ddc2b5c72ccde2b4e15d114f",
                "0xbd82b7f4f83de57b73e00aa0924a5fc4dfaf959bd252aa3fda8abdaf87713dc0",
            ],
        );
    }

    fn test_merkle_root_impl<H: MerkleHasher>(leaves: &[&'static str], expected: &'static str) {
        let leaves: Vec<H256> = leaves.iter().map(|&leaf| parse(leaf)).collect();
        let root = merkle_root::<H>(leaves.as_slice());
        assert_eq!(root, parse::<H256>(expected));
    }

    #[test]
    fn test_merkle_root_one_leaf() {
        test_merkle_root_impl::<Sha2Hash<BS>>(
            &["0x1fe9e33b7ff790473e13eb6384d61d6abb32d2e50b30a382c9767b5347f5846b"],
            "0x1fe9e33b7ff790473e13eb6384d61d6abb32d2e50b30a382c9767b5347f5846b",
        );
        test_merkle_root_impl::<Keccak256Hash<BS>>(
            &["0x1fe9e33b7ff790473e13eb6384d61d6abb32d2e50b30a382c9767b5347f5846b"],
            "0x1fe9e33b7ff790473e13eb6384d61d6abb32d2e50b30a382c9767b5347f5846b",
        );
    }

    #[test]
    fn test_merkle_root_two_leaves() {
        test_merkle_root_impl::<Sha2Hash<BS>>(
            &[
                "0x0cbb1061e55efef40fae3c8e34d301c9940889146f816edd475052fc45caf060",
                "0x2d8257d72c4b6e6bfa3a21a22c053d02d1d52b4b5e6097acfd0ed45f827d6ba4",
            ],
            "0x3af448b6612ac9931b7413ecca5d209187cd808ae2d3647099aa99fada16c955",
        );
        test_merkle_root_impl::<Keccak256Hash<BS>>(
            &[
                "0x0cbb1061e55efef40fae3c8e34d301c9940889146f816edd475052fc45caf060",
                "0x2d8257d72c4b6e6bfa3a21a22c053d02d1d52b4b5e6097acfd0ed45f827d6ba4",
            ],
            "0x6e70b19bd160e181bb367ab730eababc6e3673977f7fdce38934ee197bbcb568",
        );
    }

    #[test]
    fn test_merkle_root_four_leaves() {
        test_merkle_root_impl::<Sha2Hash<BS>>(
            &[
                "0x034acf2cced8e9744784ec9c3c626fa83f6f8ddd83faed9d5caa5ad72c91eb1c",
                "0x39dce95a2271a57f999981eef6917f3df8ad25d116c98d809a3b7da5d54805a3",
                "0x435246f701f1483adcb7037fa64b3fa027c41e13a52d1e6c502e9b71dea3ca81",
                "0x7952ed41c4062f594b53b8c548874b41a7a7f05593d7a5a313ed82c2fe1c62d7",
            ],
            "0xed171c6bd6876b93f4fc8914e0b2a8db53d7204048ad9fdff6e7127c077ca072",
        );
        test_merkle_root_impl::<Keccak256Hash<BS>>(
            &[
                "0x034acf2cced8e9744784ec9c3c626fa83f6f8ddd83faed9d5caa5ad72c91eb1c",
                "0x39dce95a2271a57f999981eef6917f3df8ad25d116c98d809a3b7da5d54805a3",
                "0x435246f701f1483adcb7037fa64b3fa027c41e13a52d1e6c502e9b71dea3ca81",
                "0x7952ed41c4062f594b53b8c548874b41a7a7f05593d7a5a313ed82c2fe1c62d7",
            ],
            "0xbd82b7f4f83de57b73e00aa0924a5fc4dfaf959bd252aa3fda8abdaf87713dc0",
        );
    }

    #[test]
    fn test_merkle_root_matches_merklify() {
        let leaves = sample_leaves();
        let hashes = make_hashes(leaves.as_slice());
        let root = merkle_root::<Sha2Hash<BS>>(leaves.as_slice());
        assert_eq!(root, *hashes.last().unwrap());
    }

    #[test]
    #[should_panic]
    fn test_merkle_root_of_no_leaves() {
        merkle_root::<Sha2Hash<BS>>(&[]);
    }

    #[test]
    #[should_panic]
    fn test_merkle_root_of_three_leaves() {
        let leaves = sample_leaves();
        merkle_root::<Sha2Hash<BS>>(&leaves[0..3]);
    }

    #[test]
    fn test_merkle_path_of_left_leaf() {
        let leaves = sample_leaves();
        let hashes = make_hashes(leaves.as_slice());
        let path = merkle_path(hashes.as_slice(), 8, 0);
        assert_eq!(path, [leaves[1], hashes[8 + 1], hashes[12 + 1]]);
    }

    #[test]
    fn test_merkle_path_of_right_leaf() {
        let leaves = sample_leaves();
        let hashes = make_hashes(leaves.as_slice());
        let path = merkle_path(hashes.as_slice(), 8, 5);
        assert_eq!(path, [leaves[4], hashes[8 + 3], hashes[12]]);
    }

    #[test]
    fn test_merkle_path_of_single_leaf() {
        let leaf = H256::from_low_u64_be(42);
        let path = merkle_path(&[leaf], 1, 0);
        assert_eq!(path, []);
    }

    #[test]
    fn test_verify_path() {
        let leaves = sample_leaves();
        let hashes = make_hashes(leaves.as_slice());
        let path = merkle_path(hashes.as_slice(), 8, 5);
        let result = verify_path::<Sha2Hash<BS>>(leaves[5], &path, 5, *hashes.last().unwrap());
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_all_paths() {
        let leaves = sample_leaves();
        let hashes = make_hashes(leaves.as_slice());
        let root_hash = *hashes.last().unwrap();
        let results: Vec<bool> = (0..8)
            .map(|index| {
                let path = merkle_path(hashes.as_slice(), 8, index);
                verify_path::<Sha2Hash<BS>>(leaves[index], &path, index, root_hash).is_ok()
            })
            .collect();
        assert_eq!(results, [true; 8]);
    }

    #[test]
    fn test_verify_path_with_wrong_index() {
        let leaves = sample_leaves();
        let hashes = make_hashes(leaves.as_slice());
        let path = merkle_path(hashes.as_slice(), 8, 5);
        let result = verify_path::<Sha2Hash<BS>>(leaves[5], &path, 4, *hashes.last().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_path_with_out_of_range_index() {
        let leaves = sample_leaves();
        let hashes = make_hashes(leaves.as_slice());
        let path = merkle_path(hashes.as_slice(), 8, 5);
        let result = verify_path::<Sha2Hash<BS>>(leaves[5], &path, 5 + 8, *hashes.last().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_path_with_wrong_leaf() {
        let leaves = sample_leaves();
        let hashes = make_hashes(leaves.as_slice());
        let path = merkle_path(hashes.as_slice(), 8, 5);
        let result = verify_path::<Sha2Hash<BS>>(leaves[4], &path, 5, *hashes.last().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_path_with_wrong_root() {
        let leaves = sample_leaves();
        let hashes = make_hashes(leaves.as_slice());
        let path = merkle_path(hashes.as_slice(), 8, 5);
        let result = verify_path::<Sha2Hash<BS>>(leaves[5], &path, 5, leaves[0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_empty_path() {
        let leaf = H256::from_low_u64_be(42);
        let result = verify_path::<Sha2Hash<BS>>(leaf, &[], 0, leaf);
        assert!(result.is_ok());
    }

    fn test_merkle_tree_impl<F: Field256, H: Hasher<F>>(
        columns: &[&[u16]],
        expected_root_hash: &'static str,
    ) {
        let tree = make_tree::<F, H>(columns);
        assert_eq!(tree.leaf(0).len(), columns.len() * F::EXTENSION_DEGREE);
        assert_eq!(tree.num_leaves(), columns[0].len());
        assert_eq!(tree.root_hash(), parse::<H256>(expected_root_hash));
    }

    #[test]
    fn test_merkle_tree_one_leaf_1() {
        test_merkle_tree_impl::<BS, Sha2Hash<BS>>(
            &[&[12]],
            "0x0bd187bc3deea1ef6c2a9ae254cf4e493f1dbbda32c79a662fc1d8437ab7e7c6",
        );
        test_merkle_tree_impl::<GL4, Sha2Hash<GL4>>(
            &[&[12]],
            "0xe76b4fc031cdd6d43d4098d336c45f5bb5a83f739d7d404f59280c4449b3d2e4",
        );
        test_merkle_tree_impl::<BS, Keccak256Hash<BS>>(
            &[&[12]],
            "0x5e70242e081756f445b5f3048611464568002e68800238dcfef718504de01782",
        );
        test_merkle_tree_impl::<GL4, Keccak256Hash<GL4>>(
            &[&[12]],
            "0xfa15cca7ea7fff7366dd498ef239d82384f949e9f7861602fda4630c9a5765ce",
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_2() {
        test_merkle_tree_impl::<BS, Sha2Hash<BS>>(
            &[&[34]],
            "0x825f71a1d38bedb88129450457f7943f988ee940aa1755aa89982e139e67047a",
        );
        test_merkle_tree_impl::<GL4, Sha2Hash<GL4>>(
            &[&[34]],
            "0x79dcebc1f36f1aef2c99656afbbfbceaddcb95f736fda75a4ec0c57844f5c690",
        );
        test_merkle_tree_impl::<BS, Keccak256Hash<BS>>(
            &[&[34]],
            "0xe6b7d9bb7fab250037d1ede2453bf936d350c03d028048fa491663a8d4a070ae",
        );
        test_merkle_tree_impl::<GL4, Keccak256Hash<GL4>>(
            &[&[34]],
            "0x57d6a6ef63adbfbf619601ce3d5427096c7872d15a1a322fe9623b6a2d8b5523",
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_two_polynomials_1() {
        test_merkle_tree_impl::<BS, Sha2Hash<BS>>(
            &[&[12], &[34]],
            "0x41c90ef8e7fa7e79e54b14cf8395f9707e9768a02aa79ce7f0f4e68668837c26",
        );
        test_merkle_tree_impl::<GL4, Sha2Hash<GL4>>(
            &[&[12], &[34]],
            "0x938b7d4fd4ce91ef9a1b6299e398ceaa4d0357b57e466c57392609e1a41a6076",
        );
        test_merkle_tree_impl::<BS, Keccak256Hash<BS>>(
            &[&[12], &[34]],
            "0x8a39979b1cf9df3dd02d420608cb73a92423a9ef2e44afb1118f5d62a1e35b59",
        );
        test_merkle_tree_impl::<GL4, Keccak256Hash<GL4>>(
            &[&[12], &[34]],
            "0xefa1eb09474dfecb19df38ea30fbfad5ae7b04cacf02a66f6a9a6185a1b738a4",
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_two_polynomials_2() {
        test_merkle_tree_impl::<BS, Sha2Hash<BS>>(
            &[&[34], &[12]],
            "0xc3dbf8cc67db17f01dd3527ab16da2120af998c7ebe3d06a2d9dff445e1adaee",
        );
        test_merkle_tree_impl::<GL4, Sha2Hash<GL4>>(
            &[&[34], &[12]],
            "0x3e53fe748dde4c5aeeb660b41bf093c4fae28a911241ceb171bec311082d757b",
        );
        test_merkle_tree_impl::<BS, Keccak256Hash<BS>>(
            &[&[34], &[12]],
            "0xa8f2b055d24c330298c32a0ed891a91378af0a495403fd5f95eca566e34a1c39",
        );
        test_merkle_tree_impl::<GL4, Keccak256Hash<GL4>>(
            &[&[34], &[12]],
            "0xa960fb20280657ea95e5ca952d0116e9d2ee740a5f6de89d4301fae34b4762d7",
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_three_polynomials_1() {
        test_merkle_tree_impl::<BS, Sha2Hash<BS>>(
            &[&[12], &[34], &[56]],
            "0x9bca77d625d9a50f1807d27c273cae980f706ef734a014c00e6a97bbb77472e3",
        );
        test_merkle_tree_impl::<GL4, Sha2Hash<GL4>>(
            &[&[12], &[34], &[56]],
            "0x9ee45931c224e429790f7c7f73b932a3e9398b2efe2874ef266990df762a1127",
        );
        test_merkle_tree_impl::<BS, Keccak256Hash<BS>>(
            &[&[12], &[34], &[56]],
            "0x11e4b3b9246b3678b751e40aac1ceb433a51be72a6c3560a49e0954d04689017",
        );
        test_merkle_tree_impl::<GL4, Keccak256Hash<GL4>>(
            &[&[12], &[34], &[56]],
            "0x6a02de3755b9c9fdbd68ff9cfcf1fa1e28037893d1c07d49fbdbe389f546501b",
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_three_polynomials_2() {
        test_merkle_tree_impl::<BS, Sha2Hash<BS>>(
            &[&[34], &[12], &[78]],
            "0xf48317b0be7caae4a15185cc8d0795c15c2bc98e18b8cbdc906270ada921fd25",
        );
        test_merkle_tree_impl::<GL4, Sha2Hash<GL4>>(
            &[&[34], &[12], &[78]],
            "0x9bd95b1d0b3592ed44d0a57f086bbf0b6dcfb1dd4dd0baec4fe4bd8b1602e5ed",
        );
        test_merkle_tree_impl::<BS, Keccak256Hash<BS>>(
            &[&[34], &[12], &[78]],
            "0xf3f5da8d03ec645ab1624876b7515f440d6dbbb5366e9e2da5218f9c603e990d",
        );
        test_merkle_tree_impl::<GL4, Keccak256Hash<GL4>>(
            &[&[34], &[12], &[78]],
            "0xe4784b964a939e32841a5c234c861d4d7df38564d91939968f2ea28f644f8168",
        );
    }

    #[test]
    fn test_merkle_tree_two_leaves_1() {
        test_merkle_tree_impl::<BS, Sha2Hash<BS>>(
            &[&[12, 34]],
            "0x2624006228d517eeda393d1440f25ed1c20887664f2444021849345167aadaf4",
        );
        test_merkle_tree_impl::<GL4, Sha2Hash<GL4>>(
            &[&[12, 34]],
            "0xb692c4a1a1046e77a0c22d21f5b652032a662d818d3af35b958a225c8140f8db",
        );
        test_merkle_tree_impl::<BS, Keccak256Hash<BS>>(
            &[&[12, 34]],
            "0xc781424ee41551ea0e56431b57519317a836b94c193ee898b39d058507217ee4",
        );
        test_merkle_tree_impl::<GL4, Keccak256Hash<GL4>>(
            &[&[12, 34]],
            "0x83e3888f9124be84325002fa7bead6f3367ed643b692ac13762aaf76ee70c0cf",
        );
    }

    #[test]
    fn test_merkle_tree_two_leaves_2() {
        test_merkle_tree_impl::<BS, Sha2Hash<BS>>(
            &[&[34, 56]],
            "0x85d6170f1dcec468c3a42f35d24b7689733abf26ff0278a9bc1edc7a9a0a7333",
        );
        test_merkle_tree_impl::<GL4, Sha2Hash<GL4>>(
            &[&[34, 56]],
            "0x767c729c25414201903c4f09098cec03e496b8409b1055ad584a643a82989bd5",
        );
        test_merkle_tree_impl::<BS, Keccak256Hash<BS>>(
            &[&[34, 56]],
            "0x00526855bdd13b55968dd71bddfca2409e77ee8c75cdc17fcdb641673558ae51",
        );
        test_merkle_tree_impl::<GL4, Keccak256Hash<GL4>>(
            &[&[34, 56]],
            "0xa6aaf73033dda9b4c6337387385b9edc1dfbcba39958c848644419ba3ae1b4eb",
        );
    }

    #[test]
    fn test_merkle_tree_two_leaves_two_polynomials_1() {
        test_merkle_tree_impl::<BS, Sha2Hash<BS>>(
            &[&[12, 56], &[34, 78]],
            "0xdc144d55dd7a9c48f00b495d30172b38db7d4dc71ee4a0feab99177e30a10d21",
        );
        test_merkle_tree_impl::<GL4, Sha2Hash<GL4>>(
            &[&[12, 56], &[34, 78]],
            "0xc665a15f4e1f3e5eed0ff3fa9655eec99ac757187657909a835025009cc39f3c",
        );
        test_merkle_tree_impl::<BS, Keccak256Hash<BS>>(
            &[&[12, 56], &[34, 78]],
            "0x086ff3e1257cbff588f5d01ce891ea152e15863c5e0d4407d4c3564d354b0614",
        );
        test_merkle_tree_impl::<GL4, Keccak256Hash<GL4>>(
            &[&[12, 56], &[34, 78]],
            "0x4a1fe0640b2abe0f0245c6918eb202605b59425a6061a59202fe2571031b3ba2",
        );
    }

    #[test]
    fn test_merkle_tree_two_leaves_two_polynomials_2() {
        test_merkle_tree_impl::<BS, Sha2Hash<BS>>(
            &[&[78, 34], &[56, 12]],
            "0xfb5a002c9ef7dad6d9b4edc690f323f4c302c665926c085dd5d8681496c937c6",
        );
        test_merkle_tree_impl::<GL4, Sha2Hash<GL4>>(
            &[&[78, 34], &[56, 12]],
            "0x7df367ae185f91e4be821d1726f09d5afa8f730c1ac24c059d4629a920b2992f",
        );
        test_merkle_tree_impl::<BS, Keccak256Hash<BS>>(
            &[&[78, 34], &[56, 12]],
            "0xb18e697b08221e2f411826d3c17c80bff6e06e90a55bcd5cc191ad67f1eb657a",
        );
        test_merkle_tree_impl::<GL4, Keccak256Hash<GL4>>(
            &[&[78, 34], &[56, 12]],
            "0x47882028397e68b3a81bedb231d0d1f8606ca142f5c61e391446ff24740d1f67",
        );
    }

    fn test_base_merkle_tree_impl<F: Field256, H: Hasher<F>>(
        columns: &[&[u16]],
        expected_root_hash: &'static str,
    ) {
        let tree = make_base_tree::<F, H>(columns);
        assert_eq!(tree.leaf(0).len(), columns.len());
        assert_eq!(tree.num_leaves(), columns[0].len());
        assert_eq!(tree.root_hash(), parse::<H256>(expected_root_hash));
    }

    #[test]
    fn test_base_merkle_tree_one_leaf_two_polynomials() {
        test_base_merkle_tree_impl::<BS, Sha2Hash<BS>>(
            &[&[12], &[34]],
            "0x96f2c772b6e63c7a4f0d7be81fb806779bbf1a7ac40293c3c546ac10ea464059",
        );
        test_base_merkle_tree_impl::<GL4, Sha2Hash<GL4>>(
            &[&[12], &[34]],
            "0x4f250af81cd03bb92c4e856aadaa89697d963b5e389aaad0b94773388a9320fc",
        );
        test_base_merkle_tree_impl::<KB8, Sha2Hash<KB8>>(
            &[&[12], &[34]],
            "0x2662d7a4f03be63b6cbcccb9421a83c88b6327a70ec7d7d2e0ffe324500e39c2",
        );
        test_base_merkle_tree_impl::<BS, Keccak256Hash<BS>>(
            &[&[12], &[34]],
            "0xb6001c17eb86995aad5b1eb8906c7d0efcf3da2f63cd2f2b2a33f14e27e6e265",
        );
        test_base_merkle_tree_impl::<GL4, Keccak256Hash<GL4>>(
            &[&[12], &[34]],
            "0xabe691bf928f2aa6d4c62eee9d5d90836f7249496dbd9d3c66d31f45ad613928",
        );
    }

    #[test]
    fn test_base_merkle_tree_two_leaves_two_polynomials() {
        test_base_merkle_tree_impl::<BS, Sha2Hash<BS>>(
            &[&[12, 56], &[34, 78]],
            "0x7eb7b67e17fc6303b4b6fc47f6e9778def9cc06b39fb0347a305061898049e0c",
        );
        test_base_merkle_tree_impl::<GL4, Sha2Hash<GL4>>(
            &[&[12, 56], &[34, 78]],
            "0x1d5e1512bc7c47706354bfc443faf2308b6f9a7cb9e67065360ffbb0b2a12d4c",
        );
        test_base_merkle_tree_impl::<KB8, Sha2Hash<KB8>>(
            &[&[12, 56], &[34, 78]],
            "0x02f1b7ea8fac7fd1b404f7cb4c8928d52e55b2d0a0faa42fe79bff948f343224",
        );
        test_base_merkle_tree_impl::<BS, Keccak256Hash<BS>>(
            &[&[12, 56], &[34, 78]],
            "0x52a2f9ef786ce91c33ddcba6ec979d35bcf640841ecad7354e6a674fd8af2cb7",
        );
        test_base_merkle_tree_impl::<GL4, Keccak256Hash<GL4>>(
            &[&[12, 56], &[34, 78]],
            "0x8b9bbfd1262bbc8311604af694df29543e8e7e537d2bb80ddc9dc05a15763cf7",
        );
    }

    #[test]
    fn test_tree_leaves() {
        let tree = make_tree::<BS, Sha2Hash<BS>>(&SAMPLE_COLUMNS);
        assert_eq!(tree.leaf(0), make_values::<BS>(&[12, 42, 78]));
        assert_eq!(tree.leaf(1), make_values::<BS>(&[34, 43, 56]));
        assert_eq!(tree.leaf(2), make_values::<BS>(&[56, 44, 34]));
        assert_eq!(tree.leaf(3), make_values::<BS>(&[78, 45, 12]));
        assert_eq!(tree.leaf(4), make_values::<BS>(&[90, 46, 90]));
        assert_eq!(tree.leaf(5), make_values::<BS>(&[12, 47, 78]));
        assert_eq!(tree.leaf(6), make_values::<BS>(&[34, 48, 56]));
        assert_eq!(tree.leaf(7), make_values::<BS>(&[56, 49, 34]));
    }

    #[test]
    fn test_query_leaf() {
        let tree = make_tree::<BS, Sha2Hash<BS>>(&SAMPLE_COLUMNS);
        let proof = tree.query(5);
        assert_eq!(proof.leaf(), make_values::<BS>(&[12, 47, 78]));
    }

    #[test]
    fn test_query_path() {
        let tree = make_tree::<BS, Sha2Hash<BS>>(&SAMPLE_COLUMNS);
        let proof = tree.query(5);
        assert_eq!(proof.len(), 3);
        assert_eq!(
            proof.path,
            [tree.hashes[4], tree.hashes[8 + 3], tree.hashes[12]]
        );
    }

    #[test]
    fn test_query_path_starts_with_sibling_leaf_hash() {
        let tree = make_tree::<BS, Sha2Hash<BS>>(&SAMPLE_COLUMNS);
        let proof = tree.query(5);
        assert_eq!(
            proof.path[0],
            WideLeafHasher::<BS, Sha2Hash<BS>>::hash(make_values(&[90, 46, 90]).as_slice())
        );
    }

    #[test]
    fn test_single_leaf_tree_query() {
        let tree = make_tree::<BS, Sha2Hash<BS>>(&[&[12], &[34]]);
        let proof = tree.query(0);
        assert_eq!(proof.leaf(), make_values::<BS>(&[12, 34]));
        assert_eq!(proof.path, []);
    }

    #[test]
    fn test_extension_tree_leaves() {
        let values = vec![
            GL4::from_base_coordinates(&[1u32, 2, 3, 4].map(GL::from)),
            GL4::from_base_coordinates(&[5u32, 6, 7, 8].map(GL::from)),
        ];
        let tree = Tree::<GL4, Sha2Hash<GL4>>::new(vec![values]);
        assert_eq!(tree.leaf(0), [1u32, 2, 3, 4].map(GL::from));
        assert_eq!(tree.leaf(1), [5u32, 6, 7, 8].map(GL::from));
    }

    #[test]
    fn test_prime_field_tree_leaves() {
        let values = make_values::<BS>(&[12, 34]);
        let tree = Tree::<BS, Sha2Hash<BS>>::new(vec![values]);
        assert_eq!(tree.leaf(0), make_values::<BS>(&[12]));
        assert_eq!(tree.leaf(1), make_values::<BS>(&[34]));
    }

    #[test]
    fn test_query_extension_leaf() {
        let tree = make_tree::<GL4, Sha2Hash<GL4>>(&SAMPLE_COLUMNS);
        let proof = tree.query(2);
        assert_eq!(
            proof.leaf(),
            make_values::<GL>(&[56, 0, 0, 0, 44, 0, 0, 0, 34, 0, 0, 0])
        );
    }

    #[test]
    fn test_proof_verify() {
        let tree = make_tree::<BS, Sha2Hash<BS>>(&SAMPLE_COLUMNS);
        let proof = tree.query(5);
        let result = proof.verify(5, tree.root_hash());
        assert!(result.is_ok());
    }

    #[test]
    fn test_proof_verify_with_wrong_index() {
        let tree = make_tree::<BS, Sha2Hash<BS>>(&SAMPLE_COLUMNS);
        let proof = tree.query(5);
        let result = proof.verify(4, tree.root_hash());
        assert!(result.is_err());
    }

    #[test]
    fn test_proof_verify_with_out_of_range_index() {
        let tree = make_tree::<BS, Sha2Hash<BS>>(&SAMPLE_COLUMNS);
        let proof = tree.query(5);
        let result = proof.verify(5 + 8, tree.root_hash());
        assert!(result.is_err());
    }

    #[test]
    fn test_proof_verify_with_wrong_root() {
        let tree = make_tree::<BS, Sha2Hash<BS>>(&SAMPLE_COLUMNS);
        let proof = tree.query(5);
        let result = proof.verify(5, tree.hashes[0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_extension_proof_as_base() {
        let tree = make_tree::<BS, Sha2Hash<BS>>(&SAMPLE_COLUMNS);
        let proof = tree.query(5);
        let result = proof.verify_base(5, tree.root_hash());
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_base_proof_as_extension() {
        let tree = make_base_tree::<BS, Sha2Hash<BS>>(&SAMPLE_COLUMNS);
        let proof = tree.query(5);
        let result = proof.verify(5, tree.root_hash());
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_misaligned_extension_proof() {
        let tree = make_base_tree::<GL4, Sha2Hash<GL4>>(&SAMPLE_COLUMNS);
        let proof = tree.query(5);
        let result = proof.verify(5, tree.root_hash());
        assert!(result.is_err());
    }

    #[test]
    fn test_single_leaf_proof_verify() {
        let tree = make_tree::<BS, Sha2Hash<BS>>(&[&[12], &[34]]);
        let proof = tree.query(0);
        let result = proof.verify(0, tree.root_hash());
        assert!(result.is_ok());
    }

    fn test_all_proofs_verify_impl<F: Field256, H: Hasher<F>>() {
        let tree = make_tree::<F, H>(&SAMPLE_COLUMNS);
        let results: Vec<bool> = (0..8)
            .map(|index| {
                let proof = tree.query(index);
                proof.verify(index, tree.root_hash()).is_ok()
            })
            .collect();
        assert_eq!(results, [true; 8]);
    }

    #[test]
    fn test_all_proofs_verify() {
        test_all_proofs_verify_impl::<BS, Sha2Hash<BS>>();
        test_all_proofs_verify_impl::<BS, Keccak256Hash<BS>>();
        test_all_proofs_verify_impl::<GL4, Sha2Hash<GL4>>();
        test_all_proofs_verify_impl::<GL4, Keccak256Hash<GL4>>();
        test_all_proofs_verify_impl::<KB8, Sha2Hash<KB8>>();
        test_all_proofs_verify_impl::<KB8, Keccak256Hash<KB8>>();
    }

    fn test_all_base_proofs_verify_impl<F: Field256, H: Hasher<F>>() {
        let tree = make_base_tree::<F, H>(&SAMPLE_COLUMNS);
        let results: Vec<bool> = (0..8)
            .map(|index| {
                let proof = tree.query(index);
                proof.verify_base(index, tree.root_hash()).is_ok()
            })
            .collect();
        assert_eq!(results, [true; 8]);
    }

    #[test]
    fn test_all_base_proofs_verify() {
        test_all_base_proofs_verify_impl::<BS, Sha2Hash<BS>>();
        test_all_base_proofs_verify_impl::<BS, Keccak256Hash<BS>>();
        test_all_base_proofs_verify_impl::<GL4, Sha2Hash<GL4>>();
        test_all_base_proofs_verify_impl::<GL4, Keccak256Hash<GL4>>();
        test_all_base_proofs_verify_impl::<KB8, Sha2Hash<KB8>>();
        test_all_base_proofs_verify_impl::<KB8, Keccak256Hash<KB8>>();
    }

    #[test]
    fn test_proof_to_proto() {
        let tree = make_tree::<BS, Sha2Hash<BS>>(&SAMPLE_COLUMNS);
        let proof = tree.query(2);
        let proto = proof.to_proto();
        assert_eq!(
            proto.leaf_values,
            [
                BS::from(56u16).to_le_bytes().to_vec(),
                BS::from(44u16).to_le_bytes().to_vec(),
                BS::from(34u16).to_le_bytes().to_vec(),
            ]
        );
        assert_eq!(
            proto.merkle_path,
            [
                tree.hashes[3].as_bytes().to_vec(),
                tree.hashes[8].as_bytes().to_vec(),
                tree.hashes[13].as_bytes().to_vec(),
            ]
        );
    }

    #[test]
    fn test_base_proof_to_proto() {
        let tree = make_base_tree::<GL4, Sha2Hash<GL4>>(&SAMPLE_COLUMNS);
        let proof = tree.query(2);
        let proto = proof.to_proto();
        assert_eq!(
            proto.leaf_values,
            [
                vec![56, 0, 0, 0, 0, 0, 0, 0],
                vec![44, 0, 0, 0, 0, 0, 0, 0],
                vec![34, 0, 0, 0, 0, 0, 0, 0],
            ]
        );
    }

    #[test]
    fn test_extension_proof_to_proto() {
        let tree = make_tree::<KB8, Sha2Hash<KB8>>(&SAMPLE_COLUMNS[0..1]);
        let proof = tree.query(2);
        let proto = proof.to_proto();
        assert_eq!(
            proto.leaf_values,
            [
                vec![56, 0, 0, 0],
                vec![0, 0, 0, 0],
                vec![0, 0, 0, 0],
                vec![0, 0, 0, 0],
                vec![0, 0, 0, 0],
                vec![0, 0, 0, 0],
                vec![0, 0, 0, 0],
                vec![0, 0, 0, 0],
            ]
        );
    }

    #[test]
    fn test_proof_from_proto() {
        let tree = make_tree::<BS, Sha2Hash<BS>>(&SAMPLE_COLUMNS);
        let proof = tree.query(2);
        let proto = proof.to_proto();
        let decoded = Proof::<BS, Sha2Hash<BS>>::from_proto(&proto).unwrap();
        assert_eq!(decoded, proof);
    }

    #[test]
    fn test_extension_proof_from_proto() {
        let tree = make_tree::<KB8, Sha2Hash<KB8>>(&SAMPLE_COLUMNS);
        let proof = tree.query(2);
        let proto = proof.to_proto();
        let decoded = Proof::<KB8, Sha2Hash<KB8>>::from_proto(&proto).unwrap();
        assert_eq!(decoded, proof);
    }

    #[test]
    fn test_proof_from_proto_with_non_canonical_value() {
        let tree = make_tree::<BS, Sha2Hash<BS>>(&SAMPLE_COLUMNS);
        let mut proto = tree.query(2).to_proto();
        let modulus: U256 = BS::MODULUS.parse().unwrap();
        proto.leaf_values[1] = modulus.to_little_endian().to_vec();
        let result = Proof::<BS, Sha2Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_proof_from_proto_with_non_canonical_base_field_value() {
        let tree = make_base_tree::<GL4, Sha2Hash<GL4>>(&SAMPLE_COLUMNS);
        let mut proto = tree.query(2).to_proto();
        proto.leaf_values[1] = vec![1, 0, 0, 0, 0xff, 0xff, 0xff, 0xff];
        let result = Proof::<GL4, Sha2Hash<GL4>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_proof_from_proto_with_long_value() {
        let tree = make_tree::<BS, Sha2Hash<BS>>(&SAMPLE_COLUMNS);
        let mut proto = tree.query(2).to_proto();
        proto.leaf_values[0].resize(33, 0);
        let result = Proof::<BS, Sha2Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_proof_from_proto_with_long_base_field_value() {
        let tree = make_base_tree::<GL4, Sha2Hash<GL4>>(&SAMPLE_COLUMNS);
        let mut proto = tree.query(2).to_proto();
        proto.leaf_values[0].push(0);
        let result = Proof::<GL4, Sha2Hash<GL4>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_proof_from_proto_with_short_hash() {
        let tree = make_tree::<BS, Sha2Hash<BS>>(&SAMPLE_COLUMNS);
        let mut proto = tree.query(2).to_proto();
        proto.merkle_path[1].pop();
        let result = Proof::<BS, Sha2Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }
}
