use crate::hash::Hasher;
use crate::merkle;
use crate::starkom::proto::pcs::v9 as proto;
use crate::utils;
use anyhow::{Result, anyhow};
use primitive_types::H256;
use starkom_ff::{Field, Field256};
use starkom_poly::Polynomial;
use std::marker::PhantomData;
use std::sync::LazyLock;

/// Domain separator tag used when deriving the Fiat-Shamir challenge for FRI folding.
static FOLD_DST: LazyLock<H256> = LazyLock::new(|| utils::make_dst(b"starkom/fri/fold"));

/// Reverses the order of the lowest `num_bits` bits of `index`.
///
/// REQUIRES: `index` must be less than `2^num_bits`.
fn reverse_bits(index: usize, num_bits: usize) -> usize {
    if num_bits > 0 {
        index.reverse_bits() >> (usize::BITS as usize - num_bits)
    } else {
        0
    }
}

/// A Merkle proof for a pair of partner FRI leaves.
///
/// The leaves of FRI trees are stored in bit-reversed order (see [`Tree`]), so partners are always
/// siblings: the proof opens one of them explicitly and the other one is the first element of the
/// Merkle path.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LeafProof<F: Field256, H: Hasher<F>> {
    /// The opened evaluation.
    value: F,
    /// The Merkle path from the leaf up to, but excluding, the root. The first element is the
    /// sibling leaf, that is the partner evaluation encoded with [`utils::encode_scalar`], and the
    /// following elements are the siblings of the ancestors of the leaf.
    path: Vec<H256>,
    _data: PhantomData<H>,
}

impl<F: Field256, H: Hasher<F>> LeafProof<F, H> {
    /// Returns the length of the Merkle path, which equals the height of the tree minus 1 (the root
    /// hash is not included in this count).
    fn len(&self) -> usize {
        self.path.len()
    }

    /// Returns the opened evaluation.
    fn value(&self) -> F {
        self.value
    }

    /// Returns the partner evaluation, that is the first element of the Merkle path.
    fn partner(&self) -> F {
        utils::decode_scalar(self.path[0]).unwrap()
    }

    /// Returns the two opened leaves.
    ///
    /// Equivalent to `(proof.value(), proof.partner())`.
    fn leaves(&self) -> (F, F) {
        (self.value, self.partner())
    }

    /// Verifies the proof against the given root hash.
    ///
    /// `index` is the index of the opened evaluation in the evaluation domain, not the position of
    /// the leaf in the tree (see [`Tree`]).
    ///
    /// REQUIRES: `index` must be less than `2^self.len()`.
    fn verify(&self, index: usize, root_hash: H256) -> Result<()> {
        merkle::verify_path::<H>(
            utils::encode_scalar(self.value),
            self.path.as_slice(),
            reverse_bits(index, self.path.len()),
            root_hash,
        )
    }

    /// Returns true iff the proof witnesses a tree whose leaves are all equal, i.e. the evaluations
    /// of a constant polynomial.
    ///
    /// That's the case if and only if every sibling equals the hash computed so far, starting from
    /// the leaf itself.
    fn is_constant(&self) -> bool {
        let mut hash = utils::encode_scalar(self.value);
        for &sibling in &self.path {
            if sibling != hash {
                return false;
            }
            hash = H::hash_binary(hash, hash);
        }
        true
    }

    /// Serializes this proof to a [`MerkleProof`](`proto::MerkleProof`) protobuf with a single leaf
    /// value.
    fn to_proto(&self) -> proto::MerkleProof {
        proto::MerkleProof {
            leaf_values: vec![self.value.to_le_bytes().to_vec()],
            merkle_path: self
                .path
                .iter()
                .map(|hash| hash.as_bytes().to_vec())
                .collect(),
        }
    }

    /// Deserializes a proof from a [`MerkleProof`](`proto::MerkleProof`) protobuf.
    fn from_proto(proto: &proto::MerkleProof) -> Result<Self> {
        let value = match proto.leaf_values.as_slice() {
            [value] => utils::load_scalar(value.as_slice()),
            values => Err(anyhow!(
                "invalid FRI leaf size (got {}, want 1)",
                values.len()
            )),
        }?;
        let path: Vec<H256> = proto
            .merkle_path
            .iter()
            .map(|hash| H::load_hash(hash.as_slice()))
            .collect::<Result<_>>()?;
        match path.first() {
            Some(&partner) => utils::decode_scalar::<F>(partner)?,
            None => return Err(anyhow!("missing FRI partner")),
        };
        Ok(Self {
            value,
            path,
            _data: PhantomData,
        })
    }
}

/// A Merkle tree whose leaves are the evaluations of a polynomial.
///
/// The evaluations are stored in bit-reversed order: the one at index `i` of the evaluation domain
/// of size N goes to the leaf at position `reverse_bits(i, log2(N))`. The partner evaluations at
/// `i` and `i + N/2` (that is at `X` and `-X`) only differ in the most significant bit of their
/// index, so they end up in sibling leaves and a single Merkle path opens both. Furthermore, their
/// parent node is at the same position as the leaf of the folded evaluation in the next round.
///
/// NOTE: bit reversal is completely abstracted away by `Tree`. The provided API exposes the
/// original evaluation domain indices, but bit reversal allows opening two partner values with a
/// single Merkle proof.
#[derive(Debug, Clone)]
struct Tree<F: Field256, H: Hasher<F>> {
    /// The whole tree, stored layer by layer as described in [`merkle::merklify`]. There are 2*N-1
    /// elements in total.
    data: Vec<H256>,
    _phantom_data: PhantomData<(F, H)>,
}

impl<F: Field256, H: Hasher<F>> Tree<F, H> {
    /// Constructs a Merkle tree from the evaluations of a polynomial, whose number must be a power
    /// of 2.
    fn new(values: &[F]) -> Self {
        let n = values.len();
        assert!(n.is_power_of_two());
        let num_bits = n.trailing_zeros() as usize;
        let mut data = Vec::with_capacity(n * 2 - 1);
        data.extend((0..n).map(|index| {
            let reverse_index = reverse_bits(index, num_bits);
            utils::encode_scalar(values[reverse_index])
        }));
        data.resize(n * 2 - 1, H256::zero());
        merkle::merklify::<H>(data.as_mut_slice(), n);
        Self {
            data,
            _phantom_data: PhantomData,
        }
    }

    /// Returns the number of leaves in the tree, corresponding to the size of the evaluation domain
    /// (always a power of 2).
    fn num_leaves(&self) -> usize {
        (self.data.len() + 1) >> 1
    }

    /// Returns the root hash of the Merkle tree.
    fn root_hash(&self) -> H256 {
        *self.data.last().unwrap()
    }

    /// Returns the evaluation at the given index of the evaluation domain.
    fn leaf(&self, index: usize) -> F {
        let num_bits = self.num_leaves().trailing_zeros() as usize;
        let reverse_index = reverse_bits(index, num_bits);
        utils::decode_scalar(self.data[reverse_index]).unwrap()
    }

    /// Returns a Merkle proof for the evaluation at the given index of the evaluation domain and
    /// its partner.
    ///
    /// NOTE: `index` is not bit-reversed, but thanks to the internal bit reversal the returned
    /// [`LeafProof`] proves both the element at `index` and its partner at `index + N/2`.
    fn query(&self, index: usize) -> LeafProof<F, H> {
        let n = self.num_leaves();
        let reverse_index = reverse_bits(index, n.trailing_zeros() as usize);
        LeafProof {
            value: self.leaf(index),
            path: merkle::merkle_path(self.data.as_slice(), n, reverse_index),
            _data: PhantomData,
        }
    }
}

/// Stores the Merkle root hashes of a FRI commitment.
///
/// Note that for low-degree testing these are strictly less than `log2(N)+1`, with N being the
/// number of committed evaluations. Once the folding process has reduced the polynomial to a
/// degree-0 one (a single constant) all subsequent folds would be identical, so we don't store
/// them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment {
    /// The first element in the array is the root of the main Merkle tree, the second one is the
    /// root of the Merkle tree from the first folding round, and so on until the last element which
    /// is the Merkle root of the last folding round.
    roots: Vec<H256>,
}

impl Commitment {
    pub(crate) fn with_roots(roots: Vec<H256>) -> Self {
        Self { roots }
    }

    /// Returns the number of stored roots, equivalent to the number of Merkle trees known to the
    /// prover. These would in turn include the base Merkle tree of the committed evaluations (over
    /// the extended domain) and one subsequent tree for every folding round. The number of folding
    /// rounds equals the log2 of the degree bound of the original polynomial. For example, if the
    /// user commits a degree<8 polynomial encoding 8 values, `len()` will return
    /// log2(8)+1 = 3+1 = 4 regardless of the blowup factor.
    pub fn len(&self) -> usize {
        self.roots.len()
    }

    /// Returns the Merkle roots of all folding rounds.
    ///
    /// The returned slice has [`Self::len()`] elements.
    pub fn roots(&self) -> &[H256] {
        self.roots.as_slice()
    }

    /// Returns the Merkle root hash of the committed polynomial, which is the first hash stored in
    /// the commitment.
    pub fn root(&self) -> H256 {
        *self.roots.first().unwrap()
    }
}

/// A single FRI query.
#[derive(Debug, Clone)]
pub struct Query<F: Field256, H: Hasher<F>> {
    /// The degree bound of the committed polynomial (always a power of 2).
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// The index of the element we're opening (the partner index is inferred automatically).
    ///
    /// Always less than the size of the extended domain.
    index: usize,
    /// Proves a pair of "partner" values at each folding round with one [`LeafProof`] for every
    /// round, opening the queried index explicitly and its partner as the sibling leaf. The proof
    /// at `folds[0]` proves the opened values.
    folds: Vec<LeafProof<F, H>>,
}

impl<F: Field256, H: Hasher<F>> Query<F, H> {
    /// Returns the two opened indices.
    pub fn indices(&self) -> (usize, usize) {
        let n = self.degree_bound << self.blowup_log2;
        (self.index, (self.index + n / 2) % n)
    }

    /// Returns the opened domain element, that is the X-coordinate of the evaluation.
    ///
    /// This is the element corresponding to the first value returned by [`Self::indices`], while
    /// the partner element can be obtained by simply negating this one.
    ///
    /// Note that we use [`Polynomial::shift_domain`] before committing polynomials, so the element
    /// returned here is a shifted power of an N-th root of unity, with
    /// `N = degree_bound * 2^blowup_factor`. The shift consists of multiplying the actual domain
    /// element by the [`MULTIPLICATIVE_GENERATOR`](`Field::MULTIPLICATIVE_GENERATOR`) of the *base*
    /// field, consistently with `shift_domain`.
    pub fn x(&self) -> F {
        Polynomial::<F>::coset_element2(self.index, self.degree_bound << self.blowup_log2).into()
    }

    /// Returns the opened evaluations of the committed polynomial.
    ///
    /// The first component of the returned tuple is the evaluation at the first index returned by
    /// [`Self::indices`], while the second component is the one at the second index.
    pub fn values(&self) -> (F, F) {
        self.folds[0].leaves()
    }

    /// Returns the number of Merkle trees in the proof: one for the original polynomial evaluations
    /// plus one for every folding round.
    ///
    /// In general these are log2(d)+1, with `d` being the degree bound of the committed polynomial.
    /// Note that for low-degree testing `d` is strictly less than the number of committed
    /// evaluations `N`.
    pub fn len(&self) -> usize {
        self.folds.len()
    }

    /// Verifies this proof against the given commitment.
    ///
    /// NOTE: for low-degree testing you also need to check that [`Self::len`] returns the log2 of
    /// the expected degree bound. This function only verifies the opened value pair across the
    /// folding structure.
    pub fn verify(&self, commitment: &Commitment) -> Result<()> {
        let mut n = self.degree_bound << self.blowup_log2;
        assert!(n.is_power_of_two());

        let mut k = n.trailing_zeros() as usize;

        let num_folds = self.folds.len();
        if num_folds > self.degree_bound.trailing_zeros() as usize + 1 {
            return Err(anyhow!("incorrect proof size"));
        }
        if commitment.len() != num_folds {
            return Err(anyhow!("wrong number of folding rounds"));
        }

        let mut index = self.index;
        let mut pos = self.folds[0].value();
        let mut step = Polynomial::<F>::domain_element2(1, n).invert_unwrap();

        for round in 0..num_folds {
            let root_hash = commitment.roots()[round];

            let proof = &self.folds[round];
            if proof.len() != k {
                return Err(anyhow!(
                    "invalid Merkle proof height (got {}, want {})",
                    proof.len(),
                    k
                ));
            }

            if proof.value() != pos {
                return Err(anyhow!("leaf value mismatch"));
            }
            proof.verify(index, root_hash)?;

            let omega_inv = step.pow_small(index);
            n /= 2;
            k -= 1;
            index %= n;

            let neg = proof.partner();
            let alpha = H::challenge(*FOLD_DST, &[root_hash]);
            pos = (pos + neg + alpha * omega_inv * (pos - neg)) * F::BaseField::TWO_INV;
            step = step.square();
        }

        if !self.folds.last().unwrap().is_constant() {
            return Err(anyhow!("the final folded polynomial is not constant"));
        }

        Ok(())
    }

    /// Serializes this query to a [`Query`](`proto::Query`) protobuf.
    pub fn to_proto(&self) -> proto::Query {
        proto::Query {
            index: self.index as u64,
            folds: self.folds.iter().map(LeafProof::to_proto).collect(),
        }
    }

    /// Deserializes a query from a [`Query`](`proto::Query`) protobuf.
    ///
    /// REQUIRES: `degree_bound` must be a power of 2, `blowup_log2` must be greater than 0, and the
    /// size of the extended domain must fit within the 2-adicity of `F::BaseField`.
    ///
    /// The query index must be less than the size of the extended domain, and every opened value,
    /// including the partner in the first element of each Merkle path, must be a canonical field
    /// element, otherwise an error is returned.
    ///
    /// NOTE: this method does not validate the proof, it only deserializes it. The caller must
    /// invoke [`Self::verify`] separately.
    pub fn from_proto(
        degree_bound: usize,
        blowup_log2: usize,
        proto: &proto::Query,
    ) -> Result<Self> {
        assert!(degree_bound.is_power_of_two());
        assert!(blowup_log2 > 0);
        assert!(degree_bound.trailing_zeros() as usize + blowup_log2 <= F::BaseField::S);
        let n = degree_bound << blowup_log2;
        if proto.index >= n as u64 {
            return Err(anyhow!(
                "invalid query index {}: the extended domain size is {n}",
                proto.index
            ));
        }
        if proto.folds.is_empty() {
            return Err(anyhow!("the FRI query has no folds"));
        }
        Ok(Self {
            degree_bound,
            blowup_log2,
            index: proto.index as usize,
            folds: proto
                .folds
                .iter()
                .map(|fold| LeafProof::from_proto(fold))
                .collect::<Result<_>>()?,
        })
    }
}

/// A FRI prover.
///
/// The struct contains the main Merkle tree built on the committed polynomial and the Merkle trees
/// of all folded polynomials up to and including the one where the polynomial has been folded into
/// a constant. Note that the final Merkle tree still has more than one leaf due to the low-degree
/// extension.
#[derive(Debug, Clone)]
pub struct Prover<F: Field256, H: Hasher<F>> {
    /// The degree bound of the committed polynomial, always a power of 2.
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// The Merkle trees of the folding argument, one for the committed polynomial and then one for
    /// every folding round.
    trees: Vec<Tree<F, H>>,
}

impl<F: Field256, H: Hasher<F>> Prover<F, H> {
    /// Performs one FRI folding round, returning the evaluations of the folded polynomial (half as
    /// many as `values`).
    ///
    /// `root_hash` is the Merkle root of `values`, from which the folding challenge is derived.
    fn fold(values: &[F], root_hash: H256) -> Vec<F> {
        let n = values.len();
        assert!(n.is_power_of_two());

        let alpha = H::challenge(*FOLD_DST, &[root_hash]);

        let omega_inv = Polynomial::<F>::domain_element2(1, n).invert_unwrap();

        let m = n / 2;
        let mut omega_inv_i = F::BaseField::ONE;

        (0..m)
            .map(|i| {
                let pos = values[i];
                let neg = values[i + m];
                let value = (pos + neg + alpha * omega_inv_i * (pos - neg)) * F::BaseField::TWO_INV;
                omega_inv_i *= omega_inv;
                value
            })
            .collect()
    }

    pub fn new(polynomial: Polynomial<F>, degree_bound: usize, blowup_log2: usize) -> Self {
        assert!(degree_bound.is_power_of_two());
        assert!(degree_bound >= polynomial.degree_bound());
        assert!(blowup_log2 > 0);

        let n = degree_bound << blowup_log2;
        assert!(n as u64 <= 1u64 << F::BaseField::S);

        let num_folds = degree_bound.trailing_zeros() as usize;
        let mut values = polynomial.shift_domain().lde2(n);
        let mut trees = Vec::with_capacity(num_folds + 1);
        trees.push(Tree::new(values.as_slice()));
        for _ in 0..num_folds {
            values = Self::fold(values.as_slice(), trees.last().unwrap().root_hash());
            trees.push(Tree::new(values.as_slice()));
        }

        Self {
            degree_bound,
            blowup_log2,
            trees,
        }
    }

    /// Returns the degree bound of the committed polynomial (always a power of 2).
    ///
    /// NOTE: the actual degree of the original polynomial may be even lower than this value because
    /// it was rounded up to the next power of 2 in order to run the FFT and FRI algorithms.
    pub fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    /// Returns the size of the extended domain, equal to `degree_bound * blowup`.
    pub fn extended_domain_size(&self) -> usize {
        self.degree_bound << self.blowup_log2
    }

    /// Alias for [`Self::extended_domain_size`].
    pub fn size(&self) -> usize {
        self.degree_bound << self.blowup_log2
    }

    /// Returns the Merkle root hash of the committed polynomial.
    ///
    /// This is equivalent to the first root stored in the commiment returned by [`Self::commit`].
    pub fn root_hash(&self) -> H256 {
        self.trees[0].root_hash()
    }

    /// Creates the FRI commitment for the committed polynomial.
    pub fn commit(&self) -> Commitment {
        Commitment {
            roots: self.trees.iter().map(Tree::root_hash).collect(),
        }
    }

    /// Builds a FRI [`Query`] for the value at the specified index of the evaluation domain.
    ///
    /// NOTE: `index` is relative to the *extended* evaluation domain, so for example if you
    /// committed to 4 evaluations with a blowup factor of 8 the range for `index` is [0, 32).
    pub fn query(&self, index: usize) -> Query<F, H> {
        let mut n = self.degree_bound << self.blowup_log2;
        assert!(index < n);

        let mut i = index;
        let mut folds = vec![];
        for tree in &self.trees {
            folds.push(tree.query(i));
            n /= 2;
            i %= n;
        }

        assert!(folds.last().unwrap().is_constant());

        Query {
            degree_bound: self.degree_bound,
            blowup_log2: self.blowup_log2,
            index,
            folds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::{Keccak256Hash, Sha2Hash};
    use starkom_bluesky::Scalar as BS;
    use starkom_goldilocks::GL4;
    use std::fmt::Debug;
    use std::str::FromStr;

    fn parse<V: FromStr<Err: Debug>>(s: &'static str) -> V {
        s.parse().unwrap()
    }

    /// Eight distinct evaluations to build trees from.
    fn sample_values() -> Vec<BS> {
        [7, 8, 11, 16, 23, 32, 43, 56]
            .into_iter()
            .map(BS::from_const)
            .collect()
    }

    /// Hashes two 32-byte nodes with an independent Keccak-256 implementation, the same way an EVM
    /// verifier does with `keccak256(abi.encodePacked(left, right))`.
    fn reference_keccak256(left: H256, right: H256) -> H256 {
        use sha3::Digest;
        let mut hasher = sha3::Keccak256::new();
        hasher.update(left.as_bytes());
        hasher.update(right.as_bytes());
        H256::from_slice(hasher.finalize().as_slice())
    }

    /// Reference implementation of a Keccak-256 FRI tree, returning all the nodes of the tree
    /// stored layer by layer as in [`merkle::merklify`].
    ///
    /// The leaves are the given values in the given order, so the caller must bit-reverse them.
    fn reference_keccak256_tree(values: &[BS]) -> Vec<H256> {
        let mut layer: Vec<H256> = values
            .iter()
            .map(|value| H256::from_slice(&value.to_u256().to_big_endian()))
            .collect();
        let mut nodes = layer.clone();
        while layer.len() > 1 {
            layer = layer
                .chunks(2)
                .map(|pair| reference_keccak256(pair[0], pair[1]))
                .collect();
            nodes.extend_from_slice(layer.as_slice());
        }
        nodes
    }

    fn test_reverse_bits_impl(index: usize, num_bits: usize, expected: usize) {
        let result = reverse_bits(index, num_bits);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_reverse_no_bits() {
        test_reverse_bits_impl(0, 0, 0);
    }

    #[test]
    fn test_reverse_one_bit() {
        test_reverse_bits_impl(0, 1, 0);
        test_reverse_bits_impl(1, 1, 1);
    }

    #[test]
    fn test_reverse_three_bits() {
        test_reverse_bits_impl(0b000, 3, 0b000);
        test_reverse_bits_impl(0b001, 3, 0b100);
        test_reverse_bits_impl(0b011, 3, 0b110);
        test_reverse_bits_impl(0b101, 3, 0b101);
        test_reverse_bits_impl(0b110, 3, 0b011);
    }

    #[test]
    fn test_reverse_many_bits() {
        test_reverse_bits_impl(0b1, 62, 1 << 61);
        test_reverse_bits_impl(0b1011, 62, 0b1101 << 58);
        test_reverse_bits_impl(usize::MAX, usize::BITS as usize, usize::MAX);
    }

    #[test]
    fn test_fold_dst() {
        assert_eq!(
            *FOLD_DST,
            parse::<H256>("0x9ffd3556faeb2cae194ce95adf6b3580f590504daa0dea56966ce4ef233844af")
        );
    }

    #[test]
    fn test_tree_size() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        assert_eq!(tree.num_leaves(), 8);
        assert_eq!(tree.data.len(), 15);
    }

    #[test]
    fn test_tree_leaves_are_bit_reversed_big_endian_values() {
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(sample_values().as_slice());
        assert_eq!(
            tree.data[0..8],
            [
                H256::from_low_u64_be(7),
                H256::from_low_u64_be(23),
                H256::from_low_u64_be(11),
                H256::from_low_u64_be(43),
                H256::from_low_u64_be(8),
                H256::from_low_u64_be(32),
                H256::from_low_u64_be(16),
                H256::from_low_u64_be(56),
            ]
        );
    }

    #[test]
    fn test_tree_matches_reference_implementation() {
        let values = sample_values();
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(values.as_slice());
        assert_eq!(
            tree.data,
            reference_keccak256_tree(&[
                values[0], values[4], values[2], values[6], values[1], values[5], values[3],
                values[7],
            ])
        );
    }

    #[test]
    fn test_tree_root_hash() {
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(sample_values().as_slice());
        let root_hash = tree.root_hash();
        assert_eq!(root_hash, tree.data[14]);
    }

    #[test]
    fn test_tree_leaf() {
        let values = sample_values();
        let tree = Tree::<BS, Sha2Hash<BS>>::new(values.as_slice());
        let leaves: Vec<BS> = (0..8).map(|index| tree.leaf(index)).collect();
        assert_eq!(leaves, values);
    }

    #[test]
    fn test_single_leaf_tree() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(&[BS::from_const(42)]);
        assert_eq!(tree.num_leaves(), 1);
        assert_eq!(tree.data, [H256::from_low_u64_be(42)]);
        assert_eq!(tree.root_hash(), H256::from_low_u64_be(42));
        assert_eq!(tree.leaf(0), BS::from_const(42));
    }

    #[test]
    fn test_query_value() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3);
        assert_eq!(proof.value(), BS::from_const(16));
    }

    #[test]
    fn test_query_path() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3);
        assert_eq!(proof.len(), 3);
        assert_eq!(proof.path, [tree.data[7], tree.data[8 + 2], tree.data[12]]);
    }

    #[test]
    fn test_query_path_starts_with_partner() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3);
        assert_eq!(proof.path[0], H256::from_low_u64_be(56));
    }

    #[test]
    fn test_query_partner() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3);
        assert_eq!(proof.partner(), BS::from_const(56));
    }

    #[test]
    fn test_partner_of_partner() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(7);
        assert_eq!(proof.partner(), BS::from_const(16));
    }

    #[test]
    fn test_single_leaf_tree_query() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(&[BS::from_const(42)]);
        let proof = tree.query(0);
        assert_eq!(proof.value(), BS::from_const(42));
        assert_eq!(proof.path, []);
    }

    #[test]
    fn test_leaf_proof_verify() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3);
        let result = proof.verify(3, tree.root_hash());
        assert!(result.is_ok());
    }

    #[test]
    fn test_all_leaf_proofs_verify() {
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(sample_values().as_slice());
        let results: Vec<bool> = (0..8)
            .map(|index| tree.query(index).verify(index, tree.root_hash()).is_ok())
            .collect();
        assert_eq!(results, [true; 8]);
    }

    #[test]
    fn test_leaf_proof_verify_with_wrong_index() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3);
        let result = proof.verify(2, tree.root_hash());
        assert!(result.is_err());
    }

    #[test]
    fn test_leaf_proof_verify_with_partner_index() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3);
        let result = proof.verify(7, tree.root_hash());
        assert!(result.is_err());
    }

    #[test]
    fn test_leaf_proof_verify_with_wrong_root() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3);
        let result = proof.verify(3, tree.data[0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_leaf_proof_verify_with_tampered_partner() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let mut proof = tree.query(3);
        proof.path[0] = H256::from_low_u64_be(57);
        let result = proof.verify(3, tree.root_hash());
        assert!(result.is_err());
    }

    #[test]
    fn test_single_leaf_proof_verify() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(&[BS::from_const(42)]);
        let proof = tree.query(0);
        let result = proof.verify(0, tree.root_hash());
        assert!(result.is_ok());
    }

    #[test]
    fn test_leaf_proof_is_not_constant() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3);
        assert!(!proof.is_constant());
    }

    #[test]
    fn test_constant_tree_proofs_are_constant() {
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(&[BS::from_const(42); 8]);
        let results: Vec<bool> = (0..8)
            .map(|index| tree.query(index).is_constant())
            .collect();
        assert_eq!(results, [true; 8]);
    }

    #[test]
    fn test_almost_constant_tree_proofs_are_not_constant() {
        let mut values = [BS::from_const(42); 8];
        values[5] = BS::from_const(43);
        let tree = Tree::<BS, Sha2Hash<BS>>::new(&values);
        let results: Vec<bool> = (0..8)
            .map(|index| tree.query(index).is_constant())
            .collect();
        assert_eq!(results, [false; 8]);
    }

    #[test]
    fn test_single_leaf_proof_is_constant() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(&[BS::from_const(42)]);
        let proof = tree.query(0);
        assert!(proof.is_constant());
    }

    #[test]
    fn test_leaf_proof_to_proto() {
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3);
        let proto = proof.to_proto();
        assert_eq!(
            proto.leaf_values,
            [BS::from_const(16).to_le_bytes().to_vec()]
        );
        assert_eq!(
            proto.merkle_path,
            [
                tree.data[7].as_bytes().to_vec(),
                tree.data[8 + 2].as_bytes().to_vec(),
                tree.data[12].as_bytes().to_vec(),
            ]
        );
    }

    #[test]
    fn test_leaf_proof_from_proto() {
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3);
        let proto = proof.to_proto();
        let decoded = LeafProof::<BS, Keccak256Hash<BS>>::from_proto(&proto).unwrap();
        assert_eq!(decoded, proof);
    }

    const DEGREE_BOUND: usize = 8;
    const BLOWUP_LOG2: usize = 2;
    const QUERY_INDEX: usize = 3;

    /// Builds a polynomial from its coefficients.
    fn make_polynomial<F: Field256>(coefficients: &[u64]) -> Polynomial<F> {
        Polynomial::with_coefficients(coefficients.iter().copied().map(F::from).collect())
    }

    /// The polynomial committed by the prover in most of the tests.
    fn honest_polynomial() -> Polynomial<BS> {
        make_polynomial(&[12, 34, 56, 78, 90, 12, 34, 56])
    }

    /// Same as [`honest_polynomial`] but with different coefficients.
    fn foreign_polynomial() -> Polynomial<BS> {
        make_polynomial(&[90, 78, 56, 34, 12, 90, 78, 56])
    }

    #[test]
    fn test_sha2_commitment_roots() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), 8, 2);
        let commitment = prover.commit();
        assert_eq!(
            commitment.roots(),
            [
                parse("0x4dbf05b032faad3a0279e11536170c9dc6c2444b65bfda0917160a556b76ab5b"),
                parse("0x3beefd65cfe8506f9b499e8397100d6d695d97af5742e70b60100f056d55307b"),
                parse("0x7d5697dea024a95e3d383fdc569fb615fa391a96f710201c970bebe336476c9b"),
                parse("0x14d99870760ebadb2f9f3b2689116c8348263851c7f2c5ff67845509618b582d"),
            ]
        );
    }

    #[test]
    fn test_keccak256_commitment_roots() {
        let prover = Prover::<BS, Keccak256Hash<BS>>::new(honest_polynomial(), 8, 2);
        let commitment = prover.commit();
        assert_eq!(
            commitment.roots(),
            [
                parse("0xf62fdcf4604e703d26974310d70bca5ef93f22d316f2083167b3bc4fd4437f8f"),
                parse("0x3e708444a31aed3a38ef8d8074873506efba437be058889e79b8499095e5eefd"),
                parse("0xbb62bc0c84acc6aa08f2bac9f342426dc3e9c284630031807c3f848f35b15dcc"),
                parse("0xcdfce461c1f186f69101a452d08021e6b6ed6613637327a5173234bafd9c3018"),
            ]
        );
    }

    #[test]
    fn test_prover_parameters() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        assert_eq!(prover.degree_bound(), DEGREE_BOUND);
        assert_eq!(prover.extended_domain_size(), DEGREE_BOUND << BLOWUP_LOG2);
        assert_eq!(prover.size(), DEGREE_BOUND << BLOWUP_LOG2);
    }

    #[test]
    fn test_prover_root_hash() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let root_hash = prover.root_hash();
        assert_eq!(root_hash, prover.commit().root());
    }

    fn test_queries_impl<F: Field256, H: Hasher<F>>(
        coefficients: &[u64],
        degree_bound: usize,
        blowup_log2: usize,
    ) {
        let polynomial = make_polynomial::<F>(coefficients);
        let prover = Prover::<F, H>::new(polynomial.clone(), degree_bound, blowup_log2);
        let commitment = prover.commit();
        let n = degree_bound << blowup_log2;
        let queries: Vec<Query<F, H>> = (0..n).map(|index| prover.query(index)).collect();
        for (index, query) in queries.iter().enumerate() {
            assert_eq!(query.indices(), (index, (index + n / 2) % n));
            assert_eq!(query.len(), degree_bound.trailing_zeros() as usize + 1);
            let x = query.x();
            assert_eq!(
                query.values(),
                (polynomial.evaluate(x), polynomial.evaluate(-x))
            );
            assert!(query.verify(&commitment).is_ok());
        }
    }

    fn test_queries(coefficients: &[u64], degree_bound: usize, blowup_log2: usize) {
        test_queries_impl::<BS, Sha2Hash<BS>>(coefficients, degree_bound, blowup_log2);
        test_queries_impl::<BS, Keccak256Hash<BS>>(coefficients, degree_bound, blowup_log2);
        test_queries_impl::<GL4, Sha2Hash<GL4>>(coefficients, degree_bound, blowup_log2);
        test_queries_impl::<GL4, Keccak256Hash<GL4>>(coefficients, degree_bound, blowup_log2);
    }

    #[test]
    fn test_constant_polynomial_blowup_2() {
        test_queries(&[12], 1, 1);
    }

    #[test]
    fn test_constant_polynomial_blowup_4() {
        test_queries(&[34], 1, 2);
    }

    #[test]
    fn test_constant_polynomial_blowup_8() {
        test_queries(&[56], 1, 3);
    }

    #[test]
    fn test_polynomial_degree_one_blowup_2() {
        test_queries(&[12, 34], 2, 1);
    }

    #[test]
    fn test_polynomial_degree_one_blowup_4() {
        test_queries(&[56, 78], 2, 2);
    }

    #[test]
    fn test_polynomial_degree_one_blowup_8() {
        test_queries(&[90, 12], 2, 3);
    }

    #[test]
    fn test_polynomial_degree_three_blowup_2() {
        test_queries(&[12, 34, 56, 78], 4, 1);
    }

    #[test]
    fn test_polynomial_degree_three_blowup_4() {
        test_queries(&[42, 43, 44, 45], 4, 2);
    }

    #[test]
    fn test_polynomial_degree_three_blowup_8() {
        test_queries(&[78, 56, 34, 12], 4, 3);
    }

    #[test]
    fn test_polynomial_degree_seven_blowup_2() {
        test_queries(&[12, 34, 56, 78, 90, 12, 34, 56], 8, 1);
    }

    #[test]
    fn test_polynomial_degree_seven_blowup_4() {
        test_queries(&[42, 43, 44, 45, 46, 47, 48, 49], 8, 2);
    }

    #[test]
    fn test_polynomial_degree_seven_blowup_8() {
        test_queries(&[56, 34, 12, 90, 78, 56, 34, 12], 8, 3);
    }

    #[test]
    fn test_polynomial_below_degree_bound_blowup_2() {
        test_queries(&[12, 34], 8, 1);
    }

    #[test]
    fn test_polynomial_below_degree_bound_blowup_4() {
        test_queries(&[56, 78], 8, 2);
    }

    #[test]
    fn test_polynomial_below_degree_bound_blowup_8() {
        test_queries(&[90, 12], 8, 3);
    }

    #[test]
    fn test_accept_untampered_proof() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let query = prover.query(QUERY_INDEX);
        let result = query.verify(&prover.commit());
        assert!(result.is_ok());
    }

    #[test]
    fn test_reject_foreign_commitment() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let query = prover.query(QUERY_INDEX);
        let foreign = Prover::<BS, Sha2Hash<BS>>::new(foreign_polynomial(), 8, 2).commit();
        let result = query.verify(&foreign);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_corrupted_root() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let mut commitment = prover.commit();
        commitment.roots[2] = H256::repeat_byte(0xAA);
        let result = prover.query(QUERY_INDEX).verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_truncated_folds() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        query.folds.truncate(3);
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_truncated_commitment() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let mut commitment = prover.commit();
        commitment.roots.truncate(3);
        let result = prover.query(QUERY_INDEX).verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_extra_folds() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        let mut donor = prover.query(QUERY_INDEX);
        query.folds.push(donor.folds.pop().unwrap());
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_folding_stopped_early() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let mut commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        commitment.roots.truncate(3);
        query.folds.truncate(3);
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_wrong_proof_height() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        let mut donor = prover.query(QUERY_INDEX);
        query.folds[0] = donor.folds.remove(1);
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_swapped_partners() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        let proof = &mut query.folds[0];
        let partner = proof.partner();
        proof.path[0] = utils::encode_scalar(proof.value);
        proof.value = partner;
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_tampered_partner() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        let proof = &mut query.folds[1];
        proof.path[0] = utils::encode_scalar(proof.partner() + BS::ONE);
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_foreign_leaf_proof() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        let mut other = prover.query(QUERY_INDEX + 4);
        query.folds[1] = other.folds.remove(1);
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_extra_path_hash() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        query.folds[0].path.push(H256::zero());
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_out_of_range_query_index() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.index = (QUERY_INDEX + (DEGREE_BOUND << BLOWUP_LOG2)) as u64;
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_huge_query_index() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.index = u64::MAX;
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_empty_merkle_path() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.folds[1].merkle_path.clear();
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_non_canonical_partner() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        let modulus: primitive_types::U256 = BS::MODULUS.parse().unwrap();
        proto.folds[2].merkle_path[0] = modulus.to_big_endian().to_vec();
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_leaf_without_values() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.folds[2].leaf_values.clear();
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_leaf_with_two_values() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        let leaf_values = &mut proto.folds[0].leaf_values;
        leaf_values.push(leaf_values[0].clone());
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_long_value() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.folds[1].leaf_values[0].resize(33, 0);
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_non_canonical_value() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        let modulus: primitive_types::U256 = BS::MODULUS.parse().unwrap();
        proto.folds[1].leaf_values[0] = modulus.to_little_endian().to_vec();
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_short_path_hash() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.folds[0].merkle_path[1].pop();
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_serialization() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let commitment = prover.commit();
        let query = prover.query(QUERY_INDEX);
        let decoded =
            Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, &query.to_proto())
                .unwrap();
        assert_eq!(decoded.indices(), query.indices());
        assert_eq!(decoded.len(), query.len());
        assert_eq!(decoded.values(), query.values());
        assert_eq!(decoded.x(), query.x());
        assert!(decoded.verify(&commitment).is_ok());
    }
}
