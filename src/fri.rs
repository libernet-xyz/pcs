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

/// A Merkle proof for a single FRI leaf.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LeafProof<F: Field256, H: Hasher<F>> {
    /// The opened evaluation.
    value: F,
    /// The Merkle path from the leaf up to, but excluding, the root. The first element is the
    /// sibling leaf, that is another evaluation encoded with [`utils::encode_scalar`], and the
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

    /// Verifies the proof against the given root hash.
    fn verify(&self, index: usize, root_hash: H256) -> Result<()> {
        merkle::verify_path::<H>(
            utils::encode_scalar(self.value),
            self.path.as_slice(),
            index,
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
    ///
    /// Returns an error if the leaf doesn't have exactly one value, if the value is not a canonical
    /// field element, or if any hash is invalid.
    fn from_proto(proto: &proto::MerkleProof) -> Result<Self> {
        Ok(Self {
            value: match proto.leaf_values.as_slice() {
                [value] => utils::load_scalar(value.as_slice()),
                values => Err(anyhow!(
                    "invalid FRI leaf size (got {}, want 1)",
                    values.len()
                )),
            }?,
            path: proto
                .merkle_path
                .iter()
                .map(|hash| H::load_hash(hash.as_slice()))
                .collect::<Result<_>>()?,
            _data: PhantomData,
        })
    }
}

/// A Merkle tree whose leaves are the evaluations of a polynomial.
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
        let mut data = Vec::with_capacity(n * 2 - 1);
        data.extend(values.iter().map(|&value| utils::encode_scalar(value)));
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

    /// Returns the evaluation stored in the i-th leaf.
    fn leaf(&self, index: usize) -> F {
        utils::decode_scalar(self.data[index]).unwrap()
    }

    /// Returns a Merkle proof for the leaf at `index`.
    fn query(&self, index: usize) -> LeafProof<F, H> {
        LeafProof {
            value: self.leaf(index),
            path: merkle::merkle_path(self.data.as_slice(), self.num_leaves(), index),
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
    index: usize,
    /// Proves a pair of "partner" values at each folding round with one [`LeafProof`] pair for
    /// every round. The pair at `folds[0]` proves the opened values.
    folds: Vec<(LeafProof<F, H>, LeafProof<F, H>)>,
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
        (self.folds[0].0.value, self.folds[0].1.value)
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

        if self.index >= n {
            return Err(anyhow!(
                "invalid query index {}: the domain size is {n}",
                self.index
            ));
        }

        let mut k = n.trailing_zeros() as usize;

        let num_folds = self.folds.len();
        if num_folds > self.degree_bound.trailing_zeros() as usize + 1 {
            return Err(anyhow!("incorrect proof size"));
        }
        if commitment.len() != num_folds {
            return Err(anyhow!("wrong number of folding rounds"));
        }

        let mut index = self.index;
        let mut pos = self.folds[0].0.value;
        let mut step = Polynomial::<F>::domain_element2(1, n).invert_unwrap();

        for round in 0..num_folds {
            let root_hash = commitment.roots()[round];

            let (left, right) = &self.folds[round];
            if left.len() != k {
                return Err(anyhow!(
                    "invalid left-hand side Merkle proof height (got {}, want {})",
                    left.len(),
                    k
                ));
            }
            if right.len() != k {
                return Err(anyhow!(
                    "invalid right-hand side Merkle proof height (got {}, want {})",
                    right.len(),
                    k
                ));
            }

            if left.value != pos {
                return Err(anyhow!("leaf value mismatch"));
            }
            left.verify(index, root_hash)?;
            right.verify((index + n / 2) % n, root_hash)?;

            let omega_inv = step.pow_small(index);
            n /= 2;
            k -= 1;
            index %= n;

            let neg = right.value;
            let alpha = H::challenge(*FOLD_DST, &[root_hash]);
            pos = (pos + neg + alpha * omega_inv * (pos - neg)) * F::BaseField::TWO_INV;
            step = step.square();
        }

        let (left, right) = self.folds.last().unwrap();
        if !left.is_constant() || !right.is_constant() {
            return Err(anyhow!("the final folded polynomial is not constant"));
        }

        Ok(())
    }

    /// Serializes this query to a [`Query`](`proto::Query`) protobuf.
    pub fn to_proto(&self) -> proto::Query {
        proto::Query {
            index: self.index as u64,
            folds: self
                .folds
                .iter()
                .map(|(left, right)| proto::query::Fold {
                    left: Some(left.to_proto()),
                    right: Some(right.to_proto()),
                })
                .collect(),
        }
    }

    /// Deserializes a query from a [`Query`](`proto::Query`) protobuf.
    ///
    /// REQUIRES: `degree_bound` must be a power of 2, `blowup_log2` must be greater than 0, and the
    /// size of the extended domain must fit within the 2-adicity of `F::BaseField`.
    ///
    /// Every opened value must be a canonical field element, otherwise an error is returned.
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
                .map(|fold| match (&fold.left, &fold.right) {
                    (Some(left), Some(right)) => {
                        Ok((LeafProof::from_proto(left)?, LeafProof::from_proto(right)?))
                    }
                    _ => Err(anyhow!("missing FRI folds")),
                })
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
    /// NOTE: `index` is relative to the *inflated* evaluation domain, so for example if you
    /// committed to 4 evaluations with a blowup factor of 8 the range for `index` is [0, 32).
    pub fn query(&self, index: usize) -> Query<F, H> {
        let mut n = self.degree_bound << self.blowup_log2;
        assert!(index < n);

        let mut i = index;
        let mut folds = vec![];
        for tree in &self.trees {
            folds.push((tree.query(i), tree.query((i + n / 2) % n)));
            n /= 2;
            i %= n;
        }

        {
            let (left, right) = folds.last().unwrap();
            assert!(left.is_constant());
            assert!(right.is_constant());
        }

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
    fn test_tree_leaves_are_big_endian_values() {
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(sample_values().as_slice());
        assert_eq!(
            tree.data[0..8],
            [
                H256::from_low_u64_be(7),
                H256::from_low_u64_be(8),
                H256::from_low_u64_be(11),
                H256::from_low_u64_be(16),
                H256::from_low_u64_be(23),
                H256::from_low_u64_be(32),
                H256::from_low_u64_be(43),
                H256::from_low_u64_be(56),
            ]
        );
    }

    #[test]
    fn test_tree_matches_reference_implementation() {
        let values = sample_values();
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(values.as_slice());
        assert_eq!(tree.data, reference_keccak256_tree(values.as_slice()));
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
        let proof = tree.query(5);
        assert_eq!(proof.value, BS::from_const(32));
    }

    #[test]
    fn test_query_path() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(5);
        assert_eq!(proof.len(), 3);
        assert_eq!(proof.path, [tree.data[4], tree.data[8 + 3], tree.data[12]]);
    }

    #[test]
    fn test_query_path_starts_with_sibling_leaf() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(5);
        assert_eq!(proof.path[0], H256::from_low_u64_be(23));
    }

    #[test]
    fn test_single_leaf_tree_query() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(&[BS::from_const(42)]);
        let proof = tree.query(0);
        assert_eq!(proof.value, BS::from_const(42));
        assert_eq!(proof.path, []);
    }

    #[test]
    fn test_leaf_proof_verify() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(5);
        let result = proof.verify(5, tree.root_hash());
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
        let proof = tree.query(5);
        let result = proof.verify(4, tree.root_hash());
        assert!(result.is_err());
    }

    #[test]
    fn test_leaf_proof_verify_with_out_of_range_index() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(5);
        let result = proof.verify(5 + 8, tree.root_hash());
        assert!(result.is_err());
    }

    #[test]
    fn test_leaf_proof_verify_with_wrong_root() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(5);
        let result = proof.verify(5, tree.data[0]);
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
        let proof = tree.query(5);
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
                tree.data[2].as_bytes().to_vec(),
                tree.data[8].as_bytes().to_vec(),
                tree.data[13].as_bytes().to_vec(),
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
                parse("0xf243320c355c40b383d412c7e1964911d8b0158e9074dcb4c1d3f8b3d47fbe40"),
                parse("0x6ec4f9a41a3d96df76e26c26acebf9b59832c42b5705595f8d9a32bea19e2592"),
                parse("0xc0da902ce501419871c92091cf5a1dc3a18345334e3dc17c5a60ce4b3b45aaeb"),
                parse("0x51f2f631f31206f27d63c68c70623c1e2bb93acd17d125b551dac8444b290eb7"),
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
                parse("0x9b4906d8f66d186d59009303c5caef59460de8a06b17c5bf09082778709f5201"),
                parse("0xe4bc0886fb6edf122ba603607a60ac1c4feb73bf2ffebf306e229a490a55d796"),
                parse("0x87b4a1f5b10087b68593b197901f40f857ad5777892fb6cc58fce34d8bc08e15"),
                parse("0x7bb73139b685ac249fc83972820d410a8be43dab72764445e5679c9a182f39f4"),
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
    fn test_reject_wrong_left_proof_height() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        let mut donor = prover.query(QUERY_INDEX);
        query.folds[0].0 = donor.folds.remove(1).0;
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_wrong_right_proof_height() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        let mut donor = prover.query(QUERY_INDEX);
        query.folds[0].1 = donor.folds.remove(1).1;
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_swapped_partners() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        let pair = &mut query.folds[0];
        std::mem::swap(&mut pair.0, &mut pair.1);
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
        query.folds[1].0 = other.folds.remove(1).0;
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_extra_path_hash() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        query.folds[0].0.path.push(H256::zero());
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_out_of_range_query_index() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        query.index = DEGREE_BOUND << BLOWUP_LOG2;
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_missing_partner_proof() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.folds[0].right = None;
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_missing_proof() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.folds[2].left = None;
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_leaf_without_values() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.folds[2].right.as_mut().unwrap().leaf_values.clear();
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_leaf_with_two_values() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        let leaf_values = &mut proto.folds[0].left.as_mut().unwrap().leaf_values;
        leaf_values.push(leaf_values[0].clone());
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_long_value() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.folds[1].left.as_mut().unwrap().leaf_values[0].resize(33, 0);
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_non_canonical_value() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        let modulus: primitive_types::U256 = BS::MODULUS.parse().unwrap();
        proto.folds[1].right.as_mut().unwrap().leaf_values[0] = modulus.to_little_endian().to_vec();
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_short_path_hash() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2);
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.folds[0].left.as_mut().unwrap().merkle_path[1].pop();
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
