use crate::poseidon;
use crate::utils;
use anyhow::{Result, anyhow};
use ff::{Field, PrimeField};
use sha2::{self, Digest};
use starkom_bluesky::Scalar;
use starkom_poly as poly;
use std::marker::PhantomData;
use std::sync::LazyLock;

type Polynomial = poly::Polynomial<Scalar>;

/// Domain separator tag used when hashing the leaves of a Merkle tree.
static LEAF_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/fri/leaf"));

/// Domain separator tag used in (internal) Merkle tree hashes.
static TREE_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/fri/tree"));

/// Domain separator tag used when deriving the Fiat-Shamir challenge for FRI folding.
static FOLD_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/fri/fold"));

/// The modular inverse of the multiplicative generator, used to correct the coset shift in the FRI
/// fold formula: the standard formula divides by `x = g * omega^i`, so the fold coefficient is
/// `alpha * g^{-1} * omega^{-i}`.
static GENERATOR_INV: LazyLock<Scalar> =
    LazyLock::new(|| Scalar::MULTIPLICATIVE_GENERATOR.invert().unwrap());

/// A generic cryptographic hash backend for use with FRI.
///
/// This trait is used for both binary Merkle trees and Fiat-Shamir challenges.
pub trait Hash {
    /// Hashes two input scalars with a DST.
    fn hash_raw(dst: Scalar, input1: Scalar, input2: Scalar) -> Scalar;

    /// Hashes two input scalars using the `TREE_DST`.
    ///
    /// This is used for hashing Merkle tree nodes.
    fn hash(input1: Scalar, input2: Scalar) -> Scalar {
        Self::hash_raw(*TREE_DST, input1, input2)
    }

    /// Hashes many input scalars.
    fn hash_many(inputs: &[Scalar]) -> Scalar;
}

/// Hashes two scalars using SHA-256.
///
/// This hashing backend is designed to be EVM-friendly so that our FRI proofs can be easily
/// verified in the EVM.
///
/// In order to convert the result to a BlueSky scalar without any significant distribution skew we
/// actually need a 512-bit hash, so that we can perform the conversion via modular reduction.
/// However we want to use the 256-bit version of SHA2 for compatibility with the EVM. So we obtain
/// a 512-bit hash as follows:
///
///   * the low 256 bits of the hash are Sha2_256(0 || input1 || input2),
///   * the high 256 bits of the hash are Sha2_256(1 || input1 || input2),
///   * the 512 bits are converted to a scalar via modular reduction.
#[derive(Debug, Default, Copy, Clone)]
pub struct Sha2Hash {}

impl Sha2Hash {
    fn hash_internal(inputs: impl IntoIterator<Item = Scalar>) -> [u8; 32] {
        let mut hasher = sha2::Sha256::default();
        for input in inputs {
            hasher.update(input.to_big_endian());
        }
        let mut bytes: [u8; 32] = hasher.finalize().into();
        bytes.reverse();
        bytes
    }
}

impl Hash for Sha2Hash {
    fn hash_raw(dst: Scalar, input1: Scalar, input2: Scalar) -> Scalar {
        let lo = Self::hash_internal([Scalar::ZERO, dst, input1, input2]);
        let hi = Self::hash_internal([Scalar::ONE, dst, input1, input2]);
        let mut bytes = [0u8; 64];
        bytes[0..32].copy_from_slice(&lo);
        bytes[32..64].copy_from_slice(&hi);
        Scalar::from_repr_wide(&bytes)
    }

    fn hash_many(inputs: &[Scalar]) -> Scalar {
        let lo = Self::hash_internal(std::iter::once(Scalar::ZERO).chain(inputs.iter().cloned()));
        let hi = Self::hash_internal(std::iter::once(Scalar::ONE).chain(inputs.iter().cloned()));
        let mut bytes = [0u8; 64];
        bytes[0..32].copy_from_slice(&lo);
        bytes[32..64].copy_from_slice(&hi);
        Scalar::from_repr_wide(&bytes)
    }
}

/// Hashes two scalars using Poseidon2.
#[derive(Debug, Default, Copy, Clone)]
pub struct Poseidon2Hash {}

impl Hash for Poseidon2Hash {
    fn hash_raw(dst: Scalar, input1: Scalar, input2: Scalar) -> Scalar {
        poseidon::hash_t4(&[dst, input1, input2])
    }

    fn hash_many(inputs: &[Scalar]) -> Scalar {
        poseidon::hash_t4(inputs)
    }
}

/// Hashes a leaf of a Merkle tree.
fn hash_leaf<H: Hash>(values: &[Scalar]) -> Scalar {
    H::hash_many(
        std::iter::once(*LEAF_DST)
            .chain(std::iter::once(Scalar::from(values.len() as u64)))
            .chain(values.iter().cloned())
            .collect::<Vec<Scalar>>()
            .as_slice(),
    )
}

/// Computes all Merkle hashes of a vector of values up to the root.
///
/// `n` is the number of values and must be a power of two.
///
/// The full Merkle tree is stored inline in the `values` vector as follows:
///
///   * the first `n` elements are the values of the original vector,
///   * the next `n / 2` elements are the hashes of the second-last layer of the tree,
///   * the next `n / 4` elements are the hashes of the third-last layer of the tree,
///   * ...
///   * the last stored element is the Merkle root.
///
/// It's the caller's responsibility to ensure the `values` array has at least `n * 2 - 1` slots so
/// that the full tree can be stored.
///
/// Note that the Merkle root will be at index `(n - 1) * 2`.
///
/// Note about usage: the Merkle trees we use in this module have scalar *vectors* for leaves, not
/// just scalars.
pub fn merklify<H: Hash>(mut values: &mut [Scalar], mut n: usize) {
    assert!(n.is_power_of_two());
    while n > 1 {
        let m = n / 2;
        for j in 0..m {
            values[n + j] = H::hash(values[j * 2], values[j * 2 + 1]);
        }
        values = &mut values[n..];
        n = m;
    }
}

/// Stores the Merkle root hashes of a FRI commitment.
///
/// Note that for low-degree testing these are *less* than log2(N), with N being the number of
/// committed evaluations. Once the folding process has reduced all polynomials to degree-0 ones
/// (that is, single constants) all subsequent folds would be identical, so we don't store them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment {
    /// The first element in the array is the root of the main Merkle tree, the second one is the
    /// root of the Merkle tree from the first folding round, and so on until the last element which
    /// is the value of the last folding round.
    roots: Vec<Scalar>,
}

impl Commitment {
    /// Returns the number of stored roots, equivalent to the number of folding rounds and therefore
    /// to the log2 of the degree bound plus one. For example, if the user commits 4 evaluations
    /// `len()` will return 3.
    pub fn len(&self) -> usize {
        self.roots.len()
    }

    /// Returns the Merkle roots of all folding rounds.
    ///
    /// The returned slice has `len()` elements.
    pub fn roots(&self) -> &[Scalar] {
        self.roots.as_slice()
    }

    /// Returns the Merkle root hash of the committed polynomial, which is the first hash stored in
    /// the commitment.
    pub fn root(&self) -> Scalar {
        *self.roots.first().unwrap()
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
pub struct LeafProof<H: Hash> {
    leaf: Vec<Scalar>,
    path: Vec<Scalar>,
    _data: PhantomData<H>,
}

impl<H: Hash> LeafProof<H> {
    /// Returns a reference to the leaf values (one for every committed polynomial).
    pub fn leaf(&self) -> &[Scalar] {
        self.leaf.as_slice()
    }

    /// Checks the leaf of this proof against the provided slice.
    ///
    /// The two must match or an error is returned.
    pub fn check_leaf(&self, expected: &[Scalar]) -> Result<()> {
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
    pub fn len(&self) -> usize {
        self.path.len()
    }

    /// Verifies the proof against the given root hash.
    pub fn verify(&self, mut index: usize, root_hash: Scalar) -> Result<()> {
        let mut hash = hash_leaf::<H>(self.leaf.as_slice());
        for sibling in &self.path {
            hash = if index & 1 != 0 {
                H::hash(*sibling, hash)
            } else {
                H::hash(hash, *sibling)
            };
            index >>= 1;
        }
        if index != 0 {
            return Err(anyhow!("invalid index"));
        }
        if hash != root_hash {
            return Err(anyhow!(
                "root hash mismatch (got {}, want {})",
                utils::format_scalar(hash),
                utils::format_scalar(root_hash)
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
    pub fn is_constant(&self) -> bool {
        let mut hash = hash_leaf::<H>(self.leaf.as_slice());
        for &sibling in &self.path {
            if sibling != hash {
                return false;
            }
            hash = H::hash(hash, hash);
        }
        true
    }
}

/// A Merkle tree whose leaves are multiple polynomial evaluations.
///
/// The tree has N leaf in total, with N being the size of the extended domain, and each leaf has K
/// polynomial evaluations, with K being the number of committed polynomials.
///
/// The internal nodes are single hashes.
#[derive(Debug, Clone)]
pub struct Tree<H: Hash> {
    /// Number of polynomials in the tree. This is the number of values in each leaf.
    num_polys: usize,
    /// The leaves of the tree (N leaves with K evaluations each).
    leaves: Vec<Vec<Scalar>>,
    /// The internal nodes of the tree. There are 2*N-1 nodes in this array, with N = number of
    /// leaves. The nodes of the bottom layer are the hashes of the corresponding leaves.
    hashes: Vec<Scalar>,
    _data: PhantomData<H>,
}

impl<H: Hash> Tree<H> {
    /// Constructs a Merkle tree from a matrix of polynomial evaluations.
    ///
    /// More than one polynomial can be batched in the same tree because we our tree leaves are
    /// vectors rather than single scalars. The only requirement is that all polynomials have the
    /// same number of evaluations (not necessarily the same degree).
    ///
    /// The provided `leaves` array has one entry for each leaf, and each leaf is a vector of K
    /// polynomial evaluations, with K = number of batched polynomials.
    ///
    /// Neither the outer array nor the inner arrays can be empty.
    pub fn from_leaves(leaves: Vec<Vec<Scalar>>) -> Self {
        let num_polys = leaves[0].len();
        assert!(num_polys > 0);
        let n = leaves.len();
        assert!(n.is_power_of_two());
        let mut hashes = vec![Scalar::ZERO; n * 2 - 1];
        for i in 0..n {
            let leaf = leaves[i].as_slice();
            assert_eq!(leaf.len(), num_polys);
            hashes[i] = hash_leaf::<H>(leaf);
        }
        merklify::<H>(hashes.as_mut_slice(), n);
        Self {
            num_polys,
            leaves,
            hashes,
            _data: Default::default(),
        }
    }

    /// Constructs a Merkle tree from a matrix of polynomial evaluations.
    ///
    /// The outer array of `values` contains one entry per committed polynomial, and each of the
    /// inner arrays represents the evaluations of a polynomial.
    ///
    /// Therefore `values` has as many elements as the number of polynomials being committed and the
    /// length of the inner arrays must equal the size of the (extended) evaluation domain.
    ///
    /// Note that the only difference between `new` and `from_leaves` is that the dimensions of the
    /// provided matrix are inverted.
    ///
    /// Neither the outer array nor the inner arrays can be empty.
    pub fn new(values: Vec<Vec<Scalar>>) -> Self {
        let k = values.len();
        assert!(k > 0);
        let n = values[0].len();
        let leaves: Vec<Vec<Scalar>> = (0..n)
            .map(|i| {
                (0..k)
                    .map(|j| {
                        assert_eq!(n, values[j].len());
                        values[j][i]
                    })
                    .collect()
            })
            .collect();
        Self::from_leaves(leaves)
    }

    /// Returns the number of polynomials stored in the tree.
    ///
    /// Each leaf of the tree has this number of values.
    pub fn num_polys(&self) -> usize {
        self.num_polys
    }

    /// Returns the number of leaves in the tree, corresponding to the size of the evaluation domain
    /// (always a power of 2).
    pub fn num_leaves(&self) -> usize {
        self.leaves.len()
    }

    /// Returns the root hash of the Merkle tree.
    pub fn root_hash(&self) -> Scalar {
        let n = self.leaves.len();
        self.hashes[(n - 1) * 2]
    }

    /// Returns a reference to the i-th leaf.
    ///
    /// Note that the leaf contains k elements, one for every committed polynomial.
    pub fn leaf(&self, index: usize) -> &[Scalar] {
        self.leaves[index].as_slice()
    }

    /// Returns a Merkle proof for the leaf at `index`.
    pub fn query(&self, mut index: usize) -> LeafProof<H> {
        let mut n = self.leaves.len();
        assert!(n.is_power_of_two());
        assert!(index < n);
        let leaf = self.leaves[index].clone();
        let mut path = Vec::with_capacity(n.trailing_zeros() as usize);
        let mut hashes = self.hashes.as_slice();
        while n > 1 {
            path.push(hashes[index ^ 1]);
            hashes = &hashes[n..];
            n /= 2;
            index >>= 1;
        }
        LeafProof {
            leaf,
            path,
            _data: Default::default(),
        }
    }

    /// Performs one FRI folding round, returning the new folded tree.
    fn fold(&self) -> Self {
        let n = self.leaves.len();
        assert!(n.is_power_of_two());

        let alpha = H::hash_raw(*FOLD_DST, self.hashes[(n - 1) * 2], Scalar::ZERO) * *GENERATOR_INV;

        let k = n.trailing_zeros();
        let omega_inv = Scalar::ROOT_OF_UNITY_INV.pow_vartime([1u64 << (Scalar::S - k), 0, 0, 0]);

        let m = n / 2;
        let mut omega_inv_i = Scalar::ONE;

        let mut leaves = Vec::with_capacity(m);
        for i in 0..m {
            let pos = self.leaves[i].as_slice();
            let neg = self.leaves[i + m].as_slice();
            leaves.push(
                pos.iter()
                    .cloned()
                    .zip(neg.iter().cloned())
                    .map(|(pos, neg)| {
                        (pos + neg + alpha * omega_inv_i * (pos - neg)) * Scalar::TWO_INV
                    })
                    .collect::<Vec<Scalar>>(),
            );
            omega_inv_i *= omega_inv;
        }

        Self::from_leaves(leaves)
    }

    /// Performs `times` FRI folding and returns an array of `times+1` trees.
    ///
    /// The first element is `self` (N leaves), the second element is the tree from the first
    /// folding round (N/2 leaves), the third element is the tree from the second folding round (N/4
    /// leaves), and so on.
    fn fold_all(self, times: usize) -> Vec<Self> {
        let mut trees = Vec::with_capacity(times + 1);
        let mut tree = self;
        for _ in 0..times {
            let folded = tree.fold();
            trees.push(tree);
            tree = folded;
        }
        trees.push(tree);
        trees
    }
}

#[derive(Debug, Clone)]
pub struct Query<H: Hash> {
    /// The number of committed evaluations.
    n: usize,
    /// The index of the element we're opening (the partner index is inferred automatically).
    index: usize,
    /// Proves a pair of "partner" values at each folding round with one `LeafProof` pair for every
    /// round. The pair at `folds[0]` proves the opened values (stored in `values`).
    folds: Vec<(LeafProof<H>, LeafProof<H>)>,
    _data: PhantomData<H>,
}

impl<H: Hash> Query<H> {
    /// Returns the two opened indices.
    pub fn indices(&self) -> (usize, usize) {
        (self.index, (self.index + self.n / 2) % self.n)
    }

    /// Returns the opened domain element, that is the X-coordinate of the evaluation.
    ///
    /// This is the element corresponding to the first value returned by `indices`, while the
    /// partner element can be obtained by simply negating this one.
    ///
    /// Note that we use `Polynomial::shifted_lde2` when committing polynomials, so the element
    /// returned here is a shifted power of an N-th root of unity, with
    /// `N = degree_bound * 2^blowup_factor`. The shift consists of multiplying the actual domain
    /// element by `Scalar::MULTIPLICATIVE_GENERATOR`, consistently with `shifted_lde2`.
    pub fn x(&self) -> Scalar {
        Polynomial::coset_element2(self.index, self.n)
    }

    /// Returns the opened evaluations, one for every committed polynomial.
    ///
    /// The first component of the returned tuple contains the evaluations at the first index
    /// returned by `indices`, while the second component contains those at the second index.
    pub fn values(&self) -> (&[Scalar], &[Scalar]) {
        (self.folds[0].0.leaf(), self.folds[0].1.leaf())
    }

    /// Returns the number of folding rounds.
    ///
    /// In general these are log2(d), with `d` being the degree bound of the committed polynomial.
    /// Note that for low-degree testing `d` is strictly less than the number of committed
    /// evaluations `N`.
    pub fn len(&self) -> usize {
        return self.folds.len();
    }

    /// Verifies this proof against the given commitment.
    ///
    /// NOTE: for low-degree testing you also need to check that `len()` returns the log2 of the
    /// expected degree bound. This function only verifies the opened value pair across the folding
    /// structure.
    pub fn verify(&self, commitment: &Commitment) -> Result<()> {
        assert!(self.n.is_power_of_two());
        assert!(self.index < self.n);
        let k = self.n.trailing_zeros();

        let folds = self.folds.as_slice();

        let h = folds.len();
        if h > k as usize + 1 {
            return Err(anyhow!("invalid proof size"));
        }
        if commitment.len() != h {
            return Err(anyhow!("wrong number of folding rounds"));
        }

        let mut m = self.n;
        let mut index = self.index;
        let mut pos = self.folds[0].0.leaf().to_vec();
        let mut step = Scalar::ROOT_OF_UNITY_INV.pow_vartime([1u64 << (Scalar::S - k), 0, 0, 0]);

        for r in 0..h {
            let (left, right) = &folds[r];
            let root_hash = commitment.roots()[r];
            let alpha = H::hash_raw(*FOLD_DST, root_hash, Scalar::ZERO) * *GENERATOR_INV;
            let neg = right.leaf();

            if 1usize << left.len() != m {
                return Err(anyhow!(
                    "invalid left-hand side Merkle proof height (got {}, want {})",
                    left.len(),
                    m.trailing_zeros()
                ));
            }
            if 1usize << right.len() != m {
                return Err(anyhow!(
                    "invalid right-hand side Merkle proof height (got {}, want {})",
                    right.len(),
                    m.trailing_zeros()
                ));
            }

            left.check_leaf(pos.as_slice())?;
            left.verify(index, root_hash)?;
            right.verify((index + m / 2) % m, root_hash)?;

            let omega_inv_i = step.pow_vartime([index as u64, 0, 0, 0]);
            m /= 2;
            index %= m;

            for i in 0..pos.len() {
                pos[i] =
                    (pos[i] + neg[i] + alpha * omega_inv_i * (pos[i] - neg[i])) * Scalar::TWO_INV;
            }
            step = step.square();
        }

        let (left, right) = folds.last().unwrap();
        if !left.is_constant() || !right.is_constant() {
            return Err(anyhow!("final folded polynomial is not constant"));
        }

        Ok(())
    }
}

/// A FRI prover.
///
/// The struct contains the main Merkle tree built on the committed polynomial(s) and the Merkle
/// trees of all folded polynomials up to and including the one where all polynomials have been
/// folded into constant ones. Note that the final Merkle tree still has more than one leaf due to
/// the low-degree extension.
#[derive(Debug, Clone)]
pub struct Prover<H: Hash> {
    /// The degree bound of the committed polynomials. This is the highest degree among the
    /// committed polynomials, plus one.
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// The Merkle trees, one for the original polynomials plus one for every folding round.
    /// `trees[0]` is the tree built over the original polynomial evaluations, `trees[1]` is the
    /// tree resulting from the first folding round, etc.
    trees: Vec<Tree<H>>,
}

impl<H: Hash> Prover<H> {
    pub fn new(polynomials: Vec<Polynomial>, degree_bound: usize, blowup_log2: usize) -> Self {
        assert!(degree_bound.is_power_of_two());
        assert!(
            polynomials
                .iter()
                .all(|polynomial| degree_bound >= polynomial.degree_bound())
        );

        let n = degree_bound << blowup_log2;
        assert!(n <= 1usize << Scalar::S);

        let main_tree = Tree::<H>::new(
            polynomials
                .into_iter()
                .map(|polynomial| polynomial.shifted_lde2(n))
                .collect(),
        );
        let trees = main_tree.fold_all(degree_bound.trailing_zeros() as usize);

        Self {
            degree_bound,
            blowup_log2,
            trees,
        }
    }

    /// Returns the degree bound of the committed polynomials (always a power of 2).
    ///
    /// NOTE: the actual degree of the original polynomials is often even lower than this value
    /// because it was rounded up to the next power of 2 in order to run the FFT and FRI algorithms.
    pub fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    /// Returns the size of the extended domain, equal to `degree_bound * 2^blowup_log2`.
    pub fn extended_domain_size(&self) -> usize {
        self.degree_bound << self.blowup_log2
    }

    /// Alias for `extended_domain_size`.
    pub fn size(&self) -> usize {
        self.degree_bound << self.blowup_log2
    }

    /// Returns the Merkle root hash of the committed polynomials.
    ///
    /// This is equivalent to the first root stored in the commiment returned by `commit()`.
    pub fn root_hash(&self) -> Scalar {
        self.trees[0].root_hash()
    }

    /// Creates the FRI commitment for the batched polynomials.
    pub fn commit(&self) -> Commitment {
        Commitment {
            roots: self.trees.iter().map(|tree| tree.root_hash()).collect(),
        }
    }

    /// Builds a FRI `Query` for the value at the specified index of the evaluation domain.
    ///
    /// NOTE: `index` is relative to the *inflated* evaluation domain, so for example if you
    /// committed to 4 evaluations with a blowup factor of 8 the range for `index` is [0, 32).
    pub fn query(&self, index: usize) -> Query<H> {
        let d = self.degree_bound;
        assert!(d.is_power_of_two());

        let n = self.degree_bound << self.blowup_log2;
        assert!(index < n);

        let mut m = n;
        let mut i = index;
        let mut folds = vec![];
        for tree in &self.trees {
            folds.push((tree.query(i), tree.query((i + m / 2) % m)));
            m /= 2;
            i %= m;
        }

        {
            let (left, right) = folds.last().unwrap();
            assert!(left.is_constant());
            assert!(right.is_constant());
        }

        Query {
            n,
            index,
            folds,
            _data: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::parse_scalar;

    #[test]
    fn test_merklify_one_sha2() {
        let mut values = vec![12.into()];
        merklify::<Sha2Hash>(&mut values, 1);
        assert_eq!(values, vec![12.into()]);
    }

    #[test]
    fn test_merklify_one_poseidon2() {
        let mut values = vec![12.into()];
        merklify::<Poseidon2Hash>(&mut values, 1);
        assert_eq!(values, vec![12.into()]);
    }

    #[test]
    fn test_merklify_two_sha2() {
        let mut values = vec![34.into(), 56.into()];
        values.resize(3, 0.into());
        merklify::<Sha2Hash>(&mut values, 2);
        assert_eq!(
            values,
            vec![
                34.into(),
                56.into(),
                parse_scalar("0x1925057b41724a77ee2aa82257dd8dbbe7dc6ee25c0a66d39865f1b61160438e")
            ]
        );
    }

    #[test]
    fn test_merklify_two_poseidon2() {
        let mut values = vec![34.into(), 56.into()];
        values.resize(3, 0.into());
        merklify::<Poseidon2Hash>(&mut values, 2);
        assert_eq!(
            values,
            vec![
                34.into(),
                56.into(),
                parse_scalar("0x2e3b901f893e1a7d2bea5145aa3e7b1b7381d97cf6f6169c916135e63c3796e6")
            ]
        );
    }

    #[test]
    fn test_merklify_four_sha2() {
        let mut values = vec![78.into(), 90.into(), 12.into(), 34.into()];
        values.resize(7, 0.into());
        merklify::<Sha2Hash>(&mut values, 4);
        assert_eq!(
            values,
            vec![
                78.into(),
                90.into(),
                12.into(),
                34.into(),
                parse_scalar("0x01ac1709c71f17e1e46474406072915ad35f485293db808081d3874462f8fc07"),
                parse_scalar("0x0e9d68b650e1e39336aca540be0eb8861a91ed35f479dd3f426873f8373772b0"),
                parse_scalar("0x249388b1c719a74f0c41935d9982cd683ea095b1a92c985a697a4c6763849ed7"),
            ]
        );
    }

    #[test]
    fn test_merklify_four_poseidon2() {
        let mut values = vec![78.into(), 90.into(), 12.into(), 34.into()];
        values.resize(7, 0.into());
        merklify::<Poseidon2Hash>(&mut values, 4);
        assert_eq!(
            values,
            vec![
                78.into(),
                90.into(),
                12.into(),
                34.into(),
                parse_scalar("0x52f3bc8f4ace8ef188a53324832de01b29de527f57a8a4084eda67d2e73a5885"),
                parse_scalar("0x5e4a742eb4809793ef06d6c6f9878040bb34f2058f3aacb3b9eca24c56203c36"),
                parse_scalar("0x4097efe42a882fcb78457cbc0549a00eeb1f923ea6ce0a210ea3b95320ec2cf1"),
            ]
        );
    }

    fn test_merkle_tree<H: Hash>(leaves: Vec<Vec<Scalar>>, expected_root_hash: Scalar) {
        let tree = Tree::<H>::from_leaves(leaves.clone());
        assert_eq!(tree.num_polys(), leaves[0].len());
        assert_eq!(tree.num_leaves(), leaves.len());
        assert_eq!(tree.root_hash(), expected_root_hash);
        for i in 0..leaves.len() {
            let leaf = &leaves[i];
            let proof = tree.query(i);
            assert!(proof.verify(i, expected_root_hash).is_ok());
            assert_eq!(proof.leaf().len(), leaf.len());
            assert!(
                proof
                    .leaf()
                    .iter()
                    .zip(leaf.iter())
                    .all(|(&lhs, &rhs)| lhs == rhs)
            );
        }
    }

    #[test]
    fn test_merkle_tree_one_leaf_1() {
        test_merkle_tree::<Sha2Hash>(
            vec![vec![12.into()]],
            parse_scalar("0x20e662747ccd53b24b82f95803effc667c97695debcce4cff135b903440f616b"),
        );
        test_merkle_tree::<Poseidon2Hash>(
            vec![vec![12.into()]],
            parse_scalar("0x7ddaad1f5a5863603fe031376b604a9b26c714c0b3488de169918a2fa7fab7a3"),
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_2() {
        test_merkle_tree::<Sha2Hash>(
            vec![vec![34.into()]],
            parse_scalar("0x2b4c08d0c960dcb7e0bc85b36122669832296335cb97ec2e47cbd660679515b7"),
        );
        test_merkle_tree::<Poseidon2Hash>(
            vec![vec![34.into()]],
            parse_scalar("0x13d7d4388040c638dd2c8f8ed1113c0c4144daa222be09a5e54fd637350895b3"),
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_two_polynomials_1() {
        test_merkle_tree::<Sha2Hash>(
            vec![vec![12.into(), 34.into()]],
            parse_scalar("0x2bd421d452e909b084c15b35ab934e06988d7589d61b10c9477eda62f00a420c"),
        );
        test_merkle_tree::<Poseidon2Hash>(
            vec![vec![12.into(), 34.into()]],
            parse_scalar("0x0d5c8dbe08ef0cd4ef927e04844ff9b7d18799ab459259e66ba7fd170a645e71"),
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_two_polynomials_2() {
        test_merkle_tree::<Sha2Hash>(
            vec![vec![34.into(), 12.into()]],
            parse_scalar("0x4a3fdf5b16a84409b2dfe335aa95597e836ab5e8cb38562f85506b3aab91f7e1"),
        );
        test_merkle_tree::<Poseidon2Hash>(
            vec![vec![34.into(), 12.into()]],
            parse_scalar("0x2f406ed7abfbf48e4294d3e3a44aacf222a0cc69f07010ce6ccd3909626da81a"),
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_three_polynomials_1() {
        test_merkle_tree::<Sha2Hash>(
            vec![vec![12.into(), 34.into(), 56.into()]],
            parse_scalar("0x4a06e35b47ecbc90a70ec2276d3167174696255526e2f73f5632b74f4f97bfd0"),
        );
        test_merkle_tree::<Poseidon2Hash>(
            vec![vec![12.into(), 34.into(), 56.into()]],
            parse_scalar("0x45bfbd23c174373739c52cf6fe45827ec7f99644b1b2160fb3869f8f00c61e9c"),
        );
    }

    #[test]
    fn test_merkle_tree_one_leaf_three_polynomials_2() {
        test_merkle_tree::<Sha2Hash>(
            vec![vec![34.into(), 12.into(), 78.into()]],
            parse_scalar("0x78217d81af1a982f2e5edd2cc3c6a4f04a16dddcfe72c4e9d769a2222c65a89f"),
        );
        test_merkle_tree::<Poseidon2Hash>(
            vec![vec![34.into(), 12.into(), 78.into()]],
            parse_scalar("0x64ed329161263f42edc834ea7ad2496e82411fe9980bfabfbcf611717af96b00"),
        );
    }

    #[test]
    fn test_merkle_tree_two_leaves_1() {
        test_merkle_tree::<Sha2Hash>(
            vec![vec![12.into()], vec![34.into()]],
            parse_scalar("0x30ea933bacd6b5f5a456e5969fc75f81714a03180751e7189e5da2b842449388"),
        );
        test_merkle_tree::<Poseidon2Hash>(
            vec![vec![12.into()], vec![34.into()]],
            parse_scalar("0x72e3aab3f8dfa490d8e0b20359d30a816ecd33073c5110aa6af56dfa72c3a868"),
        );
    }

    #[test]
    fn test_merkle_tree_two_leaves_2() {
        test_merkle_tree::<Sha2Hash>(
            vec![vec![34.into()], vec![56.into()]],
            parse_scalar("0x6f23fcc629174ff1f7c5f135a4e90326f27e07dbbaeea513f4b881b74e8332fb"),
        );
        test_merkle_tree::<Poseidon2Hash>(
            vec![vec![34.into()], vec![56.into()]],
            parse_scalar("0x597cf71ccf978d4d78ff66f9ef863515471438a7b97a05f9d20b981839fd2efa"),
        );
    }

    #[test]
    fn test_merkle_tree_two_leaves_two_polynomials_1() {
        test_merkle_tree::<Sha2Hash>(
            vec![vec![12.into(), 34.into()], vec![56.into(), 78.into()]],
            parse_scalar("0x2582a680364687a9ed97c7f950b7c5c2aa98fc73b4ca34d39c5ad816781848f7"),
        );
        test_merkle_tree::<Poseidon2Hash>(
            vec![vec![12.into(), 34.into()], vec![56.into(), 78.into()]],
            parse_scalar("0x5517bc558a108119a0487c9745e1363a6115098f803910da43e6583587c8ad33"),
        );
    }

    #[test]
    fn test_merkle_tree_two_leaves_two_polynomials_2() {
        test_merkle_tree::<Sha2Hash>(
            vec![vec![78.into(), 56.into()], vec![34.into(), 12.into()]],
            parse_scalar("0x6b61ee4a92496285fead9e24543d5806a4c3a40a9693a199c75a91c59e08c23c"),
        );
        test_merkle_tree::<Poseidon2Hash>(
            vec![vec![78.into(), 56.into()], vec![34.into(), 12.into()]],
            parse_scalar("0x6c703fabc10666cc2fcd96ac4c4fe40128bd862c31f32ec5e120bd12524b3a8d"),
        );
    }

    fn test_prover_impl<H: Hash>(
        polynomials: Vec<Polynomial>,
        degree_bound: usize,
        blowup_log2: usize,
    ) {
        let prover = Prover::<H>::new(polynomials, degree_bound, blowup_log2);
        assert_eq!(prover.degree_bound(), degree_bound);
        let n = degree_bound << blowup_log2;
        assert_eq!(prover.extended_domain_size(), n);
        let commitment = prover.commit();
        for i in 0..n {
            let query = prover.query(i);
            assert_eq!(query.indices(), (i, (i + n / 2) % n));
            assert_eq!(query.len(), degree_bound.trailing_zeros() as usize + 1);
            assert!(query.verify(&commitment).is_ok());
        }
    }

    fn test_prover(polynomials: Vec<Polynomial>, degree_bound: usize) {
        test_prover_impl::<Sha2Hash>(polynomials.clone(), degree_bound, 1);
        test_prover_impl::<Poseidon2Hash>(polynomials.clone(), degree_bound, 1);
        test_prover_impl::<Sha2Hash>(polynomials.clone(), degree_bound, 2);
        test_prover_impl::<Poseidon2Hash>(polynomials.clone(), degree_bound, 2);
        test_prover_impl::<Sha2Hash>(polynomials.clone(), degree_bound, 3);
        test_prover_impl::<Poseidon2Hash>(polynomials.clone(), degree_bound, 3);
    }

    #[test]
    fn test_one_constant_polynomial() {
        test_prover(vec![Polynomial::with_coefficients(vec![12.into()])], 1);
        test_prover(vec![Polynomial::with_coefficients(vec![34.into()])], 1);
    }

    #[test]
    fn test_two_constant_polynomials() {
        test_prover(
            vec![
                Polynomial::with_coefficients(vec![12.into()]),
                Polynomial::with_coefficients(vec![34.into()]),
            ],
            1,
        );
    }

    #[test]
    fn test_three_constant_polynomials() {
        test_prover(
            vec![
                Polynomial::with_coefficients(vec![34.into()]),
                Polynomial::with_coefficients(vec![56.into()]),
                Polynomial::with_coefficients(vec![78.into()]),
            ],
            1,
        );
    }

    #[test]
    fn test_one_polynomial_degree_one() {
        test_prover(
            vec![Polynomial::with_coefficients(vec![12.into(), 34.into()])],
            2,
        );
        test_prover(
            vec![Polynomial::with_coefficients(vec![56.into(), 78.into()])],
            2,
        );
    }

    #[test]
    fn test_two_polynomials_degree_one() {
        test_prover(
            vec![
                Polynomial::with_coefficients(vec![12.into(), 34.into()]),
                Polynomial::with_coefficients(vec![56.into(), 78.into()]),
            ],
            2,
        );
    }

    #[test]
    fn test_three_polynomials_degree_one() {
        test_prover(
            vec![
                Polynomial::with_coefficients(vec![34.into(), 56.into()]),
                Polynomial::with_coefficients(vec![56.into(), 78.into()]),
                Polynomial::with_coefficients(vec![78.into(), 90.into()]),
            ],
            2,
        );
    }

    #[test]
    fn test_one_polynomial_degree_three() {
        test_prover(
            vec![Polynomial::with_coefficients(vec![
                12.into(),
                34.into(),
                56.into(),
                78.into(),
            ])],
            4,
        );
        test_prover(
            vec![Polynomial::with_coefficients(vec![
                42.into(),
                43.into(),
                44.into(),
                45.into(),
            ])],
            4,
        );
    }

    #[test]
    fn test_two_polynomials_degree_three() {
        test_prover(
            vec![
                Polynomial::with_coefficients(vec![12.into(), 34.into(), 56.into(), 78.into()]),
                Polynomial::with_coefficients(vec![42.into(), 43.into(), 44.into(), 45.into()]),
            ],
            4,
        );
    }

    #[test]
    fn test_three_polynomials_degree_three() {
        test_prover(
            vec![
                Polynomial::with_coefficients(vec![42.into(), 43.into(), 44.into(), 45.into()]),
                Polynomial::with_coefficients(vec![12.into(), 34.into(), 56.into(), 78.into()]),
                Polynomial::with_coefficients(vec![34.into(), 56.into(), 78.into(), 90.into()]),
            ],
            4,
        );
    }
}
