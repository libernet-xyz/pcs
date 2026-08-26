use crate::fri;
use crate::hash::Hasher;
use crate::merkle::Tree;
use primitive_types::{H256, U256};
use sha2::Digest;
use starkom_ff::{Field, Field256, PrimeField};
use starkom_poly::Polynomial;
use std::marker::PhantomData;
use std::sync::LazyLock;

/// Target security level in bits.
pub const LAMBDA: usize = 128;

/// Domain separator tag for the Fiat-Shamir challenge used to derive query indices.
static QUERY_DST: LazyLock<H256> = LazyLock::new(|| {
    let mut hasher = sha3::Sha3_256::new();
    hasher.update(b"starkom/deep/query");
    H256::from_slice(hasher.finalize().as_slice())
});

/// Returns the number of FRI queries required to achieve 128-bit security using a blowup factor of
/// `2^blowup_log2`.
fn num_queries(blowup_log2: usize) -> usize {
    LAMBDA.div_ceil(blowup_log2)
}

/// Computes a random linear combination of a list of values.
///
/// `alpha` is a Fiat-Shamir challenge of some sort.
fn rlc<F: Field>(values: impl IntoIterator<Item = F>, alpha: F) -> F {
    let mut rlc = F::ZERO;
    let mut pow = F::ONE;
    for value in values.into_iter() {
        rlc += value * pow;
        pow *= alpha;
    }
    rlc
}

/// A batched DEEP-FRI polynomial commitment (see [`Committer`] for details).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment<F: PrimeField, G: Field256 + From<F>, H: Hasher<G>> {
    /// The root hashes of the Merkle trees where the evaluations of all batched polynomials are
    /// stored. There is one root hash per polynomial batch.
    tree_roots: Vec<H256>,
    /// The underlying FRI commitment.
    inner: fri::Commitment,
    _data: PhantomData<(F, G, H)>,
}

impl<F: PrimeField, G: Field256 + From<F>, H: Hasher<G>> Commitment<F, G, H> {
    /// Returns the root hashes of the Merkle trees where all batched polynomials are stored.
    pub fn tree_roots(&self) -> &[H256] {
        self.tree_roots.as_slice()
    }

    /// Encodes a `usize` into a `H256` for use in a transcript to derive the Fiat-Shamir query
    /// indices.
    fn encode_usize(value: usize) -> H256 {
        let mut bytes = [0u8; 32];
        bytes[0..8].copy_from_slice(&(value as u64).to_be_bytes());
        H256::from_slice(&bytes)
    }

    /// Returns the FRI query indices derived via Fiat-Shamir from the full commitment transcript
    /// (all polynomial and FRI Merkle root hashes).
    fn get_query_indices(&self, degree_bound: usize, blowup_log2: usize) -> Vec<usize> {
        let n = U256::from((degree_bound << blowup_log2) as u64);
        let k = num_queries(blowup_log2);
        let mut indices = Vec::with_capacity(k);
        for i in 0..k {
            let hash = H::challenge(
                *QUERY_DST,
                std::iter::once(Self::encode_usize(self.tree_roots.len()))
                    .chain(self.tree_roots.iter().copied())
                    .chain(std::iter::once(Self::encode_usize(self.inner.len())))
                    .chain(self.inner.roots().iter().copied())
                    .chain(std::iter::once(Self::encode_usize(i)))
                    .collect::<Vec<H256>>()
                    .as_slice(),
            );
            let index = hash.to_u256() % n;
            indices.push(index.as_u64() as usize);
        }
        indices
    }
}

/// Collects batches of polynomials and allows building a DEEP-FRI prover for them.
///
/// This works by building Merkle trees on the batched polynomials, one tree per batch, and
/// eventually handing everything over to a newly constructed [`Prover`] (see the [`Self::commit`]
/// method).
///
/// This two-stage Committer-Prover architecture allows getting Merkle roots for the proven
/// polynomials before running the FRI folding argument and even before batching all polynomials, so
/// that Fiat-Shamir challenges can be derived before any quotients are built.
#[derive(Debug, Clone)]
pub struct Committer<F: PrimeField, G: Field256 + From<F>, H: Hasher<G>> {
    /// The proven degree bound. The degree of all batched polynomials must be strictly less than
    /// this value.
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// All polynomials batched so far.
    polynomials: Vec<Polynomial<F>>,
    /// The Merkle trees built so far.
    ///
    /// The sum of all `num_polys` of all trees must match the number of `polynomials`.
    trees: Vec<Tree<F, G, H>>,
}

impl<F: PrimeField, G: Field256 + From<F>, H: Hasher<G>> Committer<F, G, H> {
    /// Constructs a [`Committer`] with the given degree bound, blowup factor, and first batch of
    /// polynomials.
    ///
    /// We require specifying the first batch because our DEEP-FRI protocol requires at least one
    /// committed polynomial to work.
    ///
    /// `degree_bound` must be a power of 2 less than or equal to 2^[F::S](`PrimeField::S`), and
    /// `blowup_log2` must not be zero.
    pub fn new(degree_bound: usize, blowup_log2: usize, polynomials: Vec<Polynomial<F>>) -> Self {
        assert!(degree_bound.is_power_of_two());
        assert!(blowup_log2 > 0);
        assert!(!polynomials.is_empty());
        let mut committer = Self {
            degree_bound,
            blowup_log2,
            polynomials: vec![],
            trees: vec![],
        };
        committer.add_batch(polynomials);
        committer
    }

    /// Returns the proven degree bound.
    pub fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    /// Returns the size of the extended evaluation domain.
    pub fn extended_domain_size(&self) -> usize {
        self.degree_bound << self.blowup_log2
    }

    /// Returns the number of Merkle trees constructed so far, corresponding to the number of
    /// polynomial batches.
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Returns the root hash of the i-th Merkle tree. `index` must be less than
    /// [`Self::num_trees()`].
    ///
    /// This value can be used to derive Fiat-Shamir challenges.
    pub fn root_hash(&self, index: usize) -> H256 {
        self.trees[index].root_hash()
    }

    /// Adds a batch of polynomials, returning the index of the newly created batch.
    ///
    /// The returned index can be used with the [`Self::root_hash`] method to get the Merkle root
    /// for the batch.
    ///
    /// REQUIRES: the degree of all specified polynomials must be strictly less than
    /// [`Self::degree_bound()`].
    pub fn add_batch(&mut self, polynomials: Vec<Polynomial<F>>) -> usize {
        assert!(!polynomials.is_empty());

        let degree_bound = polynomials
            .iter()
            .map(|polynomial| polynomial.degree_bound())
            .max()
            .unwrap()
            .next_power_of_two();
        assert!(degree_bound <= self.degree_bound);
        let n = self.degree_bound << self.blowup_log2;
        assert!(n.trailing_zeros() as usize <= F::S);

        let evaluations = polynomials
            .iter()
            .map(|polynomial| polynomial.clone().shift_domain().lde2(n))
            .collect::<Vec<Vec<F>>>();

        let index = self.trees.len();

        self.polynomials.extend(polynomials);
        self.trees.push(Tree::<F, G, H>::new(evaluations));

        index
    }

    // TODO
}

// TODO

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_dst() {
        assert_eq!(
            *QUERY_DST,
            "0x344dcbdbf48e4b008c5998834be6306ea62faff77441a031a89dd2d7b8a36d4a"
                .parse()
                .unwrap()
        );
    }

    // TODO
}
