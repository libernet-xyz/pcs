use crate::fri;
use crate::hash::Hasher;
use primitive_types::{H256, U256};
use sha2::Digest;
use starkom_ff::{Field, Field256, PrimeField};
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
