use crate::hash::Hash;
use crate::utils;
use starkom_bluesky::Scalar;
use starkom_ff::{Field, PrimeField};
use starkom_poly;
use std::collections::BTreeSet;
use std::marker::PhantomData;
use std::sync::LazyLock;

type Polynomial = starkom_poly::Polynomial<Scalar>;

/// Domain separator tag used when deriving the Fiat-Shamir challenge for the random linear
/// combination.
static RLC_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/rlc"));

/// Domain separator tag used when deriving the Fiat-Shamir challenge for WHIR folding.
static FOLD_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/fold"));

/// Domain separator tag used when deriving the per-round out-of-domain (OOD) challenge point.
static OOD_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/ood"));

/// Domain separator tag used when deriving the OOD combination randomness γ.
static GAMMA_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/gamma"));

/// Computes f̂(b) = ∑_{k ⊆ b} coefficients[k] for all b ∈ {0,1}^m via the zeta (sum-over-subsets)
/// transform. `coefficients` must have power-of-2 length. Bit i of b selects variable Xᵢ₊₁, so
/// b=0 means all variables are 0 and b=2^m-1 means all variables are 1.
fn build_f_table(coefficients: &[Scalar]) -> Vec<Scalar> {
    let len = coefficients.len();
    debug_assert!(len.is_power_of_two());
    let m = len.trailing_zeros() as usize;
    let mut table = coefficients.to_vec();
    for i in 0..m {
        for b in 0..len {
            if (b >> i) & 1 == 1 {
                let addend = table[b ^ (1 << i)];
                table[b] += addend;
            }
        }
    }
    table
}

/// Computes eq(b, pow(z, m)) for all b ∈ {0,1}^m, where pow(z, m) = (z, z², z⁴, …, z^{2^{m-1}}).
/// Used to add an evaluation claim at z to the weight table between folding rounds.
fn build_eq_table(z: Scalar, m: usize) -> Vec<Scalar> {
    let mut table = vec![Scalar::ONE];
    let mut z_pow = z;
    for _ in 0..m {
        let one_minus = Scalar::ONE - z_pow;
        let half = table.len();
        let mut extended = vec![Scalar::ZERO; half * 2];
        for (b, &val) in table.iter().enumerate() {
            extended[b] = val * one_minus;
            extended[b | half] = val * z_pow;
        }
        table = extended;
        z_pow = z_pow.square();
    }
    table
}

/// Interpolates a univariate polynomial of degree ≤ 2 from its values at t = 0, 1, 2.
/// Returns [a₀, a₁, a₂] such that p(t) = a₀ + a₁·t + a₂·t².
fn interpolate_degree2(h0: Scalar, h1: Scalar, h2: Scalar) -> [Scalar; 3] {
    let a0 = h0;
    let a2 = (h0 - (h1 + h1) + h2) * Scalar::TWO_INV;
    let a1 = h1 - h0 - a2;
    [a0, a1, a2]
}

/// Evaluates p(t) = a₀ + a₁·t + a₂·t² at `t`.
fn eval_degree2(coeffs: &[Scalar; 3], t: Scalar) -> Scalar {
    coeffs[0] + t * (coeffs[1] + t * coeffs[2])
}

/// A WHIR commitment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment {
    // TODO
}

#[derive(Debug, Clone)]
pub struct Committer<H: Hash<Scalar>> {
    /// The degree bound of the committed polynomial (always a power of 2).
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    // TODO
    _data: PhantomData<H>,
}

impl<H: Hash<Scalar>> Committer<H> {
    pub fn new(degree_bound: usize, blowup_log2: usize) -> Self {
        // TODO
        todo!()
    }

    /// Returns the proven degree bound.
    pub fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    /// Returns the size of the extended evaluation domain.
    pub fn extended_domain_size(&self) -> usize {
        self.degree_bound << self.blowup_log2
    }

    pub fn add_batch(&mut self, polynomials: Vec<Polynomial>) {
        // TODO
        todo!()
    }

    pub fn commit(self, points: BTreeSet<Scalar>) -> (Commitment, Prover<H>) {
        // TODO
        todo!()
    }
}

/// A WHIR prover.
#[derive(Debug)]
pub struct Prover<H: Hash<Scalar>> {
    /// The degree bound of the committed polynomial (always a power of 2).
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    // TODO
    _data: PhantomData<H>,
}
