use crate::hash::Hash;
use crate::merkle;
use crate::merkle::merkle_root;
use crate::utils;
use starkom_bluesky::Scalar;
use starkom_ff::{Field, PrimeField};
use starkom_poly;
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

/// A WHIR commitment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment {
    /// Nominal number of sumcheck sub-rounds per folding round, equal to `floor(log2(m))` where
    /// `m = log2(degree_bound)`. The final round may have fewer sub-rounds if m is not divisible by
    /// this value; the actual count per round is recoverable from `sumcheck_polynomials[r].len()`.
    num_folds_per_round: usize,
    /// Merkle roots of all committed oracles: roots[0] is the initial oracle, roots[r] is the
    /// r-th fold. Has M+1 entries in total where M is the number of folding rounds.
    roots: Vec<Scalar>,
    /// Sumcheck polynomials sent by the prover, one Vec<[Scalar; 3]> per folding round, each
    /// containing the per-sub-round univariate polynomials of degree ≤ 2 with coefficients
    /// [a₀, a₁, a₂] such that ĥ(t) = a₀ + a₁·t + a₂·t².
    sumcheck_polynomials: Vec<Vec<[Scalar; 3]>>,
    /// Prover's responses to the per-round OOD challenges, one per folding round. Entry r is the
    /// evaluation of the r-th folded oracle at z_r = H(OOD_DST, roots[r+1]).
    ood_values: Vec<Scalar>,
}

impl Commitment {
    /// Returns the number of folding rounds M. The total number of committed oracles is M+1.
    pub fn num_rounds(&self) -> usize {
        self.ood_values.len()
    }

    /// Returns the Merkle roots of all committed oracles. Has `num_rounds() + 1` entries.
    pub fn roots(&self) -> &[Scalar] {
        &self.roots
    }

    /// Returns the prover's OOD evaluations, one per folding round.
    pub fn ood_values(&self) -> &[Scalar] {
        &self.ood_values
    }

    /// Returns the Merkle root of the initial (pre-fold) oracle.
    pub fn root(&self) -> Scalar {
        self.roots[0]
    }
}

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
#[allow(dead_code)]
fn eval_degree2(coeffs: &[Scalar; 3], t: Scalar) -> Scalar {
    coeffs[0] + t * (coeffs[1] + t * coeffs[2])
}

/// A WHIR prover.
#[derive(Debug)]
pub struct Prover<H: Hash<Scalar>> {
    /// The degree bound of the committed polynomial (always a power of 2).
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// Merkle trees for all committed oracles: trees[0] is the initial oracle f₀, trees[r+1] is
    /// the oracle after the r-th folding round.
    trees: Vec<merkle::Tree<H>>,
    /// Sumcheck polynomials from the commit phase: one Vec per folding round, each holding the
    /// per-sub-round degree-≤2 coefficient triples [a₀, a₁, a₂].
    sumcheck_polynomials: Vec<Vec<[Scalar; 3]>>,
    /// OOD evaluations from the commit phase: ood_values[r] = f_{r+1}(z_r) where
    /// z_r = H(OOD_DST, trees[r+1].root_hash()).
    ood_values: Vec<Scalar>,
    _data: PhantomData<H>,
}

impl<H: Hash<Scalar>> Prover<H> {
    /// Runs the WHIR commit phase: extends all polynomials, commits them to Merkle trees, and runs
    /// the sumcheck + fold + OOD loop for each folding round.
    ///
    /// The commit phase uses a uniform initial weight polynomial ŵ₀ = 1 over {0,1}^m, making this
    /// a pure low-degree test. Evaluation claims (for the PCS) are added at prove time.
    pub fn new(mut polynomials: Vec<Polynomial>, blowup_log2: usize) -> Self {
        assert!(!polynomials.is_empty(), "at least one polynomial required");

        let degree_bound = polynomials
            .iter()
            .map(|polynomial| polynomial.degree_bound())
            .max()
            .unwrap()
            .next_power_of_two();

        for polynomial in &mut polynomials {
            polynomial.trim();
            polynomial.pad(degree_bound);
        }

        let m = degree_bound.trailing_zeros() as usize;
        // k = floor(log₂(m)), the optimal number of sumcheck sub-rounds per folding round.
        // The last round folds only m % k variables if m is not divisible by k.
        let k = if m <= 1 { m } else { m.ilog2() as usize };
        let num_rounds = if k == 0 { 0 } else { m.div_ceil(k) };
        let mut n = degree_bound << blowup_log2;
        let mut remaining_m = m;

        // Combine all polynomials into one oracle via RLC.
        let mut polynomial = {
            let rlc_challenge = H::hash_many(
                std::iter::once(*RLC_DST)
                    .chain(polynomials.iter().map(|polynomial| {
                        let values = polynomial.clone().decode2();
                        merkle_root::<H>(values.as_slice())
                    }))
                    .collect::<Vec<Scalar>>()
                    .as_slice(),
            );
            let mut pow = Scalar::ONE;
            polynomials
                .into_iter()
                .fold(Polynomial::default(), |mut accumulator, polynomial| {
                    accumulator += polynomial * pow;
                    pow *= rlc_challenge;
                    accumulator
                })
        };
        polynomial = polynomial.shift_domain();

        // Build the initial oracle and the boolean-hypercube evaluation tables.
        let initial_tree = merkle::Tree::<H>::new(polynomial.clone().lde2(n));
        let mut current_root = initial_tree.root_hash();
        let mut trees: Vec<merkle::Tree<H>> = vec![initial_tree];

        // f_table[b] = f̂(b) for b ∈ {0,1}^m; w_table[b] = ŵ₀(b) = 1 (uniform initial weight).
        let mut f_table = build_f_table(polynomial.coefficients());
        let mut w_table = vec![Scalar::ONE; f_table.len()];

        let mut sumcheck_polynomials: Vec<Vec<[Scalar; 3]>> = Vec::with_capacity(num_rounds);
        let mut ood_values: Vec<Scalar> = Vec::with_capacity(num_rounds);

        for _round in 0..num_rounds {
            // The last round may fold fewer than k variables if m is not divisible by k.
            let sub_rounds = k.min(remaining_m);
            let mut round_polys: Vec<[Scalar; 3]> = Vec::with_capacity(sub_rounds);
            // Rolling Fiat-Shamir transcript seeded from the current oracle root.
            let mut transcript = current_root;

            for _sub in 0..sub_rounds {
                let half = f_table.len() / 2;

                // Compute ĥ(t) = ∑_{j ∈ {0,1}^{m-1}} f̂(t, j) · ŵ(t, j) at t = 0, 1, 2.
                // In the flat table layout, bit 0 of the index encodes X₁: even indices have
                // X₁ = 0 and odd indices have X₁ = 1.
                let h0 =
                    (0..half).fold(Scalar::ZERO, |acc, j| acc + f_table[2 * j] * w_table[2 * j]);
                let h1 = (0..half).fold(Scalar::ZERO, |acc, j| {
                    acc + f_table[2 * j + 1] * w_table[2 * j + 1]
                });
                // Multilinear extension at t = 2: p(2) = 2·p(1) − p(0).
                let h2 = (0..half).fold(Scalar::ZERO, |acc, j| {
                    let f2 = f_table[2 * j + 1] + f_table[2 * j + 1] - f_table[2 * j];
                    let w2 = w_table[2 * j + 1] + w_table[2 * j + 1] - w_table[2 * j];
                    acc + f2 * w2
                });

                let coeffs = interpolate_degree2(h0, h1, h2);

                // Fiat-Shamir: fold challenge binds to the running transcript and ĥ's coefficients.
                transcript =
                    H::hash_many(&[*FOLD_DST, transcript, coeffs[0], coeffs[1], coeffs[2]]);
                let alpha = transcript;

                // Fold f_table and w_table: fix X₁ = alpha via multilinear interpolation.
                // new[j] = old[2j] + alpha · (old[2j+1] − old[2j])
                f_table = (0..half)
                    .map(|j| f_table[2 * j] + alpha * (f_table[2 * j + 1] - f_table[2 * j]))
                    .collect();
                w_table = (0..half)
                    .map(|j| w_table[2 * j] + alpha * (w_table[2 * j + 1] - w_table[2 * j]))
                    .collect();

                polynomial = polynomial.fold2(alpha);
                round_polys.push(coeffs);
            }

            remaining_m -= sub_rounds;
            n >>= sub_rounds;
            sumcheck_polynomials.push(round_polys);

            // Commit folded oracle and derive OOD evaluation.
            let folded_tree = merkle::Tree::<H>::new(polynomial.clone().lde2(n));
            current_root = folded_tree.root_hash();
            trees.push(folded_tree);

            let z_ood = H::hash_two(*OOD_DST, current_root, Scalar::ZERO);
            let y_ood = polynomial.evaluate(z_ood);
            ood_values.push(y_ood);

            // Add the OOD evaluation claim to the weight table for the next round:
            // ŵ_new(b) = ŵ(b) + γ · eq(b, pow(z_ood, remaining_m))
            // This encodes that f̂(pow(z_ood, remaining_m)) = y_ood, carrying it into the next
            // round's sumcheck and linking the round's sigma to y_ood.
            if remaining_m > 0 {
                let gamma = H::hash_two(*GAMMA_DST, current_root, y_ood);
                let eq_table = build_eq_table(z_ood, remaining_m);
                for (w, &eq_val) in w_table.iter_mut().zip(eq_table.iter()) {
                    *w += gamma * eq_val;
                }
            }
        }

        Self {
            degree_bound,
            blowup_log2,
            trees,
            sumcheck_polynomials,
            ood_values,
            _data: PhantomData,
        }
    }

    /// Returns the nominal number of sumcheck sub-rounds per folding round, equal to
    /// `floor(log2(log2(degree_bound)))`. The final round may fold fewer variables if
    /// `log2(degree_bound)` is not divisible by this value.
    pub fn num_folds_per_round(&self) -> usize {
        let m = self.degree_bound.trailing_zeros() as usize;
        if m <= 1 { m } else { m.ilog2() as usize }
    }

    /// Returns the proven degree bound.
    pub fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    /// Returns the size of the extended evaluation domain.
    pub fn extended_domain_size(&self) -> usize {
        self.degree_bound << self.blowup_log2
    }

    /// Constructs the WHIR commitment from the prover's commit-phase data.
    pub fn commit(&self) -> Commitment {
        Commitment {
            num_folds_per_round: self.num_folds_per_round(),
            roots: self.trees.iter().map(|tree| tree.root_hash()).collect(),
            sumcheck_polynomials: self.sumcheck_polynomials.clone(),
            ood_values: self.ood_values.clone(),
        }
    }
}
