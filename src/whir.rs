use crate::hash::Hash;
use crate::merkle::{self, Tree};
use crate::utils;
use anyhow::{Result, anyhow};
use primitive_types::U256;
use starkom_bluesky::Scalar;
use starkom_ff::{Field, Field256, PrimeField};
use starkom_poly;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::LazyLock;

type Polynomial = starkom_poly::Polynomial<Scalar>;

static RLC_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/rlc"));
static EVAL_COMBO_DST: LazyLock<Scalar> =
    LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/eval_combo"));
static SC_ALPHA_DST: LazyLock<Scalar> =
    LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/sc_alpha"));
static OOD_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/ood"));
static SHIFT_DST: LazyLock<Scalar> =
    LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/shift"));
static COMBO_DST: LazyLock<Scalar> =
    LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/combo"));
static FINAL_DST: LazyLock<Scalar> =
    LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/final"));

// --- Module-level helpers ---

fn rlc(values: &[Scalar], alpha: Scalar) -> Scalar {
    let mut result = Scalar::ZERO;
    let mut power = Scalar::ONE;
    for &v in values {
        result += v * power;
        power *= alpha;
    }
    result
}

/// h(0) + h(1) for h(X) = a[0] + a[1]·X + a[2]·X².
fn poly_sum(h: [Scalar; 3]) -> Scalar {
    h[0].double() + h[1] + h[2]
}

fn poly_eval(h: [Scalar; 3], x: Scalar) -> Scalar {
    h[0] + h[1] * x + h[2] * x.square()
}

/// eq(a, b) = (1−a)(1−b) + a·b for single-variable multilinear extension.
fn eq1(a: Scalar, b: Scalar) -> Scalar {
    Scalar::ONE - a - b + (a * b).double()
}

/// ∏ eq1(alphas[l], zs[l]).
fn eq_k(alphas: &[Scalar], zs: &[Scalar]) -> Scalar {
    alphas.iter().zip(zs).map(|(&a, &b)| eq1(a, b)).product()
}

/// Sub-vector of pow(z, ∞) starting at bit `start` with `len` entries:
/// (z^{2^start}, z^{2^{start+1}}, …, z^{2^{start+len−1}}).
fn pow_seg(z: Scalar, start: usize, len: usize) -> Vec<Scalar> {
    if len == 0 {
        return vec![];
    }
    let mut cur = z.pow_small(1usize << start);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(cur);
        cur = cur.square();
    }
    out
}

/// ∏_{r=0}^{num_rounds−1} eq_k(alphas_slice[r], pow_seg(z, r·k, k)).
///
/// Used to compute the "constant factor" of an eq constraint after substituting
/// the accumulated Fiat-Shamir fold challenges **α** in place of the fixed variables.
fn eq_prod(z: Scalar, num_rounds: usize, k: usize, alphas_slice: &[Vec<Scalar>]) -> Scalar {
    let mut result = Scalar::ONE;
    for r in 0..num_rounds {
        result *= eq_k(&alphas_slice[r], &pow_seg(z, r * k, k));
    }
    result
}

/// Evaluate a multilinear polynomial (given by its 2^m values on {0,1}^m in LSB-first order)
/// at a point in 𝔽^m, using successive variable elimination.
fn multilinear_eval(poly: &[Scalar], point: &[Scalar]) -> Scalar {
    let mut evals = poly.to_vec();
    for &alpha in point {
        let half = evals.len() / 2;
        for j in 0..half {
            // Eliminate variable i (LSB-first): fold adjacent pair.
            evals[j] = evals[j] + alpha * (evals[j + half] - evals[j]);
        }
        evals.truncate(half);
    }
    evals[0]
}

/// Verify the 2^k Merkle proofs and apply k FRI-style fold steps to compute
/// Fold(f_{prev}, **α**)(y) where y is the domain element at `query_index` in ℒ_curr
/// (the k-times-squared domain of the oracle's domain ℒ_prev = domain of size `domain_size`).
///
/// The 2^k pre-image indices in ℒ_prev are {query_index + l·n_q : l = 0,…,2^k−1}
/// where n_q = domain_size >> k is the size of ℒ_curr.
///
/// If `is_initial` is true the oracle is the per-polynomial tree (vector leaves);
/// otherwise it is a folded oracle tree (single-valued leaves).
fn compute_fold<H: Hash<Scalar>>(
    proofs: &[merkle::Proof<H>],
    query_index: usize,
    oracle_root: Scalar,
    alphas: &[Scalar],
    domain_size: usize,
    gamma: Scalar,
    is_initial: bool,
) -> Result<Scalar> {
    let k = alphas.len();
    let n_q = domain_size >> k;

    if proofs.len() != 1 << k {
        return Err(anyhow!(
            "wrong number of fold proofs: got {}, want {}",
            proofs.len(),
            1 << k
        ));
    }

    let mut values: Vec<Scalar> = Vec::with_capacity(1 << k);
    for (l, proof) in proofs.iter().enumerate() {
        let idx = query_index + l * n_q;
        proof.verify(idx, oracle_root)?;
        let val = if is_initial {
            rlc(proof.leaf(), gamma)
        } else {
            if proof.leaf().len() != 1 {
                return Err(anyhow!("expected scalar leaf for folded oracle"));
            }
            proof.leaf()[0]
        };
        values.push(val);
    }

    // Apply k fold steps (same formula as fri.rs).
    // step starts as ω_n^{−1} where n = domain_size.
    let log_n = domain_size.trailing_zeros() as u32;
    let mut step =
        Scalar::ROOT_OF_UNITY_INV.pow_u64(1u64 << (Scalar::S as u32 - log_n));
    let mut half = 1usize << (k - 1);

    for round in 0..k {
        let mut new_values = Vec::with_capacity(half);
        for j in 0..half {
            // Left pre-image index in original domain.
            let left_idx = query_index + j * (n_q << round);
            let omega_inv = step.pow_small(left_idx);
            let left = values[j];
            let right = values[j + half];
            new_values
                .push((left + right + alphas[round] * omega_inv * (left - right)) * Scalar::TWO_INV);
        }
        values = new_values;
        step = step.square();
        half >>= 1;
    }

    debug_assert_eq!(values.len(), 1);
    Ok(values[0])
}

/// A WHIR polynomial commitment.
///
/// Produced by [`Prover::commit`] and consumed by the verifier. Serves as the anchor for external
/// Fiat-Shamir derivation: the PLONK layer (or any other caller) feeds [`Self::root_hash()`] into
/// its own transcript to bind its challenges to this commitment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment {
    /// Number of committed polynomials.
    num_polys: usize,
    /// Degree bound (a power of two). All committed polynomials have degree strictly less than this
    /// value.
    degree_bound: usize,
    /// Base-2 logarithm of the blowup factor. The coset evaluation domain has
    /// `degree_bound << blowup_log2` points.
    blowup_log2: usize,
    /// Root hash of the per-polynomial Merkle tree.
    ///
    /// The tree has `degree_bound << blowup_log2` leaves; leaf `j` is the vector
    /// `[p₀(g·ωʲ), …, p_{n-1}(g·ωʲ)]` of coset-domain evaluations of all `n` committed polynomials
    /// at position `j`, where `g` is the multiplicative generator (coset shift) and `ω` is the
    /// canonical root of unity for the evaluation domain.
    ///
    /// This root cryptographically binds every polynomial to all of its coset-domain evaluations.
    /// The WHIR prover derives the RLC challenge γ by hashing this root via Fiat-Shamir, so any
    /// caller wishing to interleave their own challenges with the commitment must do so by hashing
    /// this root into their transcript before generating those challenges.
    poly_tree_root: Scalar,
}

impl Commitment {
    /// Returns the root hash of the per-polynomial Merkle tree.
    pub fn root_hash(&self) -> Scalar {
        self.poly_tree_root
    }
}

/// WHIR prover.
///
/// Call [`Prover::new`] to commit a batch of polynomials, [`Prover::commit`] to retrieve the
/// commitment for use in external Fiat-Shamir derivation, and [`Prover::open`] to produce the WHIR
/// proof once all evaluation points are known.
#[derive(Debug)]
pub struct Prover<H: Hash<Scalar>> {
    degree_bound: usize,
    blowup_log2: usize,
    /// Committed polynomials, padded to `degree_bound` coefficients and shifted via
    /// `shift_domain` so that `p.clone().lde2(n)` yields their coset-domain evaluations.
    polynomials: Vec<Polynomial>,
    /// Per-polynomial Merkle tree over the coset evaluation domain.
    poly_tree: Tree<H>,
}

impl<H: Hash<Scalar>> Prover<H> {
    /// Commits to a batch of polynomials.
    ///
    /// All polynomials are padded to the same degree bound—the next power of two at or above the
    /// maximum degree among them—and evaluated on the coset domain
    /// `{g·ωʲ : j = 0, …, n-1}` where `n = degree_bound << blowup_log2`.
    pub fn new(mut polynomials: Vec<Polynomial>, blowup_log2: usize) -> Self {
        let degree_bound = polynomials
            .iter_mut()
            .map(|polynomial| {
                polynomial.trim();
                polynomial.degree_bound()
            })
            .max()
            .unwrap()
            .next_power_of_two();

        polynomials = polynomials
            .into_iter()
            .map(|mut polynomial| {
                polynomial.pad(degree_bound);
                polynomial.shift_domain()
            })
            .collect();

        let n = degree_bound << blowup_log2;
        let poly_tree = Tree::<H>::new(polynomials.iter().map(|p| p.clone().lde2(n)).collect());

        Self {
            degree_bound,
            blowup_log2,
            polynomials,
            poly_tree,
        }
    }

    /// Returns the commitment to the currently held polynomials.
    pub fn commit(&self) -> Commitment {
        Commitment {
            num_polys: self.polynomials.len(),
            degree_bound: self.degree_bound,
            blowup_log2: self.blowup_log2,
            poly_tree_root: self.poly_tree.root_hash(),
        }
    }

    /// Produces a WHIR proof opening the committed polynomials at the given evaluation points.
    ///
    /// `points` maps each evaluation point to the vector of claimed evaluations across all
    /// committed polynomials: `points[z][i]` is the claimed value of polynomial `i` at point `z`.
    pub fn open(&self, points: BTreeMap<Scalar, Vec<Scalar>>) -> Proof<H> {
        // TODO
        todo!()
    }
}

/// A WHIR opening proof.
///
/// Produced by [`Prover::open`] and verified by [`Proof::verify`] against the corresponding
/// [`Commitment`]. The proof is self-contained: the verifier needs only the commitment and this
/// struct to reconstruct the full Fiat-Shamir transcript and execute all decision-phase checks.
///
/// The layout follows the full transcript of Construction 5.1 (§5, page 32) and its BCS
/// compilation into a non-interactive proof.
#[derive(Debug, Clone)]
pub struct Proof<H: Hash<Scalar>> {
    /// Number of committed polynomials.
    num_polys: usize,
    /// Degree bound (power of two); mirrors [`Commitment::degree_bound`].
    degree_bound: usize,
    /// Base-2 log of the blowup factor; mirrors [`Commitment::blowup_log2`].
    blowup_log2: usize,

    /// Claimed evaluation values.
    ///
    /// `points[z][i]` is the claimed value of the i-th committed polynomial at z.
    /// These define the CRS constraint `σ₀` and weight polynomial `ŵ₀` that seed the initial
    /// sumcheck (Construction 5.1, "Inputs" and step 1).
    points: BTreeMap<Scalar, Vec<Scalar>>,

    /// Merkle roots of the M-1 folded oracles f₁, …, f_{M-1} (step 2a).
    ///
    /// `fold_roots[i-1]` is the root of the Merkle tree for oracle fᵢ.
    /// The initial oracle f₀ is already committed via [`Commitment::poly_tree_root`].
    fold_roots: Vec<Scalar>,

    /// Sumcheck polynomials for every round (steps 1a and 2e).
    ///
    /// `sumcheck_polys[0]` contains k₀ entries for the initial sumcheck (step 1).
    /// `sumcheck_polys[i]` for i ≥ 1 contains kᵢ entries for main-loop round i (step 2e).
    /// Each entry `[a₀, a₁, a₂]` stores the coefficients of the univariate polynomial
    /// ĥ(X) = a₀ + a₁·X + a₂·X² (degree < d* = 3, because ŵ has degree 1 in Z and each Xⱼ).
    sumcheck_polys: Vec<Vec<[Scalar; 3]>>,

    /// Out-of-domain answers for main-loop rounds 1, …, M-1 (step 2c).
    ///
    /// `ood_answers[i-1]` = yᵢ,₀ = f̂ᵢ(pow(zᵢ,₀, mᵢ)), the MLE of fᵢ evaluated at the
    /// verifier's out-of-domain sample for round i.
    ood_answers: Vec<Scalar>,

    /// Multilinear coefficients of the final polynomial f̂_M (step 3).
    ///
    /// `final_poly[b]` = f̂_M(b) for b ∈ {0,1}^{m_M}, in lexicographic order (b as a usize).
    /// The verifier evaluates f̂_M on the boolean hypercube to check the weight constraint
    /// (decision phase step 3b/c) and fold-consistency with g_{M-1} (decision step 3a).
    final_poly: Vec<Scalar>,

    /// Merkle-proof openings for shift-query evaluations (step 2d / decision step 2b).
    ///
    /// `oracle_query_proofs[i][j]` contains 2^{kᵢ} Merkle proofs opening oracle fᵢ at the coset
    /// of positions required to evaluate gᵢ = Fold(fᵢ, **α**ᵢ) at shift query z_{i+1, j+1}.
    ///
    /// For i = 0 the oracle is the per-polynomial tree (vector leaves of width `num_polys`);
    /// for i > 0 it is the i-th folded oracle tree (single-valued leaves).
    oracle_query_proofs: Vec<Vec<Vec<merkle::Proof<H>>>>,

    /// Merkle-proof openings for the final fold-consistency checks (step 4 / decision step 3a).
    ///
    /// `final_query_proofs[l]` contains 2^{k_{M-1}} Merkle proofs opening oracle f_{M-1} at the
    /// coset needed to evaluate g_{M-1} = Fold(f_{M-1}, **α**_{M-1}) at final query r_l^fin.
    final_query_proofs: Vec<Vec<merkle::Proof<H>>>,

    _data: PhantomData<H>,
}

impl<H: Hash<Scalar>> Proof<H> {
    /// Verifies this proof against the given commitment and the evaluation claims embedded in it.
    ///
    /// Executes all decision-phase checks from Construction 5.1 (§5, page 32) and the
    /// multi-constraint batching from Construction 5.5 (§5.2, page 40):
    ///
    /// 1. Initial sumcheck claim and sub-round consistency (decision step 1).
    /// 2. Per-round sumcheck claim (step 2c) and consistency (step 2d), using Merkle-proof fold
    ///    evaluations for the shift queries (step 2b).
    /// 3. Final fold-consistency check: f̂_M(**r**_l^fin) = g_{M-1}(r_l^fin) (step 3a).
    /// 4. Weight constraint check: ∑_b ŵ_{M-1}(f̂_M(b), **α**_{M-1}, b) = ĥ_{M-1,k}(α_{M-1,k}) (step 3c).
    pub fn verify(&self, commitment: &Commitment) -> Result<()> {
        // ── Parameters ───────────────────────────────────────────────────────────────
        let poly_tree_root = commitment.poly_tree_root;
        let m0 = commitment.degree_bound.trailing_zeros() as usize; // log₂(degree bound)
        let n0 = commitment.degree_bound << commitment.blowup_log2; // |ℒ₀|

        if self.sumcheck_polys.is_empty() {
            return Err(anyhow!("empty sumcheck_polys"));
        }
        let k = self.sumcheck_polys[0].len(); // folding parameter
        if k == 0 {
            return Err(anyhow!("zero folding parameter"));
        }
        // M = total WHIR rounds (1 initial + M-1 main loop).
        let big_m = self.fold_roots.len() + 1;
        let m_final = m0.saturating_sub(k * big_m); // remaining vars in f̂_M (= 0 when k|m₀)

        // Structural length checks.
        if self.sumcheck_polys.len() != big_m {
            return Err(anyhow!(
                "sumcheck_polys length: got {}, want {}",
                self.sumcheck_polys.len(),
                big_m
            ));
        }
        for (i, round) in self.sumcheck_polys.iter().enumerate() {
            if round.len() != k {
                return Err(anyhow!("sumcheck_polys[{i}] length: got {}, want {k}", round.len()));
            }
        }
        if self.ood_answers.len() != big_m - 1 {
            return Err(anyhow!(
                "ood_answers length: got {}, want {}",
                self.ood_answers.len(),
                big_m - 1
            ));
        }
        if self.final_poly.len() != 1 << m_final {
            return Err(anyhow!(
                "final_poly length: got {}, want {}",
                self.final_poly.len(),
                1 << m_final
            ));
        }
        if self.oracle_query_proofs.len() != big_m - 1 {
            return Err(anyhow!(
                "oracle_query_proofs length: got {}, want {}",
                self.oracle_query_proofs.len(),
                big_m - 1
            ));
        }

        // ── Fiat-Shamir: initial challenges ──────────────────────────────────────────
        // γ (per-polynomial RLC challenge) — Construction 5.5, step "combination randomness"
        // applied at the oracle-commitment level.
        let gamma = H::hash_many(&[*RLC_DST, poly_tree_root]);

        // Combined per-point claims: combined_claims[j] = (z_j, Σ_i γⁱ · points[z_j][i]).
        let combined_claims: Vec<(Scalar, Scalar)> = self
            .points
            .iter()
            .map(|(z, vals)| (*z, rlc(vals, gamma)))
            .collect();

        // η (multi-constraint batching challenge for the initial eval constraints).
        let mut state = {
            let mut inputs = vec![
                *EVAL_COMBO_DST,
                poly_tree_root,
                Scalar::from(combined_claims.len() as u64),
            ];
            for &(z, v) in &combined_claims {
                inputs.push(z);
                inputs.push(v);
            }
            H::hash_many(&inputs)
        };

        // σ₀ = Σ_j η^j · combined_claims[j].1  (Construction 5.5: combined target).
        let eta = state; // saved for the weight-constraint computation at the end
        let sigma0 = rlc(&combined_claims.iter().map(|(_, v)| *v).collect::<Vec<_>>(), eta);

        // ── Round 0: initial sumcheck (decision step 1) ──────────────────────────────
        // (a) Σ_{b∈{0,1}} ĥ_{0,1}(b) = σ₀.
        if poly_sum(self.sumcheck_polys[0][0]) != sigma0 {
            return Err(anyhow!("initial sumcheck claim mismatch"));
        }

        let mut all_alphas: Vec<Vec<Scalar>> = Vec::with_capacity(big_m);
        {
            let mut round0_alphas = Vec::with_capacity(k);
            for l in 0..k {
                let h = self.sumcheck_polys[0][l];
                let alpha = H::hash_many(&[*SC_ALPHA_DST, state, h[0], h[1], h[2]]);
                state = alpha;
                round0_alphas.push(alpha);
                // (b) Σ_b ĥ_{0,ℓ}(b) = ĥ_{0,ℓ-1}(α_{0,ℓ-1}).
                if l + 1 < k
                    && poly_sum(self.sumcheck_polys[0][l + 1]) != poly_eval(h, alpha)
                {
                    return Err(anyhow!(
                        "sumcheck consistency failed in round 0 sub-round {}",
                        l + 1
                    ));
                }
            }
            all_alphas.push(round0_alphas);
        }
        let mut prev_sc_last =
            poly_eval(self.sumcheck_polys[0][k - 1], all_alphas[0][k - 1]);

        // Saved data for the weight-constraint check at the end.
        let mut ood_field_elems: Vec<Scalar> = Vec::with_capacity(big_m - 1);
        let mut shift_q_per_round: Vec<Vec<usize>> = Vec::with_capacity(big_m - 1);
        let mut gamma_per_round: Vec<Scalar> = Vec::with_capacity(big_m - 1);

        // ── Main loop (rounds i = 1, …, M-1) ─────────────────────────────────────────
        for i in 1..big_m {
            let fold_root = self.fold_roots[i - 1];
            let prev_domain_size = n0 >> (k * (i - 1)); // |ℒᵢ₋₁|
            let curr_domain_size = n0 >> (k * i); // |ℒᵢ|
            let prev_oracle_root = if i == 1 { poly_tree_root } else { self.fold_roots[i - 2] };
            let is_initial = i == 1;

            // Absorb fᵢ root → derive OOD sample z_{i,0} ∈ 𝔽.
            state = H::hash_many(&[*OOD_DST, state, fold_root]);
            let z_ood_raw = state;
            ood_field_elems.push(z_ood_raw);

            // Absorb OOD answer y_{i,0}.
            let y_i0 = self.ood_answers[i - 1];
            state = H::hash_many(&[*OOD_DST, state, y_i0]);

            // Derive shift query indices in ℒᵢ (size curr_domain_size).
            let t = self.oracle_query_proofs[i - 1].len();
            let q_mod = U256::from(curr_domain_size as u64);
            let shift_indices: Vec<usize> = (0..t)
                .map(|j| {
                    let h = H::hash_many(&[
                        *SHIFT_DST,
                        state,
                        Scalar::from(j as u64),
                    ]);
                    (h.to_u256() % q_mod).low_u64() as usize
                })
                .collect();

            // Derive γᵢ (combination challenge for this round).
            let gamma_i = H::hash_many(&[*COMBO_DST, state, Scalar::from(t as u64)]);
            state = gamma_i;
            gamma_per_round.push(gamma_i);
            shift_q_per_round.push(shift_indices.clone());

            // Compute gᵢ₋₁(zᵢ,ⱼ) for each shift query via Merkle-proof fold (decision step 2b).
            let g_vals: Vec<Scalar> = shift_indices
                .iter()
                .zip(self.oracle_query_proofs[i - 1].iter())
                .map(|(&q, proofs)| {
                    compute_fold::<H>(
                        proofs,
                        q,
                        prev_oracle_root,
                        &all_alphas[i - 1],
                        prev_domain_size,
                        gamma,
                        is_initial,
                    )
                })
                .collect::<Result<Vec<_>>>()?;

            // Decision step 2c: Σ_b ĥᵢ,₁(b) = ĥᵢ₋₁,k(αᵢ₋₁,k) + γᵢ·yᵢ,₀ + Σ_j γᵢ^{j+2}·g_vals[j].
            let rhs = {
                let mut acc = prev_sc_last + gamma_i * y_i0;
                let mut gamma_pow = gamma_i * gamma_i;
                for &g in &g_vals {
                    acc += gamma_pow * g;
                    gamma_pow *= gamma_i;
                }
                acc
            };
            if poly_sum(self.sumcheck_polys[i][0]) != rhs {
                return Err(anyhow!("sumcheck claim mismatch in round {i}"));
            }

            // Decision step 2d: sub-round consistency + alpha derivation.
            let mut round_alphas = Vec::with_capacity(k);
            for l in 0..k {
                let h = self.sumcheck_polys[i][l];
                let alpha = H::hash_many(&[*SC_ALPHA_DST, state, h[0], h[1], h[2]]);
                state = alpha;
                round_alphas.push(alpha);
                if l + 1 < k
                    && poly_sum(self.sumcheck_polys[i][l + 1]) != poly_eval(h, alpha)
                {
                    return Err(anyhow!(
                        "sumcheck consistency failed in round {i} sub-round {}",
                        l + 1
                    ));
                }
            }
            prev_sc_last = poly_eval(self.sumcheck_polys[i][k - 1], round_alphas[k - 1]);
            all_alphas.push(round_alphas);
        }

        // ── Final polynomial checks (decision step 3) ─────────────────────────────────
        let last_oracle_root = self.fold_roots.last().copied().unwrap_or(poly_tree_root);
        // ℒ_{M-1} has size n0 >> (k*(M-1)).
        let last_domain_size = n0 >> (k * (big_m - 1));
        // Final query domain ℒ_{M-1}^{(2^k)} has size last_domain_size >> k.
        let final_q_domain_size = last_domain_size >> k;
        let final_q_mod = U256::from(final_q_domain_size as u64);

        for l in 0..self.final_query_proofs.len() {
            // Derive r_l^fin index in the final query domain.
            let r_hash =
                H::hash_many(&[*FINAL_DST, state, Scalar::from(l as u64)]);
            let r_idx = (r_hash.to_u256() % final_q_mod).low_u64() as usize;

            // Compute g_{M-1}(r_l^fin) via fold on f_{M-1}.
            // When big_m == 1 the final oracle is the initial per-poly tree (vector leaves).
            let g_val = compute_fold::<H>(
                &self.final_query_proofs[l],
                r_idx,
                last_oracle_root,
                &all_alphas[big_m - 1],
                last_domain_size,
                gamma,
                big_m == 1,
            )?;

            // Decision step 3a: f̂_M(pow(r_raw, m_M)) = g_{M-1}(r_raw).
            let r_raw = Polynomial::coset_element2(r_idx, final_q_domain_size);
            let r_vec = pow_seg(r_raw, 0, m_final);
            let f_m_val = multilinear_eval(&self.final_poly, &r_vec);
            if f_m_val != g_val {
                return Err(anyhow!("final fold-consistency check failed at l={l}"));
            }
        }

        // ── Weight constraint (decision step 3c) ──────────────────────────────────────
        // Σ_{b∈{0,1}^{m_M}} ŵ_{M-1}(f̂_M(b), **α**_{M-1}, b) = prev_sc_last.
        //
        // Expands to: Σ_{contribution} const_factor · f̂_M(eval_point) = prev_sc_last,
        // where each contribution is from an initial eval constraint or an OOD/shift-query
        // constraint accumulated into ŵ.  The eval_point is the last m_M components of
        // the corresponding pow(·) vector; the const_factor encodes all fixed eq factors
        // from substituting the accumulated fold challenges.
        let weight_sum = {
            let mut sum = Scalar::ZERO;

            // Initial eval constraints: one per eval point z.
            for (j, &(z, _)) in combined_claims.iter().enumerate() {
                // eq_prod covers all M rounds (alphas[0..M]).
                let const_z = eta.pow_small(j) * eq_prod(z, big_m, k, &all_alphas);
                // pow(z, m₀)[k·M ..] = pow_seg(z, k·M, m_M).
                let eval_pt = pow_seg(z, k * big_m, m_final);
                sum += const_z * multilinear_eval(&self.final_poly, &eval_pt);
            }

            // OOD and shift-query constraints for each main-loop round i = 1,…,M-1.
            for i in 1..big_m {
                let gamma_i = gamma_per_round[i - 1];
                let curr_domain_size = n0 >> (k * i);
                // Number of rounds' worth of eq factors still to be applied: big_m - i.
                let remaining = big_m - i;
                let alphas_from_i = &all_alphas[i..];

                // j = 0: OOD sample z_{i,0} with coefficient γᵢ^1.
                {
                    let z_ood = ood_field_elems[i - 1];
                    let const_i0 = gamma_i * eq_prod(z_ood, remaining, k, alphas_from_i);
                    let eval_pt = pow_seg(z_ood, k * remaining, m_final);
                    sum += const_i0 * multilinear_eval(&self.final_poly, &eval_pt);
                }

                // j = 1,…,t: shift queries with coefficient γᵢ^{j+1} (1-indexed j).
                let mut gamma_pow = gamma_i * gamma_i; // γᵢ^2 for j=1
                for &q in &shift_q_per_round[i - 1] {
                    let z_shift = Polynomial::coset_element2(q, curr_domain_size);
                    let const_ij = gamma_pow * eq_prod(z_shift, remaining, k, alphas_from_i);
                    let eval_pt = pow_seg(z_shift, k * remaining, m_final);
                    sum += const_ij * multilinear_eval(&self.final_poly, &eval_pt);
                    gamma_pow *= gamma_i;
                }
            }

            sum
        };

        if weight_sum != prev_sc_last {
            return Err(anyhow!("weight constraint check failed"));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO
}
