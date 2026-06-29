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
static SHIFT_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/shift"));
static COMBO_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/combo"));
static FINAL_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/final"));

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
            evals[j] = evals[j] + alpha * (evals[j + half] - evals[j]);
        }
        evals.truncate(half);
    }
    evals[0]
}

/// Sum-Over-Subsets (SOS) DP: given a coefficient vector (ascending degree), return the
/// multilinear-extension truth table f̂(b) = Σ_{j ⊆ b} c_j for all b ∈ {0,1}^m.
///
/// `m` must satisfy 2^m == the padded coefficient length; coefficients shorter than 2^m are
/// zero-extended.
fn sos_dp(coeffs: &[Scalar], m: usize) -> Vec<Scalar> {
    let n = 1 << m;
    let mut table = coeffs.to_vec();
    table.resize(n, Scalar::ZERO);
    for l in 0..m {
        for j in 0..n {
            if (j >> l) & 1 == 1 {
                let prev = table[j ^ (1 << l)];
                table[j] += prev;
            }
        }
    }
    table
}

/// Compute the truth table of eq_tensor(z, m)[b] = ∏_{l=0}^{m-1} eq1(bit_l(b), z^{2^l})
/// for all b ∈ {0,1}^m (LSB-first encoding: bit 0 of b corresponds to l=0).
fn eq_tensor_table(z: Scalar, m: usize) -> Vec<Scalar> {
    let zs = pow_seg(z, 0, m);
    let mut table = vec![Scalar::ONE];
    // Process from high bit to low bit so that each new variable lands in the LSB position.
    for l in (0..m).rev() {
        let zl = zs[l];
        let one_minus_zl = Scalar::ONE - zl;
        let old_len = table.len();
        let mut new_table = Vec::with_capacity(old_len * 2);
        for &v in &table {
            new_table.push(v * one_minus_zl); // bit l = 0
            new_table.push(v * zl); // bit l = 1
        }
        table = new_table;
    }
    table
}

/// Fold a multilinear truth table in-place by eliminating variable 0 (the LSB).
///
/// After this call, `table[j]` = (1−alpha)·table_old[2j] + alpha·table_old[2j+1].
fn mle_fold(table: &mut Vec<Scalar>, alpha: Scalar) {
    let half = table.len() / 2;
    for j in 0..half {
        let v0 = table[j << 1];
        let v1 = table[(j << 1) | 1];
        table[j] = v0 + alpha * (v1 - v0);
    }
    table.truncate(half);
}

/// Compute the degree-2 sumcheck polynomial h(X) = Σ_{b∈{0,1}^{m-1}} f̂(X,b)·ŵ(X,b),
/// returned as its coefficient vector [a₀, a₁, a₂] where h(X) = a₀ + a₁X + a₂X².
fn sumcheck_poly(f_table: &[Scalar], w_table: &[Scalar]) -> [Scalar; 3] {
    let half = f_table.len() / 2;
    let mut h0 = Scalar::ZERO;
    let mut h1 = Scalar::ZERO;
    let mut h2 = Scalar::ZERO;
    for j in 0..half {
        let f0 = f_table[j << 1];
        let f1 = f_table[(j << 1) | 1];
        let w0 = w_table[j << 1];
        let w1 = w_table[(j << 1) | 1];
        h0 += f0 * w0;
        h1 += f1 * w1;
        // h(2) via multilinear extension: f(2,b) = 2f(1,b)−f(0,b)
        h2 += (f1.double() - f0) * (w1.double() - w0);
    }
    let a0 = h0;
    let a2 = (h2 - h1.double() + h0) * Scalar::TWO_INV;
    let a1 = h1 - a0 - a2;
    [a0, a1, a2]
}

/// Verify the 2^k Merkle proofs and apply k FRI-style fold steps to compute
/// Fold(f_{prev}, **α**)(y) where y is the domain element at `query_index` in the k-times-folded
/// domain.
///
/// The 2^k pre-image indices in the source domain (size `domain_size`) are
/// {query_index + l·n_q : l = 0,…,2^k−1} where n_q = domain_size >> k.
///
/// If `is_initial` is true the oracle is the per-polynomial tree (vector leaves); otherwise it is a
/// folded oracle tree (single-valued leaves).
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

    // Apply k fold steps.  `step` = ω_n^{−1} for the current domain size n.
    let log_n = domain_size.trailing_zeros() as u32;
    let mut step = Scalar::ROOT_OF_UNITY_INV.pow_u64(1u64 << (Scalar::S as u32 - log_n));
    let mut half = 1usize << (k - 1);

    for round in 0..k {
        let mut new_values = Vec::with_capacity(half);
        for j in 0..half {
            let left_idx = query_index + j * (n_q << round);
            let omega_inv = step.pow_small(left_idx);
            let left = values[j];
            let right = values[j + half];
            new_values.push(
                (left + right + alphas[round] * omega_inv * (left - right)) * Scalar::TWO_INV,
            );
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
/// Produced by [`Prover::commit`] and consumed by the verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment {
    /// Number of committed polynomials.
    num_polys: usize,
    /// Degree bound (a power of two). All committed polynomials have degree strictly less than this
    /// value.
    degree_bound: usize,
    /// Base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// Root hash of the per-polynomial Merkle tree.
    ///
    /// The tree has `degree_bound << blowup_log2` leaves; leaf `j` is the vector
    /// `[p₀(ωʲ), …, p_{n-1}(ωʲ)]` of evaluation-domain evaluations of all `n` committed
    /// polynomials at position `j`, where `ω` is the canonical root of unity for the domain.
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
    /// Number of fold2 steps per WHIR round (the sumcheck depth per round).
    k: usize,
    /// Number of shift queries per round (tuned for 128-bit security).
    t: usize,
    /// Number of final fold-consistency queries.
    num_final_queries: usize,
    /// Committed polynomials, padded to `degree_bound` coefficients.
    polynomials: Vec<Polynomial>,
    /// Per-polynomial Merkle tree over the evaluation domain {ω^j : j=0,…,n−1}.
    poly_tree: Tree<H>,
}

impl<H: Hash<Scalar>> Prover<H> {
    /// Commits to a batch of polynomials with folding parameter `k`.
    ///
    /// All polynomials are padded to the same degree bound and evaluated on the domain
    /// `{ωʲ : j = 0, …, n-1}` where `n = degree_bound << blowup_log2`.
    ///
    /// Requires `degree_bound` (after padding) to be at least `2^k` so that at least one WHIR
    /// round exists.
    pub fn new(mut polynomials: Vec<Polynomial>, blowup_log2: usize, k: usize) -> Self {
        assert!(k >= 1, "folding parameter k must be at least 1");

        let degree_bound = polynomials
            .iter_mut()
            .map(|polynomial| {
                polynomial.trim();
                polynomial.degree_bound()
            })
            .max()
            .unwrap()
            .next_power_of_two();

        let m0 = degree_bound.trailing_zeros() as usize;
        assert!(
            m0 >= k,
            "degree bound must be at least 2^k for WHIR to have at least one round"
        );

        polynomials = polynomials
            .into_iter()
            .map(|mut polynomial| {
                polynomial.pad(degree_bound);
                polynomial
            })
            .collect();

        let n = degree_bound << blowup_log2;
        let poly_tree = Tree::<H>::new(polynomials.iter().map(|p| p.clone().lde2(n)).collect());

        // t and num_final_queries tuned for 128-bit security against proximity testing.
        let t = 128usize.div_ceil(blowup_log2);
        let num_final_queries = t;

        Self {
            degree_bound,
            blowup_log2,
            k,
            t,
            num_final_queries,
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
    /// Implements Construction 5.1 (§5, page 32–34 of the WHIR paper, Arnon-Chiesa-Fenzi-Yogev
    /// 2024) compiled via BCS into a non-interactive proof via Fiat-Shamir.
    ///
    /// `points` maps each evaluation point to the vector of claimed evaluations across all
    /// committed polynomials: `points[z][i]` is the claimed value of polynomial `i` at point `z`.
    pub fn open(&self, points: BTreeMap<Scalar, Vec<Scalar>>) -> Proof<H> {
        let k = self.k;
        let t = self.t;
        let num_final_queries = self.num_final_queries;
        let m0 = self.degree_bound.trailing_zeros() as usize;
        let n0 = self.degree_bound << self.blowup_log2;
        let big_m = m0 / k; // total WHIR rounds

        let poly_tree_root = self.poly_tree.root_hash();

        // ── Fiat-Shamir: initial challenges ──────────────────────────────────────────────

        // γ (per-polynomial RLC challenge)
        let gamma = H::hash_many(&[*RLC_DST, poly_tree_root]);

        // Combined per-point claims
        let combined_claims: Vec<(Scalar, Scalar)> = points
            .iter()
            .map(|(z, vals)| (*z, rlc(vals, gamma)))
            .collect();

        // η (multi-constraint batching challenge)
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
        let eta = state;

        // ── Build combined oracle polynomial (unshifted, from original coefficients) ────

        let f_combined = {
            let mut combined = Polynomial::default();
            let mut power = Scalar::ONE;
            for poly in &self.polynomials {
                combined += poly.clone() * power;
                power *= gamma;
            }
            combined
        };

        // MLE truth table of f_combined: f_table[b] = f̂(b) for b ∈ {0,1}^m0.
        let mut f_table = sos_dp(f_combined.coefficients(), m0);

        // Weight table: w_table[b] = Σ_j η^j · eq_tensor(z_j, m0)[b].
        let mut w_table = vec![Scalar::ZERO; 1 << m0];
        {
            let mut eta_pow = Scalar::ONE;
            for &(z, _) in &combined_claims {
                let eq_z = eq_tensor_table(z, m0);
                for (b, &eq_b) in eq_z.iter().enumerate() {
                    w_table[b] += eta_pow * eq_b;
                }
                eta_pow *= eta;
            }
        }

        // ── Output collections ────────────────────────────────────────────────────────────

        let mut sumcheck_polys: Vec<Vec<[Scalar; 3]>> = Vec::with_capacity(big_m);
        let mut fold_roots: Vec<Scalar> = Vec::with_capacity(big_m - 1);
        let mut ood_answers: Vec<Scalar> = Vec::with_capacity(big_m - 1);
        let mut all_alphas: Vec<Vec<Scalar>> = Vec::with_capacity(big_m);
        let mut fold_trees: Vec<Tree<H>> = Vec::with_capacity(big_m.saturating_sub(1));
        let mut shift_idx_per_round: Vec<Vec<usize>> = Vec::with_capacity(big_m - 1);

        // Current folded polynomial in coefficient form (tracks fᵢ).
        let mut f_poly = f_combined;

        // ── Round 0: initial sumcheck ─────────────────────────────────────────────────────

        {
            let mut round_polys = Vec::with_capacity(k);
            let mut round_alphas = Vec::with_capacity(k);
            for _ in 0..k {
                let h = sumcheck_poly(&f_table, &w_table);
                let alpha = H::hash_many(&[*SC_ALPHA_DST, state, h[0], h[1], h[2]]);
                state = alpha;
                round_polys.push(h);
                round_alphas.push(alpha);
                mle_fold(&mut f_table, alpha);
                mle_fold(&mut w_table, alpha);
                f_poly = f_poly.fold2(alpha);
            }
            all_alphas.push(round_alphas);
            sumcheck_polys.push(round_polys);
        }
        let _ = poly_eval(sumcheck_polys[0][k - 1], all_alphas[0][k - 1]);

        // ── Main loop (rounds i = 1, …, M-1) ─────────────────────────────────────────────

        for i in 1..big_m {
            let m_i = m0 - k * i;
            let n_i = n0 >> (k * i);

            // Commit fᵢ (fold oracle i, evaluated on the unshifted domain {ω_{nᵢ}^j}).
            let f_i_evals = f_poly.clone().lde2(n_i);
            let f_i_tree = Tree::<H>::new(vec![f_i_evals.clone()]);
            let fold_root = f_i_tree.root_hash();
            fold_roots.push(fold_root);
            fold_trees.push(f_i_tree);

            // OOD challenge z_{i,0}
            state = H::hash_many(&[*OOD_DST, state, fold_root]);
            let z_ood = state;
            let y_i0 = f_poly.evaluate(z_ood);
            ood_answers.push(y_i0);
            state = H::hash_many(&[*OOD_DST, state, y_i0]);

            // Shift query indices in ℒᵢ (size n_i); derived without updating state.
            let q_mod = U256::from(n_i as u64);
            let shift_indices: Vec<usize> = (0..t)
                .map(|j| {
                    let h = H::hash_many(&[*SHIFT_DST, state, Scalar::from(j as u64)]);
                    (h.to_u256() % q_mod).low_u64() as usize
                })
                .collect();

            // γᵢ (combination challenge for this round)
            let gamma_i = H::hash_many(&[*COMBO_DST, state, Scalar::from(t as u64)]);
            state = gamma_i;

            shift_idx_per_round.push(shift_indices.clone());

            // Add OOD and shift-query contributions to the weight table (currently size 2^{m_i}).
            {
                let eq_ood = eq_tensor_table(z_ood, m_i);
                for (b, &eq_b) in eq_ood.iter().enumerate() {
                    w_table[b] += gamma_i * eq_b;
                }
                let mut gamma_pow = gamma_i * gamma_i;
                for &q in &shift_indices {
                    let z_shift = Polynomial::domain_element2(q, n_i);
                    let eq_shift = eq_tensor_table(z_shift, m_i);
                    for (b, &eq_b) in eq_shift.iter().enumerate() {
                        w_table[b] += gamma_pow * eq_b;
                    }
                    gamma_pow *= gamma_i;
                }
            }

            // Round i sumcheck (k sub-rounds over the current m_i-variable MLE).
            {
                let mut round_polys = Vec::with_capacity(k);
                let mut round_alphas = Vec::with_capacity(k);
                for _ in 0..k {
                    let h = sumcheck_poly(&f_table, &w_table);
                    let alpha = H::hash_many(&[*SC_ALPHA_DST, state, h[0], h[1], h[2]]);
                    state = alpha;
                    round_polys.push(h);
                    round_alphas.push(alpha);
                    mle_fold(&mut f_table, alpha);
                    mle_fold(&mut w_table, alpha);
                    f_poly = f_poly.fold2(alpha);
                }
                let _ = poly_eval(round_polys[k - 1], round_alphas[k - 1]);
                all_alphas.push(round_alphas);
                sumcheck_polys.push(round_polys);
            }
        }

        // final_poly = MLE truth table of f_M (after all k·M folds), size 2^{m_final}.
        let final_poly = f_table;

        // ── Generate shift-query Merkle proofs ────────────────────────────────────────────
        // Done AFTER fold_trees is fully built to avoid borrow conflicts.
        let mut oracle_query_proofs: Vec<Vec<Vec<merkle::Proof<H>>>> =
            Vec::with_capacity(big_m - 1);
        for i in 1..big_m {
            // Preimage count n_q in the source oracle for each target index q in ℒᵢ.
            let n_prev = n0 >> (k * (i - 1));
            let n_q = n_prev >> k; // = n_i (size of ℒᵢ)
            let prev_tree: &Tree<H> = if i == 1 {
                &self.poly_tree
            } else {
                &fold_trees[i - 2]
            };
            let mut proofs_for_round: Vec<Vec<merkle::Proof<H>>> =
                Vec::with_capacity(shift_idx_per_round[i - 1].len());
            for &q in &shift_idx_per_round[i - 1] {
                let proofs: Vec<merkle::Proof<H>> = (0..(1usize << k))
                    .map(|l| prev_tree.query(q + l * n_q))
                    .collect();
                proofs_for_round.push(proofs);
            }
            oracle_query_proofs.push(proofs_for_round);
        }

        // ── Generate final-query Merkle proofs ────────────────────────────────────────────
        let last_domain_size = n0 >> (k * (big_m - 1));
        let final_q_domain_size = last_domain_size >> k;
        let final_q_mod = U256::from(final_q_domain_size as u64);
        let n_q_final = final_q_domain_size; // = last_domain_size >> k
        let last_oracle: &Tree<H> = if big_m == 1 {
            &self.poly_tree
        } else {
            &fold_trees[big_m - 2]
        };

        let mut final_query_proofs: Vec<Vec<merkle::Proof<H>>> =
            Vec::with_capacity(num_final_queries);
        for l in 0..num_final_queries {
            let r_hash = H::hash_many(&[*FINAL_DST, state, Scalar::from(l as u64)]);
            let r_idx = (r_hash.to_u256() % final_q_mod).low_u64() as usize;
            let proofs: Vec<merkle::Proof<H>> = (0..(1usize << k))
                .map(|j| last_oracle.query(r_idx + j * n_q_final))
                .collect();
            final_query_proofs.push(proofs);
        }

        Proof {
            points,
            fold_roots,
            sumcheck_polys,
            ood_answers,
            final_poly,
            oracle_query_proofs,
            final_query_proofs,
            _data: PhantomData,
        }
    }
}

/// A WHIR opening proof.
///
/// Produced by [`Prover::open`] and verified by [`Proof::verify`] against the corresponding
/// [`Commitment`]. The proof is self-contained: the verifier needs only the commitment and this
/// struct to reconstruct the full Fiat-Shamir transcript and execute all decision-phase checks.
///
/// The layout follows Construction 5.1 (§5, page 32) compiled via BCS into a non-interactive proof.
#[derive(Debug, Clone)]
pub struct Proof<H: Hash<Scalar>> {
    /// Claimed evaluation values.
    ///
    /// `points[z][i]` is the claimed value of the i-th committed polynomial at z.
    points: BTreeMap<Scalar, Vec<Scalar>>,

    /// Merkle roots of the M-1 folded oracles f₁, …, f_{M-1} (step 2a).
    fold_roots: Vec<Scalar>,

    /// Sumcheck polynomials for every round (steps 1a and 2e).
    ///
    /// `sumcheck_polys[i]` contains k entries; each entry `[a₀, a₁, a₂]` stores the coefficients
    /// of h(X) = a₀ + a₁X + a₂X².
    sumcheck_polys: Vec<Vec<[Scalar; 3]>>,

    /// Out-of-domain answers for main-loop rounds 1, …, M-1 (step 2c).
    ood_answers: Vec<Scalar>,

    /// Multilinear values of the final polynomial f̂_M on {0,1}^{m_M}.
    final_poly: Vec<Scalar>,

    /// Merkle-proof openings for shift-query evaluations (step 2d / decision step 2b).
    ///
    /// `oracle_query_proofs[i-1][j]` contains 2^k proofs opening oracle f_{i-1} at the 2^k
    /// preimages of shift-query index j in ℒᵢ.
    oracle_query_proofs: Vec<Vec<Vec<merkle::Proof<H>>>>,

    /// Merkle-proof openings for the final fold-consistency checks (step 4 / decision step 3a).
    final_query_proofs: Vec<Vec<merkle::Proof<H>>>,

    _data: PhantomData<H>,
}

impl<H: Hash<Scalar>> Proof<H> {
    /// Verifies this proof against the given commitment and the evaluation claims embedded in it.
    ///
    /// Implements all decision-phase checks from Construction 5.1 (§5, page 32) and the
    /// multi-constraint batching from Construction 5.5 (§5.2, page 40).
    pub fn verify(&self, commitment: &Commitment) -> Result<()> {
        // ── Parameters ───────────────────────────────────────────────────────────────
        let poly_tree_root = commitment.poly_tree_root;
        let m0 = commitment.degree_bound.trailing_zeros() as usize;
        let n0 = commitment.degree_bound << commitment.blowup_log2;

        if self.sumcheck_polys.is_empty() {
            return Err(anyhow!("empty sumcheck_polys"));
        }
        let k = self.sumcheck_polys[0].len();
        if k == 0 {
            return Err(anyhow!("zero folding parameter"));
        }
        let big_m = self.fold_roots.len() + 1;
        let m_final = m0.saturating_sub(k * big_m);

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
                return Err(anyhow!(
                    "sumcheck_polys[{i}] length: got {}, want {k}",
                    round.len()
                ));
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
        let gamma = H::hash_many(&[*RLC_DST, poly_tree_root]);

        let combined_claims: Vec<(Scalar, Scalar)> = self
            .points
            .iter()
            .map(|(z, vals)| (*z, rlc(vals, gamma)))
            .collect();

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

        let eta = state;
        let sigma0 = rlc(
            &combined_claims.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
            eta,
        );

        // ── Round 0: initial sumcheck (decision step 1) ──────────────────────────────
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
                if l + 1 < k && poly_sum(self.sumcheck_polys[0][l + 1]) != poly_eval(h, alpha) {
                    return Err(anyhow!(
                        "sumcheck consistency failed in round 0 sub-round {}",
                        l + 1
                    ));
                }
            }
            all_alphas.push(round0_alphas);
        }
        let mut prev_sc_last = poly_eval(self.sumcheck_polys[0][k - 1], all_alphas[0][k - 1]);

        // Saved data for the weight-constraint check at the end.
        let mut ood_field_elems: Vec<Scalar> = Vec::with_capacity(big_m - 1);
        let mut shift_q_per_round: Vec<Vec<usize>> = Vec::with_capacity(big_m - 1);
        let mut gamma_per_round: Vec<Scalar> = Vec::with_capacity(big_m - 1);

        // ── Main loop (rounds i = 1, …, M-1) ─────────────────────────────────────────────
        for i in 1..big_m {
            let fold_root = self.fold_roots[i - 1];
            let prev_domain_size = n0 >> (k * (i - 1));
            let curr_domain_size = n0 >> (k * i);
            let prev_oracle_root = if i == 1 {
                poly_tree_root
            } else {
                self.fold_roots[i - 2]
            };
            let is_initial = i == 1;

            // Absorb fᵢ root → OOD sample z_{i,0}.
            state = H::hash_many(&[*OOD_DST, state, fold_root]);
            let z_ood_raw = state;
            ood_field_elems.push(z_ood_raw);

            // Absorb OOD answer y_{i,0}.
            let y_i0 = self.ood_answers[i - 1];
            state = H::hash_many(&[*OOD_DST, state, y_i0]);

            // Derive shift query indices in ℒᵢ.
            let t = self.oracle_query_proofs[i - 1].len();
            let q_mod = U256::from(curr_domain_size as u64);
            let shift_indices: Vec<usize> = (0..t)
                .map(|j| {
                    let h = H::hash_many(&[*SHIFT_DST, state, Scalar::from(j as u64)]);
                    (h.to_u256() % q_mod).low_u64() as usize
                })
                .collect();

            // Derive γᵢ.
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
                if l + 1 < k && poly_sum(self.sumcheck_polys[i][l + 1]) != poly_eval(h, alpha) {
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
        let last_domain_size = n0 >> (k * (big_m - 1));
        let final_q_domain_size = last_domain_size >> k;
        let final_q_mod = U256::from(final_q_domain_size as u64);

        for l in 0..self.final_query_proofs.len() {
            let r_hash = H::hash_many(&[*FINAL_DST, state, Scalar::from(l as u64)]);
            let r_idx = (r_hash.to_u256() % final_q_mod).low_u64() as usize;

            // Compute g_{M-1}(r_l^fin) via fold on f_{M-1}.
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
            let r_raw = Polynomial::domain_element2(r_idx, final_q_domain_size);
            let r_vec = pow_seg(r_raw, 0, m_final);
            let f_m_val = multilinear_eval(&self.final_poly, &r_vec);
            if f_m_val != g_val {
                return Err(anyhow!("final fold-consistency check failed at l={l}"));
            }
        }

        // ── Weight constraint (decision step 3c) ──────────────────────────────────────
        let weight_sum = {
            let mut sum = Scalar::ZERO;

            // Initial eval constraints: one per eval point z.
            for (j, &(z, _)) in combined_claims.iter().enumerate() {
                let const_z = eta.pow_small(j) * eq_prod(z, big_m, k, &all_alphas);
                let eval_pt = pow_seg(z, k * big_m, m_final);
                sum += const_z * multilinear_eval(&self.final_poly, &eval_pt);
            }

            // OOD and shift-query constraints for each main-loop round i = 1,…,M-1.
            for i in 1..big_m {
                let gamma_i = gamma_per_round[i - 1];
                let curr_domain_size = n0 >> (k * i);
                let remaining = big_m - i;
                let alphas_from_i = &all_alphas[i..];

                // OOD sample z_{i,0} with coefficient γᵢ^1.
                {
                    let z_ood = ood_field_elems[i - 1];
                    let const_i0 = gamma_i * eq_prod(z_ood, remaining, k, alphas_from_i);
                    let eval_pt = pow_seg(z_ood, k * remaining, m_final);
                    sum += const_i0 * multilinear_eval(&self.final_poly, &eval_pt);
                }

                // Shift queries with coefficient γᵢ^{j+2}.
                let mut gamma_pow = gamma_i * gamma_i;
                for &q in &shift_q_per_round[i - 1] {
                    let z_shift = Polynomial::domain_element2(q, curr_domain_size);
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
    use crate::hash;

    type Sha2Hash = hash::Sha2Hash<Scalar>;
    type Poseidon2Hash = hash::Poseidon2Hash<Scalar>;

    const fn from_const(value: u64) -> Scalar {
        Scalar::from_const(value)
    }

    fn test_open_impl<H: Hash<Scalar>>(
        polynomials: Vec<Polynomial>,
        points: &[u64],
        k: usize,
        blowup_log2: usize,
    ) {
        let eval_points: BTreeMap<Scalar, Vec<Scalar>> = points
            .iter()
            .map(|&z| {
                let z_scalar = Scalar::from(z);
                let vals = polynomials
                    .iter()
                    .map(|p| p.evaluate(z_scalar))
                    .collect::<Vec<_>>();
                (z_scalar, vals)
            })
            .collect();

        let prover = Prover::<H>::new(polynomials, blowup_log2, k);
        let commitment = prover.commit();
        let proof = prover.open(eval_points);
        assert!(proof.verify(&commitment).is_ok());
    }

    fn test_open(polynomials: Vec<Polynomial>, points: &[u64], k: usize) {
        test_open_impl::<Sha2Hash>(polynomials.clone(), points, k, 1);
        test_open_impl::<Poseidon2Hash>(polynomials.clone(), points, k, 1);
        test_open_impl::<Sha2Hash>(polynomials.clone(), points, k, 2);
        test_open_impl::<Poseidon2Hash>(polynomials, points, k, 2);
    }

    #[test]
    fn test_one_polynomial_degree_one_one_point() {
        test_open(
            vec![Polynomial::with_coefficients(vec![
                from_const(12),
                from_const(34),
            ])],
            &[123],
            1,
        );
    }

    #[test]
    fn test_one_polynomial_degree_three_one_point() {
        test_open(
            vec![Polynomial::with_coefficients(vec![
                from_const(12),
                from_const(34),
                from_const(56),
                from_const(78),
            ])],
            &[123],
            1,
        );
    }

    #[test]
    fn test_one_polynomial_degree_three_two_points() {
        test_open(
            vec![Polynomial::with_coefficients(vec![
                from_const(12),
                from_const(34),
                from_const(56),
                from_const(78),
            ])],
            &[123, 456],
            1,
        );
    }

    #[test]
    fn test_two_polynomials_degree_three_one_point() {
        test_open(
            vec![
                Polynomial::with_coefficients(vec![
                    from_const(12),
                    from_const(34),
                    from_const(56),
                    from_const(78),
                ]),
                Polynomial::with_coefficients(vec![
                    from_const(42),
                    from_const(43),
                    from_const(44),
                    from_const(45),
                ]),
            ],
            &[123],
            1,
        );
    }

    #[test]
    fn test_one_polynomial_degree_three_one_point_k2() {
        test_open(
            vec![Polynomial::with_coefficients(vec![
                from_const(12),
                from_const(34),
                from_const(56),
                from_const(78),
            ])],
            &[123],
            2,
        );
    }
}
