use crate::hash::Hash;
use crate::merkle::{self, Tree};
use crate::utils;
use anyhow::Result;
use starkom_bluesky::Scalar;
use starkom_poly;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::LazyLock;

type Polynomial = starkom_poly::Polynomial<Scalar>;

/// Domain separator tag used when deriving the Fiat-Shamir challenge for the random linear
/// combination.
static RLC_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/rlc"));

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
    pub fn verify(&self, commitment: &Commitment) -> Result<()> {
        // TODO
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO
}
