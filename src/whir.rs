use crate::hash::Hash;
use crate::merkle::Tree;
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
#[derive(Debug, Clone)]
pub struct Proof<H: Hash<Scalar>> {
    degree_bound: usize,
    blowup_log2: usize,
    // TODO
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
