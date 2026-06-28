use crate::utils;
use starkom_bluesky::Scalar;
use starkom_poly;
use std::sync::LazyLock;

type Polynomial = starkom_poly::Polynomial<Scalar>;

/// Domain separator tag used when deriving the Fiat-Shamir challenge for STIR folding.
static FOLD_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/fold"));

/// Domain separator tag used when deriving the per-round out-of-domain (OOD) challenge point.
static OOD_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/ood"));

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment {
    /// Merkle roots of all committed oracles: index 0 is the initial oracle, indices 1..=M are the
    /// M successive folds. Has M+1 entries in total where M is the number of folding rounds.
    roots: Vec<Scalar>,
    /// Prover's responses to the per-round OOD challenges, one per folding round. Entry r is the
    /// evaluation of oracle f_r at the OOD point s_r = H(OOD_DST, roots[r]).
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

// TODO
