use crate::hash::Hash;
use crate::utils;
use starkom_bluesky::Scalar;
use starkom_poly;
use std::marker::PhantomData;
use std::sync::LazyLock;

type Polynomial = starkom_poly::Polynomial<Scalar>;

/// Domain separator tag used when deriving the Fiat-Shamir challenge for STIR folding.
static FOLD_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/fold"));

/// Domain separator tag used when deriving the per-round out-of-domain (OOD) challenge point.
static OOD_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/ood"));

/// A WHIR commitment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment {
    /// Number of variables folded per round, equal to the number of sumcheck rounds per folding
    /// round. Optimal value is log2(m) where m = log2(degree_bound).
    num_folds_per_round: usize,
    /// Merkle roots of all committed oracles: index 0 is the initial oracle, indices 1..=M are the
    /// M successive folds. Has M+1 entries in total where M is the number of folding rounds.
    roots: Vec<Scalar>,
    /// Sumcheck polynomials sent by the prover, one Vec<[Scalar; 3]> per folding round, each
    /// containing `num_folds_per_round` univariate polynomials of degree < 3 (constant, linear,
    /// or quadratic).
    sumcheck_polynomials: Vec<Vec<[Scalar; 3]>>,
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

#[derive(Debug)]
pub struct Prover<H: Hash<Scalar>> {
    /// The proven degree bound. The degree of all batched polynomials must be strictly less than
    /// this value.
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    // TODO
    _data: PhantomData<H>,
}

impl<H: Hash<Scalar>> Prover<H> {
    pub fn new(polynomials: Vec<Polynomial>, blowup_log2: usize) -> Self {
        let degree_bound: usize = polynomials
            .iter()
            .map(|polynomial| polynomial.degree_bound())
            .max()
            .unwrap()
            .next_power_of_two();

        // TODO
        todo!()
    }

    pub fn num_folds_per_round(&self) -> usize {
        debug_assert!(self.degree_bound.is_power_of_two());
        self.degree_bound.trailing_zeros() as usize
    }

    pub fn commit(&self) -> Commitment {
        // TODO
        todo!()
    }

    // TODO
}

// TODO
