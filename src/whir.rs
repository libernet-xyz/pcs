use crate::hash::Hash;
use crate::utils;
use anyhow::Result;
use starkom_bluesky::Scalar;
use starkom_poly;
use std::collections::{BTreeMap, BTreeSet};
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

    /// Adds a batch of polynomials, returning the index of the newly created batch.
    ///
    /// REQUIRES: the degree of all specified polynomials must be strictly less than
    /// [`Self::degree_bound()`].
    pub fn add_batch(&mut self, polynomials: Vec<Polynomial>) -> usize {
        // TODO
        todo!()
    }

    pub fn commit(self, points: BTreeSet<Scalar>) -> (Commitment, Prover<H>) {
        // TODO
        todo!()
    }
}

/// A WHIR proof.
#[derive(Debug, Clone)]
pub struct Proof<H: Hash<Scalar>> {
    /// The proven degree bound. If the proof is valid the degree of all batched polynomials is
    /// guaranteed to be strictly less than this value.
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// Number of committed polynomials.
    num_polys: usize,
    /// The opened points. Keys are (off-domain) X-coordinates, values are the corresponding
    /// evaluations (one for every committed polynomial).
    points: BTreeMap<Scalar, Vec<Scalar>>,
    // TODO
    _data: PhantomData<H>,
}

impl<H: Hash<Scalar>> Proof<H> {
    /// Returns the proven degree bound.
    pub fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    /// Returns the base-2 logarithm of the blowup factor used in the proof.
    pub fn blowup_log2(&self) -> usize {
        self.blowup_log2
    }

    /// Returns the size of the extended evaluation domain.
    pub fn extended_domain_size(&self) -> usize {
        self.degree_bound << self.blowup_log2
    }

    /// Returns the number of committed polynomials.
    pub fn num_polys(&self) -> usize {
        self.num_polys
    }

    /// Returns a reference to the opened points. Keys are (off-domain) X-coordinates, values are
    /// the corresponding evaluations (one for every committed polynomial).
    pub fn points(&self) -> &BTreeMap<Scalar, Vec<Scalar>> {
        &self.points
    }

    /// Verifies this proof against the given commitment.
    pub fn verify(&self, commitment: &Commitment) -> Result<()> {
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

impl<H: Hash<Scalar>> Prover<H> {
    /// Returns the proven degree bound.
    pub fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    /// Returns the size of the extended evaluation domain.
    pub fn extended_domain_size(&self) -> usize {
        self.degree_bound << self.blowup_log2
    }

    /// Makes a WHIR proof opening the committed polynomials at the points specified at commitment
    /// time (see [`Committer::commit()`]).
    pub fn prove(&self, commitment: &Commitment) -> Proof<H> {
        // TODO
        todo!()
    }
}
