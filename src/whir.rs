use crate::hash::Hash;
use crate::utils;
use anyhow::Result;
use starkom_bluesky::Scalar;
use starkom_poly;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::LazyLock;

type Polynomial = starkom_poly::Polynomial<Scalar>;

/// Domain separator tag used when deriving the Fiat-Shamir challenge for the random linear combination.
static RLC_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/rlc"));

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment {
    // TODO
}

impl Commitment {
    // TODO
}

#[derive(Debug, Clone)]
pub struct Proof<H: Hash<Scalar>> {
    // TODO
    _data: PhantomData<H>,
}

impl<H: Hash<Scalar>> Prover<H> {
    pub fn verify(&self, commitment: &Commitment) -> Result<()> {
        // TODO
        todo!()
    }
}

#[derive(Debug, Default, Clone)]
pub struct Prover<H: Hash<Scalar>> {
    /// The degree bound of the committed polynomials. This is the highest degree among the
    /// committed polynomials, plus one.
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    // TODO
    _data: PhantomData<H>,
}

impl<H: Hash<Scalar>> Prover<H> {
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

        // TODO
        todo!()
    }

    pub fn commit(&self) -> Commitment {
        // TODO
        todo!()
    }

    pub fn open(&self, points: BTreeMap<Scalar, Vec<Scalar>>) -> Proof<H> {
        // TODO
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    //TODO
}
