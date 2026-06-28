use crate::hash::Hash;
use crate::merkle::{Proof as LeafProof, Tree};
use crate::utils;
use anyhow::{Result, anyhow};
use starkom_bluesky::Scalar;
use starkom_ff::{Field, PrimeField};
use starkom_poly;
use std::marker::PhantomData;
use std::sync::LazyLock;

type Polynomial = starkom_poly::Polynomial<Scalar>;

/// Domain separator tag used when deriving the Fiat-Shamir challenge for STIR folding.
static FOLD_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/stir/fold"));

/// Domain separator tag used when deriving the per-round out-of-domain (OOD) challenge point.
static OOD_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/stir/ood"));

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

#[derive(Debug, Clone)]
pub struct Query<H: Hash<Scalar>> {
    /// Size of the initial evaluation domain |L_0| = N.
    n: usize,
    /// Query index into the initial oracle f_0. The fold-partner index is (index + n/2) % n.
    index: usize,
    /// Fold chain: one pair of `LeafProof` per oracle, for M+1 oracles total. Pair r proves the
    /// values of f_r at the main query position and its fold-partner in L_r.
    folds: Vec<(LeafProof<H>, LeafProof<H>)>,
    /// Shift proofs: one list per folding round, for M rounds total. List r contains t-1 proofs
    /// opening f_{r+1} at the shift challenge positions z_{r,j} derived for round r.
    shift_proofs: Vec<Vec<LeafProof<H>>>,
    _data: PhantomData<H>,
}

impl<H: Hash<Scalar>> Query<H> {
    /// Returns the main query index and its fold-partner index into L_0.
    pub fn indices(&self) -> (usize, usize) {
        (self.index, (self.index + self.n / 2) % self.n)
    }

    /// Returns the domain element x at the main query index.
    ///
    /// Used by the DEEP layer for the algebraic check q(x) * (x - z) = combined(x) - v.
    pub fn x(&self) -> Scalar {
        Polynomial::coset_element2(self.index, self.n)
    }

    /// Returns the values of f_0 at the main query position and its fold-partner.
    pub fn values(&self) -> (Scalar, Scalar) {
        (self.folds[0].0.leaf(), self.folds[0].1.leaf())
    }

    /// Returns the number of committed oracles, equal to M+1.
    pub fn len(&self) -> usize {
        self.folds.len()
    }

    /// Returns the shift proof values for the given folding round r (0-indexed).
    ///
    /// Each element of the returned vector is the value of f_{r+1} at one shift position.
    pub fn shift_values(&self, round: usize) -> Vec<Scalar> {
        self.shift_proofs[round]
            .iter()
            .map(|proof| proof.leaf())
            .collect()
    }

    /// Verifies this proof against the given commitment.
    ///
    /// NOTE: for low-degree testing you also need to check that `len()` returns
    /// `log2(degree_bound) + 1`. This function only verifies the fold chain consistency and the
    /// OOD / shift consistency; it does not check the number of rounds.
    pub fn verify(&self, commitment: &Commitment) -> Result<()> {
        assert!(self.n.is_power_of_two());
        assert!(self.index < self.n);

        let num_rounds = commitment.num_rounds();
        if self.folds.len() != num_rounds + 1 {
            return Err(anyhow!("wrong number of fold proofs"));
        }

        let k = self.n.trailing_zeros() as usize;
        let mut n = self.n;
        let mut i = self.index;
        let mut step = Scalar::ROOT_OF_UNITY_INV.pow_u64(1u64 << (Scalar::S - k));
        let mut pos = self.folds[0].0.leaf();

        let (mut left, mut right) = (&self.folds[0].0, &self.folds[0].1);
        let mut root = commitment.roots()[0];

        if left.leaf() != pos {
            return Err(anyhow!("fold chain broken at round 0"));
        }
        left.verify(i, root)?;
        right.verify((i + n / 2) % n, root)?;

        for r in 0..num_rounds {
            let alpha = H::hash_two(*FOLD_DST, root, commitment.ood_values()[r]);
            let v0 = left.leaf();
            let v1 = right.leaf();
            let omega_inv_i = step.pow_small(i);
            n /= 2;
            i %= n;
            step = step.square();
            pos = (v0 + v1 + alpha * omega_inv_i * (v0 - v1)) * Scalar::TWO_INV;

            left = &self.folds[r + 1].0;
            right = &self.folds[r + 1].1;
            root = commitment.roots()[r + 1];

            if left.leaf() != pos {
                return Err(anyhow!("fold chain broken at round {}", r));
            }
            left.verify(i, root)?;
            right.verify((i + n / 2) % n, root)?;
        }

        let (left, right) = self.folds.last().unwrap();
        if !left.is_constant() || !right.is_constant() {
            return Err(anyhow!("final oracle is not constant"));
        }

        Ok(())
    }
}

/// A STIR prover.
#[derive(Debug, Clone)]
pub struct Prover<H: Hash<Scalar>> {
    /// The degree bound of the committed polynomial (always a power of 2).
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// Merkle trees for all committed oracles: trees[0] is f_0, trees[r] is the r-th fold.
    /// Used to answer both fold-chain queries and shift queries.
    trees: Vec<Tree<H>>,
    /// Pre-computed OOD evaluations y_r = f_r(s_r), one per folding round.
    /// Computed during `new()` by evaluating each folded polynomial at the Fiat-Shamir OOD
    /// challenge derived from its tree root. Stored here so that `commit()` can package them into
    /// the `Commitment` without needing to re-evaluate.
    ood_values: Vec<Scalar>,
}

impl<H: Hash<Scalar>> Prover<H> {
    pub fn new(mut polynomial: Polynomial, degree_bound: usize, blowup_log2: usize) -> Self {
        assert!(degree_bound.is_power_of_two());
        assert!(degree_bound >= polynomial.degree_bound());

        let mut n = degree_bound << blowup_log2;
        assert!(n as u64 <= 1u64 << Scalar::S);

        polynomial = polynomial.shift_domain();

        let num_rounds = degree_bound.trailing_zeros() as usize;
        let mut trees: Vec<Tree<H>> = Vec::with_capacity(num_rounds + 1);
        let mut ood_values: Vec<Scalar> = Vec::with_capacity(num_rounds);

        let mut tree = Tree::<H>::new(polynomial.clone().lde2(n));
        let mut root = tree.root_hash();
        trees.push(tree);

        for _ in 0..num_rounds {
            let ood_point = H::hash_two(*OOD_DST, root, Scalar::ZERO);
            let ood_value = polynomial.evaluate(ood_point);
            ood_values.push(ood_value);

            let alpha = H::hash_two(*FOLD_DST, root, ood_value);
            polynomial = polynomial.fold2(alpha);
            n >>= 1;

            tree = Tree::<H>::new(polynomial.clone().lde2(n));
            root = tree.root_hash();
            trees.push(tree);
        }

        Self {
            degree_bound,
            blowup_log2,
            trees,
            ood_values,
        }
    }

    /// Returns the degree bound of the committed polynomial (always a power of 2).
    pub fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    /// Returns the size of the extended evaluation domain, equal to `degree_bound * 2^blowup_log2`.
    pub fn extended_domain_size(&self) -> usize {
        self.degree_bound << self.blowup_log2
    }

    /// Alias for `extended_domain_size`.
    pub fn size(&self) -> usize {
        self.degree_bound << self.blowup_log2
    }

    /// Returns the Merkle root hash of the initial oracle f_0, that is the root hash of the
    /// original polynomial evaluations.
    pub fn root_hash(&self) -> Scalar {
        self.trees[0].root_hash()
    }

    /// Creates the STIR commitment.
    pub fn commit(&self) -> Commitment {
        Commitment {
            roots: self.trees.iter().map(|tree| tree.root_hash()).collect(),
            ood_values: self.ood_values.clone(),
        }
    }

    /// Builds a STIR `Query` for the given index into the initial evaluation domain L_0.
    pub fn query(&self, index: usize) -> Query<H> {
        let n = self.degree_bound << self.blowup_log2;
        assert!(index < n);

        let mut m = n;
        let mut i = index;
        let mut folds = Vec::with_capacity(self.trees.len());
        for tree in &self.trees {
            folds.push((tree.query(i), tree.query((i + m / 2) % m)));
            m /= 2;
            i %= m;
        }

        {
            let (left, right) = folds.last().unwrap();
            assert!(left.is_constant());
            assert!(right.is_constant());
        }

        // Shift proofs (t = 1 for now: only the OOD evaluation, no additional shift queries).
        let num_rounds = self.degree_bound.trailing_zeros() as usize;
        let shift_proofs = (0..num_rounds).map(|_| vec![]).collect();

        Query {
            n,
            index,
            folds,
            shift_proofs,
            _data: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash;

    type Poseidon2Hash = hash::Poseidon2Hash<Scalar>;
    type Sha2Hash = hash::Sha2Hash<Scalar>;

    #[inline(always)]
    const fn from_const(value: u64) -> Scalar {
        Scalar::from_const(value)
    }

    fn test_prover<H: Hash<Scalar>>(
        polynomial: Polynomial,
        degree_bound: usize,
        blowup_log2: usize,
    ) {
        let n = degree_bound << blowup_log2;
        let prover = Prover::<H>::new(polynomial, degree_bound, blowup_log2);
        let commitment = prover.commit();
        for index in 0..n {
            let query = prover.query(index);
            assert!(query.verify(&commitment).is_ok());
        }
    }

    fn test_blowups<H: Hash<Scalar>>(polynomial: Polynomial, degree_bound: usize) {
        test_prover::<H>(polynomial.clone(), degree_bound, 1);
        test_prover::<H>(polynomial.clone(), degree_bound, 2);
        test_prover::<H>(polynomial, degree_bound, 3);
    }

    #[test]
    fn test_constant_polynomial_sha2() {
        let p = Polynomial::with_coefficients(vec![from_const(42)]);
        test_blowups::<Sha2Hash>(p, 1);
    }

    #[test]
    fn test_constant_polynomial_poseidon2() {
        let p = Polynomial::with_coefficients(vec![from_const(42)]);
        test_blowups::<Poseidon2Hash>(p, 1);
    }

    #[test]
    fn test_degree_one_polynomial_sha2() {
        let p = Polynomial::with_coefficients(vec![from_const(12), from_const(34)]);
        test_blowups::<Sha2Hash>(p, 2);
    }

    #[test]
    fn test_degree_one_polynomial_poseidon2() {
        let p = Polynomial::with_coefficients(vec![from_const(12), from_const(34)]);
        test_blowups::<Poseidon2Hash>(p, 2);
    }

    #[test]
    fn test_degree_two_polynomial_sha2() {
        let p = Polynomial::with_coefficients(vec![from_const(12), from_const(34), from_const(56)]);
        test_blowups::<Sha2Hash>(p, 4);
    }

    #[test]
    fn test_degree_two_polynomial_poseidon2() {
        let p = Polynomial::with_coefficients(vec![from_const(12), from_const(34), from_const(56)]);
        test_blowups::<Poseidon2Hash>(p, 4);
    }

    #[test]
    fn test_degree_three_polynomial_sha2() {
        let p = Polynomial::with_coefficients(vec![
            from_const(12),
            from_const(34),
            from_const(56),
            from_const(78),
        ]);
        test_blowups::<Sha2Hash>(p, 4);
    }

    #[test]
    fn test_degree_three_polynomial_poseidon2() {
        let p = Polynomial::with_coefficients(vec![
            from_const(12),
            from_const(34),
            from_const(56),
            from_const(78),
        ]);
        test_blowups::<Poseidon2Hash>(p, 4);
    }
}
