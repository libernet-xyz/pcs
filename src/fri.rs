use crate::hash::Hasher;
use crate::merkle::{Proof as LeafProof, Tree};
use anyhow::{anyhow, Result};
use primitive_types::H256;
use sha2::Digest;
use starkom_ff::Field256;
use starkom_poly::Polynomial;
use std::sync::LazyLock;

/// Domain separator tag used when deriving the Fiat-Shamir challenge for FRI folding.
static FOLD_DST: LazyLock<H256> = LazyLock::new(|| {
    let mut hasher = sha3::Sha3_256::new();
    hasher.update(b"starkom/fri/fold");
    H256::from_slice(hasher.finalize().as_slice())
});

trait FoldableTree<F: Field256, H: Hasher<F>>: Sized {
    /// Performs one FRI folding round, returning the new folded tree.
    fn fold(&self) -> Tree<F, F, H>;

    /// Performs `times` FRI folding rounds and returns an array of `times` trees.
    ///
    /// The first element is the tree from the first folding round (N/2 leaves, with N being the
    /// number of leaves in `self`), the second element is the tree from the second folding round
    /// (N/4 leaves), and so on. If `times` is 0 no folding is performed and an empty array is
    /// returned.
    fn fold_all(self, times: usize) -> Vec<Tree<F, F, H>>;
}

impl<F: Field256, H: Hasher<F>> FoldableTree<F, H> for Tree<F, F, H> {
    fn fold(&self) -> Tree<F, F, H> {
        let num_polys = self.num_polys();
        let n = self.num_leaves();
        assert!(n.is_power_of_two());

        let alpha = H::challenge(*FOLD_DST, &[self.root_hash()]);

        let k = n.trailing_zeros() as usize;
        let omega_inv = F::ROOT_OF_UNITY_INV.pow_u64(1u64 << (F::S - k));

        let m = n / 2;
        let mut omega_inv_i = F::ONE;

        let mut leaves = vec![vec![F::ZERO; m]; num_polys];
        for i in 0..m {
            for j in 0..num_polys {
                let pos = self.leaf_value(j, i);
                let neg = self.leaf_value(j, i + m);
                leaves[j][i] = (pos + neg + alpha * omega_inv_i * (pos - neg)) * F::TWO_INV;
            }
            omega_inv_i *= omega_inv;
        }

        Self::new(leaves)
    }

    fn fold_all(self, times: usize) -> Vec<Tree<F, F, H>> {
        let mut trees = Vec::with_capacity(times + 1);
        let mut tree = self;
        for _ in 0..times {
            let folded = tree.fold();
            trees.push(tree);
            tree = folded;
        }
        trees.push(tree);
        trees
    }
}

/// Stores the Merkle root hashes of a FRI commitment.
///
/// Note that for low-degree testing these are *less* than log2(N), with N being the number of
/// committed evaluations. Once the folding process has reduced all polynomials to degree-0 ones
/// (that is, single constants) all subsequent folds would be identical, so we don't store them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment {
    /// The first element in the array is the root of the main Merkle tree, the second one is the
    /// root of the Merkle tree from the first folding round, and so on until the last element which
    /// is the value of the last folding round.
    roots: Vec<H256>,
}

impl Commitment {
    /// Returns the number of stored roots, equivalent to the number of folding rounds and therefore
    /// to the log2 of the degree bound plus one. For example, if the user commits 4 evaluations
    /// `len()` will return 3.
    pub fn len(&self) -> usize {
        self.roots.len()
    }

    /// Returns the Merkle roots of all folding rounds.
    ///
    /// The returned slice has [`Self::len()`] elements.
    pub fn roots(&self) -> &[H256] {
        self.roots.as_slice()
    }

    /// Returns the Merkle root hash of the committed polynomial, which is the first hash stored in
    /// the commitment.
    pub fn root(&self) -> H256 {
        *self.roots.first().unwrap()
    }
}

/// A single FRI query.
#[derive(Debug, Clone)]
pub struct Query<F: Field256, H: Hasher<F>> {
    /// The degree bound of the committed polynomials (always a power of 2).
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// The index of the element we're opening (the partner index is inferred automatically).
    index: usize,
    /// Proves a pair of "partner" values at each folding round with one [`LeafProof`] pair for
    /// every round. The pair at `folds[0]` proves the opened values.
    folds: Vec<(LeafProof<F, F, H>, LeafProof<F, F, H>)>,
}

impl<F: Field256, H: Hasher<F>> Query<F, H> {
    /// Returns the two opened indices.
    pub fn indices(&self) -> (usize, usize) {
        let n = self.degree_bound << self.blowup_log2;
        (self.index, (self.index + n / 2) % n)
    }

    /// Returns the opened domain element, that is the X-coordinate of the evaluation.
    ///
    /// This is the element corresponding to the first value returned by [`Self::indices`], while
    /// the partner element can be obtained by simply negating this one.
    ///
    /// Note that we use [`Polynomial::shift_domain`] before committing polynomials, so the element
    /// returned here is a shifted power of an N-th root of unity, with
    /// `N = degree_bound * 2^blowup_factor`. The shift consists of multiplying the actual domain
    /// element by [`PrimeField::MULTIPLICATIVE_GENERATOR`], consistently with `shift_domain`.
    pub fn x(&self) -> F {
        Polynomial::<F>::coset_element2(self.index, self.degree_bound << self.blowup_log2)
    }

    /// Returns the opened evaluations, one for every committed polynomial.
    ///
    /// The first component of the returned tuple contains the evaluations at the first index
    /// returned by [`Self::indices`], while the second component contains those at the second
    /// index.
    pub fn values(&self) -> (&[F], &[F]) {
        (self.folds[0].0.leaf(), self.folds[0].1.leaf())
    }

    /// Returns the number of Merkle trees in the proof: one for the original polynomial evaluations
    /// plus one for every folding round.
    ///
    /// In general these are log2(d)+1, with `d` being the degree bound of the committed polynomial.
    /// Note that for low-degree testing `d` is strictly less than the number of committed
    /// evaluations `N`.
    pub fn len(&self) -> usize {
        self.folds.len()
    }

    /// Verifies this proof against the given commitment.
    ///
    /// NOTE: for low-degree testing you also need to check that [`Self::len`] returns the log2 of
    /// the expected degree bound. This function only verifies the opened value pair across the
    /// folding structure.
    pub fn verify(&self, commitment: &Commitment) -> Result<()> {
        let mut n = self.degree_bound << self.blowup_log2;
        assert!(n.is_power_of_two());
        assert!(self.index < n);

        let mut k = n.trailing_zeros() as usize;

        let num_folds = self.folds.len();
        if num_folds > self.degree_bound.trailing_zeros() as usize + 1 {
            return Err(anyhow!("incorrect proof size"));
        }
        if commitment.len() != num_folds {
            return Err(anyhow!("wrong number of folding rounds"));
        }

        let mut index = self.index;
        let mut pos = self.folds[0].0.leaf().to_vec();
        let mut step = F::ROOT_OF_UNITY_INV.pow_u64(1u64 << (F::S - k));

        for round in 0..num_folds {
            let root_hash = commitment.roots()[round];

            let (left, right) = &self.folds[round];
            if left.len() != k {
                return Err(anyhow!(
                    "invalid left-hand side Merkle proof height (got {}, want {})",
                    left.len(),
                    k
                ));
            }
            if right.len() != k {
                return Err(anyhow!(
                    "invalid right-hand side Merkle proof height (got {}, want {})",
                    right.len(),
                    k
                ));
            }

            left.check_leaf(pos.as_slice())?;
            left.verify(index, root_hash)?;
            right.verify((index + n / 2) % n, root_hash)?;

            let omega_inv = step.pow_small(index);
            n /= 2;
            k -= 1;
            index %= n;

            let neg = right.leaf();
            let alpha = H::challenge(*FOLD_DST, &[root_hash]);
            for i in 0..pos.len() {
                pos[i] = (pos[i] + neg[i] + alpha * omega_inv * (pos[i] - neg[i])) * F::TWO_INV;
            }
            step = step.square();
        }

        let (left, right) = self.folds.last().unwrap();
        if !left.is_constant()? || !right.is_constant()? {
            return Err(anyhow!("the final folded polynomial is not constant"));
        }

        Ok(())
    }
}

/// A FRI prover.
///
/// The struct contains the main Merkle tree built on the committed polynomial(s) and the Merkle
/// trees of all folded polynomials up to and including the one where all polynomials have been
/// folded into constant ones. Note that the final Merkle tree still has more than one leaf due to
/// the low-degree extension.
#[derive(Debug, Clone)]
pub struct Prover<F: Field256, H: Hasher<F>> {
    /// The degree bound of the committed polynomials. This is the highest degree among the
    /// committed polynomials, plus one.
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// The folded Merkle trees, one for every folding round.
    trees: Vec<Tree<F, F, H>>,
}

impl<F: Field256, H: Hasher<F>> Prover<F, H> {
    pub fn new(polynomials: Vec<Polynomial<F>>, degree_bound: usize, blowup_log2: usize) -> Self {
        assert!(degree_bound.is_power_of_two());
        assert!(polynomials
            .iter()
            .all(|polynomial| degree_bound >= polynomial.degree_bound()));
        assert!(blowup_log2 > 0);

        let n = degree_bound << blowup_log2;
        assert!(n as u64 <= 1u64 << F::S);

        let main_tree = Tree::<F, F, H>::new(
            polynomials
                .into_iter()
                .map(|polynomial| polynomial.shift_domain().lde2(n))
                .collect(),
        );
        let trees = main_tree.fold_all(degree_bound.trailing_zeros() as usize);

        Self {
            degree_bound,
            blowup_log2,
            trees,
        }
    }

    /// Returns the degree bound of the committed polynomials (always a power of 2).
    ///
    /// NOTE: the actual degree of the original polynomials is often even lower than this value
    /// because it was rounded up to the next power of 2 in order to run the FFT and FRI algorithms.
    pub fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    /// Returns the size of the extended domain, equal to `degree_bound * 2^blowup_log2`.
    pub fn extended_domain_size(&self) -> usize {
        self.degree_bound << self.blowup_log2
    }

    /// Alias for [`Self::extended_domain_size`].
    pub fn size(&self) -> usize {
        self.degree_bound << self.blowup_log2
    }

    /// Returns the Merkle root hash of the committed polynomials.
    ///
    /// This is equivalent to the first root stored in the commiment returned by [`Self::commit`].
    pub fn root_hash(&self) -> H256 {
        self.trees[0].root_hash()
    }

    /// Creates the FRI commitment for the batched polynomials.
    pub fn commit(&self) -> Commitment {
        Commitment {
            roots: self.trees.iter().map(Tree::root_hash).collect(),
        }
    }

    /// Builds a FRI [`Query`] for the value at the specified index of the evaluation domain.
    ///
    /// NOTE: `index` is relative to the *inflated* evaluation domain, so for example if you
    /// committed to 4 evaluations with a blowup factor of 8 the range for `index` is [0, 32).
    pub fn query(&self, index: usize) -> Query<F, H> {
        let mut n = self.degree_bound << self.blowup_log2;
        assert!(index < n);

        let mut i = index;
        let mut folds = vec![];
        for tree in &self.trees {
            folds.push((tree.query(i), tree.query((i + n / 2) % n)));
            n /= 2;
            i %= n;
        }

        {
            let (left, right) = folds.last().unwrap();
            assert!(left.is_constant().unwrap());
            assert!(right.is_constant().unwrap());
        }

        Query {
            degree_bound: self.degree_bound,
            blowup_log2: self.blowup_log2,
            index,
            folds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::{Keccak256Hash, Sha2Hash};
    use starkom_bluesky::Scalar as BS;
    use starkom_goldilocks::GL4;

    #[test]
    fn test_fold_dst() {
        assert_eq!(
            *FOLD_DST,
            "0x9ffd3556faeb2cae194ce95adf6b3580f590504daa0dea56966ce4ef233844af"
                .parse()
                .unwrap()
        );
    }

    fn test_prover_impl<F: Field256, H: Hasher<F>>(
        polynomials: &[Polynomial<F>],
        degree_bound: usize,
        blowup_log2: usize,
    ) {
        let prover = Prover::<F, H>::new(
            polynomials.iter().cloned().collect(),
            degree_bound,
            blowup_log2,
        );
        assert_eq!(prover.degree_bound(), degree_bound);
        let n = degree_bound << blowup_log2;
        assert_eq!(prover.extended_domain_size(), n);
        let commitment = prover.commit();
        for i in 0..n {
            let query = prover.query(i);
            assert_eq!(query.indices(), (i, (i + n / 2) % n));
            assert_eq!(query.len(), degree_bound.trailing_zeros() as usize + 1);
            assert!(query.verify(&commitment).is_ok());
        }
    }

    fn test_prover(polynomials: Vec<Vec<u64>>, degree_bound: usize) {
        let bluesky_polynomials: Vec<Polynomial<BS>> = polynomials
            .iter()
            .map(|coefficients| {
                Polynomial::with_coefficients(
                    coefficients.iter().copied().map(BS::from_const).collect(),
                )
            })
            .collect();
        let goldilocks_polynomials: Vec<Polynomial<GL4>> = polynomials
            .into_iter()
            .map(|coefficients| {
                Polynomial::with_coefficients(
                    coefficients.into_iter().map(GL4::from_const).collect(),
                )
            })
            .collect();
        test_prover_impl::<BS, Sha2Hash<BS>>(&bluesky_polynomials, degree_bound, 1);
        test_prover_impl::<GL4, Sha2Hash<GL4>>(&goldilocks_polynomials, degree_bound, 1);
        test_prover_impl::<BS, Keccak256Hash<BS>>(&bluesky_polynomials, degree_bound, 1);
        test_prover_impl::<GL4, Keccak256Hash<GL4>>(&goldilocks_polynomials, degree_bound, 1);
        test_prover_impl::<BS, Sha2Hash<BS>>(&bluesky_polynomials, degree_bound, 2);
        test_prover_impl::<GL4, Sha2Hash<GL4>>(&goldilocks_polynomials, degree_bound, 2);
        test_prover_impl::<BS, Keccak256Hash<BS>>(&bluesky_polynomials, degree_bound, 2);
        test_prover_impl::<GL4, Keccak256Hash<GL4>>(&goldilocks_polynomials, degree_bound, 2);
        test_prover_impl::<BS, Sha2Hash<BS>>(&bluesky_polynomials, degree_bound, 3);
        test_prover_impl::<GL4, Sha2Hash<GL4>>(&goldilocks_polynomials, degree_bound, 3);
        test_prover_impl::<BS, Keccak256Hash<BS>>(&bluesky_polynomials, degree_bound, 3);
        test_prover_impl::<GL4, Keccak256Hash<GL4>>(&goldilocks_polynomials, degree_bound, 3);
    }

    #[test]
    fn test_one_constant_polynomial() {
        test_prover(vec![vec![12]], 1);
        test_prover(vec![vec![34]], 1);
    }

    #[test]
    fn test_two_constant_polynomials() {
        test_prover(vec![vec![12], vec![34]], 1);
    }

    #[test]
    fn test_three_constant_polynomials() {
        test_prover(vec![vec![34], vec![56], vec![78]], 1);
    }

    #[test]
    fn test_one_polynomial_degree_one() {
        test_prover(vec![vec![12, 34]], 2);
        test_prover(vec![vec![56, 78]], 2);
    }

    #[test]
    fn test_two_polynomials_degree_one() {
        test_prover(vec![vec![12, 34], vec![56, 78]], 2);
    }

    #[test]
    fn test_three_polynomials_degree_one() {
        test_prover(vec![vec![34, 56], vec![56, 78], vec![78, 90]], 2);
    }

    #[test]
    fn test_one_polynomial_degree_three() {
        test_prover(vec![vec![12, 34, 56, 78]], 4);
        test_prover(vec![vec![42, 43, 44, 45]], 4);
    }

    #[test]
    fn test_two_polynomials_degree_three() {
        test_prover(vec![vec![12, 34, 56, 78], vec![42, 43, 44, 45]], 4);
    }

    #[test]
    fn test_three_polynomials_degree_three() {
        test_prover(
            vec![
                vec![42, 43, 44, 45],
                vec![12, 34, 56, 78],
                vec![34, 56, 78, 90],
            ],
            4,
        );
    }

    #[test]
    fn test_one_polynomial_degree_seven() {
        test_prover(vec![vec![12, 34, 56, 78, 90, 12, 34]], 8);
        test_prover(vec![vec![42, 43, 44, 45, 46, 47, 48]], 8);
    }

    #[test]
    fn test_two_polynomials_degree_seven() {
        test_prover(
            vec![
                vec![12, 34, 56, 78, 90, 12, 34],
                vec![42, 43, 44, 45, 46, 47, 48],
            ],
            8,
        );
    }

    #[test]
    fn test_three_polynomials_degree_seven() {
        test_prover(
            vec![
                vec![42, 43, 44, 45, 46, 47, 48],
                vec![12, 34, 56, 78, 90, 12, 34],
                vec![34, 56, 78, 90, 78, 56, 34],
            ],
            8,
        );
    }
}
