use crate::hash::Hasher;
use crate::merkle::{Proof as LeafProof, Tree};
use anyhow::{Result, anyhow};
use primitive_types::H256;
use sha2::Digest;
use starkom_ff::{Field, Field256, PrimeField};
use starkom_poly::Polynomial;
use std::marker::PhantomData;
use std::sync::LazyLock;

/// Domain separator tag used when deriving the Fiat-Shamir challenge for FRI folding.
static FOLD_DST: LazyLock<H256> = LazyLock::new(|| {
    let mut hasher = sha3::Sha3_256::new();
    hasher.update(b"starkom/fri/fold");
    H256::from_slice(hasher.finalize().as_slice())
});

trait FoldableTree<F: PrimeField, G: Field256 + From<F>, H: Hasher<G>>: Sized {
    /// Performs one FRI folding round, returning the new folded tree.
    fn fold(&self) -> Tree<G, G, H>;

    /// Performs `times` FRI folding and returns an array of `times+1` trees.
    ///
    /// The first element is `self` (N leaves), the second element is the tree from the first
    /// folding round (N/2 leaves), the third element is the tree from the second folding round (N/4
    /// leaves), and so on.
    fn fold_all(&self, times: usize) -> Vec<Tree<G, G, H>>;
}

/// This impl uses three field parameters: B, F, and G. B is the prime field where the evaluation
/// domain is defined, F is the field used for storing leaf values in Merkle trees, and G is the
/// ~256-bit field where Fiat-Shamir challenges (`alpha` in the [`Self::fold`] algorithm) are
/// derived.
///
/// To understand their meaning and differences, consider the three following cases:
///
///   * Working with BlueSky:
///     - B = BlueSky,
///     - F = BlueSky,
///     - G = BlueSky.
///
///   * Working with Goldilocks, first Merkle tree:
///     - B = Goldilocks,
///     - F = Goldilocks,
///     - G = Goldilocks^4.
///
///   * Working with Goldilocks, folded Merkle trees:
///     - B = Goldilocks,
///     - F = Goldilocks^4,
///     - G = Goldilocks^4.
///
/// B is always the field of the original polynomial(s) and G is always the field of the folded
/// trees, but F can be either based on where we are in the folding process.
///
/// Note that it must be possible to convert B and F to G, so we require `G: From<B> + From<F>`. In
/// particular, the conversion from B to G must be a homomorphism because it needs to retain the
/// property that the n-th power of the embedded root of unity is the embedded unit
/// (`G::from(B::ROOT_OF_UNITY).pow(n) == G::from(B::ONE)`).
impl<B: PrimeField, F: Field, G: Field256 + From<B> + From<F>, H: Hasher<G>> FoldableTree<B, G, H>
    for Tree<F, G, H>
{
    fn fold(&self) -> Tree<G, G, H> {
        let num_polys = self.num_polys();
        let n = self.num_leaves();
        assert!(n.is_power_of_two());

        let alpha = H::challenge(&[*FOLD_DST, self.root_hash()]);

        let k = n.trailing_zeros() as usize;
        let omega_inv = B::ROOT_OF_UNITY_INV.pow_u64(1u64 << (B::S - k));
        let two_inv = G::from(B::TWO_INV);

        let m = n / 2;
        let mut omega_inv_i = B::ONE;

        let mut leaves = vec![vec![G::ZERO; m]; num_polys];
        for i in 0..m {
            for j in 0..num_polys {
                let pos = self.leaf_value(j, i);
                let neg = self.leaf_value(j, i + m);
                leaves[j][i] = (G::from(pos + neg)
                    + alpha * G::from(omega_inv_i) * G::from(pos - neg))
                    * two_inv;
            }
            omega_inv_i *= omega_inv;
        }

        Tree::new(leaves)
    }

    fn fold_all(&self, times: usize) -> Vec<Tree<G, G, H>> {
        let mut trees = Vec::with_capacity(times + 1);
        let mut tree = <Tree<F, G, H> as FoldableTree<B, G, H>>::fold(&self);
        for _ in 1..times {
            let folded = <Tree<G, G, H> as FoldableTree<B, G, H>>::fold(&tree);
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
pub struct Query<F: PrimeField, G: Field256 + From<F>, H: Hasher<G>> {
    /// The degree bound of the committed polynomials (always a power of 2).
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// The index of the element we're opening (the partner index is inferred automatically).
    index: usize,
    /// Proves a pair of "partner" values at each folding round with one [`LeafProof`] pair for
    /// every round. The pair at `folds[0]` proves the opened values.
    folds: Vec<(LeafProof<G, G, H>, LeafProof<G, G, H>)>,
    _data: PhantomData<F>,
}

impl<F: PrimeField, G: Field256 + From<F>, H: Hasher<G>> Query<F, G, H> {
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
    /// element by [`Scalar::MULTIPLICATIVE_GENERATOR`], consistently with `shift_domain`.
    pub fn x(&self) -> F {
        Polynomial::<F>::coset_element2(self.index, self.degree_bound << self.blowup_log2)
    }

    /// Returns the opened evaluations, one for every committed polynomial.
    ///
    /// The first component of the returned tuple contains the evaluations at the first index
    /// returned by [`Self::indices`], while the second component contains those at the second
    /// index.
    pub fn values(&self) -> (&[F], &[F]) {
        // TODO: (self.folds[0].0.leaf(), self.folds[0].1.leaf())
        todo!()
    }

    /// Returns the number of folding rounds.
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

        let k = n.trailing_zeros() as usize;

        let folds = self.folds.as_slice();

        let num_folds = folds.len();
        if num_folds > self.degree_bound.trailing_zeros() as usize + 1 {
            return Err(anyhow!("invalid proof size"));
        }
        if commitment.len() != num_folds {
            return Err(anyhow!("wrong number of folding rounds"));
        }

        let mut index = self.index;
        let mut pos = self.folds[0].0.leaf().to_vec();
        let mut step = F::ROOT_OF_UNITY_INV.pow_u64(1u64 << (F::S - k));
        let two_inv = G::from(F::TWO_INV);

        for round in 0..num_folds {
            let (left, right) = &folds[round];
            let root_hash = commitment.roots()[round];
            let alpha = H::challenge(&[*FOLD_DST, root_hash]);
            let neg: Vec<G> = right.leaf().iter().copied().map(G::from).collect();

            if 1usize << left.len() != n {
                return Err(anyhow!(
                    "invalid left-hand side Merkle proof height (got {}, want {})",
                    left.len(),
                    n.trailing_zeros()
                ));
            }
            if 1usize << right.len() != n {
                return Err(anyhow!(
                    "invalid right-hand side Merkle proof height (got {}, want {})",
                    right.len(),
                    n.trailing_zeros()
                ));
            }

            left.check_leaf(pos.as_slice())?;
            left.verify(index, root_hash)?;
            right.verify((index + n / 2) % n, root_hash)?;

            let omega_inv_i = step.pow_small(index);
            n /= 2;
            index %= n;

            for i in 0..pos.len() {
                pos[i] = (G::from(pos[i] + neg[i])
                    + alpha * G::from(omega_inv_i) * (pos[i] - neg[i]))
                    * two_inv;
            }
            step = step.square();
        }

        let (left, right) = folds.last().unwrap();
        if !left.is_constant()? || !right.is_constant()? {
            return Err(anyhow!("final folded polynomial is not constant"));
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
pub struct Prover<F: PrimeField, G: Field256 + From<F>, H: Hasher<G>> {
    /// The degree bound of the committed polynomials. This is the highest degree among the
    /// committed polynomials, plus one.
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// The initial Merkle tree containing the original polynomial evaluations.
    main_tree: Tree<F, G, H>,
    /// The folded Merkle trees, one for every folding round.
    folded_trees: Vec<Tree<G, G, H>>,
}

impl<F: PrimeField, G: Field256 + From<F>, H: Hasher<G>> Prover<F, G, H> {
    pub fn new(polynomials: Vec<Polynomial<F>>, degree_bound: usize, blowup_log2: usize) -> Self {
        assert!(degree_bound.is_power_of_two());
        assert!(
            polynomials
                .iter()
                .all(|polynomial| degree_bound >= polynomial.degree_bound())
        );
        assert!(blowup_log2 > 0);

        let n = degree_bound << blowup_log2;
        assert!(n as u64 <= 1u64 << F::S);

        let main_tree = Tree::<F, G, H>::new(
            polynomials
                .into_iter()
                .map(|polynomial| polynomial.shift_domain().lde2(n))
                .collect(),
        );
        let folded_trees = <Tree<F, G, H> as FoldableTree<F, G, H>>::fold_all(
            &main_tree,
            degree_bound.trailing_zeros() as usize,
        );

        Self {
            degree_bound,
            blowup_log2,
            main_tree,
            folded_trees,
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
        self.main_tree.root_hash()
    }

    /// Creates the FRI commitment for the batched polynomials.
    pub fn commit(&self) -> Commitment {
        Commitment {
            roots: std::iter::once(self.main_tree.root_hash())
                .chain(self.folded_trees.iter().map(Tree::root_hash))
                .collect(),
        }
    }

    // TODO
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO
}
