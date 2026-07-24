use crate::fri;
use crate::hash::HashBackend;
use crate::merkle::{Proof as LeafProof, Tree};
use crate::utils;
use anyhow::{Result, anyhow};
use primitive_types::{H256, U256};
use starkom_bluesky::Scalar;
use starkom_ff::{Field, Field256, PrimeField};
use starkom_poly;
use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;
use std::sync::LazyLock;

/// Re-exported type alias for polynomials over BlueSky. The current implementation only works on
/// that field.
pub type Polynomial = starkom_poly::Polynomial<Scalar>;

/// Target security level in bits.
pub const LAMBDA: usize = 128;

/// Domain separator tag used by [`Committer::transcript_hash`].
static TRANSCRIPT_DST: LazyLock<Scalar> =
    LazyLock::new(|| utils::hash_to_scalar(b"starkom/deep/transcript"));

/// Domain separator tag for the Fiat-Shamir challenge used to derive query indices.
static QUERY_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/deep/query"));

/// Domain separator tag for the Fiat-Shamir challenge used to build the random linear combination.
static RLC_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/deep/rlc"));

/// Returns the number of FRI queries required to achieve 128-bit security using a blowup factor of
/// `2^blowup_log2`.
fn num_queries(blowup_log2: usize) -> usize {
    LAMBDA.div_ceil(blowup_log2)
}

/// Computes a random linear combination of a list of values.
///
/// `alpha` is a Fiat-Shamir challenge of some sort.
fn rlc(values: &[Scalar], alpha: Scalar) -> Scalar {
    let mut rlc = Scalar::ZERO;
    let mut pow = Scalar::ONE;
    for &value in values {
        rlc += value * pow;
        pow *= alpha;
    }
    rlc
}

/// A batched DEEP-FRI polynomial commitment (see [`Committer`] for details).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment<H: HashBackend<Scalar>> {
    /// The root hashes of the Merkle trees where the evaluations of all batched polynomials are
    /// stored. There is one root hash per polynomial batch.
    tree_roots: Vec<H256>,
    /// The underlying FRI commitment.
    inner: fri::Commitment,
    _data: PhantomData<H>,
}

impl<H: HashBackend<Scalar>> Commitment<H> {
    /// Returns the root hashes of the Merkle trees where all batched polynomials are stored.
    pub fn tree_roots(&self) -> &[H256] {
        self.tree_roots.as_slice()
    }

    /// Hashes the first `batch_count` [tree roots](`Self::tree_roots`).
    ///
    /// The returned hash is cryptographically bound to the full transcript up to the given
    /// polynomial batch, and can be used by a verifier to recover Fiat-Shamir challenges.
    ///
    /// REQUIRES: `batch_count` must be strictly greater than 0 and less than or equal to the number
    /// of tree roots.
    ///
    /// The hashes returned by this method are compatible with [`Committer::transcript_hash`], which
    /// can be used on the prover side.
    pub fn transcript_hash(&self, batch_count: usize) -> H256 {
        assert!(batch_count > 0);
        assert!(batch_count <= self.tree_roots.len());
        H::hash_many_words(
            std::iter::once(H::encode_scalar256(*TRANSCRIPT_DST))
                .chain(std::iter::once(H::encode_usize(batch_count)))
                .chain(self.tree_roots[..batch_count].iter().copied()),
        )
        .unwrap()
    }

    /// Returns the FRI query indices derived via Fiat-Shamir from the full commitment transcript
    /// (all polynomial and FRI Merkle root hashes).
    fn get_query_indices(&self, degree_bound: usize, blowup_log2: usize) -> Vec<usize> {
        let n = U256::from((degree_bound << blowup_log2) as u64);
        let k = num_queries(blowup_log2);
        let mut indices = Vec::with_capacity(k);
        for i in 0..k {
            let hash = H::challenge(
                *QUERY_DST,
                std::iter::once(H::encode_usize(self.tree_roots.len()))
                    .chain(self.tree_roots.iter().copied())
                    .chain(std::iter::once(H::encode_usize(self.inner.len())))
                    .chain(self.inner.roots().iter().copied())
                    .chain(std::iter::once(H::encode_usize(i))),
            )
            .unwrap();
            let index = hash.to_u256() % n;
            indices.push(index.as_u64() as usize);
        }
        indices
    }
}

/// Collects batches of polynomials and allows building a DEEP-FRI prover for them.
///
/// This works by building Merkle trees on the batched polynomials, one tree per batch, and
/// eventually handing everything over to a newly constructed [`Prover`] (see the [`Self::commit`]
/// method).
///
/// This two-stage Committer-Prover architecture allows getting Merkle roots for the proven
/// polynomials before running the FRI folding argument and even before batching all polynomials, so
/// that Fiat-Shamir challenges can be derived before any quotients are built.
#[derive(Debug, Clone)]
pub struct Committer<H: HashBackend<Scalar>> {
    /// The proven degree bound. The degree of all batched polynomials must be strictly less than
    /// this value.
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// All polynomials batched so far.
    polynomials: Vec<Polynomial>,
    /// The Merkle trees built so far.
    ///
    /// The sum of all `num_polys` of all trees must match the number of `polynomials`.
    trees: Vec<Tree<H>>,
}

impl<H: HashBackend<Scalar>> Committer<H> {
    /// Constructs a [`Committer`] with the given degree bound, blowup factor, and first batch of
    /// polynomials.
    ///
    /// We require specifying the first batch because our DEEP-FRI protocol requires at least one
    /// committed polynomial to work.
    ///
    /// `degree_bound` must be a power of 2 less than or equal to 2^[`Scalar::S`], and `blowup_log2`
    /// must not be zero.
    pub fn new(degree_bound: usize, blowup_log2: usize, polynomials: Vec<Polynomial>) -> Self {
        assert!(degree_bound.is_power_of_two());
        assert!(blowup_log2 > 0);
        assert!(!polynomials.is_empty());
        let mut committer = Self {
            degree_bound,
            blowup_log2,
            polynomials: vec![],
            trees: vec![],
        };
        committer.add_batch(polynomials);
        committer
    }

    /// Returns the proven degree bound.
    pub fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    /// Returns the size of the extended evaluation domain.
    pub fn extended_domain_size(&self) -> usize {
        self.degree_bound << self.blowup_log2
    }

    /// Returns the number of Merkle trees constructed so far, corresponding to the number of
    /// polynomial batches.
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Returns the root hash of the i-th Merkle tree. `index` must be less than
    /// [`Self::num_trees()`].
    ///
    /// This value can be used to derive Fiat-Shamir challenges.
    pub fn root_hash(&self, index: usize) -> H256 {
        self.trees[index].root_hash()
    }

    /// Hashes the Merkle roots of all polynomial batches accumulated so far.
    ///
    /// The returned hash is cryptographically bound to the full transcript so far, and can be used
    /// by the caller to generate Fiat-Shamir challenges.
    ///
    /// The hashes returned by this method are compatible with [`Commitment::transcript_hash`],
    /// which can be used on the verifier side.
    pub fn transcript_hash(&self) -> H256 {
        H::hash_many_words(
            std::iter::once(H::encode_scalar256(*TRANSCRIPT_DST))
                .chain(std::iter::once(H::encode_usize(self.trees.len())))
                .chain(self.trees.iter().map(Tree::root_hash)),
        )
        .unwrap()
    }

    /// Adds a batch of polynomials, returning the index of the newly created batch.
    ///
    /// The returned index can be used with the [`Self::root_hash`] method to get the Merkle root
    /// for the batch.
    ///
    /// REQUIRES: the degree of all specified polynomials must be strictly less than
    /// [`Self::degree_bound()`].
    pub fn add_batch(&mut self, polynomials: Vec<Polynomial>) -> usize {
        assert!(!polynomials.is_empty());

        let degree_bound = polynomials
            .iter()
            .map(|polynomial| polynomial.degree_bound())
            .max()
            .unwrap()
            .next_power_of_two();
        assert!(degree_bound <= self.degree_bound);
        let n = self.degree_bound << self.blowup_log2;
        assert!(n.trailing_zeros() as usize <= Scalar::S);

        let evaluations = polynomials
            .iter()
            .map(|polynomial| polynomial.clone().shift_domain().lde2(n))
            .collect::<Vec<Vec<Scalar>>>();

        let index = self.trees.len();

        self.polynomials.extend(polynomials);
        self.trees.push(Tree::<H>::new(evaluations));

        index
    }

    /// Consumes the [`Committer`], calculates all DEEP quotients, and returns a polynomial
    /// [`Commitment`] and a DEEP-FRI [`Prover`].
    ///
    /// `points` is the set of points to open in the [`Prover`]. The contained scalars are
    /// (off-domain) X-coordinates; the corresponding Y-coordinates will be computed automatically
    /// for every batched polynomial.
    pub fn commit(self, points: BTreeSet<Scalar>) -> (Commitment<H>, Prover<H>) {
        {
            let n = self.degree_bound << self.blowup_log2;
            let g = Scalar::MULTIPLICATIVE_GENERATOR.pow_small(n);
            for &z in &points {
                // All opened points must lie outside the evaluation domain.
                assert_ne!(z.pow_small(n), g);
            }
        }

        let alpha = H::challenge(
            *RLC_DST,
            std::iter::once(H::encode_usize(self.trees.len()))
                .chain(self.trees.iter().map(Tree::root_hash))
                .chain(std::iter::once(H::encode_usize(self.polynomials.len())))
                .chain(std::iter::once(H::encode_usize(points.len())))
                .chain(
                    points
                        .iter()
                        .flat_map(|&z| {
                            std::iter::once(z).chain(
                                self.polynomials
                                    .iter()
                                    .map(|polynomial| polynomial.evaluate(z))
                                    .collect::<Vec<Scalar>>()
                                    .into_iter(),
                            )
                        })
                        .map(H::encode_scalar256),
                ),
        )
        .unwrap();

        let points: BTreeMap<Scalar, Vec<Scalar>> = points
            .iter()
            .map(|&z| {
                (
                    z,
                    self.polynomials
                        .iter()
                        .map(|polynomial| polynomial.evaluate(z))
                        .collect(),
                )
            })
            .collect();

        let combined = {
            let mut combined = Polynomial::default();
            let mut pow = Scalar::ONE;
            for polynomial in &self.polynomials {
                combined += polynomial.clone() * pow;
                pow *= alpha;
            }
            combined
        };

        let quotients = points
            .iter()
            .map(|(&z, values)| {
                let value = rlc(values.as_slice(), alpha);
                let (quotient, remainder) = (combined.clone() - value).horner(z);
                assert_eq!(remainder, Scalar::ZERO);
                quotient
            })
            .collect();

        let inner_prover = fri::Prover::<H>::new(quotients, self.degree_bound, self.blowup_log2);

        let commitment = Commitment {
            tree_roots: self.trees.iter().map(|tree| tree.root_hash()).collect(),
            inner: inner_prover.commit(),
            _data: Default::default(),
        };
        let prover = Prover {
            degree_bound: self.degree_bound,
            blowup_log2: self.blowup_log2,
            trees: self.trees,
            points,
            inner_prover,
        };
        (commitment, prover)
    }
}

/// A DEEP-FRI proof.
#[derive(Debug, Clone)]
pub struct Proof<H: HashBackend<Scalar>> {
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
    /// Merkle proofs for the points at the query positions, relative to the raw Merkle trees (not
    /// the FRI folds). The outer array has one entry for every FRI query
    /// (`openings.len() == queries.len()`), and the inner arrays contain one proof for every Merkle
    /// tree.
    openings: Vec<Vec<LeafProof<H>>>,
    /// FRI queries on the DEEP quotients. The number of queries is calculated by [`num_queries`]
    /// above and is tuned so as to achieve 128-bit security.
    queries: Vec<fri::Query<H>>,
}

impl<H: HashBackend<Scalar>> Proof<H> {
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
    pub fn verify(&self, commitment: &Commitment<H>) -> Result<()> {
        let indices = commitment.get_query_indices(self.degree_bound, self.blowup_log2);
        if self.openings.len() != indices.len() {
            return Err(anyhow!(
                "incorrect number of openings (got {}, want {})",
                self.openings.len(),
                indices.len()
            ));
        }
        if self.queries.len() != indices.len() {
            return Err(anyhow!(
                "incorrect number of queries (got {}, want {})",
                self.queries.len(),
                indices.len()
            ));
        }

        let alpha = H::challenge(
            *RLC_DST,
            std::iter::once(H::encode_usize(commitment.tree_roots().len()))
                .chain(commitment.tree_roots().iter().copied())
                .chain(std::iter::once(H::encode_usize(self.num_polys)))
                .chain(std::iter::once(H::encode_usize(self.points.len())))
                .chain(self.points.iter().flat_map(|(z, values)| {
                    std::iter::once(z)
                        .chain(values.iter())
                        .map(|&value| H::encode_scalar256(value))
                })),
        )
        .unwrap();

        for ((query, openings), &expected_index) in
            (self.queries.iter().zip(self.openings.iter())).zip(indices.iter())
        {
            let (index, _) = query.indices();
            if index != expected_index {
                return Err(anyhow!(
                    "wrong query index (got {index}, want {expected_index})",
                ));
            }

            if openings.len() != commitment.tree_roots().len() {
                return Err(anyhow!(
                    "incorrect number of openings for index {index} (got {}, want {})",
                    openings.len(),
                    commitment.tree_roots().len()
                ));
            }
            for (&root_hash, opening) in commitment.tree_roots().iter().zip(openings.iter()) {
                if 1usize << opening.len() != self.extended_domain_size() {
                    return Err(anyhow!("invalid opening for index {index}"));
                }
                opening.verify(index, root_hash)?;
            }

            if 1usize << (query.len() - 1) != self.degree_bound {
                return Err(anyhow!("invalid low-degree proof for index {index}"));
            }
            query.verify(&commitment.inner)?;

            let combined = rlc(
                openings
                    .iter()
                    .flat_map(|proof| proof.leaf().iter().cloned())
                    .collect::<Vec<Scalar>>()
                    .as_slice(),
                alpha,
            );

            let (quotients, _) = query.values();
            if quotients.len() != self.points.len() {
                return Err(anyhow!(
                    "the number of evaluation claims doesn't match the number of FRI quotients (got {}, want {})",
                    quotients.len(),
                    self.points.len()
                ));
            }

            let x = query.x();
            for ((&z, values), &quotient) in self.points.iter().zip(quotients.iter()) {
                let v = rlc(values.as_slice(), alpha);
                let numerator = combined - v;
                let denominator = x - z;
                if quotient * denominator != numerator {
                    return Err(anyhow!("algebraic check failed at query index {index}"));
                }
            }
        }

        Ok(())
    }
}

/// A DEEP-FRI prover.
///
/// [`Prover`]s are constructed by [`Committer::commit()`]; see that method for details.
#[derive(Debug, Clone)]
pub struct Prover<H: HashBackend<Scalar>> {
    /// The degree bound to prove.
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// Raw Merkle trees for the committed polynomials, one for each batch.
    trees: Vec<Tree<H>>,
    /// The opened points.
    ///
    /// The keys of the map are the (off-domain) X-coordinates of the points, while values are lists
    /// of polynomial evaluations at that point (one for every committed polynomial).
    points: BTreeMap<Scalar, Vec<Scalar>>,
    /// The underlying FRI prover for the DEEP quotients. There's one quotient for every opened
    /// point, and all quotients are batched into the same FRI folding argument.
    inner_prover: fri::Prover<H>,
}

impl<H: HashBackend<Scalar>> Prover<H> {
    /// Returns the proven degree bound.
    pub fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    /// Returns the size of the extended evaluation domain.
    pub fn extended_domain_size(&self) -> usize {
        self.degree_bound << self.blowup_log2
    }

    /// Returns the number of committed polynomials.
    pub fn num_polys(&self) -> usize {
        self.trees.iter().map(|tree| tree.num_polys()).sum()
    }

    /// Returns the number of Merkle trees, corresponding to the number of polynomial batches.
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Returns the root hash of the i-th Merkle tree. `index` must be less than
    /// [`Self::num_trees()`].
    ///
    /// This value can be used to derive Fiat-Shamir challenges.
    pub fn root_hash(&self, index: usize) -> H256 {
        self.trees[index].root_hash()
    }

    /// Returns a reference to the opened points. Keys are (off-domain) X-coordinates, values are
    /// the corresponding evaluations (one for every committed polynomial).
    pub fn points(&self) -> &BTreeMap<Scalar, Vec<Scalar>> {
        &self.points
    }

    /// Makes a DEEP-FRI proof opening the committed polynomials at the points specified at
    /// commitment time (see [`Committer::commit()`]).
    pub fn prove(&self, commitment: &Commitment<H>) -> Proof<H> {
        let indices = commitment.get_query_indices(self.degree_bound, self.blowup_log2);
        let openings = indices
            .iter()
            .map(|&index| self.trees.iter().map(|tree| tree.query(index)).collect())
            .collect();
        let queries = indices
            .iter()
            .map(|&index| self.inner_prover.query(index))
            .collect();
        Proof {
            degree_bound: self.degree_bound,
            blowup_log2: self.blowup_log2,
            num_polys: self.num_polys(),
            points: self.points.clone(),
            openings,
            queries,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash;
    use starkom_bluesky::from_const;

    type Sha2Hash = hash::Sha2Hash<Scalar>;
    type Poseidon1Hash = hash::Poseidon1Hash<Scalar>;
    type Poseidon2Hash = hash::Poseidon2Hash<Scalar>;

    fn test_prover_impl<H: HashBackend<Scalar>>(
        mut polynomial_batches: Vec<Vec<Polynomial>>,
        points: &[u64],
        degree_bound: usize,
        blowup_log2: usize,
    ) {
        let num_batches = polynomial_batches.len();
        let num_polys = polynomial_batches.iter().map(|batch| batch.len()).sum();
        let points = BTreeMap::from_iter(points.iter().cloned().map(|z| {
            (
                Scalar::from(z),
                polynomial_batches
                    .iter()
                    .flatten()
                    .map(|polynomial| polynomial.evaluate(z.into()))
                    .collect::<Vec<Scalar>>(),
            )
        }));
        let first_batch = polynomial_batches.remove(0);
        let mut committer = Committer::<H>::new(degree_bound, blowup_log2, first_batch);
        let transcript_hashes: Vec<H256> = std::iter::once(committer.transcript_hash())
            .chain(polynomial_batches.into_iter().map(|batch| {
                committer.add_batch(batch);
                committer.transcript_hash()
            }))
            .collect();
        let (commitment, prover) = committer.commit(points.iter().map(|(&z, _)| z).collect());
        assert_eq!(
            (0..num_batches)
                .map(|i| commitment.transcript_hash(i + 1))
                .collect::<Vec<H256>>(),
            transcript_hashes
        );
        assert_eq!(prover.degree_bound(), degree_bound);
        assert_eq!(prover.extended_domain_size(), degree_bound << blowup_log2);
        assert_eq!(prover.num_polys(), num_polys);
        assert_eq!(prover.num_trees(), num_batches);
        assert_eq!(*prover.points(), points);
        let proof = prover.prove(&commitment);
        assert_eq!(proof.degree_bound(), degree_bound);
        assert_eq!(proof.blowup_log2(), blowup_log2);
        assert_eq!(proof.extended_domain_size(), degree_bound << blowup_log2);
        assert_eq!(proof.num_polys(), num_polys);
        assert!(proof.verify(&commitment).is_ok());
        assert_eq!(*proof.points(), points);
    }

    fn test_prover(polynomial_batches: Vec<Vec<Polynomial>>, points: &[u64], degree_bound: usize) {
        test_prover_impl::<Sha2Hash>(polynomial_batches.clone(), points, degree_bound, 1);
        test_prover_impl::<Poseidon1Hash>(polynomial_batches.clone(), points, degree_bound, 1);
        test_prover_impl::<Poseidon2Hash>(polynomial_batches.clone(), points, degree_bound, 1);
        test_prover_impl::<Sha2Hash>(polynomial_batches.clone(), points, degree_bound, 2);
        test_prover_impl::<Poseidon1Hash>(polynomial_batches.clone(), points, degree_bound, 2);
        test_prover_impl::<Poseidon2Hash>(polynomial_batches.clone(), points, degree_bound, 2);
        test_prover_impl::<Sha2Hash>(polynomial_batches.clone(), points, degree_bound, 3);
        test_prover_impl::<Poseidon1Hash>(polynomial_batches.clone(), points, degree_bound, 3);
        test_prover_impl::<Poseidon2Hash>(polynomial_batches, points, degree_bound, 3);
    }

    #[test]
    fn test_one_constant_polynomial_one_point_1() {
        test_prover(
            vec![vec![Polynomial::with_coefficients(vec![from_const(12)])]],
            &[123],
            1,
        );
    }

    #[test]
    fn test_one_constant_polynomial_one_point_2() {
        test_prover(
            vec![vec![Polynomial::with_coefficients(vec![from_const(12)])]],
            &[321],
            1,
        );
    }

    #[test]
    fn test_one_constant_polynomial_one_point_3() {
        test_prover(
            vec![vec![Polynomial::with_coefficients(vec![from_const(34)])]],
            &[123],
            1,
        );
    }

    #[test]
    fn test_one_constant_polynomial_two_points() {
        test_prover(
            vec![vec![Polynomial::with_coefficients(vec![from_const(12)])]],
            &[123, 456],
            1,
        );
    }

    #[test]
    fn test_one_constant_polynomial_three_points() {
        test_prover(
            vec![vec![Polynomial::with_coefficients(vec![from_const(12)])]],
            &[789, 456, 123],
            1,
        );
    }

    #[test]
    fn test_one_polynomial_degree_one_one_point_1() {
        test_prover(
            vec![vec![Polynomial::with_coefficients(vec![
                from_const(12),
                from_const(34),
            ])]],
            &[123],
            2,
        );
    }

    #[test]
    fn test_one_polynomial_degree_one_one_point_2() {
        test_prover(
            vec![vec![Polynomial::with_coefficients(vec![
                from_const(12),
                from_const(34),
            ])]],
            &[321],
            2,
        );
    }

    #[test]
    fn test_one_polynomial_degree_one_one_point_3() {
        test_prover(
            vec![vec![Polynomial::with_coefficients(vec![
                from_const(34),
                from_const(56),
            ])]],
            &[123],
            2,
        );
    }

    #[test]
    fn test_one_polynomial_degree_one_two_points() {
        test_prover(
            vec![vec![Polynomial::with_coefficients(vec![
                from_const(12),
                from_const(34),
            ])]],
            &[123, 456],
            2,
        );
    }

    #[test]
    fn test_one_polynomial_degree_one_three_points() {
        test_prover(
            vec![vec![Polynomial::with_coefficients(vec![
                from_const(12),
                from_const(34),
            ])]],
            &[789, 456, 123],
            2,
        );
    }

    #[test]
    fn test_two_polynomials_degree_three_one_point_1() {
        test_prover(
            vec![vec![
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
            ]],
            &[123],
            4,
        );
    }

    #[test]
    fn test_two_polynomials_degree_three_one_point_2() {
        test_prover(
            vec![vec![
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
            ]],
            &[321],
            4,
        );
    }

    #[test]
    fn test_two_polynomials_degree_three_one_point_3() {
        test_prover(
            vec![vec![
                Polynomial::with_coefficients(vec![
                    from_const(45),
                    from_const(44),
                    from_const(43),
                    from_const(42),
                ]),
                Polynomial::with_coefficients(vec![
                    from_const(78),
                    from_const(56),
                    from_const(34),
                    from_const(12),
                ]),
            ]],
            &[123],
            4,
        );
    }

    #[test]
    fn test_two_polynomials_degree_three_two_points() {
        test_prover(
            vec![vec![
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
            ]],
            &[123, 456],
            4,
        );
    }

    #[test]
    fn test_two_polynomials_degree_three_three_points() {
        test_prover(
            vec![vec![
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
            ]],
            &[789, 456, 123],
            4,
        );
    }

    #[test]
    fn test_two_batches_one_and_one() {
        test_prover(
            vec![
                vec![Polynomial::with_coefficients(vec![
                    from_const(12),
                    from_const(34),
                    from_const(56),
                    from_const(78),
                ])],
                vec![Polynomial::with_coefficients(vec![
                    from_const(42),
                    from_const(43),
                    from_const(44),
                    from_const(45),
                ])],
            ],
            &[123, 456],
            4,
        );
    }

    #[test]
    fn test_two_batches_two_and_one() {
        test_prover(
            vec![
                vec![
                    Polynomial::with_coefficients(vec![
                        from_const(12),
                        from_const(34),
                        from_const(56),
                        from_const(78),
                    ]),
                    Polynomial::with_coefficients(vec![
                        from_const(90),
                        from_const(78),
                        from_const(56),
                        from_const(34),
                    ]),
                ],
                vec![Polynomial::with_coefficients(vec![
                    from_const(42),
                    from_const(43),
                    from_const(44),
                    from_const(45),
                ])],
            ],
            &[456, 789],
            4,
        );
    }

    #[test]
    fn test_two_batches_one_and_two() {
        test_prover(
            vec![
                vec![Polynomial::with_coefficients(vec![
                    from_const(90),
                    from_const(78),
                    from_const(56),
                    from_const(34),
                ])],
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
            ],
            &[456, 789],
            4,
        );
    }
}
