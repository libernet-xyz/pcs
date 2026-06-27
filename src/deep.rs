use crate::fri::{self, LeafProof, Tree};
use crate::hash::Hash;
use crate::utils;
use anyhow::{Result, anyhow};
use primitive_types::U256;
use starkom_bluesky::Scalar;
use starkom_ff::{Field, Field256, PrimeField};
use starkom_poly;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

/// Re-exported type alias for polynomials over BlueSky. The current implementation only works on
/// that field.
pub type Polynomial = starkom_poly::Polynomial<Scalar>;

/// Target security level in bits.
pub const LAMBDA: usize = 128;

/// Domain separator tag for the Fiat-Shamir challenge used to derive query indices.
static QUERY_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/pcs/query"));

/// Domain separator tag for the Fiat-Shamir challenge used to build the random linear combination.
static RLC_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/pcs/rlc"));

/// Domain separator tag for the Fiat-Shamir challenge used to batch DEEP quotients into one.
static BATCH_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/pcs/batch"));

/// Returns the number of STIR queries required to achieve 128-bit security using a blowup factor of
/// `2^blowup_log2` when opening `num_points` evaluation points.
fn num_queries(blowup_log2: usize, num_points: usize) -> usize {
    let extra = num_points.next_power_of_two().trailing_zeros() as usize;
    (LAMBDA + extra).div_ceil(blowup_log2)
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

/// A batched DEEP-STIR polynomial commitment (see `Committer` for details).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment {
    /// The root hashes of the Merkle trees where the evaluations of all batched polynomials are
    /// stored. There is one root hash per polynomial batch.
    tree_roots: Vec<Scalar>,
    /// The underlying STIR commitment.
    inner: fri::Commitment,
}

impl Commitment {
    /// Returns the root hashes of the Merkle trees where all batched polynomials are stored.
    pub fn tree_roots(&self) -> &[Scalar] {
        self.tree_roots.as_slice()
    }

    /// Returns the STIR query indices derived via Fiat-Shamir from the full commitment transcript
    /// (all polynomial and STIR Merkle root hashes).
    fn get_query_indices<H: Hash<Scalar>>(
        &self,
        degree_bound: usize,
        blowup_log2: usize,
        num_points: usize,
    ) -> Vec<usize> {
        let n = U256::from((degree_bound << blowup_log2) as u64);
        let k = num_queries(blowup_log2, num_points);
        let mut indices = Vec::with_capacity(k);
        for i in 0..k {
            let hash = H::hash_many(
                std::iter::once(*QUERY_DST)
                    .chain(std::iter::once(Scalar::from(self.tree_roots.len() as u64)))
                    .chain(self.tree_roots.iter().cloned())
                    .chain(std::iter::once(Scalar::from(self.inner.len() as u64)))
                    .chain(self.inner.roots().iter().cloned())
                    .chain(std::iter::once(Scalar::from(i as u64)))
                    .collect::<Vec<Scalar>>()
                    .as_slice(),
            );
            let index = hash.to_u256() % n;
            indices.push(index.as_u64() as usize);
        }
        indices
    }
}

/// Collects batches of polynomials and allows building a DEEP-STIR prover for them.
///
/// This works by building Merkle trees on the batched polynomials, one tree per batch, and
/// eventually handing everything over to a newly constructed `Prover` (see the `commit` method).
///
/// This two-stage Committer-Prover architecture allows getting Merkle roots for the proven
/// polynomials before running the STIR folding argument and even before batching all polynomials,
/// so that Fiat-Shamir challenges can be derived before any quotients are built.
#[derive(Debug, Clone)]
pub struct Committer<H: Hash<Scalar>> {
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

impl<H: Hash<Scalar>> Committer<H> {
    /// Constructs a `Committer` with the given degree bound, blowup factor, and first batch of
    /// polynomials.
    ///
    /// We require specifying the first batch because our DEEP-STIR protocol requires at least one
    /// committed polynomial to work.
    pub fn new(degree_bound: usize, blowup_log2: usize, polynomials: Vec<Polynomial>) -> Self {
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

    /// Returns the i-th Merkle tree. `index` must be less than `num_trees()`.
    pub fn tree(&self, index: usize) -> &Tree<H> {
        &self.trees[index]
    }

    /// Returns the root hash of the i-th Merkle tree. `index` must be less than `num_trees()`.
    ///
    /// This value can be used to derive Fiat-Shamir challenges.
    pub fn root_hash(&self, index: usize) -> Scalar {
        self.trees[index].root_hash()
    }

    /// Adds a batch of polynomials, returning the index of the newly created batch.
    ///
    /// The returned index can be used with the `tree` and `root_hash` methods to get the Merkle
    /// tree and root hash for the batch, respectively.
    ///
    /// REQUIRES: the degree of all specified polynomials must be strictly less than
    /// `degree_bound()`.
    pub fn add_batch(&mut self, polynomials: Vec<Polynomial>) -> usize {
        assert!(!polynomials.is_empty());
        let k = polynomials.len();

        let degree_bound = polynomials
            .iter()
            .map(|polynomial| polynomial.degree_bound())
            .max()
            .unwrap()
            .next_power_of_two();
        assert!(degree_bound <= self.degree_bound);
        let n = self.degree_bound << self.blowup_log2;
        assert!(n.trailing_zeros() as usize <= Scalar::S);

        let leaves = {
            let evaluations = polynomials
                .iter()
                .map(|polynomial| polynomial.clone().shift_domain().lde2(n))
                .collect::<Vec<Vec<Scalar>>>();
            let mut leaves: Vec<Vec<Scalar>> = vec![vec![Scalar::ZERO; k]; n];
            for i in 0..n {
                for j in 0..k {
                    leaves[i][j] = evaluations[j][i];
                }
            }
            leaves
        };

        let index = self.trees.len();

        self.polynomials.extend(polynomials);
        self.trees.push(Tree::<H>::from_leaves(leaves));

        index
    }

    /// Consumes the `Committer`, calculates all DEEP quotients, and returns a polynomial
    /// `Commitment` and a DEEP-STIR `Prover`.
    ///
    /// `points` is the set of points to open in the `Prover`. The contained scalars are
    /// (off-domain) X-coordinates; the corresponding Y-coordinates will be computed automatically
    /// for every batched polynomial.
    pub fn commit(self, points: BTreeSet<Scalar>) -> (Commitment, Prover<H>) {
        {
            let n = self.degree_bound << self.blowup_log2;
            let g = Scalar::MULTIPLICATIVE_GENERATOR.pow_small(n);
            for &z in &points {
                // All opened points must lie outside the evaluation domain.
                assert_ne!(z.pow_small(n), g);
            }
        }

        let alpha = H::hash_many(
            std::iter::once(*RLC_DST)
                .chain(std::iter::once(Scalar::from(self.trees.len() as u64)))
                .chain(self.trees.iter().map(|tree| tree.root_hash()))
                .chain(std::iter::once(Scalar::from(points.len() as u64)))
                .chain(points.iter().cloned())
                .collect::<Vec<Scalar>>()
                .as_slice(),
        );

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

        let beta = H::hash_two(*BATCH_DST, alpha, Scalar::ZERO);

        let batched_quotient = {
            let mut batched = Polynomial::default();
            let mut pow = Scalar::ONE;
            for (&z, values) in &points {
                let value = rlc(values.as_slice(), alpha);
                let (quotient, remainder) = (combined.clone() - value).horner(z);
                assert_eq!(remainder, Scalar::ZERO);
                batched += quotient * pow;
                pow *= beta;
            }
            batched
        };

        let inner_prover =
            fri::Prover::<H>::new(vec![batched_quotient], self.degree_bound, self.blowup_log2);

        let commitment = Commitment {
            tree_roots: self.trees.iter().map(|tree| tree.root_hash()).collect(),
            inner: inner_prover.commit(),
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

/// A DEEP-STIR proof.
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
    /// Merkle proofs of the opened points, relative to the raw Merkle trees (not the STIR folds).
    /// The outer array has one entry for every STIR query (`openings.len() == queries.len()`), and
    /// the inner arrays contain one proof for every Merkle tree.
    openings: Vec<Vec<LeafProof<H>>>,
    /// STIR queries on the DEEP quotients. The number of queries is calculated by `num_queries`
    /// above and is tuned so as to achieve 128-bit security.
    queries: Vec<fri::Query<H>>,
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
        let indices = commitment.get_query_indices::<H>(
            self.degree_bound,
            self.blowup_log2,
            self.points.len(),
        );
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

        let alpha = H::hash_many(
            std::iter::once(*RLC_DST)
                .chain(std::iter::once(Scalar::from(
                    commitment.tree_roots().len() as u64
                )))
                .chain(commitment.tree_roots().iter().cloned())
                .chain(std::iter::once(Scalar::from(self.points.len() as u64)))
                .chain(self.points.keys().cloned())
                .collect::<Vec<Scalar>>()
                .as_slice(),
        );
        let beta = H::hash_two(*BATCH_DST, alpha, Scalar::ZERO);

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

            let (quotient_values, _) = query.values();
            if quotient_values.len() != 1 {
                return Err(anyhow!(
                    "expected 1 batched quotient value, got {}",
                    quotient_values.len()
                ));
            }
            let q_combined = quotient_values[0];

            let x = query.x();
            let denominators: Vec<Scalar> = self.points.keys().map(|&z| x - z).collect();
            let full_product = denominators
                .iter()
                .copied()
                .fold(Scalar::ONE, |acc, d| acc * d);
            let mut rhs = Scalar::ZERO;
            let mut pow = Scalar::ONE;
            for (j, (&_z, values)) in self.points.iter().enumerate() {
                let v = rlc(values.as_slice(), alpha);
                let cofactor = {
                    let mut c = Scalar::ONE;
                    for (k, &d) in denominators.iter().enumerate() {
                        if k != j {
                            c *= d;
                        }
                    }
                    c
                };
                rhs += pow * (combined - v) * cofactor;
                pow *= beta;
            }
            if q_combined * full_product != rhs {
                return Err(anyhow!("algebraic check failed at query index {index}"));
            }
        }

        Ok(())
    }
}

/// A DEEP-STIR prover.
///
/// `Prover`s are constructed by `Committer::commit()`; see that method for details.
#[derive(Debug, Clone)]
pub struct Prover<H: Hash<Scalar>> {
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
    /// The underlying STIR prover for the DEEP quotients. There's one quotient for every opened
    /// point, and all quotients are batched into the same STIR folding argument.
    inner_prover: fri::Prover<H>,
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

    /// Returns the number of committed polynomials.
    pub fn num_polys(&self) -> usize {
        self.trees.iter().map(|tree| tree.num_polys()).sum()
    }

    /// Returns the number of Merkle trees, corresponding to the number of polynomial batches.
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Returns the i-th Merkle tree. `index` must be less than `num_trees()`.
    pub fn tree(&self, index: usize) -> &Tree<H> {
        &self.trees[index]
    }

    /// Returns the root hash of the i-th Merkle tree. `index` must be less than `num_trees()`.
    ///
    /// This value can be used to derive Fiat-Shamir challenges.
    pub fn root_hash(&self, index: usize) -> Scalar {
        self.trees[index].root_hash()
    }

    /// Returns a reference to the opened points. Keys are (off-domain) X-coordinates, values are
    /// the corresponding evaluations (one for every committed polynomial).
    pub fn points(&self) -> &BTreeMap<Scalar, Vec<Scalar>> {
        &self.points
    }

    /// Makes a DEEP-STIR proof opening the committed polynomials at the points specified at
    /// commitment time (see `Committer::commit()`).
    pub fn prove(&self, commitment: &Commitment) -> Proof<H> {
        let indices = commitment.get_query_indices::<H>(
            self.degree_bound,
            self.blowup_log2,
            self.points.len(),
        );
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

    type Sha2Hash = hash::Sha2Hash<Scalar>;
    type Poseidon2Hash = hash::Poseidon2Hash<Scalar>;

    const fn from_const(value: u64) -> Scalar {
        Scalar::from_const(value)
    }

    fn test_prover_impl<H: Hash<Scalar>>(
        polynomials: Vec<Polynomial>,
        points: &[u64],
        degree_bound: usize,
        blowup_log2: usize,
    ) {
        let num_polys = polynomials.len();
        let points = BTreeMap::from_iter(points.iter().cloned().map(|z| {
            (
                Scalar::from(z),
                polynomials
                    .iter()
                    .map(|polynomial| polynomial.evaluate(z.into()))
                    .collect::<Vec<Scalar>>(),
            )
        }));
        let committer = Committer::<H>::new(degree_bound, blowup_log2, polynomials);
        let (commitment, prover) = committer.commit(points.iter().map(|(&z, _)| z).collect());
        assert_eq!(prover.degree_bound(), degree_bound);
        assert_eq!(prover.extended_domain_size(), degree_bound << blowup_log2);
        assert_eq!(prover.num_polys(), num_polys);
        assert_eq!(prover.num_trees(), 1);
        assert_eq!(*prover.points(), points);
        let proof = prover.prove(&commitment);
        assert_eq!(proof.degree_bound(), degree_bound);
        assert_eq!(proof.blowup_log2(), blowup_log2);
        assert_eq!(proof.extended_domain_size(), degree_bound << blowup_log2);
        assert_eq!(proof.num_polys(), num_polys);
        assert!(proof.verify(&commitment).is_ok());
        assert_eq!(*proof.points(), points);
    }

    fn test_prover(polynomials: Vec<Polynomial>, points: &[u64], degree_bound: usize) {
        test_prover_impl::<Sha2Hash>(polynomials.clone(), points, degree_bound, 1);
        test_prover_impl::<Poseidon2Hash>(polynomials.clone(), points, degree_bound, 1);
        test_prover_impl::<Sha2Hash>(polynomials.clone(), points, degree_bound, 2);
        test_prover_impl::<Poseidon2Hash>(polynomials.clone(), points, degree_bound, 2);
        test_prover_impl::<Sha2Hash>(polynomials.clone(), points, degree_bound, 3);
        test_prover_impl::<Poseidon2Hash>(polynomials, points, degree_bound, 3);
    }

    #[test]
    fn test_one_constant_polynomial_one_point_1() {
        test_prover(
            vec![Polynomial::with_coefficients(vec![from_const(12)])],
            &[123],
            1,
        );
    }

    #[test]
    fn test_one_constant_polynomial_one_point_2() {
        test_prover(
            vec![Polynomial::with_coefficients(vec![from_const(12)])],
            &[321],
            1,
        );
    }

    #[test]
    fn test_one_constant_polynomial_one_point_3() {
        test_prover(
            vec![Polynomial::with_coefficients(vec![from_const(34)])],
            &[123],
            1,
        );
    }

    #[test]
    fn test_one_constant_polynomial_two_points() {
        test_prover(
            vec![Polynomial::with_coefficients(vec![from_const(12)])],
            &[123, 456],
            1,
        );
    }

    #[test]
    fn test_one_constant_polynomial_three_points() {
        test_prover(
            vec![Polynomial::with_coefficients(vec![from_const(12)])],
            &[789, 456, 123],
            1,
        );
    }

    #[test]
    fn test_one_polynomial_degree_one_one_point_1() {
        test_prover(
            vec![Polynomial::with_coefficients(vec![
                from_const(12),
                from_const(34),
            ])],
            &[123],
            2,
        );
    }

    #[test]
    fn test_one_polynomial_degree_one_one_point_2() {
        test_prover(
            vec![Polynomial::with_coefficients(vec![
                from_const(12),
                from_const(34),
            ])],
            &[321],
            2,
        );
    }

    #[test]
    fn test_one_polynomial_degree_one_one_point_3() {
        test_prover(
            vec![Polynomial::with_coefficients(vec![
                from_const(34),
                from_const(56),
            ])],
            &[123],
            2,
        );
    }

    #[test]
    fn test_one_polynomial_degree_one_two_points() {
        test_prover(
            vec![Polynomial::with_coefficients(vec![
                from_const(12),
                from_const(34),
            ])],
            &[123, 456],
            2,
        );
    }

    #[test]
    fn test_one_polynomial_degree_one_three_points() {
        test_prover(
            vec![Polynomial::with_coefficients(vec![
                from_const(12),
                from_const(34),
            ])],
            &[789, 456, 123],
            2,
        );
    }

    #[test]
    fn test_two_polynomials_degree_three_one_point_1() {
        test_prover(
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
            &[123],
            4,
        );
    }

    #[test]
    fn test_two_polynomials_degree_three_one_point_2() {
        test_prover(
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
            &[321],
            4,
        );
    }

    #[test]
    fn test_two_polynomials_degree_three_one_point_3() {
        test_prover(
            vec![
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
            ],
            &[123],
            4,
        );
    }

    #[test]
    fn test_two_polynomials_degree_three_two_points() {
        test_prover(
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
            &[123, 456],
            4,
        );
    }

    #[test]
    fn test_two_polynomials_degree_three_three_points() {
        test_prover(
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
            &[789, 456, 123],
            4,
        );
    }
}
