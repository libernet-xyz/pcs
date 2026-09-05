use crate::fri;
use crate::hash::Hasher;
use crate::merkle::{Proof as LeafProof, Tree};
use crate::utils;
use anyhow::{Result, anyhow};
use primitive_types::{H256, U256};
use starkom_ff::{Field, Field256};
use starkom_poly::Polynomial;
use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;
use std::sync::LazyLock;

/// Target security level in bits.
pub const LAMBDA: usize = 128;

/// Domain separator tag used by [`Committer::transcript_hash`].
static TRANSCRIPT_DST: LazyLock<H256> =
    LazyLock::new(|| utils::make_dst(b"starkom/deep/transcript"));

/// First domain separator tag for the Fiat-Shamir challenge used to derive query indices.
static QUERY_DST0: LazyLock<H256> = LazyLock::new(|| utils::make_dst(b"starkom/deep/query/0"));

/// Second domain separator tag for the Fiat-Shamir challenge used to derive query indices.
static QUERY_DST1: LazyLock<H256> = LazyLock::new(|| utils::make_dst(b"starkom/deep/query/1"));

/// Domain separator tag for the Fiat-Shamir challenge used to build the random linear combination.
static RLC_DST: LazyLock<H256> = LazyLock::new(|| utils::make_dst(b"starkom/deep/rlc"));

/// Returns the number of FRI queries required to achieve 128-bit security using a blowup factor of
/// `2^blowup_log2`.
fn num_queries(blowup_log2: usize) -> usize {
    LAMBDA.div_ceil(blowup_log2)
}

/// Encodes a `usize` into a `H256` for use in various transcript hashes.
fn encode_usize(value: usize) -> H256 {
    let mut bytes = [0u8; 32];
    bytes[24..32].copy_from_slice(&(value as u64).to_be_bytes());
    H256::from_slice(&bytes)
}

/// Checks that none of the given points lies inside the evaluation domain of size `n`.
///
/// Used on both sides: the prover must not be asked to open such a point, and a verifier must
/// reject a proof that claims one.
fn check_points_off_domain<F: Field256>(
    points: impl IntoIterator<Item = F>,
    n: usize,
) -> Result<()> {
    let marker = F::MULTIPLICATIVE_GENERATOR.pow_small(n);
    for z in points {
        if z.pow_small(n) == marker {
            return Err(anyhow!(
                "the opened point {z} is inside the evaluation domain"
            ));
        }
    }
    Ok(())
}

/// Computes a random linear combination of a list of values.
///
/// `alpha` is a Fiat-Shamir challenge of some sort.
fn rlc<F: Field>(values: impl IntoIterator<Item = F>, alpha: F) -> F {
    let mut rlc = F::ZERO;
    let mut pow = F::ONE;
    for value in values.into_iter() {
        rlc += value * pow;
        pow *= alpha;
    }
    rlc
}

/// A batched DEEP-FRI polynomial commitment (see [`Committer`] for details).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment<F: Field256, H: Hasher<F>> {
    /// The root hashes of the Merkle trees where the evaluations of all batched polynomials are
    /// stored. There is one root hash per polynomial batch.
    tree_roots: Vec<H256>,
    /// The underlying FRI commitment.
    inner: fri::Commitment,
    _data: PhantomData<(F, H)>,
}

impl<F: Field256, H: Hasher<F>> Commitment<F, H> {
    /// Returns the root hashes of the Merkle trees where all batched polynomials are stored.
    pub fn tree_roots(&self) -> &[H256] {
        self.tree_roots.as_slice()
    }

    /// The degree bound this commitment attests to, implied by the number of FRI folding rounds.
    /// Always equals [`Proof::degree_bound`].
    pub fn degree_bound(&self) -> usize {
        1usize << (self.inner.len() - 1)
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
        H::hash_transcript(
            *TRANSCRIPT_DST,
            std::iter::once(encode_usize(batch_count))
                .chain(self.tree_roots[..batch_count].iter().copied())
                .collect::<Vec<H256>>()
                .as_slice(),
        )
    }

    /// Returns the FRI query indices derived via Fiat-Shamir from the full commitment transcript
    /// (all polynomial and FRI Merkle root hashes).
    fn get_query_indices(&self, degree_bound: usize, blowup_log2: usize) -> Vec<usize> {
        let n = U256::from((degree_bound << blowup_log2) as u64);
        let k = num_queries(blowup_log2);
        let mut indices = Vec::with_capacity(k);
        let seed = H::hash_transcript(
            *QUERY_DST0,
            std::iter::once(encode_usize(self.tree_roots.len()))
                .chain(self.tree_roots.iter().copied())
                .chain(std::iter::once(encode_usize(self.inner.len())))
                .chain(self.inner.roots().iter().copied())
                .collect::<Vec<H256>>()
                .as_slice(),
        );
        for i in 0..k {
            let hash = H::challenge(*QUERY_DST1, &[seed, encode_usize(i)]);
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
pub struct Committer<F: Field256, H: Hasher<F>> {
    /// The proven degree bound. The degree of all batched polynomials must be less than this value.
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// All polynomials batched so far.
    polynomials: Vec<Polynomial<F>>,
    /// The Merkle trees built so far.
    ///
    /// The sum of all `num_polys` of all trees must match the number of `polynomials`.
    trees: Vec<Tree<F, H>>,
}

impl<F: Field256, H: Hasher<F>> Committer<F, H> {
    /// Constructs a [`Committer`] with the given degree bound, blowup factor, and first batch of
    /// polynomials.
    ///
    /// We require specifying the first batch because our DEEP-FRI protocol requires at least one
    /// committed polynomial to work.
    ///
    /// `degree_bound` must be a power of 2 less than or equal to 2^[F::S](`Field::S`), and
    /// `blowup_log2` must not be zero.
    pub fn new(degree_bound: usize, blowup_log2: usize, polynomials: Vec<Polynomial<F>>) -> Self {
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
        H::hash_transcript(
            *TRANSCRIPT_DST,
            std::iter::once(encode_usize(self.trees.len()))
                .chain(self.trees.iter().map(Tree::root_hash))
                .collect::<Vec<H256>>()
                .as_slice(),
        )
    }

    /// Adds a batch of polynomials, returning the index of the newly created batch.
    ///
    /// The returned index can be used with the [`Self::root_hash`] method to get the Merkle root
    /// for the batch.
    ///
    /// REQUIRES: the degree of all specified polynomials must be strictly less than
    /// [`Self::degree_bound()`].
    pub fn add_batch(&mut self, polynomials: Vec<Polynomial<F>>) -> usize {
        assert!(!polynomials.is_empty());

        let degree_bound = polynomials
            .iter()
            .map(|polynomial| polynomial.degree_bound())
            .max()
            .unwrap()
            .next_power_of_two();
        assert!(degree_bound <= self.degree_bound);
        let n = self.degree_bound << self.blowup_log2;
        assert!(n.trailing_zeros() as usize <= F::S);

        let evaluations = polynomials
            .iter()
            .map(|polynomial| polynomial.clone().shift_domain().lde2(n))
            .collect::<Vec<Vec<F>>>();

        let index = self.trees.len();

        self.polynomials.extend(polynomials);
        self.trees.push(Tree::new(evaluations));

        index
    }

    /// Consumes the [`Committer`], calculates all DEEP quotients, and returns a polynomial
    /// [`Commitment`] and a DEEP-FRI [`Prover`].
    ///
    /// `points` is the set of points to open in the [`Prover`]. The contained scalars are
    /// (off-domain) X-coordinates; the corresponding Y-coordinates will be computed automatically
    /// for every batched polynomial.
    pub fn commit(self, points: BTreeSet<F>) -> (Commitment<F, H>, Prover<F, H>) {
        check_points_off_domain(points.iter().copied(), self.extended_domain_size()).unwrap();

        let alpha = H::challenge(
            *RLC_DST,
            std::iter::once(encode_usize(self.trees.len()))
                .chain(self.trees.iter().map(Tree::root_hash))
                .chain(std::iter::once(encode_usize(self.polynomials.len())))
                .chain(std::iter::once(encode_usize(points.len())))
                .chain(points.iter().flat_map(|&z| {
                    std::iter::once(z)
                        .chain(
                            self.polynomials
                                .iter()
                                .map(|polynomial| polynomial.evaluate(z)),
                        )
                        .map(|value| H256::from_slice(&value.to_be_bytes()))
                        .collect::<Vec<H256>>()
                }))
                .collect::<Vec<H256>>()
                .as_slice(),
        );

        let points: BTreeMap<F, Vec<F>> = points
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
            let mut pow = F::ONE;
            for polynomial in &self.polynomials {
                combined += polynomial.clone() * pow;
                pow *= alpha;
            }
            combined
        };

        let quotients = points
            .iter()
            .map(|(&z, values)| {
                let value = rlc(values.iter().copied(), alpha);
                let (quotient, remainder) = (combined.clone() - value).horner(z);
                assert_eq!(remainder, F::ZERO);
                quotient
            })
            .collect();

        let inner_prover = fri::Prover::<F, H>::new(quotients, self.degree_bound, self.blowup_log2);

        let commitment = Commitment {
            tree_roots: self.trees.iter().map(Tree::root_hash).collect(),
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
pub struct Proof<F: Field256, H: Hasher<F>> {
    /// The proven degree bound. If the proof is valid the degree of all batched polynomials is
    /// guaranteed to be strictly less than this value.
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// Number of committed polynomials.
    num_polys: usize,
    /// The opened points. Keys are (off-domain) X-coordinates, values are the corresponding
    /// evaluations (one for every committed polynomial).
    points: BTreeMap<F, Vec<F>>,
    /// Merkle proofs for the points at the query positions, relative to the raw Merkle trees (not
    /// the FRI folds). The outer array has one entry for every FRI query
    /// (`openings.len() == queries.len()`), and the inner arrays contain one proof for every Merkle
    /// tree.
    openings: Vec<Vec<LeafProof<F, H>>>,
    /// FRI queries on the DEEP quotients. The number of queries is calculated by [`num_queries`]
    /// above and is tuned so as to achieve 128-bit security.
    queries: Vec<fri::Query<F, H>>,
}

impl<F: Field256, H: Hasher<F>> Proof<F, H> {
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
    pub fn points(&self) -> &BTreeMap<F, Vec<F>> {
        &self.points
    }

    /// Verifies this proof against the given commitment.
    pub fn verify(&self, commitment: &Commitment<F, H>) -> Result<()> {
        check_points_off_domain(self.points.keys().copied(), self.extended_domain_size())?;

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
            std::iter::once(encode_usize(commitment.tree_roots().len()))
                .chain(commitment.tree_roots().iter().copied())
                .chain(std::iter::once(encode_usize(self.num_polys)))
                .chain(std::iter::once(encode_usize(self.points.len())))
                .chain(self.points.iter().flat_map(|(z, values)| {
                    std::iter::once(z)
                        .chain(values.iter())
                        .map(|&value| H256::from_slice(&value.to_be_bytes()))
                }))
                .collect::<Vec<H256>>()
                .as_slice(),
        );

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
                    .flat_map(|proof| proof.leaf().iter().copied()),
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
                let v = rlc(values.iter().copied(), alpha);
                if quotient * (x - z) != combined - v {
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
pub struct Prover<F: Field256, H: Hasher<F>> {
    /// The degree bound to prove.
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// Raw Merkle trees for the committed polynomials, one for each batch.
    trees: Vec<Tree<F, H>>,
    /// The opened points.
    ///
    /// The keys of the map are the (off-domain) X-coordinates of the points, while values are lists
    /// of polynomial evaluations at that point (one for every committed polynomial).
    points: BTreeMap<F, Vec<F>>,
    /// The underlying FRI prover for the DEEP quotients. There's one quotient for every opened
    /// point, and all quotients are batched into the same FRI folding argument.
    inner_prover: fri::Prover<F, H>,
}

impl<F: Field256, H: Hasher<F>> Prover<F, H> {
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
    pub fn points(&self) -> &BTreeMap<F, Vec<F>> {
        &self.points
    }

    /// Makes a DEEP-FRI proof opening the committed polynomials at the points specified at
    /// commitment time (see [`Committer::commit()`]).
    pub fn prove(&self, commitment: &Commitment<F, H>) -> Proof<F, H> {
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

#[cfg(all(test, feature = "bluesky", feature = "goldilocks"))]
mod tests {
    use super::*;
    use crate::hash::{Keccak256Hash, Sha2Hash};
    use starkom_bluesky::Scalar as BS;
    use starkom_goldilocks::GL4;

    #[test]
    fn test_dsts() {
        assert_eq!(
            *TRANSCRIPT_DST,
            "0x09f36235476a658841de9bcdd34e1ac31ec792e41def5de31aecb4eb3bb1816b"
                .parse()
                .unwrap()
        );
        assert_eq!(
            *QUERY_DST0,
            "0xbbec7289b9fc3aade75412c031b62a769b205d1d73b29c9a06dbb91943e046bc"
                .parse()
                .unwrap()
        );
        assert_eq!(
            *QUERY_DST1,
            "0x88209086b178c9f2fb9c2f813cd229e1a2528cb4f5e6cd618710dc9869f14ac5"
                .parse()
                .unwrap()
        );
        assert_eq!(
            *RLC_DST,
            "0x688ae37e5f05871810e7c6777e1c16c55ef4b14f072357586d7a298cefc11368"
                .parse()
                .unwrap()
        );
    }

    fn test_prover_impl<F: Field256, H: Hasher<F>>(
        mut polynomial_batches: Vec<Vec<Polynomial<F>>>,
        points: &[u16],
        degree_bound: usize,
        blowup_log2: usize,
    ) {
        let num_batches = polynomial_batches.len();
        let num_polys = polynomial_batches.iter().map(|batch| batch.len()).sum();
        let points = BTreeMap::from_iter(points.iter().cloned().map(|z| {
            (
                F::from(z),
                polynomial_batches
                    .iter()
                    .flatten()
                    .map(|polynomial| polynomial.evaluate(z.into()))
                    .collect::<Vec<F>>(),
            )
        }));
        let first_batch = polynomial_batches.remove(0);
        let mut committer = Committer::<F, H>::new(degree_bound, blowup_log2, first_batch);
        let transcript_hashes: Vec<H256> = std::iter::once(committer.transcript_hash())
            .chain(polynomial_batches.into_iter().map(|batch| {
                committer.add_batch(batch);
                committer.transcript_hash()
            }))
            .collect();
        let (commitment, prover) = committer.commit(points.iter().map(|(&z, _)| z).collect());
        assert_eq!(commitment.degree_bound(), degree_bound);
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

    fn test_prover(polynomial_batches: Vec<Vec<Vec<u16>>>, points: &[u16], degree_bound: usize) {
        let bluesky_polynomial_batches: Vec<Vec<Polynomial<BS>>> = polynomial_batches
            .iter()
            .map(|batch| {
                batch
                    .iter()
                    .map(|coefficients| {
                        Polynomial::with_coefficients(
                            coefficients.iter().copied().map(BS::from).collect(),
                        )
                    })
                    .collect()
            })
            .collect();
        let goldilocks_polynomial_batches: Vec<Vec<Polynomial<GL4>>> = polynomial_batches
            .iter()
            .map(|batch| {
                batch
                    .iter()
                    .map(|coefficients| {
                        Polynomial::with_coefficients(
                            coefficients.iter().copied().map(GL4::from).collect(),
                        )
                    })
                    .collect()
            })
            .collect();
        test_prover_impl::<BS, Sha2Hash<BS>>(
            bluesky_polynomial_batches.clone(),
            points,
            degree_bound,
            1,
        );
        test_prover_impl::<GL4, Sha2Hash<GL4>>(
            goldilocks_polynomial_batches.clone(),
            points,
            degree_bound,
            1,
        );
        test_prover_impl::<BS, Sha2Hash<BS>>(
            bluesky_polynomial_batches.clone(),
            points,
            degree_bound,
            2,
        );
        test_prover_impl::<GL4, Sha2Hash<GL4>>(
            goldilocks_polynomial_batches.clone(),
            points,
            degree_bound,
            2,
        );
        test_prover_impl::<BS, Sha2Hash<BS>>(
            bluesky_polynomial_batches.clone(),
            points,
            degree_bound,
            3,
        );
        test_prover_impl::<GL4, Sha2Hash<GL4>>(
            goldilocks_polynomial_batches.clone(),
            points,
            degree_bound,
            3,
        );
        test_prover_impl::<BS, Keccak256Hash<BS>>(
            bluesky_polynomial_batches.clone(),
            points,
            degree_bound,
            2,
        );
        test_prover_impl::<GL4, Keccak256Hash<GL4>>(
            goldilocks_polynomial_batches.clone(),
            points,
            degree_bound,
            2,
        );
    }

    #[test]
    fn test_one_constant_polynomial_one_point_1() {
        test_prover(vec![vec![vec![12]]], &[123], 1);
    }

    #[test]
    fn test_one_constant_polynomial_one_point_2() {
        test_prover(vec![vec![vec![12]]], &[321], 1);
    }

    #[test]
    fn test_one_constant_polynomial_one_point_3() {
        test_prover(vec![vec![vec![34]]], &[123], 1);
    }

    #[test]
    fn test_one_constant_polynomial_two_points() {
        test_prover(vec![vec![vec![12]]], &[123, 456], 1);
    }

    #[test]
    fn test_one_constant_polynomial_three_points() {
        test_prover(vec![vec![vec![12]]], &[789, 456, 123], 1);
    }

    #[test]
    fn test_one_polynomial_degree_one_one_point_1() {
        test_prover(vec![vec![vec![12, 34]]], &[123], 2);
    }

    #[test]
    fn test_one_polynomial_degree_one_one_point_2() {
        test_prover(vec![vec![vec![12, 34]]], &[321], 2);
    }

    #[test]
    fn test_one_polynomial_degree_one_one_point_3() {
        test_prover(vec![vec![vec![34, 56]]], &[123], 2);
    }

    #[test]
    fn test_one_polynomial_degree_one_two_points() {
        test_prover(vec![vec![vec![12, 34]]], &[123, 456], 2);
    }

    #[test]
    fn test_one_polynomial_degree_one_three_points() {
        test_prover(vec![vec![vec![12, 34]]], &[789, 456, 123], 2);
    }

    #[test]
    fn test_two_polynomials_degree_three_one_point_1() {
        test_prover(
            vec![vec![vec![12, 34, 56, 78], vec![42, 43, 44, 45]]],
            &[123],
            4,
        );
    }

    #[test]
    fn test_two_polynomials_degree_three_one_point_2() {
        test_prover(
            vec![vec![vec![12, 34, 56, 78], vec![42, 43, 44, 45]]],
            &[321],
            4,
        );
    }

    #[test]
    fn test_two_polynomials_degree_three_one_point_3() {
        test_prover(
            vec![vec![vec![45, 44, 43, 42], vec![78, 56, 34, 12]]],
            &[123],
            4,
        );
    }

    #[test]
    fn test_two_polynomials_degree_three_two_points() {
        test_prover(
            vec![vec![vec![12, 34, 56, 78], vec![42, 43, 44, 45]]],
            &[123, 456],
            4,
        );
    }

    #[test]
    fn test_two_polynomials_degree_three_three_points() {
        test_prover(
            vec![vec![vec![12, 34, 56, 78], vec![42, 43, 44, 45]]],
            &[789, 456, 123],
            4,
        );
    }

    #[test]
    fn test_two_batches_one_and_one() {
        test_prover(
            vec![vec![vec![12, 34, 56, 78]], vec![vec![42, 43, 44, 45]]],
            &[123, 456],
            4,
        );
    }

    #[test]
    fn test_two_batches_two_and_one() {
        test_prover(
            vec![
                vec![vec![12, 34, 56, 78], vec![90, 78, 56, 34]],
                vec![vec![42, 43, 44, 45]],
            ],
            &[456, 789],
            4,
        );
    }

    #[test]
    fn test_two_batches_one_and_two() {
        test_prover(
            vec![
                vec![vec![90, 78, 56, 34]],
                vec![vec![12, 34, 56, 78], vec![42, 43, 44, 45]],
            ],
            &[456, 789],
            4,
        );
    }

    const ADVERSARIAL_DEGREE_BOUND: usize = 4;
    const ADVERSARIAL_BLOWUP_LOG2: usize = 2;

    fn polynomial(coefficients: &[u16]) -> Polynomial<BS> {
        Polynomial::with_coefficients(coefficients.iter().copied().map(BS::from).collect())
    }

    fn adversarial_setup() -> (Commitment<BS, Sha2Hash<BS>>, Proof<BS, Sha2Hash<BS>>) {
        let mut committer = Committer::<BS, Sha2Hash<BS>>::new(
            ADVERSARIAL_DEGREE_BOUND,
            ADVERSARIAL_BLOWUP_LOG2,
            vec![polynomial(&[12, 34, 56, 78]), polynomial(&[42, 43, 44, 45])],
        );
        committer.add_batch(vec![polynomial(&[90, 78, 56, 34])]);
        let (commitment, prover) =
            committer.commit(BTreeSet::from([BS::from(123u16), BS::from(456u16)]));
        let proof = prover.prove(&commitment);
        (commitment, proof)
    }

    fn assert_rejected(result: Result<()>, expected: &str) {
        let error = result
            .expect_err("the tampered proof was accepted")
            .to_string();
        assert!(error.contains(expected), "unexpected error: {error}");
    }

    #[test]
    fn test_accept_untampered_proof() {
        let (commitment, proof) = adversarial_setup();
        assert!(proof.verify(&commitment).is_ok());
    }

    #[test]
    fn test_reject_on_domain_point() {
        let (commitment, mut proof) = adversarial_setup();
        let values = proof.points.values().next().unwrap().clone();
        let on_domain = Polynomial::<BS>::coset_element2(0, proof.extended_domain_size());
        proof.points = BTreeMap::from([(on_domain, values)]);
        assert_rejected(proof.verify(&commitment), "is inside the evaluation domain");
    }

    #[test]
    fn test_reject_tampered_evaluation() {
        let (commitment, mut proof) = adversarial_setup();
        let z = *proof.points.keys().next().unwrap();
        proof.points.get_mut(&z).unwrap()[0] += BS::ONE;
        assert_rejected(proof.verify(&commitment), "algebraic check failed");
    }

    #[test]
    fn test_reject_foreign_commitment() {
        let (_, proof) = adversarial_setup();
        let committer = Committer::<BS, Sha2Hash<BS>>::new(
            ADVERSARIAL_DEGREE_BOUND,
            ADVERSARIAL_BLOWUP_LOG2,
            vec![polynomial(&[99, 98, 97, 96])],
        );
        let (foreign, _) = committer.commit(BTreeSet::from([BS::from(123u16)]));
        assert_rejected(proof.verify(&foreign), "wrong query index");
    }

    #[test]
    fn test_reject_missing_query() {
        let (commitment, mut proof) = adversarial_setup();
        proof.queries.pop();
        assert_rejected(proof.verify(&commitment), "incorrect number of queries");
    }

    #[test]
    fn test_reject_missing_openings() {
        let (commitment, mut proof) = adversarial_setup();
        proof.openings.pop();
        assert_rejected(proof.verify(&commitment), "incorrect number of openings");
    }

    #[test]
    fn test_reject_missing_opening_within_query() {
        let (commitment, mut proof) = adversarial_setup();
        proof.openings[0].pop();
        assert_rejected(
            proof.verify(&commitment),
            "incorrect number of openings for index",
        );
    }

    #[test]
    fn test_reject_swapped_openings() {
        let (commitment, mut proof) = adversarial_setup();
        proof.openings.swap(0, 1);
        assert_rejected(proof.verify(&commitment), "root hash mismatch");
    }

    #[test]
    fn test_reject_extra_point() {
        let (commitment, mut proof) = adversarial_setup();
        let values = proof.points.values().next().unwrap().clone();
        proof.points.insert(BS::from(789u16), values);
        assert_rejected(
            proof.verify(&commitment),
            "doesn't match the number of FRI quotients",
        );
    }

    #[test]
    fn test_reject_opening_from_wrong_domain() {
        let (commitment, mut proof) = adversarial_setup();
        let committer = Committer::<BS, Sha2Hash<BS>>::new(
            ADVERSARIAL_DEGREE_BOUND * 2,
            ADVERSARIAL_BLOWUP_LOG2,
            vec![polynomial(&[11, 22, 33, 44])],
        );
        let (other_commitment, other_prover) = committer.commit(BTreeSet::from([BS::from(123u16)]));
        let mut other_proof = other_prover.prove(&other_commitment);
        proof.openings[0][0] = other_proof.openings[0].remove(0);
        assert_rejected(proof.verify(&commitment), "invalid opening for index");
    }
}
