use crate::fri;
use crate::hash::Hasher;
use crate::merkle::{Proof as LeafProof, Tree};
use crate::starkom::proto::pcs::v9 as proto;
use crate::utils;
use anyhow::{Result, anyhow};
use primitive_types::{H256, U256};
use prost_types::Any;
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

/// Domain separator tag for the Fiat-Shamir challenge used to combine the DEEP quotients of all
/// opened points into a single polynomial.
static QUOTIENT_DST: LazyLock<H256> = LazyLock::new(|| utils::make_dst(b"starkom/deep/quotient"));

/// Domain separator tag for the proof-of-work hash used for grinding.
static POW_DST: LazyLock<H256> = LazyLock::new(|| utils::make_dst(b"starkom/deep/pow"));

/// Returns the number of FRI queries required to achieve 128-bit security using a blowup factor of
/// `2^blowup_log2` and `grinding_bits` bits of grinding.
///
/// Every query provides `blowup_log2` bits of security and grinding provides the rest. At least
/// one query is always required.
pub fn num_queries(blowup_log2: usize, grinding_bits: usize) -> usize {
    LAMBDA
        .saturating_sub(grinding_bits)
        .div_ceil(blowup_log2)
        .max(1)
}

/// The parameters of the DEEP-FRI protocol.
///
/// The prover specifies them in [`Committer::new`] and they're recorded in every [`Proof`], so
/// verifiers must check them against the expected ones (see [`Proof::options`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    /// The proven degree bound, which must be a power of 2.
    ///
    /// Honest provers commit polynomials of degree strictly less than this value, but a valid proof
    /// only guarantees a degree at most equal to it (see [`Proof::verify`]).
    pub degree_bound: usize,

    /// The base-2 logarithm of the blowup factor, which must be greater than 0.
    ///
    /// The extended evaluation domain has `degree_bound * 2^blowup_log2` elements and its size must
    /// not exceed the 2-adicity of the base field.
    pub blowup_log2: usize,

    /// The number of grinding bits.
    ///
    /// The prover grinds a nonce whose proof-of-work hash has at least this many leading zero bits,
    /// which takes `2^grinding_bits` hash evaluations on average. Every `blowup_log2` bits of
    /// grinding save one FRI query (see [`num_queries`]). 0 disables grinding.
    ///
    /// Must be at most half the number of bits of the field (see [`Self::validate`]).
    pub grinding_bits: usize,
}

impl Options {
    /// Returns the size of the extended evaluation domain, `degree_bound * 2^blowup_log2`.
    ///
    /// REQUIRES: the options must be [valid](`Self::validate`).
    pub fn extended_domain_size(&self) -> usize {
        self.degree_bound << self.blowup_log2
    }

    /// Returns the number of FRI queries required to achieve 128-bit security with these options
    /// (see [`num_queries`]).
    pub fn num_queries(&self) -> usize {
        num_queries(self.blowup_log2, self.grinding_bits)
    }

    /// Checks that these options are valid for the field `F`, returning an error otherwise.
    ///
    /// The degree bound must be a power of 2, `blowup_log2` must be greater than 0, the size of the
    /// extended evaluation domain must not exceed the 2-adicity of `F::BaseField`, and
    /// `grinding_bits` must not exceed `F::NUM_BITS / 2`.
    pub fn validate<F: Field>(&self) -> Result<()> {
        if !self.degree_bound.is_power_of_two() {
            return Err(anyhow!(
                "invalid degree bound {} (not a power of two)",
                self.degree_bound
            ));
        }
        if self.blowup_log2 == 0 {
            return Err(anyhow!("invalid blowup factor 2^0"));
        }
        let degree_bound_log2 = self.degree_bound.trailing_zeros() as usize;
        if degree_bound_log2 + self.blowup_log2 > F::BaseField::S {
            return Err(anyhow!(
                "invalid extended domain size 2^{{{}+{}}} (exceeds the 2-adicity of the field, {})",
                degree_bound_log2,
                self.blowup_log2,
                F::BaseField::S
            ));
        }
        if self.grinding_bits > F::NUM_BITS / 2 {
            return Err(anyhow!(
                "invalid number of grinding bits {} (exceeds half the field size, {})",
                self.grinding_bits,
                F::NUM_BITS / 2
            ));
        }
        Ok(())
    }
}

/// Computes the proof-of-work hash of a nonce.
///
/// `seed` is the hash of the full commitment transcript returned by [`Commitment::query_seed`], so
/// that changing any commitment requires redoing the proof of work.
fn proof_of_work<F: Field256, H: Hasher<F>>(seed: H256, nonce: u64) -> H256 {
    H::hash_transcript(*POW_DST, &[seed, utils::encode_u64(nonce)])
}

/// Checks that a proof-of-work hash, interpreted as a big-endian 256-bit integer, has at least
/// `grinding_bits` leading zero bits.
fn check_proof_of_work(pow: H256, grinding_bits: usize) -> bool {
    U256::from_big_endian(pow.as_bytes()).leading_zeros() as usize >= grinding_bits
}

/// Derives the FRI query indices from a proof-of-work hash via Fiat-Shamir.
///
/// The indices are derived from the proof-of-work hash rather than from the query seed, so that a
/// cheating prover has to find a new valid nonce for every set of indices it wants to try. A single
/// nonce covers all queries: one nonce per query would allow grinding every index independently,
/// which defeats the purpose.
fn get_query_indices<F: Field256, H: Hasher<F>>(pow: H256, options: &Options) -> Vec<usize> {
    let n = U256::from(options.extended_domain_size() as u64);
    (0..options.num_queries())
        .map(|i| {
            let hash = H::challenge(*QUERY_DST1, &[pow, utils::encode_usize(i)]);
            (hash.to_u256() % n).as_u64() as usize
        })
        .collect()
}

/// Checks that none of the given points lies inside the evaluation domain of size `n`.
///
/// Used on both sides: the prover must not be asked to open such a point, and a verifier must
/// reject a proof that claims one.
fn check_points_off_domain<F: Field256>(
    points: impl IntoIterator<Item = F>,
    n: usize,
) -> Result<()> {
    let marker: F = F::BaseField::MULTIPLICATIVE_GENERATOR.pow_small(n).into();
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

/// Derives the two Fiat-Shamir challenges of the DEEP layer.
///
/// The first challenge uses [`RLC_DST`] and is used by the caller to combine all committed
/// polynomials, while the second one uses [`QUOTIENT_DST`] and is used to combine all DEEP
/// quotients.
///
/// Both challenges are bound to all Merkle roots of the committed polynomials and to all opened
/// points and their claimed evaluations.
fn deep_challenges<F: Field256, H: Hasher<F>>(
    tree_roots: &[H256],
    num_polys: usize,
    points: &BTreeMap<F, Vec<F>>,
) -> (F, F) {
    let transcript: Vec<H256> = std::iter::once(utils::encode_usize(tree_roots.len()))
        .chain(tree_roots.iter().copied())
        .chain(std::iter::once(utils::encode_usize(num_polys)))
        .chain(std::iter::once(utils::encode_usize(points.len())))
        .chain(points.iter().flat_map(|(z, values)| {
            std::iter::once(z)
                .chain(values.iter())
                .map(|value| H256::from_slice(&value.to_be_bytes()))
        }))
        .collect();
    (
        H::challenge(*RLC_DST, transcript.as_slice()),
        H::challenge(*QUOTIENT_DST, transcript.as_slice()),
    )
}

/// Checks that the `hash_backend` field of a protobuf identifies the hash backend `T`.
fn check_hash_backend<T: prost::Name + Default>(
    hash_backend: &Option<prost_types::Any>,
) -> Result<()> {
    match hash_backend {
        Some(hash_backend) => {
            hash_backend.to_msg::<T>().map_err(|_| {
                anyhow!(
                    "incorrect hash backend (got {}, want {})",
                    hash_backend.type_url,
                    T::type_url()
                )
            })?;
            Ok(())
        }
        None => Err(anyhow!("hash backend not specified")),
    }
}

/// A batched DEEP-FRI polynomial commitment (see [`Committer`] for details).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment<F: Field256, H: Hasher<F>> {
    /// The root hashes of the Merkle trees where the evaluations of all batched polynomials are
    /// stored. There is one root hash per polynomial batch.
    tree_roots: Vec<H256>,
    /// The number of polynomials in each batch, which is also the number of evaluations stored in
    /// each leaf of the corresponding Merkle tree. There is one entry per polynomial batch.
    batch_sizes: Vec<usize>,
    /// The underlying FRI commitment.
    inner: fri::Commitment,
    _data: PhantomData<(F, H)>,
}

impl<F: Field256, H: Hasher<F>> Commitment<F, H> {
    /// Returns the root hashes of the Merkle trees where all batched polynomials are stored.
    pub fn tree_roots(&self) -> &[H256] {
        self.tree_roots.as_slice()
    }

    /// Returns the number of polynomials in each batch, which is also the number of evaluations
    /// stored in each leaf of the corresponding Merkle tree.
    ///
    /// The returned slice has one element for every [tree root](`Self::tree_roots`).
    pub fn batch_sizes(&self) -> &[usize] {
        self.batch_sizes.as_slice()
    }

    /// The degree bound this commitment attests to, implied by the number of FRI folding rounds.
    /// Always equals the [`degree_bound`](`Options::degree_bound`) of the [`Proof`].
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
            std::iter::once(utils::encode_usize(batch_count))
                .chain(self.tree_roots[..batch_count].iter().copied())
                .collect::<Vec<H256>>()
                .as_slice(),
        )
    }

    /// Hashes the full commitment transcript (all polynomial and FRI Merkle root hashes).
    ///
    /// The returned seed is the input of the proof of work, which in turn determines the FRI query
    /// indices (see [`get_query_indices`]).
    fn query_seed(&self) -> H256 {
        H::hash_transcript(
            *QUERY_DST0,
            std::iter::once(utils::encode_usize(self.tree_roots.len()))
                .chain(self.tree_roots.iter().copied())
                .chain(std::iter::once(utils::encode_usize(self.inner.len())))
                .chain(self.inner.roots().iter().copied())
                .collect::<Vec<H256>>()
                .as_slice(),
        )
    }

    /// Serializes this commitment to a [`Commitment`](`proto::Commitment`) protobuf.
    pub fn to_proto(&self) -> proto::Commitment {
        proto::Commitment {
            hash_backend: Some(Any::from_msg(&H::ProtoTag::default()).unwrap()),
            batches: self
                .tree_roots
                .iter()
                .zip(self.batch_sizes.iter())
                .map(|(root, &num_polys)| proto::commitment::Batch {
                    root: root.as_bytes().to_vec(),
                    num_polys: num_polys as u64,
                })
                .collect(),
            fri_roots: self
                .inner
                .roots()
                .iter()
                .map(|&hash| hash.as_bytes().to_vec())
                .collect(),
        }
    }

    /// Deserializes a commitment from a [`Commitment`](`proto::Commitment`) protobuf.
    pub fn from_proto(proto: &proto::Commitment) -> Result<Self> {
        check_hash_backend::<H::ProtoTag>(&proto.hash_backend)?;

        if proto.batches.is_empty() {
            return Err(anyhow!("the commitment has no polynomial batches"));
        }
        if proto.fri_roots.is_empty() {
            return Err(anyhow!("the commitment has no FRI roots"));
        }

        Ok(Self {
            tree_roots: proto
                .batches
                .iter()
                .map(|batch| H::load_hash(batch.root.as_slice()))
                .collect::<Result<_>>()?,
            batch_sizes: proto
                .batches
                .iter()
                .map(|batch| Ok(usize::try_from(batch.num_polys)?))
                .collect::<Result<_>>()?,
            inner: fri::Commitment::with_roots(
                proto
                    .fri_roots
                    .iter()
                    .map(|bytes| H::load_hash(bytes.as_slice()))
                    .collect::<Result<_>>()?,
            ),
            _data: PhantomData,
        })
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
    /// The protocol parameters. The degree of all batched polynomials must be less than the degree
    /// bound.
    options: Options,
    /// All polynomials batched so far.
    polynomials: Vec<Polynomial<F>>,
    /// The Merkle trees built so far.
    ///
    /// The sum of all `num_polys` of all trees must match the number of `polynomials`.
    trees: Vec<Tree<F, H>>,
}

impl<F: Field256, H: Hasher<F>> Committer<F, H> {
    /// Constructs a [`Committer`] with the given options and first batch of polynomials.
    ///
    /// We require specifying the first batch because our DEEP-FRI protocol requires at least one
    /// committed polynomial to work.
    ///
    /// Panics if the options are [invalid](`Options::validate`).
    pub fn new(options: Options, polynomials: Vec<Polynomial<F>>) -> Self {
        options.validate::<F>().expect("invalid DEEP-FRI options");
        assert!(!polynomials.is_empty());
        let mut committer = Self {
            options,
            polynomials: vec![],
            trees: vec![],
        };
        committer.add_batch(polynomials);
        committer
    }

    /// Returns the protocol parameters.
    pub fn options(&self) -> &Options {
        &self.options
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
            std::iter::once(utils::encode_usize(self.trees.len()))
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
    /// the [`degree_bound`](`Options::degree_bound`).
    pub fn add_batch(&mut self, polynomials: Vec<Polynomial<F>>) -> usize {
        assert!(!polynomials.is_empty());

        let degree_bound = polynomials
            .iter()
            .map(|polynomial| polynomial.degree_bound())
            .max()
            .unwrap()
            .next_power_of_two();
        assert!(degree_bound <= self.options.degree_bound);
        let n = self.options.extended_domain_size();

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
    ///
    /// The DEEP quotients of all opened points are combined into a single polynomial with a random
    /// linear combination, so the FRI folding argument always works on one polynomial regardless
    /// of the number of points.
    pub fn commit(self, points: BTreeSet<F>) -> (Commitment<F, H>, Prover<F, H>) {
        check_points_off_domain(points.iter().copied(), self.options.extended_domain_size())
            .unwrap();

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

        let tree_roots: Vec<H256> = self.trees.iter().map(Tree::root_hash).collect();
        let (alpha, beta) =
            deep_challenges::<F, H>(tree_roots.as_slice(), self.polynomials.len(), &points);

        let combined = {
            let mut combined = Polynomial::default();
            let mut pow = F::ONE;
            for polynomial in &self.polynomials {
                combined += polynomial.clone() * pow;
                pow *= alpha;
            }
            combined
        };

        let quotient = {
            let mut quotient = Polynomial::default();
            let mut pow = F::ONE;
            for (&z, values) in &points {
                let value = rlc(values.iter().copied(), alpha);
                let (point_quotient, remainder) = (combined.clone() - value).horner(z);
                assert_eq!(remainder, F::ZERO);
                quotient += point_quotient * pow;
                pow *= beta;
            }
            quotient
        };

        let inner_prover = fri::Prover::<F, H>::new(
            quotient,
            self.options.degree_bound,
            self.options.blowup_log2,
        );

        let commitment = Commitment {
            tree_roots,
            batch_sizes: self.trees.iter().map(Tree::num_polys).collect(),
            inner: inner_prover.commit(),
            _data: Default::default(),
        };
        let prover = Prover {
            options: self.options,
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
    /// The protocol parameters.
    options: Options,
    /// The proof-of-work nonce found by the prover.
    nonce: u64,
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
    /// FRI queries on the combined DEEP quotient (see [`Committer::commit`]). The number of queries
    /// is calculated by [`num_queries`] above and is tuned to achieve 128-bit security together
    /// with grinding.
    queries: Vec<fri::Query<F, H>>,
}

impl<F: Field256, H: Hasher<F>> Proof<F, H> {
    /// Returns the protocol parameters used in the proof.
    ///
    /// Callers must check them against the expected ones. Note that soundness doesn't depend on the
    /// number of grinding bits because the number of queries is automatically derived from it, but
    /// it should still be checked along with the other parameters.
    pub fn options(&self) -> &Options {
        &self.options
    }

    /// Returns the proof-of-work nonce found by the prover.
    pub fn nonce(&self) -> u64 {
        self.nonce
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
    ///
    /// If verification succeeds, the committed polynomials have degree *at most*
    /// [`Options::degree_bound`] and their evaluations at the [opened points](`Self::points`) are
    /// the ones claimed by the proof.
    ///
    /// WARNING: honest provers commit polynomials of degree *strictly less* than
    /// [`Options::degree_bound`], but a valid proof only guarantees a degree *at most* equal to it.
    /// FRI bounds the degree of the DEEP quotients `(f(X) - f(z)) / (X - z)` rather than that of
    /// the committed polynomials `f(X)`, and each quotient has one degree less than its polynomial.
    /// This slack is common to DEEP-FRI implementations (cfr. the description of the "knowledge
    /// extractor" in Section 5.5 of the ethSTARK documentation) and is harmless for protocols whose
    /// constraints only depend on the values of the polynomials over the multiplicative subgroup
    /// `H` of size `degree_bound`, such as PLONK and AIR. A polynomial of degree `degree_bound`
    /// equals one of lower degree plus a multiple of `X^degree_bound - 1`, which vanishes on `H`.
    /// Protocols whose soundness depends on the exact degree bound, such as univariate sumcheck,
    /// must take the slack into account.
    pub fn verify(&self, commitment: &Commitment<F, H>) -> Result<()> {
        check_points_off_domain(
            self.points.keys().copied(),
            self.options.extended_domain_size(),
        )?;

        if commitment.inner.len() != self.options.degree_bound.trailing_zeros() as usize + 1 {
            return Err(anyhow!(
                "the degree bound {} doesn't match the {} FRI roots of the commitment",
                self.options.degree_bound,
                commitment.inner.len()
            ));
        }

        let num_polys = commitment.batch_sizes().iter().copied().sum();
        if self.num_polys != num_polys {
            return Err(anyhow!(
                "incorrect number of polynomials (got {}, want {})",
                self.num_polys,
                num_polys
            ));
        }
        if self.points.is_empty() {
            return Err(anyhow!("the proof doesn't open any points"));
        }
        for (&z, values) in &self.points {
            if values.len() != self.num_polys {
                return Err(anyhow!(
                    "incorrect number of evaluations at {z} (got {}, want {})",
                    values.len(),
                    self.num_polys
                ));
            }
        }

        let pow = proof_of_work::<F, H>(commitment.query_seed(), self.nonce);
        if !check_proof_of_work(pow, self.options.grinding_bits) {
            return Err(anyhow!("insufficient proof of work"));
        }
        let indices = get_query_indices::<F, H>(pow, &self.options);
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

        let (alpha, beta) =
            deep_challenges::<F, H>(commitment.tree_roots(), self.num_polys, &self.points);

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
            for ((&root_hash, &batch_size), opening) in commitment
                .tree_roots()
                .iter()
                .zip(commitment.batch_sizes())
                .zip(openings.iter())
            {
                if 1usize << opening.len() != self.options.extended_domain_size() {
                    return Err(anyhow!("invalid opening for index {index}"));
                }
                if opening.leaf().len() != batch_size {
                    return Err(anyhow!(
                        "incorrect batch size at index {index} (got {}, want {})",
                        opening.leaf().len(),
                        batch_size
                    ));
                }
                opening.verify(index, root_hash)?;
            }

            if 1usize << (query.len() - 1) != self.options.degree_bound {
                return Err(anyhow!("invalid low-degree proof for index {index}"));
            }
            query.verify(&commitment.inner)?;

            let combined = rlc(
                openings
                    .iter()
                    .flat_map(|proof| proof.leaf().iter().copied()),
                alpha,
            );

            let (quotient, _) = query.values();

            let x = query.x();
            let expected = rlc(
                self.points.iter().map(|(&z, values)| {
                    let v = rlc(values.iter().copied(), alpha);
                    (combined - v) * (x - z).invert_vartime().unwrap()
                }),
                beta,
            );
            if quotient != expected {
                return Err(anyhow!("algebraic check failed at query index {index}"));
            }
        }

        Ok(())
    }

    /// Serializes this proof to a [`Proof`](`proto::Proof`) protobuf.
    pub fn to_proto(&self) -> proto::Proof {
        proto::Proof {
            hash_backend: Some(Any::from_msg(&H::ProtoTag::default()).unwrap()),
            degree_bound: self.options.degree_bound as u64,
            blowup_log2: self.options.blowup_log2 as u32,
            grinding_bits: self.options.grinding_bits as u32,
            nonce: self.nonce,
            num_polys: self.num_polys as u64,
            points: self
                .points
                .iter()
                .map(|(&z, values)| proto::proof::Point {
                    z: z.to_le_bytes().to_vec(),
                    v: values
                        .iter()
                        .map(|value| value.to_le_bytes().to_vec())
                        .collect(),
                })
                .collect(),
            openings: self
                .openings
                .iter()
                .map(|proofs| proto::proof::Opening {
                    proofs: proofs.iter().map(|proof| proof.to_proto()).collect(),
                })
                .collect(),
            queries: self.queries.iter().map(|query| query.to_proto()).collect(),
        }
    }

    /// Deserializes a proof from a [`Proof`](`proto::Proof`) protobuf.
    ///
    /// NOTE: this method does not validate the proof, it only deserializes it. The caller must
    /// invoke [`Self::verify`] separately.
    pub fn from_proto(proto: &proto::Proof) -> Result<Self> {
        check_hash_backend::<H::ProtoTag>(&proto.hash_backend)?;

        let options = Options {
            degree_bound: proto.degree_bound as usize,
            blowup_log2: proto.blowup_log2 as usize,
            grinding_bits: proto.grinding_bits as usize,
        };
        options.validate::<F>()?;

        Ok(Self {
            options,
            nonce: proto.nonce,
            num_polys: proto.num_polys as usize,
            points: proto
                .points
                .iter()
                .map(|point| {
                    Ok((
                        utils::load_scalar(point.z.as_slice())?,
                        point
                            .v
                            .iter()
                            .map(|value| utils::load_scalar(value.as_slice()))
                            .collect::<Result<_>>()?,
                    ))
                })
                .collect::<Result<_>>()?,
            openings: proto
                .openings
                .iter()
                .map(|opening| {
                    opening
                        .proofs
                        .iter()
                        .map(LeafProof::from_proto)
                        .collect::<Result<_>>()
                })
                .collect::<Result<_>>()?,
            queries: proto
                .queries
                .iter()
                .map(|query| {
                    fri::Query::from_proto(options.degree_bound, options.blowup_log2, query)
                })
                .collect::<Result<_>>()?,
        })
    }
}

/// A DEEP-FRI prover.
///
/// [`Prover`]s are constructed by [`Committer::commit()`]; see that method for details.
#[derive(Debug, Clone)]
pub struct Prover<F: Field256, H: Hasher<F>> {
    /// The protocol parameters.
    options: Options,
    /// Raw Merkle trees for the committed polynomials, one for each batch.
    trees: Vec<Tree<F, H>>,
    /// The opened points.
    ///
    /// The keys of the map are the (off-domain) X-coordinates of the points, while values are lists
    /// of polynomial evaluations at that point (one for every committed polynomial).
    points: BTreeMap<F, Vec<F>>,
    /// The underlying FRI prover for the DEEP quotient.
    inner_prover: fri::Prover<F, H>,
}

impl<F: Field256, H: Hasher<F>> Prover<F, H> {
    /// Returns the protocol parameters.
    pub fn options(&self) -> &Options {
        &self.options
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

    /// Carries out an exhaustive search for a valid nonce, stopping when one is found.
    ///
    /// This is our proof-of-work algorithm.
    fn grind_nonce(&self, seed: H256) -> (u64, H256) {
        for nonce in 0..=u64::MAX {
            let pow = proof_of_work::<F, H>(seed, nonce);
            if check_proof_of_work(pow, self.options.grinding_bits) {
                return (nonce, pow);
            }
        }
        panic!("no valid proof-of-work nonce found");
    }

    /// Makes a DEEP-FRI proof opening the committed polynomials at the points specified at
    /// commitment time (see [`Committer::commit()`]).
    ///
    /// This includes grinding a proof-of-work nonce, which takes
    /// 2^[`grinding_bits`](`Options::grinding_bits`) hash evaluations on average.
    pub fn prove(&self, commitment: &Commitment<F, H>) -> Proof<F, H> {
        let (nonce, pow) = self.grind_nonce(commitment.query_seed());
        let indices = get_query_indices::<F, H>(pow, &self.options);
        let openings = indices
            .iter()
            .map(|&index| self.trees.iter().map(|tree| tree.query(index)).collect())
            .collect();
        let queries = indices
            .iter()
            .map(|&index| self.inner_prover.query(index))
            .collect();
        Proof {
            options: self.options,
            nonce,
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
    use crate::hash::{Keccak256Hash, Sha2Hash};
    use starkom_bluesky::Scalar as BS;
    use starkom_goldilocks::GL4;
    use std::fmt::Debug;
    use std::str::FromStr;

    fn parse<V: FromStr<Err: Debug>>(s: &'static str) -> V {
        s.parse().unwrap()
    }

    /// Builds a polynomial from its coefficients.
    fn make_polynomial<F: Field256>(coefficients: &[u64]) -> Polynomial<F> {
        Polynomial::with_coefficients(coefficients.iter().copied().map(F::from).collect())
    }

    /// Builds a batch of polynomials, each one from its coefficients.
    fn make_batch<F: Field256>(batch: &[&[u64]]) -> Vec<Polynomial<F>> {
        batch
            .iter()
            .map(|coefficients| make_polynomial(coefficients))
            .collect()
    }

    /// Converts small integers into a set of points to open.
    fn make_points<F: Field256>(points: &[u64]) -> BTreeSet<F> {
        points.iter().copied().map(F::from).collect()
    }

    /// Builds a committer for the given batches of polynomials, adding them in order.
    fn make_committer<F: Field256, H: Hasher<F>>(
        options: Options,
        batches: &[&[&[u64]]],
    ) -> Committer<F, H> {
        let mut committer = Committer::new(options, make_batch(batches[0]));
        for batch in &batches[1..] {
            committer.add_batch(make_batch(batch));
        }
        committer
    }

    /// The protocol parameters used in most tests.
    const OPTIONS: Options = Options {
        degree_bound: 4,
        blowup_log2: 2,
        grinding_bits: 0,
    };

    /// Same as [`OPTIONS`] but with grinding.
    const GROUND_OPTIONS: Options = Options {
        grinding_bits: 8,
        ..OPTIONS
    };

    /// Two batches of polynomials of degree 3, the first with two polynomials and the second with
    /// one.
    const SAMPLE_BATCHES: [&[&[u64]]; 2] = [
        &[&[12, 34, 56, 78], &[42, 43, 44, 45]],
        &[&[90, 78, 56, 34]],
    ];

    /// The points opened in most tests.
    const SAMPLE_POINTS: [u64; 2] = [123, 456];

    /// Commits to [`SAMPLE_BATCHES`] with the given options and proves their evaluations at
    /// [`SAMPLE_POINTS`].
    fn sample_proof(options: Options) -> (Commitment<BS, Sha2Hash<BS>>, Proof<BS, Sha2Hash<BS>>) {
        let committer = make_committer(options, &SAMPLE_BATCHES);
        let (commitment, prover) = committer.commit(make_points(&SAMPLE_POINTS));
        let proof = prover.prove(&commitment);
        (commitment, proof)
    }

    /// Returns the smallest nonce greater than `proof.nonce` that satisfies (`valid == true`) or
    /// fails (`valid == false`) the proof of work of `proof` against `commitment`.
    fn find_nonce(
        commitment: &Commitment<BS, Sha2Hash<BS>>,
        proof: &Proof<BS, Sha2Hash<BS>>,
        valid: bool,
    ) -> u64 {
        let seed = commitment.query_seed();
        (proof.nonce + 1..)
            .find(|&nonce| {
                let hash = proof_of_work::<BS, Sha2Hash<BS>>(seed, nonce);
                check_proof_of_work(hash, proof.options.grinding_bits) == valid
            })
            .unwrap()
    }

    #[test]
    fn test_transcript_dst() {
        assert_eq!(
            *TRANSCRIPT_DST,
            parse("0x09f36235476a658841de9bcdd34e1ac31ec792e41def5de31aecb4eb3bb1816b")
        );
    }

    #[test]
    fn test_query_dst0() {
        assert_eq!(
            *QUERY_DST0,
            parse("0xbbec7289b9fc3aade75412c031b62a769b205d1d73b29c9a06dbb91943e046bc")
        );
    }

    #[test]
    fn test_query_dst1() {
        assert_eq!(
            *QUERY_DST1,
            parse("0x88209086b178c9f2fb9c2f813cd229e1a2528cb4f5e6cd618710dc9869f14ac5")
        );
    }

    #[test]
    fn test_rlc_dst() {
        assert_eq!(
            *RLC_DST,
            parse("0x688ae37e5f05871810e7c6777e1c16c55ef4b14f072357586d7a298cefc11368")
        );
    }

    #[test]
    fn test_quotient_dst() {
        assert_eq!(
            *QUOTIENT_DST,
            parse("0xa7f91ddaa061a34288a9146bf5afc157211da6ec200650dd93daa9cb4c5b4808")
        );
    }

    #[test]
    fn test_pow_dst() {
        assert_eq!(
            *POW_DST,
            parse("0xd659450ee50ebd3de60604eff7a514c5873b63b66073b1164b1cb61b7b113a51")
        );
    }

    fn test_num_queries_impl(blowup_log2: usize, grinding_bits: usize, expected: usize) {
        let result = num_queries(blowup_log2, grinding_bits);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_num_queries_without_grinding() {
        test_num_queries_impl(1, 0, 128);
        test_num_queries_impl(4, 0, 32);
        test_num_queries_impl(6, 0, 22);
    }

    #[test]
    fn test_num_queries_with_grinding() {
        test_num_queries_impl(4, 1, 32);
        test_num_queries_impl(4, 4, 31);
        test_num_queries_impl(4, 20, 27);
        test_num_queries_impl(6, 20, 18);
    }

    #[test]
    fn test_num_queries_with_full_grinding() {
        test_num_queries_impl(4, 128, 1);
        test_num_queries_impl(4, 256, 1);
    }

    #[test]
    fn test_options_extended_domain_size() {
        let options = Options {
            degree_bound: 4096,
            blowup_log2: 6,
            grinding_bits: 20,
        };
        assert_eq!(options.extended_domain_size(), 1 << 18);
    }

    #[test]
    fn test_options_num_queries_without_grinding() {
        let options = Options {
            degree_bound: 4096,
            blowup_log2: 6,
            grinding_bits: 0,
        };
        assert_eq!(options.num_queries(), 22);
    }

    #[test]
    fn test_options_num_queries_with_grinding() {
        let options = Options {
            degree_bound: 4096,
            blowup_log2: 6,
            grinding_bits: 20,
        };
        assert_eq!(options.num_queries(), 18);
    }

    fn test_valid_options_impl<F: Field>(
        degree_bound: usize,
        blowup_log2: usize,
        grinding_bits: usize,
    ) {
        let options = Options {
            degree_bound,
            blowup_log2,
            grinding_bits,
        };
        let result = options.validate::<F>();
        assert!(result.is_ok());
    }

    fn test_invalid_options_impl<F: Field>(
        degree_bound: usize,
        blowup_log2: usize,
        grinding_bits: usize,
    ) {
        let options = Options {
            degree_bound,
            blowup_log2,
            grinding_bits,
        };
        let result = options.validate::<F>();
        assert!(result.is_err());
    }

    #[test]
    fn test_valid_options() {
        test_valid_options_impl::<BS>(4, 2, 8);
        test_valid_options_impl::<GL4>(4, 2, 8);
    }

    #[test]
    fn test_options_with_largest_domain() {
        test_valid_options_impl::<BS>(1 << 59, 3, 0);
        test_valid_options_impl::<GL4>(1 << 29, 3, 0);
    }

    #[test]
    fn test_options_with_most_grinding() {
        test_valid_options_impl::<BS>(4, 2, 127);
        test_valid_options_impl::<GL4>(4, 2, 128);
    }

    #[test]
    fn test_options_with_zero_degree_bound() {
        test_invalid_options_impl::<BS>(0, 2, 0);
        test_invalid_options_impl::<GL4>(0, 2, 0);
    }

    #[test]
    fn test_options_with_degree_bound_not_power_of_two() {
        test_invalid_options_impl::<BS>(6, 2, 0);
        test_invalid_options_impl::<GL4>(6, 2, 0);
    }

    #[test]
    fn test_options_with_zero_blowup_factor() {
        test_invalid_options_impl::<BS>(4, 0, 0);
        test_invalid_options_impl::<GL4>(4, 0, 0);
    }

    #[test]
    fn test_options_with_oversized_domain() {
        test_invalid_options_impl::<BS>(1 << 60, 3, 0);
        test_invalid_options_impl::<GL4>(1 << 30, 3, 0);
    }

    #[test]
    fn test_options_with_excessive_grinding() {
        test_invalid_options_impl::<BS>(4, 2, 128);
        test_invalid_options_impl::<GL4>(4, 2, 129);
    }

    #[test]
    fn test_proof_of_work_sha2() {
        let hash = proof_of_work::<BS, Sha2Hash<BS>>(H256::repeat_byte(0x12), 34);
        assert_eq!(
            hash,
            parse("0x751c67edba4c1538ab9e01f62307319cbc97f2bd598d3a55686031fb84e96de6")
        );
    }

    #[test]
    fn test_proof_of_work_keccak256() {
        let hash = proof_of_work::<BS, Keccak256Hash<BS>>(H256::repeat_byte(0x12), 34);
        assert_eq!(
            hash,
            parse("0x4bd47978101f127ba6b5f99dff54c6110e351615a65e4f8a9fbfc9003ceaa5ec")
        );
    }

    fn test_check_proof_of_work_impl(hash: H256, grinding_bits: usize, expected: bool) {
        let result = check_proof_of_work(hash, grinding_bits);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_check_proof_of_work_without_leading_zeros() {
        let hash = H256::repeat_byte(0xFF);
        test_check_proof_of_work_impl(hash, 0, true);
        test_check_proof_of_work_impl(hash, 1, false);
    }

    #[test]
    fn test_check_proof_of_work_with_19_leading_zeros() {
        let hash = parse("0x00001fffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
        test_check_proof_of_work_impl(hash, 19, true);
        test_check_proof_of_work_impl(hash, 20, false);
    }

    #[test]
    fn test_check_proof_of_work_with_255_leading_zeros() {
        let hash = H256::from_low_u64_be(1);
        test_check_proof_of_work_impl(hash, 255, true);
        test_check_proof_of_work_impl(hash, 256, false);
    }

    #[test]
    fn test_check_zero_proof_of_work() {
        let hash = H256::zero();
        test_check_proof_of_work_impl(hash, 256, true);
        test_check_proof_of_work_impl(hash, 257, false);
    }

    #[test]
    fn test_query_indices() {
        let options = Options {
            degree_bound: 4,
            blowup_log2: 4,
            grinding_bits: 104,
        };
        let indices = get_query_indices::<BS, Sha2Hash<BS>>(H256::repeat_byte(0x34), &options);
        assert_eq!(indices, [27, 59, 7, 21, 14, 33]);
    }

    #[test]
    fn test_points_off_domain() {
        let points = make_points::<BS>(&SAMPLE_POINTS);
        let result = check_points_off_domain(points, 16);
        assert!(result.is_ok());
    }

    #[test]
    fn test_point_on_domain() {
        let points = [BS::from_const(123), Polynomial::<BS>::coset_element2(3, 16)];
        let result = check_points_off_domain(points, 16);
        assert!(result.is_err());
    }

    #[test]
    fn test_rlc() {
        let values = [BS::from_const(1), BS::from_const(2), BS::from_const(3)];
        let result = rlc(values, BS::from_const(10));
        assert_eq!(result, BS::from_const(321));
    }

    #[test]
    fn test_rlc_of_no_values() {
        let result = rlc([], BS::from_const(10));
        assert_eq!(result, BS::ZERO);
    }

    #[test]
    fn test_deep_challenges() {
        let tree_roots = [H256::repeat_byte(0x56), H256::repeat_byte(0x78)];
        let points = BTreeMap::from([
            (
                BS::from_const(123),
                vec![BS::from_const(1), BS::from_const(2), BS::from_const(3)],
            ),
            (
                BS::from_const(456),
                vec![BS::from_const(4), BS::from_const(5), BS::from_const(6)],
            ),
        ]);
        let (alpha, beta) = deep_challenges::<BS, Sha2Hash<BS>>(&tree_roots, 3, &points);
        assert_eq!(
            alpha,
            parse("0x097f51584f3018cfb58caa2989196f63076af96bff20325da17bf923ef269335")
        );
        assert_eq!(
            beta,
            parse("0x152a37a010a775d0c161debac79b64a26f915125a5e1fff14b875fc51d9301f7")
        );
    }

    #[test]
    #[should_panic]
    fn test_committer_with_invalid_options() {
        Committer::<BS, Sha2Hash<BS>>::new(
            Options {
                degree_bound: 6,
                blowup_log2: 2,
                grinding_bits: 0,
            },
            make_batch(&[&[12, 34]]),
        );
    }

    #[test]
    #[should_panic]
    fn test_committer_without_polynomials() {
        Committer::<BS, Sha2Hash<BS>>::new(OPTIONS, vec![]);
    }

    #[test]
    fn test_committer_options() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        assert_eq!(*committer.options(), OPTIONS);
    }

    #[test]
    fn test_committer_num_trees() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        assert_eq!(committer.num_trees(), 2);
    }

    #[test]
    fn test_committer_root_hashes() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        assert_eq!(
            committer.root_hash(0),
            parse("0x8a560267cca9607a73ad5186d575443499a19685f460be07ff3928387dcb5a48")
        );
        assert_eq!(
            committer.root_hash(1),
            parse("0x917d1c95f6c177b6a0ff720557fa391335d0e6710f0029c60bbef44234a03b85")
        );
    }

    #[test]
    fn test_committer_transcript_hash_after_first_batch() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES[0..1]);
        let hash = committer.transcript_hash();
        assert_eq!(
            hash,
            parse("0x608c3f1c8a1354d04f2b596ff716e064e3d98d59ff9523bb2e8ecd1061c81816")
        );
    }

    #[test]
    fn test_committer_transcript_hash_after_second_batch() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let hash = committer.transcript_hash();
        assert_eq!(
            hash,
            parse("0xabc67f0c49806951ed936b32229b6fac2095f57a4524231fde67f63932692719")
        );
    }

    #[test]
    fn test_add_batch() {
        let mut committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let index = committer.add_batch(make_batch(&[&[12, 34]]));
        assert_eq!(index, 2);
        assert_eq!(committer.num_trees(), 3);
    }

    #[test]
    #[should_panic]
    fn test_add_empty_batch() {
        let mut committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        committer.add_batch(vec![]);
    }

    #[test]
    #[should_panic]
    fn test_add_batch_exceeding_degree_bound() {
        let mut committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        committer.add_batch(make_batch(&[&[12, 34, 56, 78, 90]]));
    }

    #[test]
    #[should_panic]
    fn test_commit_point_on_domain() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let z = Polynomial::<BS>::coset_element2(3, OPTIONS.extended_domain_size());
        committer.commit(BTreeSet::from([z]));
    }

    #[test]
    fn test_commitment_tree_roots() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit(make_points(&SAMPLE_POINTS));
        assert_eq!(
            commitment.tree_roots(),
            [
                parse::<H256>("0x8a560267cca9607a73ad5186d575443499a19685f460be07ff3928387dcb5a48"),
                parse("0x917d1c95f6c177b6a0ff720557fa391335d0e6710f0029c60bbef44234a03b85")
            ]
        );
    }

    #[test]
    fn test_commitment_batch_sizes() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit(make_points(&SAMPLE_POINTS));
        assert_eq!(commitment.batch_sizes(), [2, 1]);
    }

    #[test]
    fn test_commitment_degree_bound() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit(make_points(&SAMPLE_POINTS));
        assert_eq!(commitment.degree_bound(), 4);
    }

    #[test]
    fn test_commitment_fri_roots() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit(make_points(&SAMPLE_POINTS));
        assert_eq!(
            commitment.inner.roots(),
            [
                parse::<H256>("0x1cfc697bb66e15c8357f7a749010a210ae829f520944e21e4df00cfa7212ec8c"),
                parse("0xc61b2df672d90a4aef984d6b7d485f673cec75f1868d39e6ba659a4ec3883f6f"),
                parse("0xb3eb7eed49ad1119daa31ad838a2a4bb9905657869e618cc497611bbadbbd074"),
            ]
        );
    }

    #[test]
    fn test_commitment_transcript_hash_of_first_batch() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit(make_points(&SAMPLE_POINTS));
        let hash = commitment.transcript_hash(1);
        assert_eq!(
            hash,
            parse("0x608c3f1c8a1354d04f2b596ff716e064e3d98d59ff9523bb2e8ecd1061c81816")
        );
    }

    #[test]
    fn test_commitment_transcript_hash_of_all_batches() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit(make_points(&SAMPLE_POINTS));
        let hash = commitment.transcript_hash(2);
        assert_eq!(
            hash,
            parse("0xabc67f0c49806951ed936b32229b6fac2095f57a4524231fde67f63932692719")
        );
    }

    #[test]
    #[should_panic]
    fn test_commitment_transcript_hash_of_no_batches() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit(make_points(&SAMPLE_POINTS));
        commitment.transcript_hash(0);
    }

    #[test]
    #[should_panic]
    fn test_commitment_transcript_hash_of_too_many_batches() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit(make_points(&SAMPLE_POINTS));
        commitment.transcript_hash(3);
    }

    #[test]
    fn test_commitment_query_seed() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit(make_points(&SAMPLE_POINTS));
        let seed = commitment.query_seed();
        assert_eq!(
            seed,
            parse("0x085c3c13977971933e24213578667842fb22b1068841a1ce7edbccaaeb7ded6d")
        );
    }

    #[test]
    fn test_commitment_to_proto() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit(make_points(&SAMPLE_POINTS));
        let proto = commitment.to_proto();
        assert_eq!(
            proto.batches,
            [
                proto::commitment::Batch {
                    root: commitment.tree_roots[0].as_bytes().to_vec(),
                    num_polys: 2,
                },
                proto::commitment::Batch {
                    root: commitment.tree_roots[1].as_bytes().to_vec(),
                    num_polys: 1,
                },
            ]
        );
        assert_eq!(
            proto.fri_roots,
            [
                commitment.inner.roots()[0].as_bytes().to_vec(),
                commitment.inner.roots()[1].as_bytes().to_vec(),
                commitment.inner.roots()[2].as_bytes().to_vec(),
            ]
        );
    }

    #[test]
    fn test_commitment_from_proto() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit(make_points(&SAMPLE_POINTS));
        let proto = commitment.to_proto();
        let decoded = Commitment::<BS, Sha2Hash<BS>>::from_proto(&proto).unwrap();
        assert_eq!(decoded, commitment);
    }

    #[test]
    fn test_reject_commitment_without_batches() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit(make_points(&SAMPLE_POINTS));
        let mut proto = commitment.to_proto();
        proto.batches.clear();
        let result = Commitment::<BS, Sha2Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_commitment_without_fri_roots() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit(make_points(&SAMPLE_POINTS));
        let mut proto = commitment.to_proto();
        proto.fri_roots.clear();
        let result = Commitment::<BS, Sha2Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_commitment_with_short_root() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit(make_points(&SAMPLE_POINTS));
        let mut proto = commitment.to_proto();
        proto.batches[1].root.pop();
        let result = Commitment::<BS, Sha2Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_commitment_with_short_fri_root() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit(make_points(&SAMPLE_POINTS));
        let mut proto = commitment.to_proto();
        proto.fri_roots[1].pop();
        let result = Commitment::<BS, Sha2Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_commitment_without_hash_backend() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit(make_points(&SAMPLE_POINTS));
        let mut proto = commitment.to_proto();
        proto.hash_backend = None;
        let result = Commitment::<BS, Sha2Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_commitment_with_wrong_hash_backend() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit(make_points(&SAMPLE_POINTS));
        let proto = commitment.to_proto();
        let result = Commitment::<BS, Keccak256Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_prover_options() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (_, prover) = committer.commit(make_points(&SAMPLE_POINTS));
        assert_eq!(*prover.options(), OPTIONS);
    }

    #[test]
    fn test_prover_num_polys() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (_, prover) = committer.commit(make_points(&SAMPLE_POINTS));
        assert_eq!(prover.num_polys(), 3);
    }

    #[test]
    fn test_prover_num_trees() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (_, prover) = committer.commit(make_points(&SAMPLE_POINTS));
        assert_eq!(prover.num_trees(), 2);
    }

    #[test]
    fn test_prover_root_hashes() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (_, prover) = committer.commit(make_points(&SAMPLE_POINTS));
        assert_eq!(
            prover.root_hash(0),
            parse("0x8a560267cca9607a73ad5186d575443499a19685f460be07ff3928387dcb5a48")
        );
        assert_eq!(
            prover.root_hash(1),
            parse("0x917d1c95f6c177b6a0ff720557fa391335d0e6710f0029c60bbef44234a03b85")
        );
    }

    #[test]
    fn test_prover_points() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (_, prover) = committer.commit(make_points(&SAMPLE_POINTS));
        assert_eq!(
            *prover.points(),
            BTreeMap::from([
                (
                    BS::from_const(123),
                    vec![
                        BS::from_const(145999044),
                        BS::from_const(84410022),
                        BS::from_const(64126386),
                    ]
                ),
                (
                    BS::from_const(456),
                    vec![
                        BS::from_const(7407527580),
                        BS::from_const(4276015554),
                        BS::from_const(3235519818),
                    ]
                ),
            ])
        );
    }

    #[test]
    fn test_proof_options() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, prover) = committer.commit(make_points(&SAMPLE_POINTS));
        let proof = prover.prove(&commitment);
        assert_eq!(*proof.options(), OPTIONS);
    }

    #[test]
    fn test_proof_num_polys() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, prover) = committer.commit(make_points(&SAMPLE_POINTS));
        let proof = prover.prove(&commitment);
        assert_eq!(proof.num_polys(), 3);
    }

    #[test]
    fn test_proof_points() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, prover) = committer.commit(make_points(&SAMPLE_POINTS));
        let proof = prover.prove(&commitment);
        assert_eq!(proof.points(), prover.points());
    }

    #[test]
    fn test_proof_nonce_without_grinding() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, prover) = committer.commit(make_points(&SAMPLE_POINTS));
        let proof = prover.prove(&commitment);
        assert_eq!(proof.nonce(), 0);
    }

    #[test]
    fn test_proof_nonce_with_grinding() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(GROUND_OPTIONS, &SAMPLE_BATCHES);
        let (commitment, prover) = committer.commit(make_points(&SAMPLE_POINTS));
        let proof = prover.prove(&commitment);
        assert_eq!(proof.nonce(), 70);
    }

    #[test]
    fn test_proof_size_without_grinding() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, prover) = committer.commit(make_points(&SAMPLE_POINTS));
        let proof = prover.prove(&commitment);
        assert_eq!(proof.openings.len(), 64);
        assert_eq!(proof.queries.len(), 64);
    }

    #[test]
    fn test_proof_size_with_grinding() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(GROUND_OPTIONS, &SAMPLE_BATCHES);
        let (commitment, prover) = committer.commit(make_points(&SAMPLE_POINTS));
        let proof = prover.prove(&commitment);
        assert_eq!(proof.openings.len(), 60);
        assert_eq!(proof.queries.len(), 60);
    }

    fn test_honest_proof_impl<F: Field256, H: Hasher<F>>(
        batches: &[&[&[u64]]],
        points: &[u64],
        options: Options,
    ) {
        let committer = make_committer::<F, H>(options, batches);
        let (commitment, prover) = committer.commit(make_points(points));
        let proof = prover.prove(&commitment);
        let result = proof.verify(&commitment);
        assert!(result.is_ok());
    }

    fn test_honest_proof_with_options(batches: &[&[&[u64]]], points: &[u64], options: Options) {
        test_honest_proof_impl::<BS, Sha2Hash<BS>>(batches, points, options);
        test_honest_proof_impl::<BS, Keccak256Hash<BS>>(batches, points, options);
        test_honest_proof_impl::<GL4, Sha2Hash<GL4>>(batches, points, options);
        test_honest_proof_impl::<GL4, Keccak256Hash<GL4>>(batches, points, options);
    }

    fn test_honest_proof(batches: &[&[&[u64]]], points: &[u64], degree_bound: usize) {
        test_honest_proof_with_options(
            batches,
            points,
            Options {
                degree_bound,
                ..OPTIONS
            },
        );
    }

    #[test]
    fn test_one_constant_polynomial_one_point_1() {
        test_honest_proof(&[&[&[12]]], &[123], 1);
    }

    #[test]
    fn test_one_constant_polynomial_one_point_2() {
        test_honest_proof(&[&[&[12]]], &[321], 1);
    }

    #[test]
    fn test_one_constant_polynomial_one_point_3() {
        test_honest_proof(&[&[&[34]]], &[123], 1);
    }

    #[test]
    fn test_one_constant_polynomial_two_points() {
        test_honest_proof(&[&[&[12]]], &[123, 456], 1);
    }

    #[test]
    fn test_one_constant_polynomial_three_points() {
        test_honest_proof(&[&[&[12]]], &[789, 456, 123], 1);
    }

    #[test]
    fn test_one_polynomial_degree_one_one_point_1() {
        test_honest_proof(&[&[&[12, 34]]], &[123], 2);
    }

    #[test]
    fn test_one_polynomial_degree_one_one_point_2() {
        test_honest_proof(&[&[&[12, 34]]], &[321], 2);
    }

    #[test]
    fn test_one_polynomial_degree_one_one_point_3() {
        test_honest_proof(&[&[&[34, 56]]], &[123], 2);
    }

    #[test]
    fn test_one_polynomial_degree_one_two_points() {
        test_honest_proof(&[&[&[12, 34]]], &[123, 456], 2);
    }

    #[test]
    fn test_one_polynomial_degree_one_three_points() {
        test_honest_proof(&[&[&[12, 34]]], &[789, 456, 123], 2);
    }

    #[test]
    fn test_two_polynomials_degree_three_one_point_1() {
        test_honest_proof(&[&[&[12, 34, 56, 78], &[42, 43, 44, 45]]], &[123], 4);
    }

    #[test]
    fn test_two_polynomials_degree_three_one_point_2() {
        test_honest_proof(&[&[&[12, 34, 56, 78], &[42, 43, 44, 45]]], &[321], 4);
    }

    #[test]
    fn test_two_polynomials_degree_three_one_point_3() {
        test_honest_proof(&[&[&[45, 44, 43, 42], &[78, 56, 34, 12]]], &[123], 4);
    }

    #[test]
    fn test_two_polynomials_degree_three_two_points() {
        test_honest_proof(&[&[&[12, 34, 56, 78], &[42, 43, 44, 45]]], &[123, 456], 4);
    }

    #[test]
    fn test_two_polynomials_degree_three_three_points() {
        test_honest_proof(
            &[&[&[12, 34, 56, 78], &[42, 43, 44, 45]]],
            &[789, 456, 123],
            4,
        );
    }

    #[test]
    fn test_two_batches_one_and_one() {
        test_honest_proof(
            &[&[&[12, 34, 56, 78]], &[&[42, 43, 44, 45]]],
            &[123, 456],
            4,
        );
    }

    #[test]
    fn test_two_batches_two_and_one() {
        test_honest_proof(
            &[
                &[&[12, 34, 56, 78], &[90, 78, 56, 34]],
                &[&[42, 43, 44, 45]],
            ],
            &[456, 789],
            4,
        );
    }

    #[test]
    fn test_two_batches_one_and_two() {
        test_honest_proof(
            &[
                &[&[90, 78, 56, 34]],
                &[&[12, 34, 56, 78], &[42, 43, 44, 45]],
            ],
            &[456, 789],
            4,
        );
    }

    #[test]
    fn test_polynomial_below_degree_bound() {
        test_honest_proof(&[&[&[12, 34]]], &[123, 456], 8);
    }

    #[test]
    fn test_honest_proof_blowup_2() {
        test_honest_proof_with_options(
            &[&[&[12]]],
            &[123],
            Options {
                degree_bound: 1,
                blowup_log2: 1,
                grinding_bits: 0,
            },
        );
        test_honest_proof_with_options(
            &SAMPLE_BATCHES,
            &SAMPLE_POINTS,
            Options {
                blowup_log2: 1,
                ..OPTIONS
            },
        );
    }

    #[test]
    fn test_honest_proof_blowup_8() {
        test_honest_proof_with_options(
            &[&[&[12]]],
            &[123],
            Options {
                degree_bound: 1,
                blowup_log2: 3,
                grinding_bits: 0,
            },
        );
        test_honest_proof_with_options(
            &SAMPLE_BATCHES,
            &SAMPLE_POINTS,
            Options {
                blowup_log2: 3,
                ..OPTIONS
            },
        );
    }

    #[test]
    fn test_honest_proof_with_grinding() {
        test_honest_proof_with_options(
            &[&[&[12]]],
            &[123],
            Options {
                degree_bound: 1,
                ..GROUND_OPTIONS
            },
        );
        test_honest_proof_with_options(&SAMPLE_BATCHES, &SAMPLE_POINTS, GROUND_OPTIONS);
    }

    #[test]
    fn test_accept_untampered_proof() {
        let (commitment, proof) = sample_proof(OPTIONS);
        let result = proof.verify(&commitment);
        assert!(result.is_ok());
    }

    #[test]
    fn test_accept_untampered_ground_proof() {
        let (commitment, proof) = sample_proof(GROUND_OPTIONS);
        let result = proof.verify(&commitment);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reject_mismatched_degree_bound() {
        let (commitment, mut proof) = sample_proof(OPTIONS);
        proof.options.degree_bound *= 2;
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_on_domain_point() {
        let (commitment, mut proof) = sample_proof(OPTIONS);
        let values = proof.points[&BS::from_const(123)].clone();
        let z = Polynomial::<BS>::coset_element2(0, OPTIONS.extended_domain_size());
        proof.points = BTreeMap::from([(z, values)]);
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_tampered_evaluation() {
        let (commitment, mut proof) = sample_proof(OPTIONS);
        proof.points.get_mut(&BS::from_const(123)).unwrap()[0] += BS::ONE;
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_no_points() {
        let (commitment, mut proof) = sample_proof(OPTIONS);
        proof.points.clear();
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_missing_evaluation() {
        let (commitment, mut proof) = sample_proof(OPTIONS);
        proof.points.get_mut(&BS::from_const(123)).unwrap().pop();
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_extra_evaluation() {
        let (commitment, mut proof) = sample_proof(OPTIONS);
        proof
            .points
            .get_mut(&BS::from_const(123))
            .unwrap()
            .push(BS::ZERO);
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_extra_point() {
        let (commitment, mut proof) = sample_proof(OPTIONS);
        let values = proof.points[&BS::from_const(123)].clone();
        proof.points.insert(BS::from_const(789), values);
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_swapped_evaluations() {
        let (commitment, mut proof) = sample_proof(OPTIONS);
        let values = proof.points[&BS::from_const(123)].clone();
        let other_values = proof.points.insert(BS::from_const(456), values).unwrap();
        proof.points.insert(BS::from_const(123), other_values);
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_foreign_commitment() {
        let (_, proof) = sample_proof(OPTIONS);
        let committer = make_committer::<BS, Sha2Hash<BS>>(
            OPTIONS,
            &[
                &[&[99, 98, 97, 96], &[95, 94, 93, 92]],
                &[&[91, 90, 89, 88]],
            ],
        );
        let (foreign, _) = committer.commit(make_points(&SAMPLE_POINTS));
        let result = proof.verify(&foreign);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_misreported_batch_sizes() {
        let (mut commitment, proof) = sample_proof(OPTIONS);
        commitment.batch_sizes = vec![1, 2];
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_wrong_number_of_polynomials() {
        let (commitment, mut proof) = sample_proof(OPTIONS);
        proof.num_polys += 1;
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_missing_query() {
        let (commitment, mut proof) = sample_proof(OPTIONS);
        proof.queries.pop();
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_missing_openings() {
        let (commitment, mut proof) = sample_proof(OPTIONS);
        proof.openings.pop();
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_missing_opening_within_query() {
        let (commitment, mut proof) = sample_proof(OPTIONS);
        proof.openings[0].pop();
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_swapped_openings() {
        let (commitment, mut proof) = sample_proof(OPTIONS);
        proof.openings.swap(0, 1);
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_opening_from_wrong_domain() {
        let (commitment, mut proof) = sample_proof(OPTIONS);
        let committer = make_committer::<BS, Sha2Hash<BS>>(
            Options {
                degree_bound: 8,
                ..OPTIONS
            },
            &[&[&[11, 22, 33, 44]]],
        );
        let (other_commitment, other_prover) = committer.commit(make_points(&[123]));
        let mut other_proof = other_prover.prove(&other_commitment);
        proof.openings[0][0] = other_proof.openings[0].remove(0);
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_invalid_nonce() {
        let (commitment, mut proof) = sample_proof(GROUND_OPTIONS);
        proof.nonce = find_nonce(&commitment, &proof, false);
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_other_valid_nonce() {
        let (commitment, mut proof) = sample_proof(GROUND_OPTIONS);
        proof.nonce = find_nonce(&commitment, &proof, true);
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_overstated_grinding() {
        let (commitment, mut proof) = sample_proof(GROUND_OPTIONS);
        proof.options.grinding_bits = BS::NUM_BITS / 2;
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_understated_grinding() {
        let (commitment, mut proof) = sample_proof(GROUND_OPTIONS);
        proof.options.grinding_bits = 0;
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_proof_to_proto() {
        let (_, proof) = sample_proof(GROUND_OPTIONS);
        let proto = proof.to_proto();
        assert_eq!(proto.degree_bound, 4);
        assert_eq!(proto.blowup_log2, 2);
        assert_eq!(proto.grinding_bits, 8);
        assert_eq!(proto.nonce, 70);
        assert_eq!(proto.num_polys, 3);
        assert_eq!(
            proto.points,
            [
                proto::proof::Point {
                    z: BS::from_const(123).to_le_bytes().to_vec(),
                    v: vec![
                        BS::from_const(145999044).to_le_bytes().to_vec(),
                        BS::from_const(84410022).to_le_bytes().to_vec(),
                        BS::from_const(64126386).to_le_bytes().to_vec(),
                    ],
                },
                proto::proof::Point {
                    z: BS::from_const(456).to_le_bytes().to_vec(),
                    v: vec![
                        BS::from_const(7407527580).to_le_bytes().to_vec(),
                        BS::from_const(4276015554).to_le_bytes().to_vec(),
                        BS::from_const(3235519818).to_le_bytes().to_vec(),
                    ],
                },
            ]
        );
        assert_eq!(proto.openings.len(), 60);
        assert_eq!(proto.queries.len(), 60);
    }

    #[test]
    fn test_proof_from_proto() {
        let (_, proof) = sample_proof(GROUND_OPTIONS);
        let proto = proof.to_proto();
        let decoded = Proof::<BS, Sha2Hash<BS>>::from_proto(&proto).unwrap();
        assert_eq!(decoded.to_proto(), proto);
    }

    #[test]
    fn test_accept_decoded_proof() {
        let (commitment, proof) = sample_proof(GROUND_OPTIONS);
        let commitment_proto = commitment.to_proto();
        let proof_proto = proof.to_proto();
        let commitment = Commitment::<BS, Sha2Hash<BS>>::from_proto(&commitment_proto).unwrap();
        let proof = Proof::<BS, Sha2Hash<BS>>::from_proto(&proof_proto).unwrap();
        let result = proof.verify(&commitment);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reject_zero_degree_bound() {
        let (_, proof) = sample_proof(OPTIONS);
        let mut proto = proof.to_proto();
        proto.degree_bound = 0;
        let result = Proof::<BS, Sha2Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_zero_blowup_factor() {
        let (_, proof) = sample_proof(OPTIONS);
        let mut proto = proof.to_proto();
        proto.blowup_log2 = 0;
        let result = Proof::<BS, Sha2Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_oversized_domain() {
        let (_, proof) = sample_proof(OPTIONS);
        let mut proto = proof.to_proto();
        proto.blowup_log2 = 64;
        let result = Proof::<BS, Sha2Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_excessive_grinding() {
        let (_, proof) = sample_proof(GROUND_OPTIONS);
        let mut proto = proof.to_proto();
        proto.grinding_bits = 128;
        let result = Proof::<BS, Sha2Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_non_canonical_point() {
        let (_, proof) = sample_proof(OPTIONS);
        let mut proto = proof.to_proto();
        let modulus: U256 = BS::MODULUS.parse().unwrap();
        proto.points[0].z = modulus.to_little_endian().to_vec();
        let result = Proof::<BS, Sha2Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_non_canonical_evaluation() {
        let (_, proof) = sample_proof(OPTIONS);
        let mut proto = proof.to_proto();
        let modulus: U256 = BS::MODULUS.parse().unwrap();
        proto.points[1].v[2] = modulus.to_little_endian().to_vec();
        let result = Proof::<BS, Sha2Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_proof_without_hash_backend() {
        let (_, proof) = sample_proof(OPTIONS);
        let mut proto = proof.to_proto();
        proto.hash_backend = None;
        let result = Proof::<BS, Sha2Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_proof_with_wrong_hash_backend() {
        let (_, proof) = sample_proof(OPTIONS);
        let proto = proof.to_proto();
        let result = Proof::<BS, Keccak256Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }
}
