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
            vec![quotient],
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

            let (quotients, _) = query.values();
            if quotients.len() != 1 {
                return Err(anyhow!(
                    "the FRI query at index {index} must open exactly one polynomial (got {})",
                    quotients.len()
                ));
            }

            let x = query.x();
            let expected = rlc(
                self.points.iter().map(|(&z, values)| {
                    let v = rlc(values.iter().copied(), alpha);
                    (combined - v) * (x - z).invert_vartime().unwrap()
                }),
                beta,
            );
            if quotients[0] != expected {
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

    #[test]
    fn test_dsts() {
        assert_eq!(
            *TRANSCRIPT_DST,
            parse("0x09f36235476a658841de9bcdd34e1ac31ec792e41def5de31aecb4eb3bb1816b")
        );
        assert_eq!(
            *QUERY_DST0,
            parse("0xbbec7289b9fc3aade75412c031b62a769b205d1d73b29c9a06dbb91943e046bc")
        );
        assert_eq!(
            *QUERY_DST1,
            parse("0x88209086b178c9f2fb9c2f813cd229e1a2528cb4f5e6cd618710dc9869f14ac5")
        );
        assert_eq!(
            *RLC_DST,
            parse("0x688ae37e5f05871810e7c6777e1c16c55ef4b14f072357586d7a298cefc11368")
        );
        assert_eq!(
            *QUOTIENT_DST,
            parse("0xa7f91ddaa061a34288a9146bf5afc157211da6ec200650dd93daa9cb4c5b4808")
        );
        assert_eq!(
            *POW_DST,
            parse("0xd659450ee50ebd3de60604eff7a514c5873b63b66073b1164b1cb61b7b113a51")
        );
    }

    #[test]
    fn test_num_queries() {
        assert_eq!(num_queries(1, 0), 128);
        assert_eq!(num_queries(4, 0), 32);
        assert_eq!(num_queries(6, 0), 22);
        assert_eq!(num_queries(4, 20), 27);
        assert_eq!(num_queries(6, 20), 18);
        assert_eq!(num_queries(4, 1), 32);
        assert_eq!(num_queries(4, 4), 31);
        assert_eq!(num_queries(4, 128), 1);
        assert_eq!(num_queries(4, 256), 1);
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
    fn test_options_num_queries() {
        let options = Options {
            degree_bound: 4096,
            blowup_log2: 6,
            grinding_bits: 0,
        };
        assert_eq!(options.num_queries(), 22);
        assert_eq!(
            Options {
                grinding_bits: 20,
                ..options
            }
            .num_queries(),
            18
        );
    }

    #[test]
    fn test_valid_options() {
        let options = Options {
            degree_bound: 4,
            blowup_log2: 2,
            grinding_bits: 8,
        };
        assert!(options.validate::<BS>().is_ok());
        assert!(options.validate::<GL4>().is_ok());
        // The largest domain allowed by the 2-adicity of BlueSky (62) and Goldilocks (32).
        assert!(
            Options {
                degree_bound: 1 << 59,
                blowup_log2: 3,
                grinding_bits: 0,
            }
            .validate::<BS>()
            .is_ok()
        );
        assert!(
            Options {
                degree_bound: 1 << 29,
                blowup_log2: 3,
                grinding_bits: 0,
            }
            .validate::<GL4>()
            .is_ok()
        );
        // The most grinding allowed by BlueSky (255 bits) and Goldilocks^4 (256 bits).
        assert!(
            Options {
                grinding_bits: 127,
                ..options
            }
            .validate::<BS>()
            .is_ok()
        );
        assert!(
            Options {
                grinding_bits: 128,
                ..options
            }
            .validate::<GL4>()
            .is_ok()
        );
    }

    fn assert_invalid_options<F: Field>(options: Options, expected: &str) {
        let error = options
            .validate::<F>()
            .expect_err("the invalid options were accepted")
            .to_string();
        assert!(error.contains(expected), "unexpected error: {error}");
    }

    #[test]
    fn test_invalid_options() {
        let options = Options {
            degree_bound: 4,
            blowup_log2: 2,
            grinding_bits: 0,
        };
        assert_invalid_options::<BS>(
            Options {
                degree_bound: 0,
                ..options
            },
            "not a power of two",
        );
        assert_invalid_options::<BS>(
            Options {
                degree_bound: 6,
                ..options
            },
            "not a power of two",
        );
        assert_invalid_options::<BS>(
            Options {
                blowup_log2: 0,
                ..options
            },
            "invalid blowup factor",
        );
        assert_invalid_options::<BS>(
            Options {
                degree_bound: 1 << 60,
                blowup_log2: 3,
                grinding_bits: 0,
            },
            "invalid extended domain size",
        );
        assert_invalid_options::<GL4>(
            Options {
                degree_bound: 1 << 30,
                blowup_log2: 3,
                grinding_bits: 0,
            },
            "invalid extended domain size",
        );
        assert_invalid_options::<BS>(
            Options {
                grinding_bits: 128,
                ..options
            },
            "invalid number of grinding bits",
        );
        assert_invalid_options::<GL4>(
            Options {
                grinding_bits: 129,
                ..options
            },
            "invalid number of grinding bits",
        );
    }

    #[test]
    #[should_panic(expected = "invalid DEEP-FRI options")]
    fn test_committer_rejects_invalid_options() {
        Committer::<BS, Sha2Hash<BS>>::new(
            Options {
                degree_bound: 6,
                blowup_log2: 2,
                grinding_bits: 0,
            },
            vec![polynomial(&[12, 34])],
        );
    }

    #[test]
    fn test_check_proof_of_work() {
        let zero = H256::zero();
        let one = H256::from_low_u64_be(1);
        let top_bit = H256::from_slice(&(U256::one() << 255).to_big_endian());
        let byte = H256::from_slice(&(U256::one() << 236).to_big_endian());
        assert!(check_proof_of_work(top_bit, 0));
        assert!(!check_proof_of_work(top_bit, 1));
        assert!(check_proof_of_work(byte, 19));
        assert!(!check_proof_of_work(byte, 20));
        assert!(check_proof_of_work(one, 255));
        assert!(!check_proof_of_work(one, 256));
        assert!(check_proof_of_work(zero, 256));
        assert!(!check_proof_of_work(zero, 257));
    }

    fn test_prover_impl<F: Field256, H: Hasher<F>>(
        polynomial_batches: Vec<Vec<Polynomial<F>>>,
        points: &[u16],
        degree_bound: usize,
        blowup_log2: usize,
    ) {
        test_prover_with_grinding_impl::<F, H>(
            polynomial_batches.clone(),
            points,
            degree_bound,
            blowup_log2,
            0,
        );
        test_prover_with_grinding_impl::<F, H>(
            polynomial_batches,
            points,
            degree_bound,
            blowup_log2,
            8,
        );
    }

    fn test_prover_with_grinding_impl<F: Field256, H: Hasher<F>>(
        mut polynomial_batches: Vec<Vec<Polynomial<F>>>,
        points: &[u16],
        degree_bound: usize,
        blowup_log2: usize,
        grinding_bits: usize,
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
        let options = Options {
            degree_bound,
            blowup_log2,
            grinding_bits,
        };
        let mut committer = Committer::<F, H>::new(options, first_batch);
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
        assert_eq!(prover.options().degree_bound, degree_bound);
        assert_eq!(
            prover.options().extended_domain_size(),
            degree_bound << blowup_log2
        );
        assert_eq!(prover.num_polys(), num_polys);
        assert_eq!(prover.num_trees(), num_batches);
        assert_eq!(*prover.points(), points);
        let proof = prover.prove(&commitment);
        assert_eq!(proof.options().degree_bound, degree_bound);
        assert_eq!(proof.options().blowup_log2, blowup_log2);
        assert_eq!(
            proof.options().extended_domain_size(),
            degree_bound << blowup_log2
        );
        assert_eq!(proof.options().grinding_bits, grinding_bits);
        assert_eq!(proof.num_polys(), num_polys);
        assert_eq!(*proof.options(), options);
        assert!(check_proof_of_work(
            proof_of_work::<F, H>(commitment.query_seed(), proof.nonce()),
            grinding_bits
        ));
        assert_eq!(proof.queries.len(), num_queries(blowup_log2, grinding_bits));
        assert!(
            proof
                .queries
                .iter()
                .all(|query| query.values().0.len() == 1)
        );
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
    const ADVERSARIAL_GRINDING_BITS: usize = 8;

    fn polynomial(coefficients: &[u16]) -> Polynomial<BS> {
        Polynomial::with_coefficients(coefficients.iter().copied().map(BS::from).collect())
    }

    fn adversarial_setup() -> (Commitment<BS, Sha2Hash<BS>>, Proof<BS, Sha2Hash<BS>>) {
        adversarial_setup_with_grinding(0)
    }

    fn grinding_setup() -> (Commitment<BS, Sha2Hash<BS>>, Proof<BS, Sha2Hash<BS>>) {
        adversarial_setup_with_grinding(ADVERSARIAL_GRINDING_BITS)
    }

    fn adversarial_setup_with_grinding(
        grinding_bits: usize,
    ) -> (Commitment<BS, Sha2Hash<BS>>, Proof<BS, Sha2Hash<BS>>) {
        let mut committer = Committer::<BS, Sha2Hash<BS>>::new(
            Options {
                degree_bound: ADVERSARIAL_DEGREE_BOUND,
                blowup_log2: ADVERSARIAL_BLOWUP_LOG2,
                grinding_bits,
            },
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
    fn test_reject_mismatched_degree_bound() {
        let (commitment, mut proof) = adversarial_setup();
        proof.options.degree_bound *= 2;
        assert_rejected(proof.verify(&commitment), "FRI roots of the commitment");
    }

    #[test]
    fn test_reject_zero_degree_bound() {
        let (_, proof) = adversarial_setup();
        let mut proto = proof.to_proto();
        proto.degree_bound = 0;
        assert_rejected(
            Proof::<BS, Sha2Hash<BS>>::from_proto(&proto).map(|_| ()),
            "not a power of two",
        );
    }

    #[test]
    fn test_reject_zero_blowup_factor() {
        let (_, proof) = adversarial_setup();
        let mut proto = proof.to_proto();
        proto.blowup_log2 = 0;
        assert_rejected(
            Proof::<BS, Sha2Hash<BS>>::from_proto(&proto).map(|_| ()),
            "invalid blowup factor",
        );
    }

    #[test]
    fn test_reject_oversized_domain() {
        let (_, proof) = adversarial_setup();
        let mut proto = proof.to_proto();
        proto.blowup_log2 = 64;
        assert_rejected(
            Proof::<BS, Sha2Hash<BS>>::from_proto(&proto).map(|_| ()),
            "invalid extended domain size",
        );
    }

    #[test]
    fn test_reject_on_domain_point() {
        let (commitment, mut proof) = adversarial_setup();
        let values = proof.points.values().next().unwrap().clone();
        let on_domain = Polynomial::<BS>::coset_element2(0, proof.options().extended_domain_size());
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
    fn test_reject_no_points() {
        let (commitment, mut proof) = adversarial_setup();
        proof.points.clear();
        assert_rejected(
            proof.verify(&commitment),
            "the proof doesn't open any points",
        );
    }

    #[test]
    fn test_reject_missing_evaluation() {
        let (commitment, mut proof) = adversarial_setup();
        let z = *proof.points.keys().next().unwrap();
        proof.points.get_mut(&z).unwrap().pop();
        assert_rejected(proof.verify(&commitment), "incorrect number of evaluations");
    }

    #[test]
    fn test_reject_extra_evaluation() {
        let (commitment, mut proof) = adversarial_setup();
        let z = *proof.points.keys().next().unwrap();
        proof.points.get_mut(&z).unwrap().push(BS::ZERO);
        assert_rejected(proof.verify(&commitment), "incorrect number of evaluations");
    }

    #[test]
    fn test_reject_foreign_commitment() {
        let (_, proof) = adversarial_setup();
        let mut committer = Committer::<BS, Sha2Hash<BS>>::new(
            Options {
                degree_bound: ADVERSARIAL_DEGREE_BOUND,
                blowup_log2: ADVERSARIAL_BLOWUP_LOG2,
                grinding_bits: 0,
            },
            vec![polynomial(&[99, 98, 97, 96]), polynomial(&[95, 94, 93, 92])],
        );
        committer.add_batch(vec![polynomial(&[91, 90, 89, 88])]);
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
        assert_rejected(proof.verify(&commitment), "algebraic check failed");
    }

    #[test]
    fn test_reject_swapped_evaluations() {
        let (commitment, mut proof) = adversarial_setup();
        let mut values: Vec<Vec<BS>> = proof.points.values().cloned().collect();
        values.reverse();
        for (evaluations, swapped) in proof.points.values_mut().zip(values) {
            *evaluations = swapped;
        }
        assert_rejected(proof.verify(&commitment), "algebraic check failed");
    }

    #[test]
    fn test_reject_opening_from_wrong_domain() {
        let (commitment, mut proof) = adversarial_setup();
        let committer = Committer::<BS, Sha2Hash<BS>>::new(
            Options {
                degree_bound: ADVERSARIAL_DEGREE_BOUND * 2,
                blowup_log2: ADVERSARIAL_BLOWUP_LOG2,
                grinding_bits: 0,
            },
            vec![polynomial(&[11, 22, 33, 44])],
        );
        let (other_commitment, other_prover) = committer.commit(BTreeSet::from([BS::from(123u16)]));
        let mut other_proof = other_prover.prove(&other_commitment);
        proof.openings[0][0] = other_proof.openings[0].remove(0);
        assert_rejected(proof.verify(&commitment), "invalid opening for index");
    }

    #[test]
    fn test_reject_misreported_batch_sizes() {
        let (mut commitment, proof) = adversarial_setup();
        commitment.batch_sizes = vec![1, 2];
        assert_rejected(proof.verify(&commitment), "incorrect batch size");
    }

    #[test]
    fn test_reject_wrong_number_of_polynomials() {
        let (commitment, mut proof) = adversarial_setup();
        proof.num_polys += 1;
        assert_rejected(proof.verify(&commitment), "incorrect number of polynomials");
    }

    #[test]
    fn test_reject_commitment_without_batches() {
        let (commitment, _) = adversarial_setup();
        let mut proto = commitment.to_proto();
        proto.batches.clear();
        assert_rejected(
            Commitment::<BS, Sha2Hash<BS>>::from_proto(&proto).map(|_| ()),
            "the commitment has no polynomial batches",
        );
    }

    #[test]
    fn test_reject_commitment_without_fri_roots() {
        let (commitment, _) = adversarial_setup();
        let mut proto = commitment.to_proto();
        proto.fri_roots.clear();
        assert_rejected(
            Commitment::<BS, Sha2Hash<BS>>::from_proto(&proto).map(|_| ()),
            "the commitment has no FRI roots",
        );
    }

    #[test]
    fn test_reject_commitment_without_hash_backend() {
        let (commitment, _) = adversarial_setup();
        let mut proto = commitment.to_proto();
        proto.hash_backend = None;
        assert_rejected(
            Commitment::<BS, Sha2Hash<BS>>::from_proto(&proto).map(|_| ()),
            "hash backend not specified",
        );
    }

    #[test]
    fn test_reject_proof_without_hash_backend() {
        let (_, proof) = adversarial_setup();
        let mut proto = proof.to_proto();
        proto.hash_backend = None;
        assert_rejected(
            Proof::<BS, Sha2Hash<BS>>::from_proto(&proto).map(|_| ()),
            "hash backend not specified",
        );
    }

    #[test]
    fn test_reject_commitment_with_wrong_hash_backend() {
        let (commitment, _) = adversarial_setup();
        assert_rejected(
            Commitment::<BS, Keccak256Hash<BS>>::from_proto(&commitment.to_proto()).map(|_| ()),
            "incorrect hash backend",
        );
    }

    #[test]
    fn test_reject_proof_with_wrong_hash_backend() {
        let (_, proof) = adversarial_setup();
        assert_rejected(
            Proof::<BS, Keccak256Hash<BS>>::from_proto(&proof.to_proto()).map(|_| ()),
            "incorrect hash backend",
        );
    }

    #[test]
    fn test_commitment_serialization() {
        let mut committer = Committer::<BS, Sha2Hash<BS>>::new(
            Options {
                degree_bound: 4,
                blowup_log2: 1,
                grinding_bits: 0,
            },
            vec![polynomial(&[12, 34, 56, 78]), polynomial(&[42, 43, 44, 45])],
        );
        committer.add_batch(vec![polynomial(&[90, 78, 56, 34])]);
        let (commitment, _) =
            committer.commit(BTreeSet::from([BS::from(123u16), BS::from(456u16)]));
        assert_eq!(
            commitment.tree_roots(),
            [
                parse("0xc47ed760edd5fc63ecb5c86240553f7e2fdabd0003ca62ab40d79e230ead9eb1"),
                parse("0x13880d0002db3411382d144a40f86201f3b84b55c317bcb0b583f166100e8b37"),
            ]
        );
        assert_eq!(commitment.batch_sizes(), [2, 1]);
        assert_eq!(
            commitment.transcript_hash(1),
            parse("0x8e58a20287b9d5f4743c73d0d57b2b1dcb5846f4b658d5463a393b0b44dafcfc")
        );
        assert_eq!(
            commitment.transcript_hash(2),
            parse("0x60d1b1e6e174c834b3ea7412cf09cdb5f3949f7b63f0c8f3f841b1cedc58c56a")
        );
        let proto = commitment.to_proto();
        let commitment = Commitment::<BS, Sha2Hash<BS>>::from_proto(&proto).unwrap();
        assert_eq!(commitment.degree_bound(), 4);
        assert_eq!(
            commitment.tree_roots(),
            [
                parse("0xc47ed760edd5fc63ecb5c86240553f7e2fdabd0003ca62ab40d79e230ead9eb1"),
                parse("0x13880d0002db3411382d144a40f86201f3b84b55c317bcb0b583f166100e8b37"),
            ]
        );
        assert_eq!(commitment.batch_sizes(), [2, 1]);
        assert_eq!(
            commitment.transcript_hash(1),
            parse("0x8e58a20287b9d5f4743c73d0d57b2b1dcb5846f4b658d5463a393b0b44dafcfc")
        );
        assert_eq!(
            commitment.transcript_hash(2),
            parse("0x60d1b1e6e174c834b3ea7412cf09cdb5f3949f7b63f0c8f3f841b1cedc58c56a")
        );
    }

    #[test]
    fn test_proof_serialization() {
        let mut committer = Committer::<BS, Sha2Hash<BS>>::new(
            Options {
                degree_bound: 4,
                blowup_log2: 3,
                grinding_bits: 8,
            },
            vec![polynomial(&[12, 34, 56, 78]), polynomial(&[42, 43, 44, 45])],
        );
        committer.add_batch(vec![polynomial(&[90, 78, 56, 34])]);
        let (commitment, prover) =
            committer.commit(BTreeSet::from([BS::from(123u16), BS::from(456u16)]));
        let proof = prover.prove(&commitment);
        let nonce = proof.nonce();
        let commitment_proto = commitment.to_proto();
        let proof_proto = proof.to_proto();
        let commitment = Commitment::<BS, Sha2Hash<BS>>::from_proto(&commitment_proto).unwrap();
        let proof = Proof::<BS, Sha2Hash<BS>>::from_proto(&proof_proto).unwrap();
        assert_eq!(proof.options().degree_bound, 4);
        assert_eq!(proof.options().blowup_log2, 3);
        assert_eq!(proof.options().grinding_bits, 8);
        assert_eq!(proof.num_polys(), 3);
        assert_eq!(proof.nonce(), nonce);
        assert_eq!(
            proof.points().keys().copied().collect::<BTreeSet<BS>>(),
            BTreeSet::from([BS::from(123u16), BS::from(456u16)])
        );
        assert!(proof.verify(&commitment).is_ok());
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
                let pow = proof_of_work::<BS, Sha2Hash<BS>>(seed, nonce);
                check_proof_of_work(pow, proof.options.grinding_bits) == valid
            })
            .unwrap()
    }

    #[test]
    fn test_accept_ground_proof() {
        let (commitment, proof) = grinding_setup();
        assert_eq!(
            proof.queries.len(),
            num_queries(ADVERSARIAL_BLOWUP_LOG2, proof.options().grinding_bits)
        );
        assert!(proof.queries.len() < num_queries(ADVERSARIAL_BLOWUP_LOG2, 0));
        assert!(proof.verify(&commitment).is_ok());
    }

    #[test]
    fn test_reject_invalid_nonce() {
        let (commitment, mut proof) = grinding_setup();
        proof.nonce = find_nonce(&commitment, &proof, false);
        assert_rejected(proof.verify(&commitment), "insufficient proof of work");
    }

    #[test]
    fn test_reject_other_valid_nonce() {
        let (commitment, mut proof) = grinding_setup();
        proof.nonce = find_nonce(&commitment, &proof, true);
        assert_rejected(proof.verify(&commitment), "wrong query index");
    }

    #[test]
    fn test_reject_overstated_grinding() {
        let (commitment, mut proof) = grinding_setup();
        proof.options.grinding_bits = BS::NUM_BITS / 2;
        assert_rejected(proof.verify(&commitment), "insufficient proof of work");
    }

    #[test]
    fn test_reject_understated_grinding() {
        let (commitment, mut proof) = grinding_setup();
        proof.options.grinding_bits = 0;
        assert_rejected(proof.verify(&commitment), "incorrect number of openings");
    }

    #[test]
    fn test_reject_proof_with_excessive_grinding() {
        let (_, proof) = grinding_setup();
        let mut proto = proof.to_proto();
        proto.grinding_bits = 128;
        assert_rejected(
            Proof::<BS, Sha2Hash<BS>>::from_proto(&proto).map(|_| ()),
            "invalid number of grinding bits",
        );
    }

    #[test]
    fn test_ground_proof_serialization() {
        let (commitment, proof) = grinding_setup();
        let proto = proof.to_proto();
        assert_eq!(proto.grinding_bits, ADVERSARIAL_GRINDING_BITS as u32);
        assert_eq!(proto.nonce, proof.nonce());
        let decoded = Proof::<BS, Sha2Hash<BS>>::from_proto(&proto).unwrap();
        assert_eq!(decoded.options().grinding_bits, ADVERSARIAL_GRINDING_BITS);
        assert_eq!(decoded.nonce(), proof.nonce());
        assert!(decoded.verify(&commitment).is_ok());
    }
}
