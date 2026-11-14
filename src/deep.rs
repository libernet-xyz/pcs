use crate::fri;
use crate::hash::Hasher;
use crate::merkle::{Proof as LeafProof, Tree};
use crate::scpf;
use crate::starkom::proto::pcs::v10 as proto;
use crate::utils;
use anyhow::{Result, anyhow};
use primitive_types::{H256, U256};
use prost_types::Any;
use starkom_ff::{Field, Field256, PrimeField256};
use starkom_poly::Polynomial;
use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;
use std::sync::LazyLock;

/// The number of bits of a proof-of-work hash, which is also the most leading zero bits it can
/// have.
pub const PROOF_OF_WORK_BITS: usize = 256;

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

/// Schema identifier of the SCPF data of commitments (see [`Commitment::to_scpf`]).
static COMMITMENT_SCHEMA_IDENTIFIER: LazyLock<H256> =
    LazyLock::new(|| scpf::schema_identifier("starkom.proto.pcs.v10.Commitment"));

/// Schema identifier of the SCPF data of proofs (see [`Proof::to_scpf`]).
static PROOF_SCHEMA_IDENTIFIER: LazyLock<H256> =
    LazyLock::new(|| scpf::schema_identifier("starkom.proto.pcs.v10.Proof"));

/// Returns the number of FRI queries required to achieve `security_bits` bits of security using a
/// blowup factor of `2^blowup_log2` and `grinding_bits` bits of grinding.
///
/// Every query provides `blowup_log2` bits of security and grinding provides the rest. At least
/// one query is always required.
pub fn num_queries(security_bits: usize, blowup_log2: usize, grinding_bits: usize) -> usize {
    security_bits
        .saturating_sub(grinding_bits)
        .div_ceil(blowup_log2)
        .max(1)
}

/// The parameters of the DEEP-FRI protocol.
///
/// The prover specifies them in [`Committer::new`] and they're recorded in every [`Proof`], so
/// verifiers must check them against the expected ones (see [`Proof::options`] and
/// [`Options::satisfies`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    /// The proven degree bound, which must be a power of 2.
    ///
    /// Honest provers commit polynomials of degree strictly less than this value, but a valid proof
    /// only guarantees a degree at most equal to it (see [`Proof::verify`]).
    pub degree_bound: usize,

    /// The target security level in bits.
    ///
    /// Every FRI query provides [`blowup_log2`](`Self::blowup_log2`) bits of security and grinding
    /// provides [`grinding_bits`](`Self::grinding_bits`), so a proof has the smallest number of
    /// queries that reaches this many bits (see [`num_queries`]). Any value is valid.
    ///
    /// The prover chooses it, so verifiers must check it against the expected one (see
    /// [`Self::satisfies`]).
    pub security_bits: usize,

    /// The base-2 logarithm of the blowup factor, which must be greater than 0.
    ///
    /// The extended evaluation domain has `degree_bound * 2^blowup_log2` elements and its size must
    /// not exceed the 2-adicity of the base field.
    pub blowup_log2: usize,

    /// The number of FRI folds per round, which must be greater than 0.
    ///
    /// This is the base-2 logarithm of the folding arity: for example, with a fold rate of 3 every
    /// FRI round folds the polynomial 3 times, dividing its degree bound by 2^3 = 8, and every
    /// query opens a coset of 8 evaluations in each Merkle tree. Every round performs exactly this
    /// many folds (see [`Self::max_rounds`]).
    ///
    /// Higher rates mean fewer Merkle trees and therefore fewer Merkle paths per query, but larger
    /// cosets. The optimum depends on the size of the extended domain, and 3 is a good default.
    pub fold_rate: usize,

    /// The maximum number of FRI folding rounds.
    ///
    /// Since every round performs exactly [`fold_rate`](`Self::fold_rate`) folds, the actual number
    /// of rounds is the minimum between this value and `log2(degree_bound) / fold_rate`, rounded
    /// down (see [`Self::num_rounds`]). Rather than being folded further, the polynomial resulting
    /// from the last round is sent in the clear, so it has degree less than
    /// `degree_bound / 2^(rounds * fold_rate)`. With 0 there's no folding at all and the whole DEEP
    /// quotient is sent in the clear.
    ///
    /// Every round costs a Merkle proof per query, while every coefficient of the folded quotient
    /// is sent only once, so the optimum stops folding when the number of coefficients is roughly
    /// comparable to the number of queries times the size of a round's Merkle proof (a few hundred
    /// coefficients in practice). Any value is valid.
    pub max_rounds: usize,

    /// The number of grinding bits.
    ///
    /// The prover grinds a nonce whose proof-of-work hash has at least this many leading zero bits,
    /// which takes `2^grinding_bits` hash evaluations on average. Every `blowup_log2` bits of
    /// grinding save one FRI query (see [`num_queries`]). 0 disables grinding.
    ///
    /// Must be at most [`PROOF_OF_WORK_BITS`], the number of bits of the proof-of-work hash,
    /// because no hash can satisfy more (see [`Self::validate`]). Values above
    /// [`security_bits`](`Self::security_bits`) are valid but save no further queries, because at
    /// least one query is always required.
    pub grinding_bits: usize,
}

impl Options {
    /// Returns the size of the extended evaluation domain, `degree_bound * 2^blowup_log2`.
    ///
    /// REQUIRES: the options must be [valid](`Self::validate`).
    pub fn extended_domain_size(&self) -> usize {
        self.degree_bound << self.blowup_log2
    }

    /// Returns the number of FRI queries required to achieve
    /// [`security_bits`](`Self::security_bits`) bits of security with these options (see
    /// [`num_queries`]).
    pub fn num_queries(&self) -> usize {
        num_queries(self.security_bits, self.blowup_log2, self.grinding_bits)
    }

    /// Returns the number of FRI folding rounds (see [`Self::max_rounds`]), which is also the
    /// number of Merkle trees of the FRI folding argument.
    ///
    /// REQUIRES: the options must be [valid](`Self::validate`).
    pub fn num_rounds(&self) -> usize {
        fri::num_rounds(self.degree_bound, self.fold_rate, self.max_rounds)
    }

    /// Returns true iff these options are acceptable to a verifier that expects `other`: the
    /// security level and the number of grinding bits must be greater than or equal to the expected
    /// ones, and all other parameters must be equal.
    ///
    /// The number of queries is derived from both (see [`num_queries`]), so a proof with a higher
    /// security level than expected is at least as sound. Soundness doesn't depend on the number of
    /// grinding bits, but more grinding bits mean fewer queries, so a proof with more grinding bits
    /// than expected is never more expensive to verify.
    pub fn satisfies(&self, other: &Self) -> bool {
        self.security_bits >= other.security_bits
            && self.grinding_bits >= other.grinding_bits
            && Self {
                security_bits: other.security_bits,
                grinding_bits: other.grinding_bits,
                ..*self
            }
            .eq(other)
    }

    /// Checks that these options are valid for the field `F`, returning an error otherwise.
    ///
    /// The degree bound must be a power of 2, `blowup_log2` and `fold_rate` must be greater than 0,
    /// the size of the extended evaluation domain must not exceed the 2-adicity of `F::BaseField`,
    /// and `grinding_bits` must not exceed [`PROOF_OF_WORK_BITS`]. `security_bits` and `max_rounds`
    /// can have any value.
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
        if self.fold_rate == 0 {
            return Err(anyhow!("invalid fold rate 0"));
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
        if self.grinding_bits > PROOF_OF_WORK_BITS {
            return Err(anyhow!(
                "invalid number of grinding bits {} (exceeds {PROOF_OF_WORK_BITS})",
                self.grinding_bits
            ));
        }
        Ok(())
    }
}

/// Computes the proof-of-work hash of a nonce.
///
/// `seed` is the hash of the full proof transcript returned by [`query_seed`], so that changing
/// anything sent before the queries requires redoing the proof of work.
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

/// Encodes the transcript of the DEEP layer, that is all Merkle roots of the committed polynomials,
/// all opened points, and their claimed evaluations.
fn deep_transcript<F: Field256>(
    tree_roots: &[H256],
    num_polys: usize,
    points: &BTreeMap<F, Vec<F>>,
) -> impl Iterator<Item = H256> {
    std::iter::once(utils::encode_usize(tree_roots.len()))
        .chain(tree_roots.iter().copied())
        .chain(std::iter::once(utils::encode_usize(num_polys)))
        .chain(std::iter::once(utils::encode_usize(points.len())))
        .chain(points.iter().flat_map(|(z, values)| {
            std::iter::once(z)
                .chain(values.iter())
                .map(|value| H256::from_slice(&value.to_be_bytes()))
        }))
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
    let transcript: Vec<H256> = deep_transcript(tree_roots, num_polys, points).collect();
    (
        H::challenge(*RLC_DST, transcript.as_slice()),
        H::challenge(*QUOTIENT_DST, transcript.as_slice()),
    )
}

/// Hashes the full transcript of a proof up to the queries: all Merkle roots of the committed
/// polynomials, all opened points and their claimed evaluations, all FRI roots, and the
/// coefficients of the folded quotient.
///
/// The returned seed is the input of the proof of work, which in turn determines the FRI query
/// indices (see [`get_query_indices`]).
fn query_seed<F: Field256, H: Hasher<F>>(
    tree_roots: &[H256],
    num_polys: usize,
    points: &BTreeMap<F, Vec<F>>,
    fri_commitment: &fri::Commitment<F>,
) -> H256 {
    let folded_quotient = fri_commitment.folded_polynomial();
    H::hash_transcript(
        *QUERY_DST0,
        deep_transcript(tree_roots, num_polys, points)
            .chain(std::iter::once(utils::encode_usize(fri_commitment.len())))
            .chain(fri_commitment.roots().iter().copied())
            .chain(std::iter::once(utils::encode_usize(folded_quotient.len())))
            .chain(
                folded_quotient
                    .coefficients()
                    .iter()
                    .map(|coefficient| H256::from_slice(&coefficient.to_be_bytes())),
            )
            .collect::<Vec<H256>>()
            .as_slice(),
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

/// Evaluates a polynomial over the base field at a point of the extension field.
fn evaluate_base_polynomial<F: Field256>(polynomial: &Polynomial<F::BaseField>, z: F) -> F {
    polynomial
        .coefficients()
        .iter()
        .rev()
        .fold(F::ZERO, |value, &coefficient| value * z + coefficient)
}

/// The polynomials of a committed batch, which are either all over the base field or all over the
/// ~256-bit field.
#[derive(Debug, Clone)]
enum BatchPolynomials<F: Field256> {
    Base(Vec<Polynomial<F::BaseField>>),
    Extension(Vec<Polynomial<F>>),
}

/// A committed batch: its polynomials and the Merkle tree of their evaluations.
#[derive(Debug, Clone)]
struct Batch<F: Field256, H: Hasher<F>> {
    polynomials: BatchPolynomials<F>,
    tree: Tree<F, H>,
}

impl<F: Field256, H: Hasher<F>> Batch<F, H> {
    /// Returns the commitment to the batch.
    fn commitment(&self) -> BatchCommitment {
        BatchCommitment {
            base_field: matches!(self.polynomials, BatchPolynomials::Base(_)),
            num_polys: self.num_polys(),
            root: self.tree.root_hash(),
        }
    }

    /// Returns the number of polynomials in the batch.
    fn num_polys(&self) -> usize {
        match &self.polynomials {
            BatchPolynomials::Base(polynomials) => polynomials.len(),
            BatchPolynomials::Extension(polynomials) => polynomials.len(),
        }
    }

    /// Returns the root hash of the Merkle tree of the batch.
    fn root_hash(&self) -> H256 {
        self.tree.root_hash()
    }

    /// Returns a Merkle proof for the leaf at `index`.
    fn query(&self, index: usize) -> LeafProof<F, H> {
        self.tree.query(index)
    }

    /// Evaluates all polynomials of the batch at `z`.
    fn evaluate(&self, z: F) -> Vec<F> {
        match &self.polynomials {
            BatchPolynomials::Base(polynomials) => polynomials
                .iter()
                .map(|polynomial| evaluate_base_polynomial(polynomial, z))
                .collect(),
            BatchPolynomials::Extension(polynomials) => polynomials
                .iter()
                .map(|polynomial| polynomial.evaluate(z))
                .collect(),
        }
    }

    /// Computes the random linear combination of all polynomials of all `batches`, in order: the
    /// i-th polynomial is multiplied by `alpha^i`.
    fn combine<'a>(batches: impl IntoIterator<Item = &'a Self>, alpha: F) -> Polynomial<F>
    where
        Self: 'a,
    {
        let mut combined = Polynomial::<F>::default();
        let mut pow = F::ONE;
        for batch in batches {
            match &batch.polynomials {
                BatchPolynomials::Base(polynomials) => {
                    for polynomial in polynomials {
                        combined += Polynomial::with_coefficients(
                            polynomial
                                .coefficients()
                                .iter()
                                .map(|&coefficient| pow * coefficient)
                                .collect(),
                        );
                        pow *= alpha;
                    }
                }
                BatchPolynomials::Extension(polynomials) => {
                    for polynomial in polynomials {
                        combined += polynomial.clone() * pow;
                        pow *= alpha;
                    }
                }
            }
        }
        combined
    }
}

/// The commitment to a single batch of polynomials (see [`Commitment`]).
///
/// The Merkle root also determines the other fields, so a proof never verifies against a commitment
/// with the wrong ones. However, callers must still check the expected shape (number of polynomials
/// and whether they're defined over the base field or the extension field).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BatchCommitment {
    /// Whether the polynomials of the batch are over the base field.
    base_field: bool,
    /// Number of polynomials in the batch.
    num_polys: usize,
    /// Root hash of the Merkle tree where the evaluations of the polynomials are stored.
    root: H256,
}

impl BatchCommitment {
    /// Returns whether the polynomials of the batch are over the base field `F::BaseField` (see
    /// [`Committer::add_base_batch`]) rather than over the ~256-bit field `F` (see
    /// [`Committer::add_batch`]).
    pub fn base_field(&self) -> bool {
        self.base_field
    }

    /// Returns the number of polynomials in the batch.
    ///
    /// Every polynomial over `F` is committed as its `T` coordinate polynomials over
    /// `F::BaseField` where `T` is [`F::EXTENSION_DEGREE`](`Field::EXTENSION_DEGREE`), so every
    /// leaf of the Merkle tree of the batch has `num_polys` values for a base-field batch and
    /// `T * num_polys` values for a batch over `F`.
    pub fn num_polys(&self) -> usize {
        self.num_polys
    }

    /// Returns the root hash of the Merkle tree where the evaluations of the polynomials of the
    /// batch are stored.
    pub fn root(&self) -> H256 {
        self.root
    }
}

/// A batched DEEP-FRI polynomial commitment (see [`Committer`] for details).
///
/// The commitment only depends on the committed polynomials, not on the points opened by a
/// [`Proof`], so the points can be chosen after committing (see [`Prover::prove`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment<F: Field256, H: Hasher<F>> {
    /// The commitments to the polynomial batches, in commitment order.
    batches: Vec<BatchCommitment>,
    _data: PhantomData<(F, H)>,
}

impl<F: Field256, H: Hasher<F>> Commitment<F, H> {
    /// Returns the commitments to the polynomial batches, in commitment order.
    pub fn batches(&self) -> &[BatchCommitment] {
        self.batches.as_slice()
    }

    /// Returns the number of polynomial batches, which is also the number of Merkle trees.
    pub fn num_batches(&self) -> usize {
        self.batches.len()
    }

    /// Returns the total number of committed polynomials, i.e. the sum of the `num_polys` of all
    /// [batches](`Self::batches`), which is also the number of evaluations that a [`Proof`] opens
    /// at every point.
    pub fn num_polys(&self) -> usize {
        self.batches.iter().map(|batch| batch.num_polys).sum()
    }

    /// Returns the root hashes of the Merkle trees of all batches, in commitment order.
    fn tree_roots(&self) -> Vec<H256> {
        self.batches.iter().map(BatchCommitment::root).collect()
    }

    /// Hashes the Merkle roots of the first `batch_count` [batches](`Self::batches`).
    ///
    /// The returned hash is cryptographically bound to the full transcript up to the given
    /// polynomial batch, and can be used by a verifier to recover Fiat-Shamir challenges.
    ///
    /// REQUIRES: `batch_count` must be strictly greater than 0 and less than or equal to the number
    /// of batches.
    ///
    /// The hashes returned by this method are compatible with [`Committer::transcript_hash`], which
    /// can be used on the prover side.
    pub fn transcript_hash(&self, batch_count: usize) -> H256 {
        assert!(batch_count > 0);
        assert!(batch_count <= self.num_batches());
        H::hash_transcript(
            *TRANSCRIPT_DST,
            std::iter::once(utils::encode_usize(batch_count))
                .chain(
                    self.batches[..batch_count]
                        .iter()
                        .map(BatchCommitment::root),
                )
                .collect::<Vec<H256>>()
                .as_slice(),
        )
    }

    /// Serializes this commitment to a [`Commitment`](`proto::Commitment`) protobuf.
    pub fn to_proto(&self) -> proto::Commitment {
        proto::Commitment {
            hash_backend: Some(Any::from_msg(&H::ProtoTag::default()).unwrap()),
            batches: self
                .batches
                .iter()
                .map(|batch| proto::commitment::Batch {
                    base_field: batch.base_field,
                    num_polys: batch.num_polys as u64,
                    root: batch.root.as_bytes().to_vec(),
                })
                .collect(),
        }
    }

    /// Deserializes a commitment from a [`Commitment`](`proto::Commitment`) protobuf.
    ///
    /// Returns an error if the total number of polynomials, or the number of values of the leaves
    /// of any batch, doesn't fit in a `usize`.
    pub fn from_proto(proto: &proto::Commitment) -> Result<Self> {
        check_hash_backend::<H::ProtoTag>(&proto.hash_backend)?;

        if proto.batches.is_empty() {
            return Err(anyhow!("the commitment has no polynomial batches"));
        }

        let batches: Vec<BatchCommitment> = proto
            .batches
            .iter()
            .map(|batch| {
                let num_polys = usize::try_from(batch.num_polys)?;
                if !batch.base_field && num_polys.checked_mul(F::EXTENSION_DEGREE).is_none() {
                    return Err(anyhow!(
                        "the leaves of a batch of {num_polys} polynomials are too large"
                    ));
                }
                Ok(BatchCommitment {
                    base_field: batch.base_field,
                    num_polys,
                    root: H::load_hash(batch.root.as_slice())?,
                })
            })
            .collect::<Result<_>>()?;

        if batches
            .iter()
            .try_fold(0usize, |total, batch| total.checked_add(batch.num_polys))
            .is_none()
        {
            return Err(anyhow!("the commitment has too many polynomials"));
        }

        Ok(Self {
            batches,
            _data: PhantomData,
        })
    }
}

impl<F: PrimeField256, H: Hasher<F>> Commitment<F, H> {
    /// Serializes this commitment to SCPF, the Starkom Compact Proof Format (see
    /// [`Proof::to_scpf`]).
    ///
    /// The schema identifier is the Keccak-256 hash of `starkom.proto.pcs.v10.Commitment`, and the
    /// stream holds the fields of the [`Commitment`](`proto::Commitment`) protobuf:
    ///
    /// 1. the number of batches, as a varint;
    /// 2. for every batch in commitment order, `base_field` as a varint (1 for true and 0 for
    ///    false), then `num_polys` as a varint, then the Merkle root as a pool reference.
    ///
    /// The hash backend isn't encoded, so decoders must know it in advance.
    pub fn to_scpf(&self) -> Vec<u8> {
        let mut encoder = scpf::Encoder::default();
        encoder.push_usize(self.batches.len());
        for batch in &self.batches {
            encoder.push_bool(batch.base_field);
            encoder.push_usize(batch.num_polys);
            encoder.push_hash(batch.root);
        }
        encoder.finish(*COMMITMENT_SCHEMA_IDENTIFIER)
    }
}

/// Collects batches of polynomials and allows building a DEEP-FRI prover for them.
///
/// This works by building Merkle trees on the batched polynomials, one tree per batch, and
/// eventually handing everything over to a newly constructed [`Prover`] (see the [`Self::commit`]
/// method).
///
/// All Fiat-Shamir challenges and opened points are elements of the ~256-bit field `F`, but every
/// batch of committed polynomials can be defined either over `F` (see [`Self::add_batch`]) or over
/// its base field `F::BaseField`, e.g. Goldilocks^4 and Goldilocks (see [`Self::add_base_batch`]).
/// Committing polynomials over the base field costs less, because their Merkle leaves are `T` times
/// smaller (`T` being the extension degree of `F`), and proves that they're over the base field.
/// When `F` is a prime field (e.g. BlueSky) `F::BaseField` and `F` coincide, and so do the two
/// kinds of batch.
///
/// This two-stage Committer-Prover architecture allows getting Merkle roots for the proven
/// polynomials before batching all of them, so that Fiat-Shamir challenges can be derived between
/// batches (see [`Self::transcript_hash`]). The opened points don't need to be known at commitment
/// time either: the [`Prover`] receives them later (see [`Prover::prove`]).
#[derive(Debug, Clone)]
pub struct Committer<F: Field256, H: Hasher<F>> {
    /// The protocol parameters. The degree of all batched polynomials must be less than the degree
    /// bound.
    options: Options,
    /// All polynomial batches added so far, along with their Merkle trees.
    batches: Vec<Batch<F, H>>,
}

impl<F: Field256, H: Hasher<F>> Committer<F, H> {
    /// Constructs a [`Committer`] with the given options and no polynomials.
    ///
    /// Our DEEP-FRI protocol requires at least one committed polynomial to work, so at least one
    /// batch must be added (see [`Self::add_batch`] and [`Self::add_base_batch`]) before calling
    /// [`Self::commit`].
    ///
    /// Panics if the options are [invalid](`Options::validate`).
    pub fn new(options: Options) -> Self {
        options.validate::<F>().expect("invalid DEEP-FRI options");
        Self {
            options,
            batches: vec![],
        }
    }

    /// Returns the protocol parameters.
    pub fn options(&self) -> &Options {
        &self.options
    }

    /// Returns the number of polynomial batches added so far, which is also the number of Merkle
    /// trees.
    pub fn num_batches(&self) -> usize {
        self.batches.len()
    }

    /// Returns the Merkle root of the i-th batch. `index` must be less than
    /// [`Self::num_batches()`].
    ///
    /// This value can be used to derive Fiat-Shamir challenges.
    pub fn root_hash(&self, index: usize) -> H256 {
        self.batches[index].root_hash()
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
            std::iter::once(utils::encode_usize(self.batches.len()))
                .chain(self.batches.iter().map(Batch::root_hash))
                .collect::<Vec<H256>>()
                .as_slice(),
        )
    }

    /// Adds a batch of polynomials over `F`, returning the index of the newly created batch.
    ///
    /// The returned index can be used with the [`Self::root_hash`] method to get the Merkle root
    /// for the batch.
    ///
    /// REQUIRES: the batch must not be empty, and the degree of all specified polynomials must be
    /// strictly less than the [`degree_bound`](`Options::degree_bound`).
    pub fn add_batch(&mut self, polynomials: Vec<Polynomial<F>>) -> usize {
        let n = self.check_batch(polynomials.iter().map(Polynomial::degree_bound));
        let evaluations = polynomials
            .iter()
            .map(|polynomial| polynomial.clone().shift_domain().lde2(n))
            .collect();
        self.push_batch(Batch {
            polynomials: BatchPolynomials::Extension(polynomials),
            tree: Tree::new(evaluations),
        })
    }

    /// Adds a batch of polynomials over the base field `F::BaseField`, returning the index of the
    /// newly created batch.
    ///
    /// The leaves of the Merkle tree of a base-field batch are elements of `F::BaseField` by
    /// construction, and so are the coefficients of the polynomials that a successful verification
    /// binds them to: a polynomial whose degree is less than the number of points of the
    /// (base-field) evaluation domain where it agrees with `F::BaseField`-valued evaluations is the
    /// interpolant of those evaluations, so its coefficients are in `F::BaseField`.
    ///
    /// The returned index can be used with the [`Self::root_hash`] method to get the Merkle root
    /// for the batch.
    ///
    /// REQUIRES: the batch must not be empty, and the degree of all specified polynomials must be
    /// strictly less than the [`degree_bound`](`Options::degree_bound`).
    pub fn add_base_batch(&mut self, polynomials: Vec<Polynomial<F::BaseField>>) -> usize {
        let n = self.check_batch(polynomials.iter().map(Polynomial::degree_bound));
        let evaluations = polynomials
            .iter()
            .map(|polynomial| polynomial.clone().shift_domain().lde2(n))
            .collect();
        self.push_batch(Batch {
            polynomials: BatchPolynomials::Base(polynomials),
            tree: Tree::new_from_base(evaluations),
        })
    }

    /// Checks the degree bounds of the polynomials of a new batch, returning the size of the
    /// extended evaluation domain.
    ///
    /// Panics if the batch is empty or if any polynomial exceeds the degree bound.
    fn check_batch(&self, degree_bounds: impl Iterator<Item = usize>) -> usize {
        let degree_bound = degree_bounds
            .max()
            .expect("empty polynomial batch")
            .next_power_of_two();
        assert!(degree_bound <= self.options.degree_bound);
        self.options.extended_domain_size()
    }

    /// Records a new batch, returning its index.
    fn push_batch(&mut self, batch: Batch<F, H>) -> usize {
        let index = self.batches.len();
        self.batches.push(batch);
        index
    }

    /// Consumes the [`Committer`] and returns a polynomial [`Commitment`] and a DEEP-FRI [`Prover`]
    /// for the committed polynomials.
    ///
    /// The commitment only depends on the committed polynomials, so the points to open can be
    /// chosen afterwards, for example by deriving them from the commitment via Fiat-Shamir or by
    /// letting an interactive verifier request them (see [`Prover::prove`]).
    ///
    /// Panics if no batches have been added.
    pub fn commit(self) -> (Commitment<F, H>, Prover<F, H>) {
        assert!(!self.batches.is_empty(), "no polynomial batches");
        let commitment = Commitment {
            batches: self.batches.iter().map(Batch::commitment).collect(),
            _data: PhantomData,
        };
        let prover = Prover {
            options: self.options,
            batches: self.batches,
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
    /// The opened points. Keys are (off-domain) X-coordinates, values are the corresponding
    /// evaluations (one for every committed polynomial).
    points: BTreeMap<F, Vec<F>>,
    /// The FRI commitment to the combined DEEP quotient of all opened points (see
    /// [`Prover::prove`]), that is the Merkle roots of the folding rounds and the folded quotient.
    fri_commitment: fri::Commitment<F>,
    /// Merkle proofs for the points at the query positions, relative to the raw Merkle trees (not
    /// the FRI folds). The outer array has one entry for every FRI query
    /// (`openings.len() == queries.len()`), and the inner arrays contain one proof for every Merkle
    /// tree.
    openings: Vec<Vec<LeafProof<F, H>>>,
    /// FRI queries on the combined DEEP quotient (see [`Prover::prove`]). The number of queries is
    /// calculated by [`num_queries`] above and is tuned to achieve [`Options::security_bits`] bits
    /// of security together with grinding.
    queries: Vec<fri::Query<F, H>>,
}

impl<F: Field256, H: Hasher<F>> Proof<F, H> {
    /// Returns the protocol parameters used in the proof.
    ///
    /// Callers must check them against the expected ones, typically with [`Options::satisfies`].
    /// In particular the security level is chosen by the prover, so a proof that isn't checked can
    /// have as few as one query. Soundness doesn't depend on the number of grinding bits because
    /// the number of queries is automatically derived from it, but it should still be checked
    /// along with the other parameters.
    pub fn options(&self) -> &Options {
        &self.options
    }

    /// Returns the proof-of-work nonce found by the prover.
    pub fn nonce(&self) -> u64 {
        self.nonce
    }

    /// Returns a reference to the opened points. Keys are (off-domain) X-coordinates, values are
    /// the corresponding evaluations (one for every committed polynomial).
    pub fn points(&self) -> &BTreeMap<F, Vec<F>> {
        &self.points
    }

    /// Verifies this proof against the given commitment.
    ///
    /// If verification succeeds, the committed polynomials have degree *at most*
    /// [`Options::degree_bound`], the ones of the base-field batches are over the base field (see
    /// [`Committer::add_base_batch`]), and their evaluations at the [opened points](`Self::points`)
    /// are the ones claimed by the proof.
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
    ///
    /// NOTE: the proof proves the evaluations at whatever points it contains, so callers must check
    /// that the [opened points](`Self::points`) are the ones they expect.
    pub fn verify(&self, commitment: &Commitment<F, H>) -> Result<()> {
        check_points_off_domain(
            self.points.keys().copied(),
            self.options.extended_domain_size(),
        )?;

        if self.fri_commitment.len() != self.options.num_rounds() {
            return Err(anyhow!(
                "the options (degree bound {}, fold rate {}, max rounds {}) don't match the {} FRI roots",
                self.options.degree_bound,
                self.options.fold_rate,
                self.options.max_rounds,
                self.fri_commitment.len()
            ));
        }

        let num_polys = commitment.num_polys();
        if self.points.is_empty() {
            return Err(anyhow!("the proof doesn't open any points"));
        }
        for (&z, values) in &self.points {
            if values.len() != num_polys {
                return Err(anyhow!(
                    "incorrect number of evaluations at {z} (got {}, want {num_polys})",
                    values.len()
                ));
            }
        }

        // The number of queries must be checked before deriving the query indices, so that the work
        // of deriving them is bounded by the size of the proof even if its security level is huge.
        let num_queries = self.options.num_queries();
        if self.openings.len() != num_queries {
            return Err(anyhow!(
                "incorrect number of openings (got {}, want {})",
                self.openings.len(),
                num_queries
            ));
        }
        if self.queries.len() != num_queries {
            return Err(anyhow!(
                "incorrect number of queries (got {}, want {})",
                self.queries.len(),
                num_queries
            ));
        }

        let tree_roots = commitment.tree_roots();
        let seed = query_seed::<F, H>(
            tree_roots.as_slice(),
            num_polys,
            &self.points,
            &self.fri_commitment,
        );
        let pow = proof_of_work::<F, H>(seed, self.nonce);
        if !check_proof_of_work(pow, self.options.grinding_bits) {
            return Err(anyhow!("insufficient proof of work"));
        }
        let indices = get_query_indices::<F, H>(pow, &self.options);

        let (alpha, beta) = deep_challenges::<F, H>(tree_roots.as_slice(), num_polys, &self.points);

        for ((query, openings), &expected_index) in
            (self.queries.iter().zip(self.openings.iter())).zip(indices.iter())
        {
            let index = query.index();
            if index != expected_index {
                return Err(anyhow!(
                    "wrong query index (got {index}, want {expected_index})",
                ));
            }

            if openings.len() != commitment.num_batches() {
                return Err(anyhow!(
                    "incorrect number of openings for index {index} (got {}, want {})",
                    openings.len(),
                    commitment.num_batches()
                ));
            }
            let mut values = Vec::with_capacity(num_polys);
            for (batch, opening) in commitment.batches().iter().zip(openings.iter()) {
                let height = self.options.extended_domain_size().trailing_zeros() as usize;
                if opening.len() != height {
                    return Err(anyhow!("invalid opening for index {index}"));
                }
                let leaf_size = if batch.base_field {
                    batch.num_polys
                } else {
                    batch.num_polys * F::EXTENSION_DEGREE
                };
                if opening.leaf().len() != leaf_size {
                    return Err(anyhow!(
                        "incorrect leaf size at index {index} (got {}, want {leaf_size})",
                        opening.leaf().len()
                    ));
                }
                if batch.base_field {
                    opening.verify_base(index, batch.root)?;
                    values.extend(opening.leaf().iter().map(|&value| F::from(value)));
                } else {
                    opening.verify(index, batch.root)?;
                    values.extend(utils::unflatten::<F>(opening.leaf()));
                }
            }

            if query.len() != self.options.num_rounds() {
                return Err(anyhow!("invalid low-degree proof for index {index}"));
            }

            let combined = rlc(values, alpha);

            let x = query.x();
            let quotient = rlc(
                self.points.iter().map(|(&z, values)| {
                    let v = rlc(values.iter().copied(), alpha);
                    (combined - v) * (x - z).invert_vartime().unwrap()
                }),
                beta,
            );
            query.verify(&self.fri_commitment, quotient)?;
        }

        Ok(())
    }

    /// Serializes this proof to a [`Proof`](`proto::Proof`) protobuf.
    pub fn to_proto(&self) -> proto::Proof {
        proto::Proof {
            hash_backend: Some(Any::from_msg(&H::ProtoTag::default()).unwrap()),
            degree_bound: self.options.degree_bound as u64,
            security_bits: self.options.security_bits as u32,
            blowup_log2: self.options.blowup_log2 as u32,
            fold_rate: self.options.fold_rate as u32,
            max_rounds: self.options.max_rounds as u64,
            grinding_bits: self.options.grinding_bits as u32,
            nonce: self.nonce,
            points: self
                .points
                .iter()
                .map(|(&z, values)| proto::proof::Point {
                    z: utils::store_scalar(z),
                    v: values
                        .iter()
                        .map(|&value| utils::store_scalar(value))
                        .collect(),
                })
                .collect(),
            fri_roots: self
                .fri_commitment
                .roots()
                .iter()
                .map(|&hash| hash.as_bytes().to_vec())
                .collect(),
            folded_quotient: self
                .fri_commitment
                .folded_polynomial()
                .coefficients()
                .iter()
                .map(|&coefficient| utils::store_scalar(coefficient))
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
            security_bits: proto.security_bits as usize,
            blowup_log2: proto.blowup_log2 as usize,
            fold_rate: proto.fold_rate as usize,
            max_rounds: proto.max_rounds as usize,
            grinding_bits: proto.grinding_bits as usize,
        };
        options.validate::<F>()?;

        Ok(Self {
            options,
            nonce: proto.nonce,
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
            fri_commitment: fri::Commitment::new(
                proto
                    .fri_roots
                    .iter()
                    .map(|bytes| H::load_hash(bytes.as_slice()))
                    .collect::<Result<_>>()?,
                Polynomial::with_coefficients(
                    proto
                        .folded_quotient
                        .iter()
                        .map(|coefficient| utils::load_scalar(coefficient.as_slice()))
                        .collect::<Result<_>>()?,
                ),
            ),
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
                    fri::Query::from_proto(
                        options.degree_bound,
                        options.blowup_log2,
                        options.fold_rate,
                        options.max_rounds,
                        query,
                    )
                })
                .collect::<Result<_>>()?,
        })
    }
}

impl<F: PrimeField256, H: Hasher<F>> Proof<F, H> {
    /// Serializes this proof to SCPF, the Starkom Compact Proof Format.
    ///
    /// SCPF is the binary format of the proofs sent to EVM verifiers, and it minimizes the gas
    /// cost of the calldata and of decoding it (see
    /// [`FORMAT.md`](https://github.com/libernet-xyz/scpf/blob/main/FORMAT.md)). It only supports
    /// prime fields of about 256 bits, e.g. BlueSky. The data is made of a header, a pool of
    /// 32-byte words, and a stream in which unsigned integers are varints, field elements are
    /// inline 32-byte words holding their big-endian integer values, and hashes are pool
    /// references, i.e. the varint positions of their words in the pool.
    ///
    /// The schema identifier is the Keccak-256 hash of `starkom.proto.pcs.v10.Proof`, and the
    /// stream holds the fields of the [`Proof`](`proto::Proof`) protobuf in this order:
    ///
    /// 1. `degree_bound`, `security_bits`, `blowup_log2`, `fold_rate`, `max_rounds`,
    ///    `grinding_bits`, and `nonce`, as varints;
    /// 2. the number of opened points as a varint, then for every point in increasing order its
    ///    X-coordinate `z` followed by its evaluations `v`, as inline words;
    /// 3. `fri_roots` as pool references, then the number of coefficients of `folded_quotient` as
    ///    a varint, followed by the coefficients as inline words;
    /// 4. for every query, the opening of every batch in commitment order (the leaf values as
    ///    inline words, then the Merkle path as pool references), followed by the folds of all
    ///    the rounds in order (the coset values as inline words, then the Merkle path as pool
    ///    references).
    ///
    /// The openings and the queries are interleaved rather than following each other as in the
    /// protobuf, so that a verifier can check every query as soon as it reads it.
    ///
    /// The values that the decoder can derive aren't encoded: the number of FRI roots and of
    /// queries, the number of coset values, and the number of hashes of every Merkle path (from
    /// the protocol parameters), the number of evaluations of every point and of values of every
    /// leaf (from the [`Commitment`]), and the query indices (from the proof of work). The hash
    /// backend isn't encoded either, so decoders must know it in advance.
    ///
    /// NOTE: decoders rely on the shape of the proof that they derive, so the data of a proof
    /// that doesn't have it, e.g. a malformed one deserialized with [`Self::from_proto`], doesn't
    /// decode to the same proof.
    pub fn to_scpf(&self) -> Vec<u8> {
        let mut encoder = scpf::Encoder::default();
        encoder.push_usize(self.options.degree_bound);
        encoder.push_usize(self.options.security_bits);
        encoder.push_usize(self.options.blowup_log2);
        encoder.push_usize(self.options.fold_rate);
        encoder.push_usize(self.options.max_rounds);
        encoder.push_usize(self.options.grinding_bits);
        encoder.push_varint(self.nonce);
        encoder.push_usize(self.points.len());
        for (&z, values) in &self.points {
            encoder.push_scalar(z);
            encoder.push_scalars(values.as_slice());
        }
        self.fri_commitment.encode_scpf(&mut encoder);
        for (proofs, query) in self.openings.iter().zip(&self.queries) {
            for proof in proofs {
                proof.encode_scpf(&mut encoder);
            }
            query.encode_scpf(&mut encoder);
        }
        encoder.finish(*PROOF_SCHEMA_IDENTIFIER)
    }
}

/// A DEEP-FRI prover.
///
/// [`Prover`]s are constructed by [`Committer::commit()`]; see that method for details.
#[derive(Debug, Clone)]
pub struct Prover<F: Field256, H: Hasher<F>> {
    /// The protocol parameters.
    options: Options,
    /// The committed polynomial batches, along with their Merkle trees.
    batches: Vec<Batch<F, H>>,
}

impl<F: Field256, H: Hasher<F>> Prover<F, H> {
    /// Returns the protocol parameters.
    pub fn options(&self) -> &Options {
        &self.options
    }

    /// Returns the number of committed polynomials.
    pub fn num_polys(&self) -> usize {
        self.batches.iter().map(Batch::num_polys).sum()
    }

    /// Returns the number of polynomial batches, which is also the number of Merkle trees.
    pub fn num_batches(&self) -> usize {
        self.batches.len()
    }

    /// Returns the root hash of the Merkle tree of the i-th batch. `index` must be less than
    /// [`Self::num_batches()`].
    ///
    /// This value can be used to derive Fiat-Shamir challenges.
    pub fn root_hash(&self, index: usize) -> H256 {
        self.batches[index].root_hash()
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

    /// Makes a DEEP-FRI proof opening the committed polynomials at the given points.
    ///
    /// `points` contains (off-domain) X-coordinates; the corresponding Y-coordinates are computed
    /// automatically for every committed polynomial. The DEEP quotients of all opened points are
    /// combined into a single polynomial with a random linear combination, so the FRI folding
    /// argument always works on one polynomial regardless of the number of points.
    ///
    /// This method can be called several times, so that the same [`Commitment`] can be opened at
    /// different points by different proofs. However, every proof reveals more information about
    /// the committed polynomials (the opened evaluations as well as the evaluations at the queried
    /// positions of the extended domain), so a commitment that must be zero-knowledge must be
    /// opened only once.
    ///
    /// This includes grinding a proof-of-work nonce, which takes
    /// 2^[`grinding_bits`](`Options::grinding_bits`) hash evaluations on average.
    ///
    /// Panics if any of the points lies inside the evaluation domain.
    pub fn prove(&self, points: BTreeSet<F>) -> Proof<F, H> {
        check_points_off_domain(points.iter().copied(), self.options.extended_domain_size())
            .unwrap();

        let num_polys = self.num_polys();
        let points: BTreeMap<F, Vec<F>> = points
            .into_iter()
            .map(|z| {
                let mut values = Vec::with_capacity(num_polys);
                for batch in &self.batches {
                    values.append(&mut batch.evaluate(z));
                }
                (z, values)
            })
            .collect();

        let tree_roots: Vec<H256> = self.batches.iter().map(Batch::root_hash).collect();
        let (alpha, beta) = deep_challenges::<F, H>(tree_roots.as_slice(), num_polys, &points);

        let combined = Batch::combine(&self.batches, alpha);

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

        let fri_prover = fri::Prover::<F, H>::new(
            quotient,
            self.options.degree_bound,
            self.options.blowup_log2,
            self.options.fold_rate,
            self.options.max_rounds,
        );
        let fri_commitment = fri_prover.commit();

        let seed = query_seed::<F, H>(tree_roots.as_slice(), num_polys, &points, &fri_commitment);
        let (nonce, pow) = self.grind_nonce(seed);
        let indices = get_query_indices::<F, H>(pow, &self.options);
        let openings = indices
            .iter()
            .map(|&index| {
                self.batches
                    .iter()
                    .map(|batch| batch.query(index))
                    .collect()
            })
            .collect();
        let queries = indices
            .iter()
            .map(|&index| fri_prover.query(index))
            .collect();
        Proof {
            options: self.options,
            nonce,
            points,
            fri_commitment,
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
    use starkom_goldilocks::{GL, GL4};
    use starkom_koalabear::KB8;
    use std::fmt::Debug;
    use std::str::FromStr;

    fn parse<V: FromStr<Err: Debug>>(s: &'static str) -> V {
        s.parse().unwrap()
    }

    /// Builds a polynomial from its coefficients.
    fn make_polynomial<F: Field>(coefficients: &[u64]) -> Polynomial<F> {
        Polynomial::with_coefficients(
            coefficients
                .iter()
                .map(|&coefficient| F::try_from(coefficient as usize).unwrap())
                .collect(),
        )
    }

    /// Builds a batch of polynomials, each one from its coefficients.
    fn make_batch<F: Field>(batch: &[&[u64]]) -> Vec<Polynomial<F>> {
        batch
            .iter()
            .map(|coefficients| make_polynomial(coefficients))
            .collect()
    }

    /// Converts small integers into a set of points to open.
    fn make_points<F: Field256>(points: &[u64]) -> BTreeSet<F> {
        points.iter().copied().map(F::from).collect()
    }

    /// Builds a committer for the given batches of polynomials, adding them in order. Every batch
    /// is over the base field iff the corresponding flag in `base_fields` is set.
    fn make_mixed_committer<F: Field256, H: Hasher<F>>(
        options: Options,
        batches: &[&[&[u64]]],
        base_fields: &[bool],
    ) -> Committer<F, H> {
        assert_eq!(batches.len(), base_fields.len());
        let mut committer = Committer::new(options);
        for (batch, &base_field) in batches.iter().zip(base_fields) {
            if base_field {
                committer.add_base_batch(make_batch(batch));
            } else {
                committer.add_batch(make_batch(batch));
            }
        }
        committer
    }

    /// Builds a committer for the given batches of polynomials over `F` (rather than over its base
    /// field), adding them in order.
    fn make_committer<F: Field256, H: Hasher<F>>(
        options: Options,
        batches: &[&[&[u64]]],
    ) -> Committer<F, H> {
        make_mixed_committer(options, batches, &vec![false; batches.len()])
    }

    /// The protocol parameters used in most tests.
    const OPTIONS: Options = Options {
        degree_bound: 4,
        security_bits: 128,
        blowup_log2: 2,
        fold_rate: 1,
        max_rounds: usize::MAX,
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

    /// A commitment and a proof using the SHA2 backend.
    type SampleProof<F> = (Commitment<F, Sha2Hash<F>>, Proof<F, Sha2Hash<F>>);

    /// Commits to [`SAMPLE_BATCHES`] with the given options and proves their evaluations at
    /// [`SAMPLE_POINTS`].
    fn sample_proof(options: Options) -> SampleProof<BS> {
        let committer = make_committer(options, &SAMPLE_BATCHES);
        let (commitment, prover) = committer.commit();
        let proof = prover.prove(make_points(&SAMPLE_POINTS));
        (commitment, proof)
    }

    /// The `base_field` flags of [`SAMPLE_BATCHES`] in the tests with mixed batches.
    const MIXED_BASE_FIELDS: [bool; 2] = [true, false];

    /// Commits to [`SAMPLE_BATCHES`] over Goldilocks^4, the first batch over the base field and the
    /// second one over the extension field (see [`MIXED_BASE_FIELDS`]), and proves their
    /// evaluations at [`SAMPLE_POINTS`].
    fn sample_mixed_proof() -> SampleProof<GL4> {
        let committer = make_mixed_committer(OPTIONS, &SAMPLE_BATCHES, &MIXED_BASE_FIELDS);
        let (commitment, prover) = committer.commit();
        let proof = prover.prove(make_points(&SAMPLE_POINTS));
        (commitment, proof)
    }

    /// Returns the smallest nonce greater than `proof.nonce` that satisfies (`valid == true`) or
    /// fails (`valid == false`) the proof of work of `proof` against `commitment`.
    fn find_nonce(
        commitment: &Commitment<BS, Sha2Hash<BS>>,
        proof: &Proof<BS, Sha2Hash<BS>>,
        valid: bool,
    ) -> u64 {
        let seed = query_seed::<BS, Sha2Hash<BS>>(
            commitment.tree_roots().as_slice(),
            commitment.num_polys(),
            &proof.points,
            &proof.fri_commitment,
        );
        ((proof.nonce + 1)..)
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

    fn test_num_queries_impl(
        security_bits: usize,
        blowup_log2: usize,
        grinding_bits: usize,
        expected: usize,
    ) {
        let result = num_queries(security_bits, blowup_log2, grinding_bits);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_num_queries_without_grinding() {
        test_num_queries_impl(128, 1, 0, 128);
        test_num_queries_impl(128, 4, 0, 32);
        test_num_queries_impl(128, 6, 0, 22);
    }

    #[test]
    fn test_num_queries_with_grinding() {
        test_num_queries_impl(128, 4, 1, 32);
        test_num_queries_impl(128, 4, 4, 31);
        test_num_queries_impl(128, 4, 20, 27);
        test_num_queries_impl(128, 6, 20, 18);
    }

    #[test]
    fn test_num_queries_with_full_grinding() {
        test_num_queries_impl(128, 4, 128, 1);
        test_num_queries_impl(128, 4, 256, 1);
        test_num_queries_impl(128, 4, 257, 1);
        test_num_queries_impl(128, 4, usize::MAX, 1);
    }

    #[test]
    fn test_num_queries_with_different_security_levels() {
        test_num_queries_impl(0, 4, 0, 1);
        test_num_queries_impl(1, 4, 0, 1);
        test_num_queries_impl(64, 4, 0, 16);
        test_num_queries_impl(96, 1, 26, 70);
        test_num_queries_impl(100, 3, 0, 34);
        test_num_queries_impl(256, 4, 0, 64);
        test_num_queries_impl(256, 4, 20, 59);
        test_num_queries_impl(256, 8, 20, 30);
        test_num_queries_impl(256, 4, 250, 2);
        test_num_queries_impl(256, 4, 256, 1);
    }

    #[test]
    fn test_options_extended_domain_size() {
        let options = Options {
            degree_bound: 4096,
            security_bits: 128,
            blowup_log2: 6,
            fold_rate: 1,
            max_rounds: usize::MAX,
            grinding_bits: 20,
        };
        assert_eq!(options.extended_domain_size(), 1 << 18);
    }

    #[test]
    fn test_options_num_queries_without_grinding() {
        let options = Options {
            degree_bound: 4096,
            security_bits: 128,
            blowup_log2: 6,
            fold_rate: 1,
            max_rounds: usize::MAX,
            grinding_bits: 0,
        };
        assert_eq!(options.num_queries(), 22);
    }

    #[test]
    fn test_options_num_queries_with_grinding() {
        let options = Options {
            degree_bound: 4096,
            security_bits: 128,
            blowup_log2: 6,
            fold_rate: 1,
            max_rounds: usize::MAX,
            grinding_bits: 20,
        };
        assert_eq!(options.num_queries(), 18);
    }

    fn test_options_num_queries_impl(security_bits: usize, grinding_bits: usize, expected: usize) {
        let options = Options {
            degree_bound: 4096,
            security_bits,
            blowup_log2: 6,
            fold_rate: 1,
            max_rounds: usize::MAX,
            grinding_bits,
        };
        assert_eq!(options.num_queries(), expected);
    }

    #[test]
    fn test_options_num_queries_with_different_security_levels() {
        test_options_num_queries_impl(64, 0, 11);
        test_options_num_queries_impl(64, 20, 8);
        test_options_num_queries_impl(256, 0, 43);
        test_options_num_queries_impl(256, 20, 40);
    }

    fn test_options_num_rounds_impl(
        degree_bound: usize,
        fold_rate: usize,
        max_rounds: usize,
        expected: usize,
    ) {
        let options = Options {
            degree_bound,
            security_bits: 128,
            blowup_log2: 6,
            fold_rate,
            max_rounds,
            grinding_bits: 20,
        };
        assert_eq!(options.num_rounds(), expected);
    }

    #[test]
    fn test_options_num_rounds() {
        test_options_num_rounds_impl(1, 3, usize::MAX, 0);
        test_options_num_rounds_impl(4096, 1, usize::MAX, 12);
        test_options_num_rounds_impl(4096, 3, usize::MAX, 4);
        test_options_num_rounds_impl(4096, 5, usize::MAX, 2);
        test_options_num_rounds_impl(4096, 12, usize::MAX, 1);
        test_options_num_rounds_impl(4096, 20, usize::MAX, 0);
    }

    #[test]
    fn test_options_num_rounds_with_limit() {
        test_options_num_rounds_impl(4096, 1, 0, 0);
        test_options_num_rounds_impl(4096, 3, 2, 2);
        test_options_num_rounds_impl(16384, 3, 2, 2);
        test_options_num_rounds_impl(16384, 3, 9, 4);
    }

    fn test_valid_options_impl<F: Field>(
        degree_bound: usize,
        blowup_log2: usize,
        fold_rate: usize,
        grinding_bits: usize,
    ) {
        let options = Options {
            degree_bound,
            security_bits: 128,
            blowup_log2,
            fold_rate,
            max_rounds: usize::MAX,
            grinding_bits,
        };
        let result = options.validate::<F>();
        assert!(result.is_ok());
    }

    fn test_invalid_options_impl<F: Field>(
        degree_bound: usize,
        blowup_log2: usize,
        fold_rate: usize,
        grinding_bits: usize,
    ) {
        let options = Options {
            degree_bound,
            security_bits: 128,
            blowup_log2,
            fold_rate,
            max_rounds: usize::MAX,
            grinding_bits,
        };
        let result = options.validate::<F>();
        assert!(result.is_err());
    }

    #[test]
    fn test_valid_options() {
        test_valid_options_impl::<BS>(4, 2, 1, 8);
        test_valid_options_impl::<GL4>(4, 2, 1, 8);
    }

    #[test]
    fn test_options_with_largest_domain() {
        test_valid_options_impl::<BS>(1 << 59, 3, 1, 0);
        test_valid_options_impl::<GL4>(1 << 29, 3, 1, 0);
    }

    #[test]
    fn test_options_with_full_grinding() {
        test_valid_options_impl::<BS>(4, 2, 1, 128);
        test_valid_options_impl::<BS>(4, 2, 1, 256);
        test_valid_options_impl::<GL4>(4, 2, 1, 128);
        test_valid_options_impl::<GL4>(4, 2, 1, 256);
    }

    #[test]
    fn test_options_with_zero_degree_bound() {
        test_invalid_options_impl::<BS>(0, 2, 1, 0);
        test_invalid_options_impl::<GL4>(0, 2, 1, 0);
    }

    #[test]
    fn test_options_with_degree_bound_not_power_of_two() {
        test_invalid_options_impl::<BS>(6, 2, 1, 0);
        test_invalid_options_impl::<GL4>(6, 2, 1, 0);
    }

    #[test]
    fn test_options_with_zero_blowup_log2() {
        test_invalid_options_impl::<BS>(4, 0, 1, 0);
        test_invalid_options_impl::<GL4>(4, 0, 1, 0);
    }

    #[test]
    fn test_options_with_oversized_domain() {
        test_invalid_options_impl::<BS>(1 << 60, 3, 1, 0);
        test_invalid_options_impl::<GL4>(1 << 30, 3, 1, 0);
    }

    #[test]
    fn test_options_with_large_fold_rate() {
        test_valid_options_impl::<BS>(4, 2, 10, 0);
        test_valid_options_impl::<GL4>(4, 2, 10, 0);
    }

    #[test]
    fn test_options_with_zero_fold_rate() {
        test_invalid_options_impl::<BS>(4, 2, 0, 0);
        test_invalid_options_impl::<GL4>(4, 2, 0, 0);
    }

    #[test]
    fn test_options_with_unsatisfiable_grinding() {
        test_invalid_options_impl::<BS>(4, 2, 1, 257);
        test_invalid_options_impl::<BS>(4, 2, 1, usize::MAX);
        test_invalid_options_impl::<GL4>(4, 2, 1, 257);
        test_invalid_options_impl::<GL4>(4, 2, 1, usize::MAX);
    }

    /// The protocol parameters expected by the verifier in the `Options::satisfies` tests.
    const EXPECTED_OPTIONS: Options = Options {
        degree_bound: 4096,
        security_bits: 128,
        blowup_log2: 6,
        fold_rate: 3,
        max_rounds: 2,
        grinding_bits: 20,
    };

    #[test]
    fn test_equal_options_satisfy() {
        assert!(EXPECTED_OPTIONS.satisfies(&EXPECTED_OPTIONS));
    }

    #[test]
    fn test_options_with_more_grinding_bits_satisfy() {
        let options = Options {
            grinding_bits: 21,
            ..EXPECTED_OPTIONS
        };
        assert!(options.satisfies(&EXPECTED_OPTIONS));
    }

    #[test]
    fn test_options_with_full_grinding_satisfy() {
        let options = Options {
            grinding_bits: PROOF_OF_WORK_BITS,
            ..EXPECTED_OPTIONS
        };
        assert!(options.satisfies(&EXPECTED_OPTIONS));
    }

    #[test]
    fn test_options_with_fewer_grinding_bits_dont_satisfy() {
        let options = Options {
            grinding_bits: 19,
            ..EXPECTED_OPTIONS
        };
        assert!(!options.satisfies(&EXPECTED_OPTIONS));
    }

    #[test]
    fn test_options_without_grinding_dont_satisfy() {
        let options = Options {
            grinding_bits: 0,
            ..EXPECTED_OPTIONS
        };
        assert!(!options.satisfies(&EXPECTED_OPTIONS));
    }

    #[test]
    fn test_options_with_higher_security_level_satisfy() {
        let options = Options {
            security_bits: 256,
            ..EXPECTED_OPTIONS
        };
        assert!(options.satisfies(&EXPECTED_OPTIONS));
    }

    #[test]
    fn test_options_with_lower_security_level_dont_satisfy() {
        let options = Options {
            security_bits: 127,
            ..EXPECTED_OPTIONS
        };
        assert!(!options.satisfies(&EXPECTED_OPTIONS));
    }

    #[test]
    fn test_options_with_higher_security_level_and_fewer_grinding_bits_dont_satisfy() {
        let options = Options {
            security_bits: 256,
            grinding_bits: 19,
            ..EXPECTED_OPTIONS
        };
        assert!(!options.satisfies(&EXPECTED_OPTIONS));
    }

    fn test_options_dont_satisfy_impl(options: Options) {
        assert!(!options.satisfies(&EXPECTED_OPTIONS));
        assert!(
            !Options {
                grinding_bits: 21,
                ..options
            }
            .satisfies(&EXPECTED_OPTIONS)
        );
        assert!(
            !Options {
                security_bits: 256,
                ..options
            }
            .satisfies(&EXPECTED_OPTIONS)
        );
    }

    #[test]
    fn test_options_with_different_degree_bound_dont_satisfy() {
        test_options_dont_satisfy_impl(Options {
            degree_bound: 2048,
            ..EXPECTED_OPTIONS
        });
        test_options_dont_satisfy_impl(Options {
            degree_bound: 8192,
            ..EXPECTED_OPTIONS
        });
    }

    #[test]
    fn test_options_with_different_blowup_dont_satisfy() {
        test_options_dont_satisfy_impl(Options {
            blowup_log2: 5,
            ..EXPECTED_OPTIONS
        });
        test_options_dont_satisfy_impl(Options {
            blowup_log2: 7,
            ..EXPECTED_OPTIONS
        });
    }

    #[test]
    fn test_options_with_different_fold_rate_dont_satisfy() {
        test_options_dont_satisfy_impl(Options {
            fold_rate: 2,
            ..EXPECTED_OPTIONS
        });
        test_options_dont_satisfy_impl(Options {
            fold_rate: 4,
            ..EXPECTED_OPTIONS
        });
    }

    #[test]
    fn test_options_with_different_max_rounds_dont_satisfy() {
        test_options_dont_satisfy_impl(Options {
            max_rounds: 1,
            ..EXPECTED_OPTIONS
        });
        test_options_dont_satisfy_impl(Options {
            max_rounds: 3,
            ..EXPECTED_OPTIONS
        });
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
        test_check_proof_of_work_impl(hash, usize::MAX, false);
    }

    #[test]
    fn test_query_indices() {
        let options = Options {
            degree_bound: 4,
            security_bits: 128,
            blowup_log2: 4,
            fold_rate: 1,
            max_rounds: usize::MAX,
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

    /// The Merkle roots of the committed polynomials used to test the Fiat-Shamir transcript.
    fn transcript_tree_roots() -> [H256; 2] {
        [H256::repeat_byte(0x56), H256::repeat_byte(0x78)]
    }

    /// The opened points and evaluations (of three polynomials) used to test the Fiat-Shamir
    /// transcript.
    fn transcript_points() -> BTreeMap<BS, Vec<BS>> {
        BTreeMap::from([
            (
                BS::from_const(123),
                vec![BS::from_const(1), BS::from_const(2), BS::from_const(3)],
            ),
            (
                BS::from_const(456),
                vec![BS::from_const(4), BS::from_const(5), BS::from_const(6)],
            ),
        ])
    }

    /// A FRI commitment with two roots and a folded quotient of two coefficients, used to test the
    /// Fiat-Shamir transcript.
    fn transcript_fri_commitment() -> fri::Commitment<BS> {
        fri::Commitment::new(
            vec![H256::repeat_byte(0x9a), H256::repeat_byte(0xbc)],
            Polynomial::with_coefficients(vec![BS::from_const(7), BS::from_const(8)]),
        )
    }

    #[test]
    fn test_deep_challenges() {
        let (alpha, beta) =
            deep_challenges::<BS, Sha2Hash<BS>>(&transcript_tree_roots(), 3, &transcript_points());
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
    fn test_query_seed() {
        let seed = query_seed::<BS, Sha2Hash<BS>>(
            &transcript_tree_roots(),
            3,
            &transcript_points(),
            &transcript_fri_commitment(),
        );
        assert_eq!(
            seed,
            parse("0xc3749a4732349eda6ce66d199ac4fa709842b54fa6e99c88ed80fb2d75f411c5")
        );
    }

    #[test]
    fn test_query_seed_binds_points() {
        let tree_roots = transcript_tree_roots();
        let points = transcript_points();
        let mut other_points = points.clone();
        let values = other_points.remove(&BS::from_const(456)).unwrap();
        other_points.insert(BS::from_const(789), values);
        let fri_commitment = transcript_fri_commitment();
        let seed = query_seed::<BS, Sha2Hash<BS>>(&tree_roots, 3, &points, &fri_commitment);
        let other_seed =
            query_seed::<BS, Sha2Hash<BS>>(&tree_roots, 3, &other_points, &fri_commitment);
        assert_ne!(seed, other_seed);
    }

    #[test]
    fn test_query_seed_binds_evaluations() {
        let tree_roots = transcript_tree_roots();
        let points = transcript_points();
        let mut other_points = points.clone();
        other_points.get_mut(&BS::from_const(456)).unwrap()[2] += BS::ONE;
        let fri_commitment = transcript_fri_commitment();
        let seed = query_seed::<BS, Sha2Hash<BS>>(&tree_roots, 3, &points, &fri_commitment);
        let other_seed =
            query_seed::<BS, Sha2Hash<BS>>(&tree_roots, 3, &other_points, &fri_commitment);
        assert_ne!(seed, other_seed);
    }

    #[test]
    #[should_panic]
    fn test_committer_with_invalid_options() {
        Committer::<BS, Sha2Hash<BS>>::new(Options {
            degree_bound: 6,
            security_bits: 128,
            blowup_log2: 2,
            fold_rate: 1,
            max_rounds: usize::MAX,
            grinding_bits: 0,
        });
    }

    #[test]
    #[should_panic]
    fn test_commit_without_batches() {
        Committer::<BS, Sha2Hash<BS>>::new(OPTIONS).commit();
    }

    #[test]
    fn test_committer_options() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        assert_eq!(*committer.options(), OPTIONS);
    }

    #[test]
    fn test_committer_num_batches() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        assert_eq!(committer.num_batches(), 2);
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
        assert_eq!(committer.num_batches(), 3);
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
    fn test_add_base_batch() {
        let mut committer = make_committer::<GL4, Sha2Hash<GL4>>(OPTIONS, &SAMPLE_BATCHES);
        let index = committer.add_base_batch(make_batch(&[&[12, 34]]));
        assert_eq!(index, 2);
        assert_eq!(committer.num_batches(), 3);
    }

    #[test]
    #[should_panic]
    fn test_add_empty_base_batch() {
        let mut committer = make_committer::<GL4, Sha2Hash<GL4>>(OPTIONS, &SAMPLE_BATCHES);
        committer.add_base_batch(vec![]);
    }

    #[test]
    #[should_panic]
    fn test_add_base_batch_exceeding_degree_bound() {
        let mut committer = make_committer::<GL4, Sha2Hash<GL4>>(OPTIONS, &SAMPLE_BATCHES);
        committer.add_base_batch(make_batch(&[&[12, 34, 56, 78, 90]]));
    }

    /// A batch of four polynomials over Goldilocks and a batch of one polynomial over Goldilocks^4
    /// whose coordinates are those four polynomials have the same leaf values on the wire, but
    /// different roots.
    #[test]
    fn test_reinterpreted_batch_has_different_root() {
        let mut base_committer = Committer::<GL4, Sha2Hash<GL4>>::new(OPTIONS);
        base_committer.add_base_batch(make_batch(&[&[12, 34], &[56, 78], &[90, 12], &[34, 56]]));
        let mut extension_committer = Committer::<GL4, Sha2Hash<GL4>>::new(OPTIONS);
        extension_committer.add_batch(vec![Polynomial::with_coefficients(vec![
            GL4::from_base_coordinates(&[12u32, 56, 90, 34].map(GL::from)),
            GL4::from_base_coordinates(&[34u32, 78, 12, 56].map(GL::from)),
        ])]);
        assert_eq!(
            base_committer.batches[0].query(5).leaf(),
            extension_committer.batches[0].query(5).leaf()
        );
        assert_ne!(
            base_committer.root_hash(0),
            extension_committer.root_hash(0)
        );
    }

    #[test]
    fn test_commitment_batches() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit();
        assert_eq!(
            commitment.batches(),
            [
                BatchCommitment {
                    base_field: false,
                    num_polys: 2,
                    root: parse(
                        "0x8a560267cca9607a73ad5186d575443499a19685f460be07ff3928387dcb5a48"
                    ),
                },
                BatchCommitment {
                    base_field: false,
                    num_polys: 1,
                    root: parse(
                        "0x917d1c95f6c177b6a0ff720557fa391335d0e6710f0029c60bbef44234a03b85"
                    ),
                },
            ]
        );
    }

    #[test]
    fn test_commitment_num_batches() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit();
        assert_eq!(commitment.num_batches(), 2);
    }

    #[test]
    fn test_commitment_batches_with_base_batch() {
        let committer = make_mixed_committer::<GL4, Sha2Hash<GL4>>(
            OPTIONS,
            &SAMPLE_BATCHES,
            &MIXED_BASE_FIELDS,
        );
        let roots = [committer.root_hash(0), committer.root_hash(1)];
        let (commitment, _) = committer.commit();
        assert_eq!(
            commitment.batches(),
            [
                BatchCommitment {
                    base_field: true,
                    num_polys: 2,
                    root: roots[0],
                },
                BatchCommitment {
                    base_field: false,
                    num_polys: 1,
                    root: roots[1],
                },
            ]
        );
    }

    #[test]
    fn test_batch_commitment_accessors() {
        let (commitment, _) = sample_mixed_proof();
        let batch = commitment.batches()[0];
        assert!(batch.base_field());
        assert_eq!(batch.num_polys(), 2);
        assert_eq!(batch.root(), commitment.batches[0].root);
    }

    #[test]
    fn test_commitment_with_base_batch_to_proto() {
        let (commitment, _) = sample_mixed_proof();
        let proto = commitment.to_proto();
        assert_eq!(
            proto.batches,
            [
                proto::commitment::Batch {
                    root: commitment.batches[0].root.as_bytes().to_vec(),
                    num_polys: 2,
                    base_field: true,
                },
                proto::commitment::Batch {
                    root: commitment.batches[1].root.as_bytes().to_vec(),
                    num_polys: 1,
                    base_field: false,
                },
            ]
        );
    }

    #[test]
    fn test_commitment_with_base_batch_from_proto() {
        let (commitment, _) = sample_mixed_proof();
        let proto = commitment.to_proto();
        let decoded = Commitment::<GL4, Sha2Hash<GL4>>::from_proto(&proto).unwrap();
        assert_eq!(decoded, commitment);
    }

    #[test]
    fn test_commitment_transcript_hash_of_first_batch() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit();
        let hash = commitment.transcript_hash(1);
        assert_eq!(
            hash,
            parse("0x608c3f1c8a1354d04f2b596ff716e064e3d98d59ff9523bb2e8ecd1061c81816")
        );
    }

    #[test]
    fn test_commitment_transcript_hash_of_all_batches() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit();
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
        let (commitment, _) = committer.commit();
        commitment.transcript_hash(0);
    }

    #[test]
    #[should_panic]
    fn test_commitment_transcript_hash_of_too_many_batches() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit();
        commitment.transcript_hash(3);
    }

    #[test]
    fn test_commitment_to_proto() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit();
        let proto = commitment.to_proto();
        assert_eq!(
            proto.batches,
            [
                proto::commitment::Batch {
                    root: commitment.batches[0].root.as_bytes().to_vec(),
                    num_polys: 2,
                    base_field: false,
                },
                proto::commitment::Batch {
                    root: commitment.batches[1].root.as_bytes().to_vec(),
                    num_polys: 1,
                    base_field: false,
                },
            ]
        );
    }

    #[test]
    fn test_commitment_from_proto() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit();
        let proto = commitment.to_proto();
        let decoded = Commitment::<BS, Sha2Hash<BS>>::from_proto(&proto).unwrap();
        assert_eq!(decoded, commitment);
    }

    #[test]
    fn test_reject_commitment_without_batches() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit();
        let mut proto = commitment.to_proto();
        proto.batches.clear();
        let result = Commitment::<BS, Sha2Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_commitment_with_short_root() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit();
        let mut proto = commitment.to_proto();
        proto.batches[1].root.pop();
        let result = Commitment::<BS, Sha2Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_commitment_without_hash_backend() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit();
        let mut proto = commitment.to_proto();
        proto.hash_backend = None;
        let result = Commitment::<BS, Sha2Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_commitment_with_wrong_hash_backend() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit();
        let proto = commitment.to_proto();
        let result = Commitment::<BS, Keccak256Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    /// The leaves of the batch over Goldilocks^4 have 4 values per polynomial.
    #[test]
    fn test_commitment_with_largest_leaves_from_proto() {
        let (commitment, _) = sample_mixed_proof();
        let mut proto = commitment.to_proto();
        proto.batches[1].num_polys = (usize::MAX / 4) as u64;
        let decoded = Commitment::<GL4, Sha2Hash<GL4>>::from_proto(&proto).unwrap();
        assert_eq!(decoded.batches()[1].num_polys(), usize::MAX / 4);
    }

    /// The leaves of the batch over Goldilocks^4 would have more than `usize::MAX` values.
    #[test]
    fn test_reject_commitment_with_oversized_leaves() {
        let (commitment, _) = sample_mixed_proof();
        let mut proto = commitment.to_proto();
        proto.batches[1].num_polys = (usize::MAX / 4 + 1) as u64;
        let result = Commitment::<GL4, Sha2Hash<GL4>>::from_proto(&proto);
        assert!(result.is_err());
    }

    /// The leaves of the base-field batch have one value per polynomial, so only the total number
    /// of polynomials limits its size.
    #[test]
    fn test_commitment_with_largest_base_batch_from_proto() {
        let (commitment, _) = sample_mixed_proof();
        let mut proto = commitment.to_proto();
        proto.batches[0].num_polys = (usize::MAX - 1) as u64;
        let decoded = Commitment::<GL4, Sha2Hash<GL4>>::from_proto(&proto).unwrap();
        assert_eq!(decoded.num_polys(), usize::MAX);
    }

    #[test]
    fn test_reject_commitment_with_too_many_polynomials() {
        let (commitment, _) = sample_mixed_proof();
        let mut proto = commitment.to_proto();
        proto.batches[0].num_polys = usize::MAX as u64;
        let result = Commitment::<GL4, Sha2Hash<GL4>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_prover_options() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (_, prover) = committer.commit();
        assert_eq!(*prover.options(), OPTIONS);
    }

    #[test]
    fn test_prover_num_polys() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (_, prover) = committer.commit();
        assert_eq!(prover.num_polys(), 3);
    }

    #[test]
    fn test_prover_num_batches() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (_, prover) = committer.commit();
        assert_eq!(prover.num_batches(), 2);
    }

    #[test]
    fn test_prover_root_hashes() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (_, prover) = committer.commit();
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
    #[should_panic]
    fn test_prove_point_on_domain() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (_, prover) = committer.commit();
        let z = Polynomial::<BS>::coset_element2(3, OPTIONS.extended_domain_size());
        prover.prove(BTreeSet::from([z]));
    }

    #[test]
    fn test_proof_points() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (_, prover) = committer.commit();
        let proof = prover.prove(make_points(&SAMPLE_POINTS));
        assert_eq!(
            *proof.points(),
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
    fn test_proof_points_of_base_batch() {
        let (_, proof) = sample_mixed_proof();
        assert_eq!(
            *proof.points(),
            BTreeMap::from([
                (
                    GL4::from(123u64),
                    vec![
                        GL4::from(145999044u64),
                        GL4::from(84410022u64),
                        GL4::from(64126386u64),
                    ]
                ),
                (
                    GL4::from(456u64),
                    vec![
                        GL4::from(7407527580u64),
                        GL4::from(4276015554u64),
                        GL4::from(3235519818u64),
                    ]
                ),
            ])
        );
    }

    #[test]
    fn test_proof_points_of_base_batch_at_extension_point() {
        let committer = make_mixed_committer::<GL4, Sha2Hash<GL4>>(
            OPTIONS,
            &SAMPLE_BATCHES,
            &MIXED_BASE_FIELDS,
        );
        let (_, prover) = committer.commit();
        let z = GL4::from_base_coordinates(&[1u32, 2, 3, 4].map(GL::from));
        let proof = prover.prove(BTreeSet::from([z]));
        assert_eq!(
            proof.points()[&z],
            [
                make_polynomial::<GL4>(&[12, 34, 56, 78]).evaluate(z),
                make_polynomial::<GL4>(&[42, 43, 44, 45]).evaluate(z),
                make_polynomial::<GL4>(&[90, 78, 56, 34]).evaluate(z),
            ]
        );
    }

    #[test]
    fn test_proof_options() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (_, prover) = committer.commit();
        let proof = prover.prove(make_points(&SAMPLE_POINTS));
        assert_eq!(*proof.options(), OPTIONS);
    }

    #[test]
    fn test_commitment_num_polys() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, _) = committer.commit();
        assert_eq!(commitment.num_polys(), 3);
    }

    #[test]
    fn test_commitment_num_polys_with_base_batch() {
        let (commitment, _) = sample_mixed_proof();
        assert_eq!(commitment.num_polys(), 3);
    }

    #[test]
    fn test_proof_fri_roots() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (_, prover) = committer.commit();
        let proof = prover.prove(make_points(&SAMPLE_POINTS));
        assert_eq!(
            proof.fri_commitment.roots(),
            [
                parse::<H256>("0x1cfc697bb66e15c8357f7a749010a210ae829f520944e21e4df00cfa7212ec8c"),
                parse("0xc61b2df672d90a4aef984d6b7d485f673cec75f1868d39e6ba659a4ec3883f6f"),
            ]
        );
    }

    #[test]
    fn test_proof_folded_quotient() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (_, prover) = committer.commit();
        let proof = prover.prove(make_points(&SAMPLE_POINTS));
        assert_eq!(
            proof.fri_commitment.folded_polynomial().coefficients(),
            [parse::<BS>(
                "0x0e9f1ad4b8e8061af017e701db60283baf50808a86959003141baed02ddbf544"
            )]
        );
    }

    #[test]
    fn test_proof_nonce_without_grinding() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (_, prover) = committer.commit();
        let proof = prover.prove(make_points(&SAMPLE_POINTS));
        assert_eq!(proof.nonce(), 0);
    }

    #[test]
    fn test_proof_nonce_with_grinding() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(GROUND_OPTIONS, &SAMPLE_BATCHES);
        let (_, prover) = committer.commit();
        let proof = prover.prove(make_points(&SAMPLE_POINTS));
        assert_eq!(proof.nonce(), 204);
    }

    #[test]
    fn test_proof_size_without_grinding() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (_, prover) = committer.commit();
        let proof = prover.prove(make_points(&SAMPLE_POINTS));
        assert_eq!(proof.openings.len(), 64);
        assert_eq!(proof.queries.len(), 64);
    }

    #[test]
    fn test_proof_size_with_grinding() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(GROUND_OPTIONS, &SAMPLE_BATCHES);
        let (_, prover) = committer.commit();
        let proof = prover.prove(make_points(&SAMPLE_POINTS));
        assert_eq!(proof.openings.len(), 60);
        assert_eq!(proof.queries.len(), 60);
    }

    fn test_proof_num_queries_impl(security_bits: usize, grinding_bits: usize, expected: usize) {
        let options = Options {
            security_bits,
            grinding_bits,
            ..OPTIONS
        };
        let (commitment, proof) = sample_proof(options);
        let result = proof.verify(&commitment);
        assert_eq!(proof.options().security_bits, security_bits);
        assert_eq!(proof.openings.len(), expected);
        assert_eq!(proof.queries.len(), expected);
        assert!(result.is_ok());
    }

    #[test]
    fn test_proof_num_queries_with_different_security_levels() {
        test_proof_num_queries_impl(0, 0, 1);
        test_proof_num_queries_impl(3, 0, 2);
        test_proof_num_queries_impl(64, 0, 32);
        test_proof_num_queries_impl(128, 0, 64);
        test_proof_num_queries_impl(256, 0, 128);
        test_proof_num_queries_impl(8, 8, 1);
        test_proof_num_queries_impl(128, 8, 60);
        test_proof_num_queries_impl(256, 8, 124);
    }

    fn test_honest_proof_impl<F: Field256, H: Hasher<F>>(
        batches: &[&[&[u64]]],
        base_fields: &[bool],
        points: &[u64],
        options: Options,
    ) {
        let committer = make_mixed_committer::<F, H>(options, batches, base_fields);
        let (commitment, prover) = committer.commit();
        let proof = prover.prove(make_points(points));
        let result = proof.verify(&commitment);
        assert!(result.is_ok());
    }

    fn test_honest_proof_with_fields(
        batches: &[&[&[u64]]],
        base_fields: &[bool],
        points: &[u64],
        options: Options,
    ) {
        test_honest_proof_impl::<BS, Sha2Hash<BS>>(batches, base_fields, points, options);
        test_honest_proof_impl::<BS, Keccak256Hash<BS>>(batches, base_fields, points, options);
        test_honest_proof_impl::<GL4, Sha2Hash<GL4>>(batches, base_fields, points, options);
        test_honest_proof_impl::<GL4, Keccak256Hash<GL4>>(batches, base_fields, points, options);
        test_honest_proof_impl::<KB8, Sha2Hash<KB8>>(batches, base_fields, points, options);
        test_honest_proof_impl::<KB8, Keccak256Hash<KB8>>(batches, base_fields, points, options);
    }

    /// Tests the given proof with every field and hash backend, committing all batches over the
    /// extension field and then all batches over the base field.
    fn test_honest_proof_with_options(batches: &[&[&[u64]]], points: &[u64], options: Options) {
        let num_batches = batches.len();
        test_honest_proof_with_fields(batches, &vec![false; num_batches], points, options);
        test_honest_proof_with_fields(batches, &vec![true; num_batches], points, options);
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
    fn test_base_and_extension_batches() {
        test_honest_proof_with_fields(&SAMPLE_BATCHES, &MIXED_BASE_FIELDS, &SAMPLE_POINTS, OPTIONS);
    }

    #[test]
    fn test_extension_and_base_batches() {
        test_honest_proof_with_fields(&SAMPLE_BATCHES, &[false, true], &SAMPLE_POINTS, OPTIONS);
    }

    #[test]
    fn test_three_mixed_batches() {
        test_honest_proof_with_fields(
            &[
                &[&[12, 34, 56, 78], &[42, 43]],
                &[&[90, 78, 56, 34]],
                &[&[1, 2, 3, 4], &[5, 6, 7, 8], &[9]],
            ],
            &[true, false, true],
            &[123, 456, 789],
            GROUND_OPTIONS,
        );
    }

    #[test]
    fn test_honest_proof_blowup_2() {
        test_honest_proof_with_options(
            &[&[&[12]]],
            &[123],
            Options {
                degree_bound: 1,
                security_bits: 128,
                blowup_log2: 1,
                fold_rate: 1,
                max_rounds: usize::MAX,
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
                security_bits: 128,
                blowup_log2: 3,
                fold_rate: 1,
                max_rounds: usize::MAX,
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
    fn test_several_proofs_for_the_same_commitment() {
        let committer = make_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES);
        let (commitment, prover) = committer.commit();
        let proof1 = prover.prove(make_points(&SAMPLE_POINTS));
        let proof2 = prover.prove(make_points(&[789]));
        assert!(proof1.verify(&commitment).is_ok());
        assert!(proof2.verify(&commitment).is_ok());
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
    fn test_honest_proof_with_fold_rate_2() {
        test_honest_proof_with_options(
            &SAMPLE_BATCHES,
            &SAMPLE_POINTS,
            Options {
                fold_rate: 2,
                ..OPTIONS
            },
        );
        test_honest_proof_with_options(
            &[&[&[12, 34, 56, 78, 90, 12, 34, 56]]],
            &[123, 456],
            Options {
                degree_bound: 8,
                fold_rate: 2,
                ..OPTIONS
            },
        );
    }

    #[test]
    fn test_honest_proof_with_fold_rate_3() {
        test_honest_proof_with_options(
            &[&[&[12, 34, 56, 78, 90, 12, 34, 56]]],
            &[123, 456],
            Options {
                degree_bound: 8,
                fold_rate: 3,
                ..OPTIONS
            },
        );
        test_honest_proof_with_options(
            &[
                &[&[12, 34, 56, 78, 90, 12, 34, 56], &[42, 43]],
                &[&[1, 2, 3, 4]],
            ],
            &[123, 456],
            Options {
                degree_bound: 16,
                fold_rate: 3,
                ..GROUND_OPTIONS
            },
        );
    }

    #[test]
    fn test_honest_proof_without_rounds() {
        test_honest_proof_with_options(
            &SAMPLE_BATCHES,
            &SAMPLE_POINTS,
            Options {
                max_rounds: 0,
                ..OPTIONS
            },
        );
        test_honest_proof_with_options(
            &SAMPLE_BATCHES,
            &SAMPLE_POINTS,
            Options {
                max_rounds: 0,
                ..GROUND_OPTIONS
            },
        );
    }

    #[test]
    fn test_honest_proof_with_early_stop() {
        test_honest_proof_with_options(
            &[&[&[12, 34, 56, 78, 90, 12, 34, 56]]],
            &[123, 456],
            Options {
                degree_bound: 8,
                max_rounds: 1,
                ..OPTIONS
            },
        );
        test_honest_proof_with_options(
            &[
                &[&[12, 34, 56, 78, 90, 12, 34, 56], &[42, 43]],
                &[&[1, 2, 3, 4]],
            ],
            &[123, 456],
            Options {
                degree_bound: 16,
                fold_rate: 2,
                max_rounds: 1,
                ..GROUND_OPTIONS
            },
        );
    }

    #[test]
    fn test_reject_mismatched_max_rounds() {
        let (commitment, mut proof) = sample_proof(OPTIONS);
        proof.options.max_rounds = 1;
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_tampered_folded_quotient_without_rounds() {
        let (commitment, mut proof) = sample_proof(Options {
            max_rounds: 0,
            ..OPTIONS
        });
        let mut coefficients = proof
            .fri_commitment
            .folded_polynomial()
            .coefficients()
            .to_vec();
        coefficients[1] += BS::ONE;
        proof.fri_commitment =
            fri::Commitment::new(vec![], Polynomial::with_coefficients(coefficients));
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_tampered_fri_root() {
        let (commitment, mut proof) = sample_proof(OPTIONS);
        let mut roots = proof.fri_commitment.roots().to_vec();
        roots[1] = H256::repeat_byte(0x42);
        proof.fri_commitment =
            fri::Commitment::new(roots, proof.fri_commitment.folded_polynomial().clone());
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_mismatched_fold_rate() {
        let (commitment, mut proof) = sample_proof(OPTIONS);
        proof.options.fold_rate = 2;
        let result = proof.verify(&commitment);
        assert!(result.is_err());
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
        let (foreign, _) = committer.commit();
        let result = proof.verify(&foreign);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_misreported_batch_sizes() {
        let (mut commitment, proof) = sample_proof(OPTIONS);
        commitment.batches[0].num_polys = 1;
        commitment.batches[1].num_polys = 2;
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    /// The base-field batch of two polynomials is reported as a batch of two polynomials over
    /// Goldilocks^4, whose leaves would have 8 values rather than 2.
    #[test]
    fn test_reject_base_batch_reported_as_extension() {
        let (mut commitment, proof) = sample_mixed_proof();
        commitment.batches[0].base_field = false;
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    /// The batch of one polynomial over Goldilocks^4 is reported as a base-field batch of one
    /// polynomial, whose leaves would have 1 value rather than 4.
    #[test]
    fn test_reject_extension_batch_reported_as_base() {
        let (mut commitment, proof) = sample_mixed_proof();
        commitment.batches[1].base_field = true;
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    /// With BlueSky a base-field batch and a batch over the extension field have the same leaves,
    /// so only the different leaf DSTs make the relabeled commitment fail.
    #[test]
    fn test_reject_relabeled_batches() {
        let committer =
            make_mixed_committer::<BS, Sha2Hash<BS>>(OPTIONS, &SAMPLE_BATCHES, &MIXED_BASE_FIELDS);
        let (commitment, prover) = committer.commit();
        let proof = prover.prove(make_points(&SAMPLE_POINTS));
        let mut base_relabeled = commitment.clone();
        base_relabeled.batches[0].base_field = false;
        let mut extension_relabeled = commitment.clone();
        extension_relabeled.batches[1].base_field = true;
        assert!(proof.verify(&commitment).is_ok());
        assert!(proof.verify(&base_relabeled).is_err());
        assert!(proof.verify(&extension_relabeled).is_err());
    }

    #[test]
    fn test_reject_tampered_base_leaf_value() {
        let (commitment, proof) = sample_mixed_proof();
        let mut proto = proof.to_proto();
        proto.openings[3].proofs[0].leaf_values[1] = vec![42, 0, 0, 0, 0, 0, 0, 0];
        let proof = Proof::<GL4, Sha2Hash<GL4>>::from_proto(&proto).unwrap();
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_tampered_extension_leaf_coordinate() {
        let (commitment, proof) = sample_mixed_proof();
        let mut proto = proof.to_proto();
        proto.openings[3].proofs[1].leaf_values[2] = vec![42, 0, 0, 0, 0, 0, 0, 0];
        let proof = Proof::<GL4, Sha2Hash<GL4>>::from_proto(&proto).unwrap();
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
        let (_, other_prover) = committer.commit();
        let mut other_proof = other_prover.prove(make_points(&[123]));
        proof.openings[0][0] = other_proof.openings[0].remove(0);
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    /// The Merkle path has 64 more hashes than the height of the tree, so `1 << path.len()` would
    /// wrap around to the domain size.
    #[test]
    fn test_reject_oversized_opening_path() {
        let (commitment, proof) = sample_proof(OPTIONS);
        let mut proto = proof.to_proto();
        proto.openings[0].proofs[0]
            .merkle_path
            .extend(vec![vec![0; 32]; 64]);
        let proof = Proof::<BS, Sha2Hash<BS>>::from_proto(&proto).unwrap();
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
        proof.options.grinding_bits = 128;
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_overstated_security_level() {
        let (commitment, mut proof) = sample_proof(OPTIONS);
        proof.options.security_bits = 130;
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_understated_security_level() {
        let (commitment, mut proof) = sample_proof(OPTIONS);
        proof.options.security_bits = 126;
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    /// Deriving the query indices before checking the number of queries would try to derive 2^63
    /// indices here.
    #[test]
    fn test_reject_huge_security_level() {
        let (commitment, mut proof) = sample_proof(OPTIONS);
        proof.options.security_bits = usize::MAX;
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_unsatisfiable_grinding() {
        let (commitment, mut proof) = sample_proof(GROUND_OPTIONS);
        proof.options.grinding_bits = 257;
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_maximum_grinding() {
        let (commitment, mut proof) = sample_proof(GROUND_OPTIONS);
        proof.options.grinding_bits = usize::MAX;
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    #[should_panic]
    fn test_committer_with_unsatisfiable_grinding() {
        make_committer::<BS, Sha2Hash<BS>>(
            Options {
                grinding_bits: 257,
                ..OPTIONS
            },
            &SAMPLE_BATCHES,
        );
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
        assert_eq!(proto.security_bits, 128);
        assert_eq!(proto.blowup_log2, 2);
        assert_eq!(proto.fold_rate, 1);
        assert_eq!(proto.max_rounds, u64::MAX);
        assert_eq!(proto.grinding_bits, 8);
        assert_eq!(proto.nonce, 204);
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
        assert_eq!(
            proto.fri_roots,
            [
                proof.fri_commitment.roots()[0].as_bytes().to_vec(),
                proof.fri_commitment.roots()[1].as_bytes().to_vec(),
            ]
        );
        assert_eq!(
            proto.folded_quotient,
            [proof.fri_commitment.folded_polynomial().coefficients()[0]
                .to_le_bytes()
                .to_vec()]
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
    fn test_proof_with_base_batch_to_proto() {
        let (_, proof) = sample_mixed_proof();
        let proto = proof.to_proto();
        let opening = &proto.openings[0];
        assert_eq!(opening.proofs.len(), 2);
        assert_eq!(opening.proofs[0].leaf_values.len(), 2);
        assert_eq!(opening.proofs[1].leaf_values.len(), 4);
        assert!(
            opening
                .proofs
                .iter()
                .all(|proof| proof.leaf_values.iter().all(|value| value.len() == 8))
        );
        assert_eq!(proto.points[0].z.len(), 32);
        assert!(proto.points[0].v.iter().all(|value| value.len() == 32));
    }

    #[test]
    fn test_accept_decoded_proof_with_base_batch() {
        let (commitment, proof) = sample_mixed_proof();
        let commitment_proto = commitment.to_proto();
        let proof_proto = proof.to_proto();
        let commitment = Commitment::<GL4, Sha2Hash<GL4>>::from_proto(&commitment_proto).unwrap();
        let proof = Proof::<GL4, Sha2Hash<GL4>>::from_proto(&proof_proto).unwrap();
        let result = proof.verify(&commitment);
        assert!(result.is_ok());
    }

    #[test]
    fn test_koalabear_proof_to_proto() {
        let committer = make_mixed_committer::<KB8, Keccak256Hash<KB8>>(
            OPTIONS,
            &SAMPLE_BATCHES,
            &MIXED_BASE_FIELDS,
        );
        let (_, prover) = committer.commit();
        let proof = prover.prove(make_points(&SAMPLE_POINTS));
        let proto = proof.to_proto();
        assert_eq!(proto.points[0].z.len(), 31);
        assert!(proto.points[0].v.iter().all(|value| value.len() == 31));
        assert!(proto.folded_quotient.iter().all(|value| value.len() == 31));
        assert!(
            proto.openings[0]
                .proofs
                .iter()
                .all(|proof| proof.leaf_values.iter().all(|value| value.len() == 4))
        );
        assert!(
            proto.queries[0]
                .folds
                .iter()
                .all(|fold| fold.coset_values.iter().all(|value| value.len() == 31))
        );
    }

    #[test]
    fn test_accept_decoded_koalabear_proof() {
        let committer = make_mixed_committer::<KB8, Keccak256Hash<KB8>>(
            GROUND_OPTIONS,
            &SAMPLE_BATCHES,
            &MIXED_BASE_FIELDS,
        );
        let (commitment, prover) = committer.commit();
        let proof = prover.prove(make_points(&SAMPLE_POINTS));
        let commitment_proto = commitment.to_proto();
        let proof_proto = proof.to_proto();
        let commitment =
            Commitment::<KB8, Keccak256Hash<KB8>>::from_proto(&commitment_proto).unwrap();
        let proof = Proof::<KB8, Keccak256Hash<KB8>>::from_proto(&proof_proto).unwrap();
        let result = proof.verify(&commitment);
        assert!(result.is_ok());
    }

    #[test]
    fn test_proof_security_level_from_proto() {
        let (_, proof) = sample_proof(OPTIONS);
        let mut proto = proof.to_proto();
        proto.security_bits = 256;
        let decoded = Proof::<BS, Sha2Hash<BS>>::from_proto(&proto).unwrap();
        assert_eq!(decoded.options().security_bits, 256);
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
    fn test_accept_decoded_proof_with_fold_rate_2() {
        let (commitment, proof) = sample_proof(Options {
            fold_rate: 2,
            ..GROUND_OPTIONS
        });
        let commitment_proto = commitment.to_proto();
        let proof_proto = proof.to_proto();
        let commitment = Commitment::<BS, Sha2Hash<BS>>::from_proto(&commitment_proto).unwrap();
        let proof = Proof::<BS, Sha2Hash<BS>>::from_proto(&proof_proto).unwrap();
        let result = proof.verify(&commitment);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reject_zero_fold_rate() {
        let (_, proof) = sample_proof(OPTIONS);
        let mut proto = proof.to_proto();
        proto.fold_rate = 0;
        let result = Proof::<BS, Sha2Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_zero_blowup_log2() {
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
    fn test_reject_unsatisfiable_grinding_in_proto() {
        let (_, proof) = sample_proof(GROUND_OPTIONS);
        let mut proto = proof.to_proto();
        proto.grinding_bits = 257;
        let result = Proof::<BS, Sha2Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_maximum_grinding_in_proto() {
        let (_, proof) = sample_proof(GROUND_OPTIONS);
        let mut proto = proof.to_proto();
        proto.grinding_bits = u32::MAX;
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
    fn test_reject_non_canonical_base_leaf_value() {
        let (_, proof) = sample_mixed_proof();
        let mut proto = proof.to_proto();
        proto.openings[0].proofs[0].leaf_values[0] = vec![1, 0, 0, 0, 0xff, 0xff, 0xff, 0xff];
        let result = Proof::<GL4, Sha2Hash<GL4>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_long_base_leaf_value() {
        let (_, proof) = sample_mixed_proof();
        let mut proto = proof.to_proto();
        proto.openings[0].proofs[0].leaf_values[0].push(0);
        let result = Proof::<GL4, Sha2Hash<GL4>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_proof_without_fri_roots() {
        // The number of FRI roots is only checked by `verify`.
        let (commitment, proof) = sample_proof(OPTIONS);
        let mut proto = proof.to_proto();
        proto.fri_roots.clear();
        let proof = Proof::<BS, Sha2Hash<BS>>::from_proto(&proto).unwrap();
        let result = proof.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_short_fri_root() {
        let (_, proof) = sample_proof(OPTIONS);
        let mut proto = proof.to_proto();
        proto.fri_roots[1].pop();
        let result = Proof::<BS, Sha2Hash<BS>>::from_proto(&proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_non_canonical_folded_quotient() {
        let (_, proof) = sample_proof(OPTIONS);
        let mut proto = proof.to_proto();
        let modulus: U256 = BS::MODULUS.parse().unwrap();
        proto.folded_quotient[0] = modulus.to_little_endian().to_vec();
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

    /// Returns the header of SCPF data with the given schema identifier and pool size.
    fn scpf_header(schema_identifier: H256, pool_size: u32) -> Vec<u8> {
        [
            &b"SCPF"[..],
            &[0, 0, 0, 0][..],
            &scpf::VERSION[..],
            &[0, 0, 0, 0, 0][..],
            schema_identifier.as_bytes(),
            &pool_size.to_be_bytes()[..],
            &[0; 12][..],
        ]
        .concat()
    }

    /// Returns the protobuf encoding of a small BlueSky element (little-endian bytes).
    fn proto_scalar(value: u64) -> Vec<u8> {
        BS::from_const(value).to_le_bytes().to_vec()
    }

    /// Returns the SCPF encoding of a small BlueSky element (an inline big-endian word).
    fn scpf_scalar(value: u64) -> [u8; 32] {
        BS::from_const(value).to_be_bytes()
    }

    /// Returns the Keccak-256 hash of `data`.
    fn keccak256(data: &[u8]) -> H256 {
        use sha3::Digest;
        H256::from_slice(sha3::Keccak256::digest(data).as_slice())
    }

    /// Returns a hash made of 32 copies of the given byte.
    fn hash(byte: u8) -> H256 {
        H256::repeat_byte(byte)
    }

    /// Returns the protobuf encoding of the hashes made of 32 copies of the given bytes.
    fn proto_hashes(bytes: &[u8]) -> Vec<Vec<u8>> {
        bytes.iter().map(|&byte| vec![byte; 32]).collect()
    }

    /// A proof with small values deserialized from a protobuf, with one batch, one point, one FRI
    /// round, and two queries. The two batch openings share hash 0xbb and the two folds share hash
    /// 0xdd, while the other hashes appear once each.
    fn small_proof_proto() -> proto::Proof {
        let opening = |leaf: u64, path: [u8; 2]| proto::proof::Opening {
            proofs: vec![proto::MerkleProof {
                leaf_values: vec![proto_scalar(leaf)],
                merkle_path: proto_hashes(&path),
            }],
        };
        let query = |index: u64, coset: [u64; 2]| proto::Query {
            index,
            folds: vec![proto::query::LeafProof {
                coset_values: coset.map(proto_scalar).to_vec(),
                merkle_path: proto_hashes(&[0xdd]),
            }],
        };
        proto::Proof {
            hash_backend: Some(Any::from_msg(&proto::KeccakHashBackend::default()).unwrap()),
            degree_bound: 4,
            security_bits: 128,
            blowup_log2: 2,
            fold_rate: 1,
            max_rounds: 1,
            grinding_bits: 8,
            nonce: 300,
            points: vec![proto::proof::Point {
                z: proto_scalar(123),
                v: vec![proto_scalar(45)],
            }],
            fri_roots: proto_hashes(&[0x11]),
            folded_quotient: vec![proto_scalar(7)],
            openings: vec![opening(10, [0xaa, 0xbb]), opening(11, [0xcc, 0xbb])],
            queries: vec![query(5, [20, 21]), query(9, [22, 23])],
        }
    }

    #[test]
    fn test_commitment_schema_identifier() {
        assert_eq!(
            *COMMITMENT_SCHEMA_IDENTIFIER,
            parse("0xbd47f9a987065018ccd907dfa7d41b8eef2c2c88acddffc327829aed3e313b70")
        );
    }

    #[test]
    fn test_proof_schema_identifier() {
        assert_eq!(
            *PROOF_SCHEMA_IDENTIFIER,
            parse("0xb7fcbc946ada95241d640499d14b4c6824a850187a7cd259c954bb722be8c086")
        );
    }

    #[test]
    fn test_commitment_to_scpf() {
        let committer = make_mixed_committer::<BS, Keccak256Hash<BS>>(
            OPTIONS,
            &SAMPLE_BATCHES,
            &MIXED_BASE_FIELDS,
        );
        let (commitment, _) = committer.commit();
        let data = commitment.to_scpf();
        assert_eq!(
            data,
            [
                &scpf_header(*COMMITMENT_SCHEMA_IDENTIFIER, 2)[..],
                commitment.batches()[0].root().as_bytes(),
                commitment.batches()[1].root().as_bytes(),
                // The number of batches, then every batch: base field flag, polynomials, root.
                &[2, 1, 2, 0, 0, 1, 1][..],
            ]
            .concat()
        );
    }

    #[test]
    fn test_commitment_from_proto_to_scpf() {
        let commitment = Commitment::<BS, Sha2Hash<BS>>::from_proto(&proto::Commitment {
            hash_backend: Some(Any::from_msg(&proto::Sha2HashBackend::default()).unwrap()),
            batches: vec![
                proto::commitment::Batch {
                    base_field: false,
                    num_polys: 300,
                    root: vec![0xaa; 32],
                },
                proto::commitment::Batch {
                    base_field: true,
                    num_polys: 1,
                    root: vec![0xbb; 32],
                },
                proto::commitment::Batch {
                    base_field: false,
                    num_polys: 2,
                    root: vec![0xaa; 32],
                },
            ],
        })
        .unwrap();
        let data = commitment.to_scpf();
        assert_eq!(
            data,
            [
                &scpf_header(*COMMITMENT_SCHEMA_IDENTIFIER, 2)[..],
                hash(0xaa).as_bytes(),
                hash(0xbb).as_bytes(),
                &[3, 0, 0x81, 0x2c, 0, 1, 1, 1, 0, 2, 0][..],
            ]
            .concat()
        );
    }

    #[test]
    fn test_proof_to_scpf() {
        let proof = Proof::<BS, Keccak256Hash<BS>>::from_proto(&small_proof_proto()).unwrap();
        let data = proof.to_scpf();
        // The pool has the two hashes referenced twice first, then the others: 0xbb, 0xdd, 0x11,
        // 0xaa, 0xcc.
        assert_eq!(
            data,
            [
                &scpf_header(*PROOF_SCHEMA_IDENTIFIER, 5)[..],
                hash(0xbb).as_bytes(),
                hash(0xdd).as_bytes(),
                hash(0x11).as_bytes(),
                hash(0xaa).as_bytes(),
                hash(0xcc).as_bytes(),
                // Degree bound, security bits (128 takes two bytes), blowup, fold rate, max rounds,
                // grinding bits, nonce (300 takes two bytes), and number of points.
                &[4, 0x80, 0x80, 2, 1, 1, 8, 0x81, 0x2c, 1][..],
                &scpf_scalar(123)[..],
                &scpf_scalar(45)[..],
                // The FRI root and the number of coefficients of the folded quotient.
                &[2, 1][..],
                &scpf_scalar(7)[..],
                // First query: the batch opening, then the fold.
                &scpf_scalar(10)[..],
                &[3, 0][..],
                &scpf_scalar(20)[..],
                &scpf_scalar(21)[..],
                &[1][..],
                // Second query.
                &scpf_scalar(11)[..],
                &[4, 0][..],
                &scpf_scalar(22)[..],
                &scpf_scalar(23)[..],
                &[1][..],
            ]
            .concat()
        );
    }

    #[test]
    fn test_proof_without_rounds_to_scpf() {
        let mut proto = small_proof_proto();
        proto.max_rounds = 0;
        proto.fri_roots.clear();
        proto.folded_quotient = vec![proto_scalar(7), proto_scalar(8)];
        for query in &mut proto.queries {
            query.folds.clear();
        }
        let proof = Proof::<BS, Keccak256Hash<BS>>::from_proto(&proto).unwrap();
        let data = proof.to_scpf();
        assert_eq!(
            data,
            [
                &scpf_header(*PROOF_SCHEMA_IDENTIFIER, 3)[..],
                hash(0xbb).as_bytes(),
                hash(0xaa).as_bytes(),
                hash(0xcc).as_bytes(),
                &[4, 0x80, 0x80, 2, 1, 0, 8, 0x81, 0x2c, 1][..],
                &scpf_scalar(123)[..],
                &scpf_scalar(45)[..],
                // No FRI roots, then the two coefficients of the folded quotient.
                &[2][..],
                &scpf_scalar(7)[..],
                &scpf_scalar(8)[..],
                &scpf_scalar(10)[..],
                &[1, 0][..],
                &scpf_scalar(11)[..],
                &[2, 0][..],
            ]
            .concat()
        );
    }

    #[test]
    fn test_proof_with_unordered_points_to_scpf() {
        let mut proto = small_proof_proto();
        proto.points = vec![
            proto::proof::Point {
                z: proto_scalar(456),
                v: vec![proto_scalar(67)],
            },
            proto::proof::Point {
                z: proto_scalar(123),
                v: vec![proto_scalar(45)],
            },
        ];
        proto.openings.clear();
        proto.queries.clear();
        let proof = Proof::<BS, Keccak256Hash<BS>>::from_proto(&proto).unwrap();
        let data = proof.to_scpf();
        // The points are encoded in increasing order.
        assert_eq!(
            data,
            [
                &scpf_header(*PROOF_SCHEMA_IDENTIFIER, 1)[..],
                hash(0x11).as_bytes(),
                &[4, 0x80, 0x80, 2, 1, 1, 8, 0x81, 0x2c, 2][..],
                &scpf_scalar(123)[..],
                &scpf_scalar(45)[..],
                &scpf_scalar(456)[..],
                &scpf_scalar(67)[..],
                &[0, 1][..],
                &scpf_scalar(7)[..],
            ]
            .concat()
        );
    }

    #[test]
    fn test_sample_proof_to_scpf() {
        let committer = make_committer::<BS, Keccak256Hash<BS>>(GROUND_OPTIONS, &SAMPLE_BATCHES);
        let (_, prover) = committer.commit();
        let proof = prover.prove(make_points(&SAMPLE_POINTS));
        let data = proof.to_scpf();
        assert_eq!(data.len(), 17216);
        assert_eq!(
            keccak256(data.as_slice()),
            parse("0x3b6914d37314ce7011868cf14716101de02dbd9c0228c4d26517d1f1773046d1")
        );
    }
}
