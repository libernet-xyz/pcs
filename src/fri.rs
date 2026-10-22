use crate::hash::Hasher;
use crate::merkle;
use crate::starkom::proto::pcs::v9 as proto;
use crate::utils;
use anyhow::{Result, anyhow};
use primitive_types::H256;
use starkom_ff::{Field, Field256};
use starkom_poly::Polynomial;
use std::marker::PhantomData;
use std::sync::LazyLock;

/// Domain separator tag used when deriving the Fiat-Shamir challenge for FRI folding.
static FOLD_DST: LazyLock<H256> = LazyLock::new(|| utils::make_dst(b"starkom/fri/fold"));

/// Returns the number of rounds of the folding argument.
///
/// Every round performs exactly `fold_rate` folds, so there can be at most
/// `log2(degree_bound) / fold_rate` rounds (rounded down), and their number is further capped by
/// `max_rounds`. Rather than being folded further, the polynomial resulting from the last round is
/// sent in the clear as the [folded quotient](`Commitment::folded_quotient`), which has degree less
/// than `degree_bound / 2^(rounds * fold_rate)`.
///
/// REQUIRES: `degree_bound` must be a power of 2 and `fold_rate` must be greater than 0.
pub fn num_rounds(degree_bound: usize, fold_rate: usize, max_rounds: usize) -> usize {
    assert!(degree_bound.is_power_of_two());
    assert!(fold_rate > 0);
    std::cmp::min(
        max_rounds,
        degree_bound.trailing_zeros() as usize / fold_rate,
    )
}

/// Reverses the order of the lowest `num_bits` bits of `index`.
///
/// REQUIRES: `index` must be less than `2^num_bits`.
fn reverse_bits(index: usize, num_bits: usize) -> usize {
    if num_bits > 0 {
        index.reverse_bits() >> (usize::BITS as usize - num_bits)
    } else {
        0
    }
}

/// A Merkle proof for a coset of FRI leaves.
///
/// The leaves of FRI trees are stored in bit-reversed order (see [`Tree`]), so the `2^r`
/// evaluations folded together by a round with fold rate `r` are adjacent: they're the leaves of a
/// subtree of height `r`. The proof contains all of them and (allowing the verifier to recompute
/// the subtree root), along with the Merkle path from the subtree root up to the root of the tree.
///
/// The proof authenticates the coset as a whole. In a tree of height `h`, the evaluation at index
/// `i` is at position `reverse_bits(i, h) mod 2^r` within the coset.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LeafProof<F: Field256, H: Hasher<F>> {
    /// The evaluations of the coset in bit-reversed order. Their number is always a power of 2.
    coset: Vec<F>,
    /// The Merkle path from the root of the coset subtree up to, but excluding, the root of the
    /// tree.
    path: Vec<H256>,
    _data: PhantomData<H>,
}

impl<F: Field256, H: Hasher<F>> LeafProof<F, H> {
    /// Returns the fold rate implied by the coset, that is the log2 of its size, which is also the
    /// height of the opened subtree.
    fn fold_rate(&self) -> usize {
        self.coset.len().trailing_zeros() as usize
    }

    /// Returns the height of the full tree, that is the log2 of its number of leaves.
    fn height(&self) -> usize {
        self.fold_rate() + self.path.len()
    }

    /// Returns the evaluations of the coset in bit-reversed order.
    fn coset(&self) -> &[F] {
        self.coset.as_slice()
    }

    /// Verifies the proof against the given root hash.
    ///
    /// `index` is the index of any evaluation of the coset in the evaluation domain, not the
    /// position of its leaf in the tree (see [`Tree`]).
    ///
    /// REQUIRES: `index` must be less than `2^self.height()`.
    fn verify(&self, index: usize, root_hash: H256) -> Result<()> {
        let subtree_root = merkle::merkle_root::<H>(
            self.coset
                .iter()
                .map(|&value| utils::encode_scalar(value))
                .collect::<Vec<H256>>()
                .as_slice(),
        );
        let index = reverse_bits(index, self.height()) >> self.fold_rate();
        merkle::verify_path::<H>(subtree_root, self.path.as_slice(), index, root_hash)
    }

    /// Serializes this proof to a [`LeafProof`](`proto::query::LeafProof`) protobuf.
    fn to_proto(&self) -> proto::query::LeafProof {
        proto::query::LeafProof {
            coset_values: self
                .coset
                .iter()
                .map(|value| value.to_le_bytes().to_vec())
                .collect(),
            merkle_path: self
                .path
                .iter()
                .map(|hash| hash.as_bytes().to_vec())
                .collect(),
        }
    }

    /// Deserializes a proof for a coset of size `2^num_folds` from a
    /// [`LeafProof`](`proto::query::LeafProof`) protobuf.
    ///
    /// Returns an error if the number of coset values is not `2^num_folds`, if any value is not a
    /// canonical field element, or if any hash is invalid.
    fn from_proto(proto: &proto::query::LeafProof, num_folds: usize) -> Result<Self> {
        if proto.coset_values.len() != 1 << num_folds {
            return Err(anyhow!(
                "invalid FRI coset size (got {}, want {})",
                proto.coset_values.len(),
                1 << num_folds
            ));
        }
        Ok(Self {
            coset: proto
                .coset_values
                .iter()
                .map(|value| utils::load_scalar(value.as_slice()))
                .collect::<Result<_>>()?,
            path: proto
                .merkle_path
                .iter()
                .map(|hash| H::load_hash(hash.as_slice()))
                .collect::<Result<_>>()?,
            _data: PhantomData,
        })
    }
}

/// A Merkle tree whose leaves are the evaluations of a polynomial.
///
/// The leaves are the evaluations themselves rather than hashes of them: each leaf is an evaluation
/// encoded with [`utils::encode_scalar`]. The internal nodes are computed as described in
/// [`merkle::merklify`], hashing pairs of children with
/// [`MerkleHasher::hash_binary`](`crate::hash::MerkleHasher::hash_binary`).
///
/// The evaluations are stored in bit-reversed order: the one at index `i` of the evaluation domain
/// of size N goes to the leaf at position `reverse_bits(i, log2(N))`. The partner evaluations at
/// `i` and `i + j * N / 2^r` (for all `j` in `[0..2^r)`) only differ in the `r` most significant
/// bits of their index, with `r` being the fold rate. So all partner values form a coset and end up
/// in adjacent leaves, and a single [`LeafProof`] opens all of them.
///
/// When `r = 1` this reduces to binary FRI with only two partner values at each round, located at
/// indices `i` and `i + n/2`. In that case a [`LeafProof`] has the shape of a regular Merkle proof.
///
/// In addition to the above, the root of that subtree is at the same position as the leaf of the
/// folded evaluation in the next round.
///
/// NOTE: bit reversal is completely abstracted away by `Tree`. The provided API exposes the
/// original evaluation domain indices, but bit reversal allows opening a whole coset with a single
/// Merkle proof (see [`LeafProof`]).
#[derive(Debug, Clone)]
struct Tree<F: Field256, H: Hasher<F>> {
    /// The whole tree, stored layer by layer as described in [`merkle::merklify`]. There are 2*N-1
    /// elements in total.
    data: Vec<H256>,
    _phantom_data: PhantomData<(F, H)>,
}

impl<F: Field256, H: Hasher<F>> Tree<F, H> {
    /// Constructs a Merkle tree from the evaluations of a polynomial, whose number must be a power
    /// of 2.
    fn new(values: &[F]) -> Self {
        let n = values.len();
        assert!(n.is_power_of_two());
        let num_bits = n.trailing_zeros() as usize;
        let mut data = Vec::with_capacity(n * 2 - 1);
        data.extend((0..n).map(|index| {
            let reverse_index = reverse_bits(index, num_bits);
            utils::encode_scalar(values[reverse_index])
        }));
        data.resize(n * 2 - 1, H256::zero());
        merkle::merklify::<H>(data.as_mut_slice(), n);
        Self {
            data,
            _phantom_data: PhantomData,
        }
    }

    /// Returns the number of leaves in the tree, corresponding to the size of the evaluation domain
    /// (always a power of 2).
    fn num_leaves(&self) -> usize {
        (self.data.len() + 1) >> 1
    }

    /// Returns the root hash of the Merkle tree.
    fn root_hash(&self) -> H256 {
        *self.data.last().unwrap()
    }

    /// Returns a Merkle proof for the coset of `2^fold_rate` evaluations folded together by a round
    /// with the given fold rate, containing the evaluation at the given index of the domain.
    ///
    /// NOTE: `index` is not bit-reversed, but thanks to the internal bit reversal the returned
    /// [`LeafProof`] proves all the evaluations of the coset with a single Merkle path. For
    /// example, with `fold_rate = 1` it proves the evaluation at `index` and its partner at
    /// `index + N/2`.
    ///
    /// REQUIRES: `fold_rate` must not exceed the height of the tree.
    fn query(&self, index: usize, fold_rate: usize) -> LeafProof<F, H> {
        let n = self.num_leaves();
        let reverse_index = reverse_bits(index, n.trailing_zeros() as usize);
        let size = 1 << fold_rate;
        let start = reverse_index & !(size - 1);
        LeafProof {
            coset: self.data[start..(start + size)]
                .iter()
                .map(|&leaf| utils::decode_scalar(leaf).unwrap())
                .collect(),
            path: merkle::merkle_path(self.data.as_slice(), n, reverse_index)[fold_rate..].to_vec(),
            _data: PhantomData,
        }
    }
}

/// A FRI commitment.
///
/// It consists of the Merkle roots of the folding rounds, one for every round, and of the
/// polynomial resulting from the last round, which is sent in the clear rather than being folded
/// further. With zero rounds there are no Merkle roots and the folded quotient is the committed
/// polynomial itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment<F: Field256> {
    /// The Merkle roots of the folding rounds. The first one is the root of the Merkle tree of the
    /// committed polynomial, and every subsequent one is the root of the Merkle tree of the
    /// polynomial folded by the previous rounds.
    roots: Vec<H256>,
    /// The polynomial resulting from the last folding round.
    folded_quotient: Polynomial<F>,
}

impl<F: Field256> Commitment<F> {
    pub(crate) fn new(roots: Vec<H256>, folded_quotient: Polynomial<F>) -> Self {
        Self {
            roots,
            folded_quotient,
        }
    }

    /// Returns the number of stored roots, which is the number of folding rounds. For example, if
    /// the user commits a degree<8 polynomial with a fold rate of 1 and no limit on the number of
    /// rounds, `len()` will return log2(8) = 3 regardless of the blowup factor.
    pub fn len(&self) -> usize {
        self.roots.len()
    }

    /// Returns the Merkle roots of all folding rounds.
    ///
    /// The returned slice has [`Self::len()`] elements.
    pub fn roots(&self) -> &[H256] {
        self.roots.as_slice()
    }

    /// Returns the folded quotient, that is the polynomial resulting from the last folding round.
    ///
    /// Its coefficients are those of the domain-shifted polynomial (see
    /// [`Polynomial::shift_domain`]), so they must be evaluated over the folded domain, not at the
    /// shifted domain points.
    pub fn folded_quotient(&self) -> &Polynomial<F> {
        &self.folded_quotient
    }
}

/// A single FRI query.
#[derive(Debug, Clone)]
pub struct Query<F: Field256, H: Hasher<F>> {
    /// The degree bound of the committed polynomial (always a power of 2).
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// The number of folds per round.
    fold_rate: usize,
    /// The maximum number of folding rounds (see [`num_rounds`]).
    max_rounds: usize,
    /// The index of the element we're opening.
    ///
    /// This is always less than the size of the extended domain.
    ///
    /// NOTE: this is the natural index, not the bit-reversed one.
    index: usize,
    /// Proves the coset of the queried index folded by each round with one [`LeafProof`] for every
    /// round.
    folds: Vec<LeafProof<F, H>>,
}

impl<F: Field256, H: Hasher<F>> Query<F, H> {
    /// Returns the opened index of the extended domain.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns the opened domain element, that is the X-coordinate of the evaluation.
    ///
    /// Note that we use [`Polynomial::shift_domain`] before committing polynomials, so the element
    /// returned here is a shifted power of an N-th root of unity, with
    /// `N = degree_bound * 2^blowup_log2`. The shift consists of multiplying the actual domain
    /// element by the [`MULTIPLICATIVE_GENERATOR`](`Field::MULTIPLICATIVE_GENERATOR`) of the *base*
    /// field, consistently with `shift_domain`.
    pub fn x(&self) -> F {
        Polynomial::<F>::coset_element2(self.index, self.degree_bound << self.blowup_log2).into()
    }

    /// Returns the number of Merkle proofs in the query, which is the number of folding rounds
    /// (see [`num_rounds`]).
    pub fn len(&self) -> usize {
        self.folds.len()
    }

    /// Folds a coset of evaluations into a single evaluation of the folded polynomial.
    ///
    /// `values` contains the `2^r` evaluations of a coset of the evaluation domain of size `n`,
    /// in bit-reversed order (see [`Tree`]): the evaluation at index `base_index + j * n / 2^r` is
    /// in position `reverse_bits(j, r)`, with `base_index < n / 2^r`. The coset is folded `r` times
    /// with the challenges `alpha`, `alpha^2`, `alpha^4`, and so on, which is equivalent to folding
    /// it once with arity `2^r` and challenge `alpha`. The result is the evaluation of the folded
    /// polynomial at index `base_index` of the evaluation domain of size `n / 2^r`.
    fn fold_coset(mut values: Vec<F>, base_index: usize, mut n: usize, mut alpha: F) -> F {
        while values.len() > 1 {
            // Leaves `2*i` and `2*i+1` contain the evaluations at `X` and `-X`. The former is the
            // one at the lower index, `base_index + reverse_bits(i, log2(half)) * stride`.
            let half = values.len() / 2;
            let num_bits = half.trailing_zeros() as usize;
            let stride = n / 2 / half;
            let omega_inv = Polynomial::<F>::domain_element2(1, n).invert_unwrap();
            values = (0..half)
                .map(|i| {
                    let pos = values[i * 2];
                    let neg = values[i * 2 + 1];
                    let index = base_index + reverse_bits(i, num_bits) * stride;
                    (pos + neg + alpha * omega_inv.pow_small(index) * (pos - neg))
                        * F::BaseField::TWO_INV
                })
                .collect();
            n /= 2;
            alpha = alpha.square();
        }
        values[0]
    }

    /// Verifies this proof against the given commitment, with `value` being the expected evaluation
    /// of the committed polynomial at the opened domain element [`Self::x`].
    ///
    /// The caller derives `value` on its own (DEEP-FRI computes it from the openings of the
    /// committed polynomials), and every round derives the expected evaluation of the next one by
    /// folding its coset.
    ///
    /// The number of folding rounds and the maximum number of coefficients of the folded quotient
    /// are determined by the degree bound, the fold rate, and the maximum number of rounds (see
    /// [`num_rounds`]). Every round checks that the opened coset contains the expected evaluation,
    /// checks the coset against the Merkle root of the round, and folds it. The result of the last
    /// round, or `value` itself if there are zero rounds, must match the evaluation of the folded
    /// quotient.
    ///
    /// If verification succeeds, the committed polynomial has degree less than `2^(rounds * r)`
    /// times the number of coefficients of the folded quotient, with `r` being the fold rate, and
    /// therefore less than the degree bound.
    pub fn verify(&self, commitment: &Commitment<F>, mut value: F) -> Result<()> {
        let mut n = self.degree_bound << self.blowup_log2;
        assert!(n.is_power_of_two());

        let mut k = n.trailing_zeros() as usize;

        let num_rounds = num_rounds(self.degree_bound, self.fold_rate, self.max_rounds);
        if self.folds.len() != num_rounds {
            return Err(anyhow!("incorrect proof size"));
        }
        if commitment.len() != num_rounds {
            return Err(anyhow!("wrong number of folding rounds"));
        }
        let final_degree_bound = self.degree_bound >> (num_rounds * self.fold_rate);
        if commitment.folded_quotient().len() > final_degree_bound {
            return Err(anyhow!(
                "invalid folded quotient size (got {} coefficients, want at most {})",
                commitment.folded_quotient().len(),
                final_degree_bound
            ));
        }

        let mut index = self.index;

        for (proof, &root_hash) in self.folds.iter().zip(commitment.roots()) {
            if proof.fold_rate() != self.fold_rate {
                return Err(anyhow!(
                    "invalid fold rate (got {}, want {})",
                    proof.fold_rate(),
                    self.fold_rate
                ));
            }
            if proof.height() != k {
                return Err(anyhow!(
                    "invalid Merkle proof height (got {}, want {})",
                    proof.height(),
                    k
                ));
            }

            let offset = reverse_bits(index, k) & ((1 << self.fold_rate) - 1);
            if proof.coset()[offset] != value {
                return Err(anyhow!("leaf value mismatch"));
            }
            proof.verify(index, root_hash)?;

            let alpha = H::challenge(*FOLD_DST, &[root_hash]);
            let folded_n = n >> self.fold_rate;
            value = Self::fold_coset(proof.coset().to_vec(), index % folded_n, n, alpha);
            n = folded_n;
            k -= self.fold_rate;
            index %= n;
        }

        if commitment
            .folded_quotient()
            .evaluate_on_two_adic_domain(index, n)
            != value
        {
            return Err(anyhow!("folded quotient mismatch"));
        }

        Ok(())
    }

    /// Serializes this query to a [`Query`](`proto::Query`) protobuf.
    pub fn to_proto(&self) -> proto::Query {
        proto::Query {
            index: self.index as u64,
            folds: self.folds.iter().map(LeafProof::to_proto).collect(),
        }
    }

    /// Deserializes a query from a [`Query`](`proto::Query`) protobuf.
    ///
    /// REQUIRES: `degree_bound` must be a power of 2, `blowup_log2` and `fold_rate` must be greater
    /// than 0, and the size of the extended domain must fit within the 2-adicity of
    /// `F::BaseField`.
    ///
    /// The query index must be less than the size of the extended domain, there must be exactly
    /// one Merkle proof for every folding round (see [`num_rounds`]), every Merkle proof must
    /// contain the whole coset folded by its round, and every opened value must be a canonical
    /// field element, otherwise an error is returned.
    ///
    /// NOTE: this method does not validate the proof, it only deserializes it. The caller must
    /// invoke [`Self::verify`] separately.
    pub fn from_proto(
        degree_bound: usize,
        blowup_log2: usize,
        fold_rate: usize,
        max_rounds: usize,
        proto: &proto::Query,
    ) -> Result<Self> {
        assert!(degree_bound.is_power_of_two());
        assert!(blowup_log2 > 0);
        assert!(fold_rate > 0);
        assert!(degree_bound.trailing_zeros() as usize + blowup_log2 <= F::BaseField::S);
        let n = degree_bound << blowup_log2;
        if proto.index >= n as u64 {
            return Err(anyhow!(
                "invalid query index {}: the extended domain size is {n}",
                proto.index
            ));
        }
        if proto.folds.len() != num_rounds(degree_bound, fold_rate, max_rounds) {
            return Err(anyhow!("incorrect proof size"));
        }
        Ok(Self {
            degree_bound,
            blowup_log2,
            fold_rate,
            max_rounds,
            index: proto.index as usize,
            folds: proto
                .folds
                .iter()
                .map(|fold| LeafProof::from_proto(fold, fold_rate))
                .collect::<Result<_>>()?,
        })
    }
}

/// A FRI prover.
///
/// The struct contains the Merkle tree of the committed polynomial and the Merkle trees of the
/// polynomials folded by every round but the last one, so there's one tree for every round, plus
/// the polynomial resulting from the last round. Note that, since every round performs exactly
/// `fold_rate` folds, the folded quotient is not necessarily a constant.
#[derive(Debug, Clone)]
pub struct Prover<F: Field256, H: Hasher<F>> {
    /// The degree bound of the committed polynomial, always a power of 2.
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// The number of folds per round.
    fold_rate: usize,
    /// The maximum number of folding rounds (see [`num_rounds`]).
    max_rounds: usize,
    /// The Merkle trees of the folding argument, one for every round: the first one stores the
    /// evaluations of the committed polynomial, and every subsequent one those of the polynomial
    /// folded by the previous rounds.
    trees: Vec<Tree<F, H>>,
    /// The polynomial resulting from the last folding round.
    folded_quotient: Polynomial<F>,
}

impl<F: Field256, H: Hasher<F>> Prover<F, H> {
    /// Folds the given evaluations once with the given challenge, returning the evaluations of the
    /// folded polynomial (half as many as `values`).
    fn fold(values: &[F], alpha: F) -> Vec<F> {
        let n = values.len();
        assert!(n.is_power_of_two());

        let omega_inv = Polynomial::<F>::domain_element2(1, n).invert_unwrap();

        let m = n / 2;
        let mut omega_inv_i = F::BaseField::ONE;

        (0..m)
            .map(|i| {
                let pos = values[i];
                let neg = values[i + m];
                let value = (pos + neg + alpha * omega_inv_i * (pos - neg)) * F::BaseField::TWO_INV;
                omega_inv_i *= omega_inv;
                value
            })
            .collect()
    }

    /// Folds the coefficients of a polynomial once with the given challenge.
    ///
    /// This is the coefficient form of [`Self::fold`]: writing the polynomial as
    /// `P(X) = P_even(X^2) + X * P_odd(X^2)`, the result is `P_even(X) + alpha * P_odd(X)`.
    ///
    /// REQUIRES: the number of coefficients must be even.
    fn fold_coefficients(coefficients: &[F], alpha: F) -> Vec<F> {
        coefficients
            .chunks(2)
            .map(|pair| pair[0] + alpha * pair[1])
            .collect()
    }

    /// Commits to a polynomial of degree less than `degree_bound`, which must be a power of 2, over
    /// an extended domain `2^blowup_log2` times larger.
    ///
    /// Every round folds the polynomial exactly `fold_rate` times, and the number of rounds is
    /// determined by `max_rounds` as described in [`num_rounds`]. A round folds the evaluations `r`
    /// times with the challenges `alpha`, `alpha^2`, `alpha^4`, and so on, with `alpha` derived
    /// from the Merkle root of the round. That is equivalent to folding them once with arity `2^r`
    /// and challenge `alpha`.
    pub fn new(
        polynomial: Polynomial<F>,
        degree_bound: usize,
        blowup_log2: usize,
        fold_rate: usize,
        max_rounds: usize,
    ) -> Self {
        assert!(degree_bound.is_power_of_two());
        assert!(degree_bound >= polynomial.degree_bound());
        assert!(blowup_log2 > 0);
        assert!(fold_rate > 0);

        let n = degree_bound << blowup_log2;
        assert!(n as u64 <= 1u64 << F::BaseField::S);

        let num_rounds = num_rounds(degree_bound, fold_rate, max_rounds);
        let polynomial = polynomial.shift_domain().trim().pad(degree_bound);
        let mut values = polynomial.clone().lde2(n);
        let mut coefficients = polynomial.take();
        let mut trees = Vec::with_capacity(num_rounds);
        for _ in 0..num_rounds {
            let tree = Tree::new(values.as_slice());
            let mut alpha = H::challenge(*FOLD_DST, &[tree.root_hash()]);
            trees.push(tree);
            for _ in 0..fold_rate {
                values = Self::fold(values.as_slice(), alpha);
                coefficients = Self::fold_coefficients(coefficients.as_slice(), alpha);
                alpha = alpha.square();
            }
        }

        Self {
            degree_bound,
            blowup_log2,
            fold_rate,
            max_rounds,
            trees,
            folded_quotient: Polynomial::with_coefficients(coefficients),
        }
    }

    /// Returns the degree bound of the committed polynomial (always a power of 2).
    ///
    /// NOTE: the actual degree of the original polynomial may be even lower than this value because
    /// it was rounded up to the next power of 2 in order to run the FFT and FRI algorithms.
    pub fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    /// Returns the number of folds per round.
    pub fn fold_rate(&self) -> usize {
        self.fold_rate
    }

    /// Returns the maximum number of folding rounds (see [`num_rounds`]).
    pub fn max_rounds(&self) -> usize {
        self.max_rounds
    }

    /// Returns the size of the extended domain, equal to `degree_bound * blowup`.
    pub fn extended_domain_size(&self) -> usize {
        self.degree_bound << self.blowup_log2
    }

    /// Alias for [`Self::extended_domain_size`].
    pub fn size(&self) -> usize {
        self.degree_bound << self.blowup_log2
    }

    /// Creates the FRI commitment for the committed polynomial.
    pub fn commit(&self) -> Commitment<F> {
        Commitment {
            roots: self.trees.iter().map(Tree::root_hash).collect(),
            folded_quotient: self.folded_quotient.clone(),
        }
    }

    /// Builds a FRI [`Query`] for the value at the specified index of the evaluation domain.
    ///
    /// NOTE: `index` is relative to the *extended* evaluation domain, so for example if you
    /// committed to 4 evaluations with a blowup factor of 8 the range for `index` is [0, 32).
    pub fn query(&self, index: usize) -> Query<F, H> {
        let mut n = self.degree_bound << self.blowup_log2;
        assert!(index < n);

        let mut i = index;
        let mut folds = Vec::with_capacity(self.trees.len());
        for tree in &self.trees {
            folds.push(tree.query(i, self.fold_rate));
            n >>= self.fold_rate;
            i %= n;
        }

        Query {
            degree_bound: self.degree_bound,
            blowup_log2: self.blowup_log2,
            fold_rate: self.fold_rate,
            max_rounds: self.max_rounds,
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
    use std::fmt::Debug;
    use std::str::FromStr;

    fn parse<V: FromStr<Err: Debug>>(s: &'static str) -> V {
        s.parse().unwrap()
    }

    /// Eight distinct evaluations to build trees from.
    fn sample_values() -> Vec<BS> {
        [7, 8, 11, 16, 23, 32, 43, 56]
            .into_iter()
            .map(BS::from_const)
            .collect()
    }

    /// Hashes two 32-byte nodes with an independent Keccak-256 implementation, the same way an EVM
    /// verifier does with `keccak256(abi.encodePacked(left, right))`.
    fn reference_keccak256(left: H256, right: H256) -> H256 {
        use sha3::Digest;
        let mut hasher = sha3::Keccak256::new();
        hasher.update(left.as_bytes());
        hasher.update(right.as_bytes());
        H256::from_slice(hasher.finalize().as_slice())
    }

    /// Reference implementation of a Keccak-256 FRI tree, returning all the nodes of the tree
    /// stored layer by layer as in [`merkle::merklify`].
    ///
    /// The leaves are the given values in the given order, so the caller must bit-reverse them.
    fn reference_keccak256_tree(values: &[BS]) -> Vec<H256> {
        let mut layer: Vec<H256> = values
            .iter()
            .map(|value| H256::from_slice(&value.to_u256().to_big_endian()))
            .collect();
        let mut nodes = layer.clone();
        while layer.len() > 1 {
            layer = layer
                .chunks(2)
                .map(|pair| reference_keccak256(pair[0], pair[1]))
                .collect();
            nodes.extend_from_slice(layer.as_slice());
        }
        nodes
    }

    fn test_reverse_bits_impl(index: usize, num_bits: usize, expected: usize) {
        let result = reverse_bits(index, num_bits);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_reverse_no_bits() {
        test_reverse_bits_impl(0, 0, 0);
    }

    #[test]
    fn test_reverse_one_bit() {
        test_reverse_bits_impl(0, 1, 0);
        test_reverse_bits_impl(1, 1, 1);
    }

    #[test]
    fn test_reverse_three_bits() {
        test_reverse_bits_impl(0b000, 3, 0b000);
        test_reverse_bits_impl(0b001, 3, 0b100);
        test_reverse_bits_impl(0b011, 3, 0b110);
        test_reverse_bits_impl(0b101, 3, 0b101);
        test_reverse_bits_impl(0b110, 3, 0b011);
    }

    #[test]
    fn test_reverse_many_bits() {
        test_reverse_bits_impl(0b1, 62, 1 << 61);
        test_reverse_bits_impl(0b1011, 62, 0b1101 << 58);
        test_reverse_bits_impl(usize::MAX, usize::BITS as usize, usize::MAX);
    }

    fn test_num_rounds_impl(
        degree_bound: usize,
        fold_rate: usize,
        max_rounds: usize,
        expected: usize,
    ) {
        let result = num_rounds(degree_bound, fold_rate, max_rounds);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_num_rounds_without_limit() {
        test_num_rounds_impl(1, 1, usize::MAX, 0);
        test_num_rounds_impl(8, 1, usize::MAX, 3);
        test_num_rounds_impl(64, 3, usize::MAX, 2);
        test_num_rounds_impl(16384, 3, usize::MAX, 4);
    }

    #[test]
    fn test_num_rounds_with_limit() {
        test_num_rounds_impl(8, 1, 0, 0);
        test_num_rounds_impl(8, 1, 2, 2);
        test_num_rounds_impl(16384, 3, 2, 2);
        test_num_rounds_impl(16384, 3, 4, 4);
        test_num_rounds_impl(16384, 3, 5, 4);
    }

    #[test]
    fn test_num_rounds_with_rate_above_folds() {
        test_num_rounds_impl(4, 3, usize::MAX, 0);
        test_num_rounds_impl(128, 8, usize::MAX, 0);
    }

    #[test]
    #[should_panic]
    fn test_num_rounds_with_rate_0() {
        num_rounds(8, 0, usize::MAX);
    }

    #[test]
    #[should_panic]
    fn test_num_rounds_with_invalid_degree_bound() {
        num_rounds(6, 1, usize::MAX);
    }

    #[test]
    fn test_fold_coefficients() {
        let coefficients = [
            BS::from_const(1),
            BS::from_const(2),
            BS::from_const(3),
            BS::from_const(4),
        ];
        let folded = Prover::<BS, Sha2Hash<BS>>::fold_coefficients(
            coefficients.as_slice(),
            BS::from_const(10),
        );
        assert_eq!(folded, [BS::from_const(21), BS::from_const(43)]);
    }

    #[test]
    fn test_fold_coefficients_matches_fold() {
        let polynomial = Polynomial::with_coefficients(sample_values());
        let values = polynomial.clone().lde2(32);
        let alpha = BS::from_const(5);
        let folded =
            Prover::<BS, Sha2Hash<BS>>::fold_coefficients(polynomial.coefficients(), alpha);
        assert_eq!(
            Polynomial::with_coefficients(folded).lde2(16),
            Prover::<BS, Sha2Hash<BS>>::fold(values.as_slice(), alpha)
        );
    }

    fn test_fold_coset_impl(index: usize, fold_rate: usize) {
        let values = sample_values();
        let alpha = BS::from_const(5);
        let mut folded = values.clone();
        let mut challenge = alpha;
        for _ in 0..fold_rate {
            folded = Prover::<BS, Sha2Hash<BS>>::fold(folded.as_slice(), challenge);
            challenge = challenge * challenge;
        }
        let base_index = index % folded.len();
        let proof = Tree::<BS, Sha2Hash<BS>>::new(values.as_slice()).query(index, fold_rate);
        let result =
            Query::<BS, Sha2Hash<BS>>::fold_coset(proof.coset().to_vec(), base_index, 8, alpha);
        assert_eq!(result, folded[base_index]);
    }

    #[test]
    fn test_fold_coset_matches_prover() {
        test_fold_coset_impl(3, 1);
        test_fold_coset_impl(6, 1);
        test_fold_coset_impl(3, 2);
        test_fold_coset_impl(6, 2);
        test_fold_coset_impl(3, 3);
    }

    #[test]
    fn test_fold_dst() {
        assert_eq!(
            *FOLD_DST,
            parse::<H256>("0x9ffd3556faeb2cae194ce95adf6b3580f590504daa0dea56966ce4ef233844af")
        );
    }

    #[test]
    fn test_tree_size() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        assert_eq!(tree.num_leaves(), 8);
        assert_eq!(tree.data.len(), 15);
    }

    #[test]
    fn test_tree_leaves_are_bit_reversed_big_endian_values() {
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(sample_values().as_slice());
        assert_eq!(
            tree.data[0..8],
            [
                H256::from_low_u64_be(7),
                H256::from_low_u64_be(23),
                H256::from_low_u64_be(11),
                H256::from_low_u64_be(43),
                H256::from_low_u64_be(8),
                H256::from_low_u64_be(32),
                H256::from_low_u64_be(16),
                H256::from_low_u64_be(56),
            ]
        );
    }

    #[test]
    fn test_tree_matches_reference_implementation() {
        let values = sample_values();
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(values.as_slice());
        assert_eq!(
            tree.data,
            reference_keccak256_tree(&[
                values[0], values[4], values[2], values[6], values[1], values[5], values[3],
                values[7],
            ])
        );
    }

    #[test]
    fn test_tree_root_hash() {
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(sample_values().as_slice());
        let root_hash = tree.root_hash();
        assert_eq!(root_hash, tree.data[14]);
    }

    #[test]
    fn test_tree_single_value_cosets() {
        let values = sample_values();
        let tree = Tree::<BS, Sha2Hash<BS>>::new(values.as_slice());
        let cosets: Vec<Vec<BS>> = (0..8)
            .map(|index| tree.query(index, 0).coset().to_vec())
            .collect();
        assert_eq!(
            cosets,
            values.iter().map(|&value| vec![value]).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_single_leaf_tree() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(&[BS::from_const(42)]);
        assert_eq!(tree.num_leaves(), 1);
        assert_eq!(tree.data, [H256::from_low_u64_be(42)]);
        assert_eq!(tree.root_hash(), H256::from_low_u64_be(42));
    }

    #[test]
    fn test_query_pair() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3, 1);
        assert_eq!(proof.coset(), [BS::from_const(16), BS::from_const(56)]);
    }

    #[test]
    fn test_query_pair_of_partner() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(7, 1);
        assert_eq!(proof.coset(), [BS::from_const(16), BS::from_const(56)]);
    }

    #[test]
    fn test_query_pair_path() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3, 1);
        assert_eq!(proof.height(), 3);
        assert_eq!(proof.path, [tree.data[8 + 2], tree.data[12]]);
    }

    #[test]
    fn test_query_coset() {
        // Index 3 of 8 belongs to the coset of indices 1, 3, 5, 7, stored in bit-reversed order at
        // positions 4 to 7.
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3, 2);
        assert_eq!(
            proof.coset(),
            [
                BS::from_const(8),
                BS::from_const(32),
                BS::from_const(16),
                BS::from_const(56),
            ]
        );
    }

    #[test]
    fn test_query_coset_path() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3, 2);
        assert_eq!(proof.height(), 3);
        assert_eq!(proof.path, [tree.data[12]]);
    }

    #[test]
    fn test_query_whole_tree_coset() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3, 3);
        assert_eq!(proof.height(), 3);
        assert_eq!(
            proof.coset(),
            [
                BS::from_const(7),
                BS::from_const(23),
                BS::from_const(11),
                BS::from_const(43),
                BS::from_const(8),
                BS::from_const(32),
                BS::from_const(16),
                BS::from_const(56),
            ]
        );
        assert_eq!(proof.path, []);
    }

    #[test]
    fn test_single_leaf_tree_query() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(&[BS::from_const(42)]);
        let proof = tree.query(0, 0);
        assert_eq!(proof.coset(), [BS::from_const(42)]);
        assert_eq!(proof.path, []);
    }

    #[test]
    fn test_leaf_proof_verify() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3, 1);
        let result = proof.verify(3, tree.root_hash());
        assert!(result.is_ok());
    }

    #[test]
    fn test_all_leaf_proofs_verify() {
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(sample_values().as_slice());
        let results: Vec<bool> = (0..8)
            .map(|index| tree.query(index, 1).verify(index, tree.root_hash()).is_ok())
            .collect();
        assert_eq!(results, [true; 8]);
    }

    #[test]
    fn test_leaf_proof_verify_with_wrong_index() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3, 1);
        let result = proof.verify(2, tree.root_hash());
        assert!(result.is_err());
    }

    #[test]
    fn test_leaf_proof_verify_with_partner_index() {
        // The proof authenticates the whole coset, so it's valid for the partner index too.
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3, 1);
        let result = proof.verify(7, tree.root_hash());
        assert!(result.is_ok());
    }

    #[test]
    fn test_leaf_proof_verify_with_wrong_root() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3, 1);
        let result = proof.verify(3, tree.data[0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_leaf_proof_verify_with_tampered_partner() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let mut proof = tree.query(3, 1);
        proof.coset[1] = BS::from_const(57);
        let result = proof.verify(3, tree.root_hash());
        assert!(result.is_err());
    }

    #[test]
    fn test_leaf_proof_verify_with_tampered_value() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let mut proof = tree.query(3, 1);
        proof.coset[0] = BS::from_const(17);
        let result = proof.verify(3, tree.root_hash());
        assert!(result.is_err());
    }

    #[test]
    fn test_coset_proof_verify() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3, 2);
        let result = proof.verify(3, tree.root_hash());
        assert!(result.is_ok());
    }

    fn test_all_coset_proofs_verify_impl(fold_rate: usize) {
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(sample_values().as_slice());
        let results: Vec<bool> = (0..8)
            .map(|index| {
                let proof = tree.query(index, fold_rate);
                proof.verify(index, tree.root_hash()).is_ok()
            })
            .collect();
        assert_eq!(results, [true; 8]);
    }

    #[test]
    fn test_all_coset_proofs_verify() {
        test_all_coset_proofs_verify_impl(2);
        test_all_coset_proofs_verify_impl(3);
    }

    #[test]
    fn test_coset_proof_verify_with_other_index_of_the_coset() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3, 2);
        let result = proof.verify(1, tree.root_hash());
        assert!(result.is_ok());
    }

    #[test]
    fn test_coset_proof_verify_with_index_of_another_coset() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3, 2);
        let result = proof.verify(2, tree.root_hash());
        assert!(result.is_err());
    }

    #[test]
    fn test_coset_proof_verify_with_tampered_leaf() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let mut proof = tree.query(3, 2);
        proof.coset[1] = BS::from_const(33);
        let result = proof.verify(3, tree.root_hash());
        assert!(result.is_err());
    }

    #[test]
    fn test_coset_proof_verify_with_swapped_leaves() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let mut proof = tree.query(3, 2);
        proof.coset.swap(0, 1);
        let result = proof.verify(3, tree.root_hash());
        assert!(result.is_err());
    }

    #[test]
    fn test_single_leaf_proof_verify() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(&[BS::from_const(42)]);
        let proof = tree.query(0, 0);
        let result = proof.verify(0, tree.root_hash());
        assert!(result.is_ok());
    }

    #[test]
    fn test_leaf_proof_to_proto() {
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3, 1);
        let proto = proof.to_proto();
        assert_eq!(
            proto.coset_values,
            [
                BS::from_const(16).to_le_bytes().to_vec(),
                BS::from_const(56).to_le_bytes().to_vec(),
            ]
        );
        assert_eq!(
            proto.merkle_path,
            [
                tree.data[8 + 2].as_bytes().to_vec(),
                tree.data[12].as_bytes().to_vec(),
            ]
        );
    }

    #[test]
    fn test_leaf_proof_from_proto() {
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3, 1);
        let proto = proof.to_proto();
        let decoded = LeafProof::<BS, Keccak256Hash<BS>>::from_proto(&proto, 1).unwrap();
        assert_eq!(decoded, proof);
    }

    #[test]
    fn test_coset_proof_from_proto() {
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3, 2);
        let proto = proof.to_proto();
        let decoded = LeafProof::<BS, Keccak256Hash<BS>>::from_proto(&proto, 2).unwrap();
        assert_eq!(decoded, proof);
    }

    #[test]
    fn test_coset_proof_from_proto_with_missing_value() {
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(sample_values().as_slice());
        let mut proto = tree.query(3, 3).to_proto();
        proto.coset_values.pop();
        let result = LeafProof::<BS, Keccak256Hash<BS>>::from_proto(&proto, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_coset_proof_from_proto_with_extra_value() {
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(sample_values().as_slice());
        let mut proto = tree.query(3, 2).to_proto();
        proto
            .coset_values
            .push(BS::from_const(42).to_le_bytes().to_vec());
        let result = LeafProof::<BS, Keccak256Hash<BS>>::from_proto(&proto, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_coset_proof_from_proto_with_non_canonical_value() {
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(sample_values().as_slice());
        let mut proto = tree.query(3, 2).to_proto();
        let modulus: primitive_types::U256 = BS::MODULUS.parse().unwrap();
        proto.coset_values[2] = modulus.to_little_endian().to_vec();
        let result = LeafProof::<BS, Keccak256Hash<BS>>::from_proto(&proto, 2);
        assert!(result.is_err());
    }

    const DEGREE_BOUND: usize = 8;
    const BLOWUP_LOG2: usize = 2;
    const FOLD_RATE: usize = 1;
    const MAX_ROUNDS: usize = usize::MAX;
    const QUERY_INDEX: usize = 3;

    /// Builds a polynomial from its coefficients.
    fn make_polynomial<F: Field256>(coefficients: &[u64]) -> Polynomial<F> {
        Polynomial::with_coefficients(coefficients.iter().copied().map(F::from).collect())
    }

    /// The polynomial committed by the prover in most of the tests.
    fn honest_polynomial() -> Polynomial<BS> {
        make_polynomial(&[12, 34, 56, 78, 90, 12, 34, 56])
    }

    /// The evaluation of [`honest_polynomial`] at the element of the extended domain at index
    /// [`QUERY_INDEX`].
    fn honest_value() -> BS {
        let x = Polynomial::<BS>::coset_element2(QUERY_INDEX, DEGREE_BOUND << BLOWUP_LOG2);
        honest_polynomial().evaluate(x)
    }

    /// Same as [`honest_polynomial`] but with different coefficients.
    fn foreign_polynomial() -> Polynomial<BS> {
        make_polynomial(&[90, 78, 56, 34, 12, 90, 78, 56])
    }

    fn test_commitment_impl<H: Hasher<BS>>(
        fold_rate: usize,
        max_rounds: usize,
        expected_roots: &[&'static str],
        expected_folded_quotient: &[&'static str],
    ) {
        let prover = Prover::<BS, H>::new(honest_polynomial(), 8, 2, fold_rate, max_rounds);
        let commitment = prover.commit();
        assert_eq!(
            commitment.roots(),
            expected_roots
                .iter()
                .map(|&root| parse(root))
                .collect::<Vec<H256>>()
        );
        assert_eq!(
            commitment.folded_quotient().coefficients(),
            expected_folded_quotient
                .iter()
                .map(|&coefficient| parse(coefficient))
                .collect::<Vec<BS>>()
        );
    }

    #[test]
    fn test_commitment_with_fold_rate_1() {
        test_commitment_impl::<Sha2Hash<BS>>(
            1,
            usize::MAX,
            &[
                "0x4dbf05b032faad3a0279e11536170c9dc6c2444b65bfda0917160a556b76ab5b",
                "0x3beefd65cfe8506f9b499e8397100d6d695d97af5742e70b60100f056d55307b",
                "0x7d5697dea024a95e3d383fdc569fb615fa391a96f710201c970bebe336476c9b",
            ],
            &["0x498284459f1ebaa29b74fa6160113aff504a94794da2deb4190b7050842f686c"],
        );
        test_commitment_impl::<Keccak256Hash<BS>>(
            1,
            usize::MAX,
            &[
                "0xf62fdcf4604e703d26974310d70bca5ef93f22d316f2083167b3bc4fd4437f8f",
                "0x3e708444a31aed3a38ef8d8074873506efba437be058889e79b8499095e5eefd",
                "0xbb62bc0c84acc6aa08f2bac9f342426dc3e9c284630031807c3f848f35b15dcc",
            ],
            &["0x5f39da2f621ee22a45dc105e400ff563f2c6ffaebfae9914734b7aa7f0b2df4f"],
        );
    }

    #[test]
    fn test_commitment_with_fold_rate_2() {
        test_commitment_impl::<Sha2Hash<BS>>(
            2,
            usize::MAX,
            &["0x4dbf05b032faad3a0279e11536170c9dc6c2444b65bfda0917160a556b76ab5b"],
            &[
                "0x223cdfe19f534f19b38372a42464ceb4a7136427c6140aa80ed3f34096d03266",
                "0x478b9dab28cff46eae9d5fd746db792992acdb77600a7686a5f035cd9fca7712",
            ],
        );
        test_commitment_impl::<Keccak256Hash<BS>>(
            2,
            usize::MAX,
            &["0xf62fdcf4604e703d26974310d70bca5ef93f22d316f2083167b3bc4fd4437f8f"],
            &[
                "0x34da8614b0f50cfe28a1e96719f05dc4dc18e7201e5dd32b93a2b782cf6ac201",
                "0x3f55dd48fe5538bcc9f934e9a33f8bd484e3cd7d34744ac08f266d262146df7f",
            ],
        );
    }

    #[test]
    fn test_commitment_with_fold_rate_3() {
        test_commitment_impl::<Sha2Hash<BS>>(
            3,
            usize::MAX,
            &["0x4dbf05b032faad3a0279e11536170c9dc6c2444b65bfda0917160a556b76ab5b"],
            &["0x3f2a2db74fbf3875efc75d2b0fc2a7fa6b3b3ba0c3774168548036effacefffd"],
        );
        test_commitment_impl::<Keccak256Hash<BS>>(
            3,
            usize::MAX,
            &["0xf62fdcf4604e703d26974310d70bca5ef93f22d316f2083167b3bc4fd4437f8f"],
            &["0x77fcd1e8fad3d79dd58d54f22a33f97014ed1161713ceb7728d98e0f5aa11905"],
        );
    }

    #[test]
    fn test_commitment_with_one_round() {
        test_commitment_impl::<Sha2Hash<BS>>(
            1,
            1,
            &["0x4dbf05b032faad3a0279e11536170c9dc6c2444b65bfda0917160a556b76ab5b"],
            &[
                "0x5113579bb66be96da73beb99454bad6a8b372e4a8bee85cb896353fa3c4ceebd",
                "0x480a0a00b13c1e9ffee51392c28ae94358b45d0a7c440f844013941ecdb4ca47",
                "0x63d7d7b3e4bfd83ffbbfd520c4db3268388b6d9d511b00ad004b4d62c8695a86",
                "0x2b084d50ea188d28bac1cb99b68fa4322681591974a01ac377a699af54fc8e7a",
            ],
        );
        test_commitment_impl::<Keccak256Hash<BS>>(
            1,
            1,
            &["0xf62fdcf4604e703d26974310d70bca5ef93f22d316f2083167b3bc4fd4437f8f"],
            &[
                "0x1c7f11b5aab72663576b493b95048e8d6dc05332911790c251dcfcc295d40566",
                "0x443b8e49b780a8b9ff2749fedb14685e1a6e9e9d83719729fcbe826adc61ed0e",
                "0x066ee82f37eed7b7adf992c085b0f3e409fccd8c462f7374737a445fed2aa534",
                "0x6e8b2584d2d8f9b54912366797f9d0a1179d01858dec9e008b0dd3b76a689771",
            ],
        );
    }

    #[test]
    fn test_prover_parameters() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        assert_eq!(prover.degree_bound(), DEGREE_BOUND);
        assert_eq!(prover.fold_rate(), FOLD_RATE);
        assert_eq!(prover.max_rounds(), MAX_ROUNDS);
        assert_eq!(prover.extended_domain_size(), DEGREE_BOUND << BLOWUP_LOG2);
        assert_eq!(prover.size(), DEGREE_BOUND << BLOWUP_LOG2);
    }

    fn test_queries_impl<F: Field256, H: Hasher<F>>(
        coefficients: &[u64],
        degree_bound: usize,
        blowup_log2: usize,
        fold_rate: usize,
        max_rounds: usize,
        num_rounds: usize,
    ) {
        let polynomial = make_polynomial::<F>(coefficients);
        let prover = Prover::<F, H>::new(
            polynomial.clone(),
            degree_bound,
            blowup_log2,
            fold_rate,
            max_rounds,
        );
        let commitment = prover.commit();
        let n = degree_bound << blowup_log2;
        let queries: Vec<Query<F, H>> = (0..n).map(|index| prover.query(index)).collect();
        assert_eq!(commitment.len(), num_rounds);
        assert_eq!(
            commitment.folded_quotient().len(),
            degree_bound >> (num_rounds * fold_rate)
        );
        for (index, query) in queries.iter().enumerate() {
            assert_eq!(query.index(), index);
            assert_eq!(query.len(), num_rounds);
            assert!(
                query
                    .verify(&commitment, polynomial.evaluate(query.x()))
                    .is_ok()
            );
        }
    }

    /// Runs [`test_queries_impl`] with every combination of field and hash backend.
    fn test_queries_with_options(
        coefficients: &[u64],
        degree_bound: usize,
        blowup_log2: usize,
        fold_rate: usize,
        max_rounds: usize,
        num_rounds: usize,
    ) {
        test_queries_impl::<BS, Sha2Hash<BS>>(
            coefficients,
            degree_bound,
            blowup_log2,
            fold_rate,
            max_rounds,
            num_rounds,
        );
        test_queries_impl::<BS, Keccak256Hash<BS>>(
            coefficients,
            degree_bound,
            blowup_log2,
            fold_rate,
            max_rounds,
            num_rounds,
        );
        test_queries_impl::<GL4, Sha2Hash<GL4>>(
            coefficients,
            degree_bound,
            blowup_log2,
            fold_rate,
            max_rounds,
            num_rounds,
        );
        test_queries_impl::<GL4, Keccak256Hash<GL4>>(
            coefficients,
            degree_bound,
            blowup_log2,
            fold_rate,
            max_rounds,
            num_rounds,
        );
    }

    /// Runs [`test_queries_impl`] with every combination of field and hash backend, with a fold
    /// rate of 1 and no limit on the number of rounds.
    fn test_queries(coefficients: &[u64], degree_bound: usize, blowup_log2: usize) {
        let num_rounds = degree_bound.trailing_zeros() as usize;
        test_queries_with_options(
            coefficients,
            degree_bound,
            blowup_log2,
            1,
            usize::MAX,
            num_rounds,
        );
    }

    /// Runs [`test_queries_impl`] with every combination of field and hash backend, with no limit
    /// on the number of rounds.
    fn test_queries_with_fold_rate(
        coefficients: &[u64],
        degree_bound: usize,
        blowup_log2: usize,
        fold_rate: usize,
        num_rounds: usize,
    ) {
        test_queries_with_options(
            coefficients,
            degree_bound,
            blowup_log2,
            fold_rate,
            usize::MAX,
            num_rounds,
        );
    }

    #[test]
    fn test_constant_polynomial_blowup_2() {
        test_queries(&[12], 1, 1);
    }

    #[test]
    fn test_constant_polynomial_blowup_4() {
        test_queries(&[34], 1, 2);
    }

    #[test]
    fn test_constant_polynomial_blowup_8() {
        test_queries(&[56], 1, 3);
    }

    #[test]
    fn test_polynomial_degree_one_blowup_2() {
        test_queries(&[12, 34], 2, 1);
    }

    #[test]
    fn test_polynomial_degree_one_blowup_4() {
        test_queries(&[56, 78], 2, 2);
    }

    #[test]
    fn test_polynomial_degree_one_blowup_8() {
        test_queries(&[90, 12], 2, 3);
    }

    #[test]
    fn test_polynomial_degree_three_blowup_2() {
        test_queries(&[12, 34, 56, 78], 4, 1);
    }

    #[test]
    fn test_polynomial_degree_three_blowup_4() {
        test_queries(&[42, 43, 44, 45], 4, 2);
    }

    #[test]
    fn test_polynomial_degree_three_blowup_8() {
        test_queries(&[78, 56, 34, 12], 4, 3);
    }

    #[test]
    fn test_polynomial_degree_seven_blowup_2() {
        test_queries(&[12, 34, 56, 78, 90, 12, 34, 56], 8, 1);
    }

    #[test]
    fn test_polynomial_degree_seven_blowup_4() {
        test_queries(&[42, 43, 44, 45, 46, 47, 48, 49], 8, 2);
    }

    #[test]
    fn test_polynomial_degree_seven_blowup_8() {
        test_queries(&[56, 34, 12, 90, 78, 56, 34, 12], 8, 3);
    }

    #[test]
    fn test_polynomial_below_degree_bound_blowup_2() {
        test_queries(&[12, 34], 8, 1);
    }

    #[test]
    fn test_polynomial_below_degree_bound_blowup_4() {
        test_queries(&[56, 78], 8, 2);
    }

    #[test]
    fn test_polynomial_below_degree_bound_blowup_8() {
        test_queries(&[90, 12], 8, 3);
    }

    #[test]
    fn test_constant_polynomial_fold_rate_3() {
        test_queries_with_fold_rate(&[12], 1, 2, 3, 0);
    }

    #[test]
    fn test_polynomial_degree_one_fold_rate_3() {
        test_queries_with_fold_rate(&[12, 34], 2, 2, 3, 0);
    }

    #[test]
    fn test_polynomial_degree_three_fold_rate_2() {
        test_queries_with_fold_rate(&[12, 34, 56, 78], 4, 1, 2, 1);
    }

    #[test]
    fn test_polynomial_degree_seven_fold_rate_2() {
        test_queries_with_fold_rate(&[12, 34, 56, 78, 90, 12, 34, 56], 8, 2, 2, 1);
    }

    #[test]
    fn test_polynomial_degree_seven_fold_rate_3() {
        test_queries_with_fold_rate(&[42, 43, 44, 45, 46, 47, 48, 49], 8, 1, 3, 1);
    }

    #[test]
    fn test_polynomial_degree_seven_fold_rate_4() {
        test_queries_with_fold_rate(&[56, 34, 12, 90, 78, 56, 34, 12], 8, 2, 4, 0);
    }

    #[test]
    fn test_polynomial_degree_fifteen_fold_rate_3() {
        test_queries_with_fold_rate(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            16,
            1,
            3,
            1,
        );
    }

    #[test]
    fn test_polynomial_below_degree_bound_fold_rate_2() {
        test_queries_with_fold_rate(&[90, 12], 8, 2, 2, 1);
    }

    #[test]
    fn test_polynomial_degree_seven_zero_rounds() {
        test_queries_with_options(&[12, 34, 56, 78, 90, 12, 34, 56], 8, 2, 1, 0, 0);
    }

    #[test]
    fn test_polynomial_degree_seven_one_round() {
        test_queries_with_options(&[12, 34, 56, 78, 90, 12, 34, 56], 8, 2, 1, 1, 1);
    }

    #[test]
    fn test_polynomial_degree_seven_two_rounds() {
        test_queries_with_options(&[12, 34, 56, 78, 90, 12, 34, 56], 8, 1, 1, 2, 2);
    }

    #[test]
    fn test_polynomial_degree_fifteen_fold_rate_2_one_round() {
        test_queries_with_options(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            16,
            2,
            2,
            1,
            1,
        );
    }

    #[test]
    fn test_polynomial_below_degree_bound_zero_rounds() {
        test_queries_with_options(&[90, 12], 8, 2, 3, 0, 0);
    }

    #[test]
    fn test_accept_untampered_proof() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let query = prover.query(QUERY_INDEX);
        let result = query.verify(&prover.commit(), honest_value());
        assert!(result.is_ok());
    }

    #[test]
    fn test_reject_foreign_commitment() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let query = prover.query(QUERY_INDEX);
        let foreign =
            Prover::<BS, Sha2Hash<BS>>::new(foreign_polynomial(), 8, 2, 1, MAX_ROUNDS).commit();
        let result = query.verify(&foreign, honest_value());
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_corrupted_root() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let mut commitment = prover.commit();
        commitment.roots[2] = H256::repeat_byte(0xAA);
        let result = prover
            .query(QUERY_INDEX)
            .verify(&commitment, honest_value());
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_truncated_folds() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        query.folds.truncate(2);
        let result = query.verify(&commitment, honest_value());
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_truncated_commitment() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let mut commitment = prover.commit();
        commitment.roots.truncate(2);
        let result = prover
            .query(QUERY_INDEX)
            .verify(&commitment, honest_value());
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_extra_folds() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        let mut donor = prover.query(QUERY_INDEX);
        query.folds.push(donor.folds.pop().unwrap());
        let result = query.verify(&commitment, honest_value());
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_folding_stopped_early() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let mut commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        commitment.roots.truncate(2);
        query.folds.truncate(2);
        let result = query.verify(&commitment, honest_value());
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_tampered_folded_quotient() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let mut commitment = prover.commit();
        let mut coefficients = commitment.folded_quotient.take();
        coefficients[0] += BS::ONE;
        commitment.folded_quotient = Polynomial::with_coefficients(coefficients);
        let result = prover
            .query(QUERY_INDEX)
            .verify(&commitment, honest_value());
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_oversized_folded_quotient() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let mut commitment = prover.commit();
        let mut coefficients = commitment.folded_quotient.take();
        coefficients.push(BS::ZERO);
        commitment.folded_quotient = Polynomial::with_coefficients(coefficients);
        let result = prover
            .query(QUERY_INDEX)
            .verify(&commitment, honest_value());
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_oversized_folded_quotient_without_rounds() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2, 1, 0);
        let mut commitment = prover.commit();
        let mut coefficients = commitment.folded_quotient.take();
        coefficients.push(BS::ZERO);
        commitment.folded_quotient = Polynomial::with_coefficients(coefficients);
        let result = prover
            .query(QUERY_INDEX)
            .verify(&commitment, honest_value());
        assert!(result.is_err());
    }

    #[test]
    fn test_accept_trimmed_folded_quotient() {
        // The polynomial has degree less than 2, so after one round its folded quotient has 4
        // coefficients of which only the first is non-zero. A shorter folded quotient proves a
        // stronger degree bound.
        let polynomial = make_polynomial(&[12, 34]);
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(polynomial.clone(), DEGREE_BOUND, BLOWUP_LOG2, 1, 1);
        let mut commitment = prover.commit();
        commitment.folded_quotient = commitment.folded_quotient.clone().trim();
        let query = prover.query(QUERY_INDEX);
        let result = query.verify(&commitment, polynomial.evaluate(query.x()));
        assert!(result.is_ok());
    }

    #[test]
    fn test_zero_rounds_folded_quotient() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2, 1, 0);
        let commitment = prover.commit();
        assert_eq!(commitment.len(), 0);
        assert_eq!(
            *commitment.folded_quotient(),
            honest_polynomial().shift_domain()
        );
    }

    #[test]
    fn test_reject_wrong_proof_height() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        let mut donor = prover.query(QUERY_INDEX);
        query.folds[0] = donor.folds.remove(1);
        let result = query.verify(&commitment, honest_value());
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_wrong_value() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let commitment = prover.commit();
        let query = prover.query(QUERY_INDEX);
        let result = query.verify(&commitment, honest_value() + BS::ONE);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_wrong_value_without_rounds() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2, 1, 0);
        let commitment = prover.commit();
        let query = prover.query(QUERY_INDEX);
        let result = query.verify(&commitment, honest_value() + BS::ONE);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_swapped_partners() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        query.folds[0].coset.swap(0, 1);
        let result = query.verify(&commitment, honest_value());
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_tampered_partner() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        query.folds[1].coset[1] += BS::ONE;
        let result = query.verify(&commitment, honest_value());
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_foreign_leaf_proof() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        let mut other = prover.query(QUERY_INDEX + 4);
        query.folds[1] = other.folds.remove(1);
        let result = query.verify(&commitment, honest_value());
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_extra_path_hash() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        query.folds[0].path.push(H256::zero());
        let result = query.verify(&commitment, honest_value());
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_out_of_range_query_index() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.index = (QUERY_INDEX + (DEGREE_BOUND << BLOWUP_LOG2)) as u64;
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
            &proto,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_huge_query_index() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.index = u64::MAX;
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
            &proto,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_short_merkle_path() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        query.folds[1].path.pop();
        let result = query.verify(&commitment, honest_value());
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_non_canonical_partner() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        let modulus: primitive_types::U256 = BS::MODULUS.parse().unwrap();
        proto.folds[2].coset_values[1] = modulus.to_little_endian().to_vec();
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
            &proto,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_leaf_without_values() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.folds[2].coset_values.clear();
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
            &proto,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_leaf_with_three_values() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        let coset_values = &mut proto.folds[0].coset_values;
        coset_values.push(coset_values[0].clone());
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
            &proto,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_long_value() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.folds[1].coset_values[0].resize(33, 0);
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
            &proto,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_non_canonical_value() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        let modulus: primitive_types::U256 = BS::MODULUS.parse().unwrap();
        proto.folds[1].coset_values[0] = modulus.to_little_endian().to_vec();
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
            &proto,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_short_path_hash() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.folds[0].merkle_path[1].pop();
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
            &proto,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_tampered_coset_leaf() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            3,
            MAX_ROUNDS,
        );
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        let proof = &mut query.folds[0];
        proof.coset[4] += BS::ONE;
        let result = query.verify(&commitment, honest_value());
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_swapped_coset_leaves() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            3,
            MAX_ROUNDS,
        );
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        query.folds[0].coset.swap(1, 2);
        let result = query.verify(&commitment, honest_value());
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_wrong_coset_size() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            3,
            MAX_ROUNDS,
        );
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        query.fold_rate = 2;
        let result = query.verify(&commitment, honest_value());
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_query_with_wrong_fold_rate() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            3,
            MAX_ROUNDS,
        );
        let proto = prover.query(QUERY_INDEX).to_proto();
        let result =
            Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, 1, MAX_ROUNDS, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_too_many_folds_for_fold_rate() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let proto = prover.query(QUERY_INDEX).to_proto();
        let result =
            Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, 3, MAX_ROUNDS, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_coset_serialization() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            3,
            MAX_ROUNDS,
        );
        let query = prover.query(QUERY_INDEX);
        let decoded = Query::<BS, Sha2Hash<BS>>::from_proto(
            DEGREE_BOUND,
            BLOWUP_LOG2,
            3,
            MAX_ROUNDS,
            &query.to_proto(),
        )
        .unwrap();
        assert_eq!(decoded.to_proto(), query.to_proto());
    }

    #[test]
    fn test_serialization() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
        );
        let commitment = prover.commit();
        let query = prover.query(QUERY_INDEX);
        let decoded = Query::<BS, Sha2Hash<BS>>::from_proto(
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            MAX_ROUNDS,
            &query.to_proto(),
        )
        .unwrap();
        assert_eq!(decoded.index(), query.index());
        assert_eq!(decoded.len(), query.len());
        assert_eq!(decoded.x(), query.x());
        assert!(decoded.verify(&commitment, honest_value()).is_ok());
    }
}
