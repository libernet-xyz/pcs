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

/// Returns the number of folds performed by each round of the folding argument.
///
/// `num_folds` is the total number of folds, that is the log2 of the degree bound. Every round
/// performs `fold_rate` folds except possibly the last one, which performs *up to* `fold_rate`
/// folds. Performing the larger rounds first minimizes the total length of the Merkle paths,
/// because they shrink the trees of the subsequent rounds.
///
/// REQUIRES: `fold_rate` must be greater than 0.
pub(crate) fn fold_schedule(num_folds: usize, fold_rate: usize) -> Vec<usize> {
    assert!(fold_rate > 0);
    let mut schedule = vec![fold_rate; num_folds / fold_rate];
    if !num_folds.is_multiple_of(fold_rate) {
        schedule.push(num_folds % fold_rate);
    }
    schedule
}

/// Returns the log2 of the size of the coset opened in the given Merkle tree of the folding
/// argument.
///
/// Every tree but the last one opens the coset folded by its round (see [`fold_schedule`]), while
/// the last tree, which only needs to witness that the final polynomial is constant, opens a pair
/// of partners.
fn coset_size_log2(schedule: &[usize], tree: usize, num_trees: usize) -> usize {
    if tree + 1 < num_trees {
        schedule[tree]
    } else {
        1
    }
}

/// Folds a coset of evaluations into a single evaluation of the folded polynomial.
///
/// `values` contains the `2^r` evaluations of a coset of the evaluation domain of size `n`, in leaf
/// order (see [`Tree`]): the evaluation at index `base_index + j * n / 2^r` is in position
/// `reverse_bits(j, r)`, with `base_index < n / 2^r`. The coset is folded `r` times with the
/// challenges `alpha`, `alpha^2`, `alpha^4`, and so on, which is equivalent to folding it once with
/// arity `2^r` and challenge `alpha`. The result is the evaluation of the folded polynomial at
/// index `base_index` of the evaluation domain of size `n / 2^r`.
fn fold_coset<F: Field256>(mut values: Vec<F>, base_index: usize, mut n: usize, mut alpha: F) -> F {
    while values.len() > 1 {
        // Leaves `2*i` and `2*i+1` contain the evaluations at `X` and `-X`. The former is the one
        // at the lower index, `base_index + reverse_bits(i, log2(half)) * stride`.
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
        alpha = alpha * alpha;
    }
    values[0]
}

/// Returns the position of the evaluation at `index` within its coset of size `2^coset_size_log2`,
/// in a Merkle tree of the given height (see [`Tree`]).
///
/// REQUIRES: `index` must be less than `2^height`.
fn coset_offset(index: usize, height: usize, coset_size_log2: usize) -> usize {
    reverse_bits(index, height) & ((1 << coset_size_log2) - 1)
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
/// subtree of height `r`. The proof contains all of them, from which the verifier recomputes the
/// subtree root, and the Merkle path from the subtree root up to the root of the tree.
///
/// Note that the proof authenticates the coset as a whole: it's valid for any index of the coset,
/// and it's up to the caller to pick the evaluation at the queried index (see [`coset_offset`]).
#[derive(Debug, Clone, PartialEq, Eq)]
struct LeafProof<F: Field256, H: Hasher<F>> {
    /// The evaluations of the coset in leaf order. Their number is always a power of 2.
    coset: Vec<F>,
    /// The Merkle path from the root of the coset subtree up to, but excluding, the root of the
    /// tree.
    path: Vec<H256>,
    _data: PhantomData<H>,
}

impl<F: Field256, H: Hasher<F>> LeafProof<F, H> {
    /// Returns the log2 of the size of the coset.
    fn coset_size_log2(&self) -> usize {
        self.coset.len().trailing_zeros() as usize
    }

    /// Returns the height of the tree, that is the log2 of its number of leaves.
    fn height(&self) -> usize {
        self.coset_size_log2() + self.path.len()
    }

    /// Returns the evaluations of the coset in leaf order.
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
        let size = self.coset.len();
        let mut subtree = Vec::with_capacity(size * 2 - 1);
        subtree.extend(self.coset.iter().map(|&value| utils::encode_scalar(value)));
        subtree.resize(size * 2 - 1, H256::zero());
        merkle::merklify::<H>(subtree.as_mut_slice(), size);
        merkle::verify_path::<H>(
            *subtree.last().unwrap(),
            self.path.as_slice(),
            reverse_bits(index, self.height()) >> self.coset_size_log2(),
            root_hash,
        )
    }

    /// Returns true iff the proof witnesses a tree whose leaves are all equal, i.e. the evaluations
    /// of a constant polynomial.
    ///
    /// That's the case if and only if all the evaluations of the coset are equal and every sibling
    /// equals the hash computed so far, starting from the root of the coset subtree.
    fn is_constant(&self) -> bool {
        let value = self.coset[0];
        if self.coset.iter().any(|&other| other != value) {
            return false;
        }
        let mut hash = utils::encode_scalar(value);
        for _ in 0..self.coset_size_log2() {
            hash = H::hash_binary(hash, hash);
        }
        for &sibling in &self.path {
            if sibling != hash {
                return false;
            }
            hash = H::hash_binary(hash, hash);
        }
        true
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

    /// Deserializes a proof for a coset of size `2^coset_size_log2` from a
    /// [`LeafProof`](`proto::query::LeafProof`) protobuf.
    ///
    /// Returns an error if the number of coset values is not `2^coset_size_log2`, if any value is
    /// not a canonical field element, or if any hash is invalid.
    fn from_proto(proto: &proto::query::LeafProof, coset_size_log2: usize) -> Result<Self> {
        let size = 1 << coset_size_log2;
        if proto.coset_values.len() != size {
            return Err(anyhow!(
                "invalid FRI coset size (got {}, want {})",
                proto.coset_values.len(),
                size
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
/// `i` and `i + N/2` (that is at `X` and `-X`) only differ in the most significant bit of their
/// index, so they end up in sibling leaves and a single Merkle path opens both. More generally, the
/// `2^r` evaluations of a coset folded by a round with fold rate `r`, at the indices `i + j*N/2^r`,
/// only differ in their `r` most significant bits, so they're the leaves of a subtree of height
/// `r`. Furthermore, the root of that subtree is at the same position as the leaf of the folded
/// evaluation in the next round.
///
/// NOTE: bit reversal is completely abstracted away by `Tree`. The provided API exposes the
/// original evaluation domain indices, but bit reversal allows opening a whole coset with a single
/// Merkle proof.
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

    /// Returns a Merkle proof for the coset of size `2^coset_size_log2` containing the evaluation
    /// at the given index of the evaluation domain.
    ///
    /// NOTE: `index` is not bit-reversed, but thanks to the internal bit reversal the returned
    /// [`LeafProof`] proves all the evaluations of the coset with a single Merkle path. For
    /// example, with `coset_size_log2 = 1` it proves the evaluation at `index` and its partner at
    /// `index + N/2`.
    ///
    /// REQUIRES: `coset_size_log2` must not exceed the height of the tree.
    fn query(&self, index: usize, coset_size_log2: usize) -> LeafProof<F, H> {
        let n = self.num_leaves();
        let reverse_index = reverse_bits(index, n.trailing_zeros() as usize);
        let size = 1 << coset_size_log2;
        let start = reverse_index & !(size - 1);
        LeafProof {
            coset: self.data[start..(start + size)]
                .iter()
                .map(|&leaf| utils::decode_scalar(leaf).unwrap())
                .collect(),
            path: merkle::merkle_path(self.data.as_slice(), n, reverse_index)[coset_size_log2..]
                .to_vec(),
            _data: PhantomData,
        }
    }
}

/// Stores the Merkle root hashes of a FRI commitment.
///
/// Note that for low-degree testing these are strictly less than `log2(N)+1`, with N being the
/// number of committed evaluations. Once the folding process has reduced the polynomial to a
/// degree-0 one (a single constant) all subsequent folds would be identical, so we don't store
/// them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment {
    /// The first element in the array is the root of the main Merkle tree, the second one is the
    /// root of the Merkle tree from the first folding round, and so on until the last element which
    /// is the Merkle root of the last folding round.
    roots: Vec<H256>,
}

impl Commitment {
    pub(crate) fn with_roots(roots: Vec<H256>) -> Self {
        Self { roots }
    }

    /// Returns the number of stored roots, equivalent to the number of Merkle trees known to the
    /// prover. These would in turn include the base Merkle tree of the committed evaluations (over
    /// the extended domain) and one subsequent tree for every folding round. The number of folding
    /// rounds is the log2 of the degree bound of the original polynomial divided by the fold rate,
    /// rounded up. For example, if the user commits a degree<8 polynomial encoding 8 values,
    /// `len()` will return log2(8)+1 = 3+1 = 4 with a fold rate of 1, or 2 with a fold rate of 3,
    /// regardless of the blowup factor.
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
    /// The degree bound of the committed polynomial (always a power of 2).
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// The number of folds per round (see [`fold_schedule`]).
    fold_rate: usize,
    /// The index of the element we're opening (the partner index is inferred automatically).
    ///
    /// Always less than the size of the extended domain.
    index: usize,
    /// Proves the coset of the queried index folded by each round with one [`LeafProof`] for every
    /// Merkle tree. The proof at `folds[0]` proves the opened values.
    folds: Vec<LeafProof<F, H>>,
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
    /// element by the [`MULTIPLICATIVE_GENERATOR`](`Field::MULTIPLICATIVE_GENERATOR`) of the *base*
    /// field, consistently with `shift_domain`.
    pub fn x(&self) -> F {
        Polynomial::<F>::coset_element2(self.index, self.degree_bound << self.blowup_log2).into()
    }

    /// Returns the opened evaluations of the committed polynomial.
    ///
    /// The first component of the returned tuple is the evaluation at the first index returned by
    /// [`Self::indices`], while the second component is the one at the second index.
    pub fn values(&self) -> (F, F) {
        let proof = &self.folds[0];
        let height = (self.degree_bound << self.blowup_log2).trailing_zeros() as usize;
        let offset = coset_offset(self.index, height, proof.coset_size_log2());
        let coset = proof.coset();
        (coset[offset], coset[offset ^ 1])
    }

    /// Returns the number of Merkle trees in the proof: one for the original polynomial evaluations
    /// plus one for every folding round.
    ///
    /// In general these are ceil(log2(d)/r)+1, with `d` being the degree bound of the committed
    /// polynomial and `r` the fold rate, that is the number of folds per round. Note that for
    /// low-degree testing `d` is strictly less than the number of committed evaluations `N`.
    pub fn len(&self) -> usize {
        self.folds.len()
    }

    /// Verifies this proof against the given commitment.
    ///
    /// NOTE: for low-degree testing you also need to check that [`Self::len`] returns the number of
    /// Merkle trees expected for the degree bound. This function only verifies the opened values
    /// across the folding structure.
    pub fn verify(&self, commitment: &Commitment) -> Result<()> {
        let mut n = self.degree_bound << self.blowup_log2;
        assert!(n.is_power_of_two());

        let mut k = n.trailing_zeros() as usize;

        let schedule = fold_schedule(self.degree_bound.trailing_zeros() as usize, self.fold_rate);
        let num_trees = self.folds.len();
        if num_trees > schedule.len() + 1 {
            return Err(anyhow!("incorrect proof size"));
        }
        if commitment.len() != num_trees {
            return Err(anyhow!("wrong number of folding rounds"));
        }

        let mut index = self.index;
        // The evaluation at the queried index expected from the previous round.
        let mut expected: Option<F> = None;

        for round in 0..num_trees {
            let root_hash = commitment.roots()[round];

            let proof = &self.folds[round];
            let coset_size_log2 = coset_size_log2(schedule.as_slice(), round, num_trees);
            if proof.coset_size_log2() != coset_size_log2 {
                return Err(anyhow!(
                    "invalid FRI coset size 2^{} (want 2^{})",
                    proof.coset_size_log2(),
                    coset_size_log2
                ));
            }
            if proof.height() != k {
                return Err(anyhow!(
                    "invalid Merkle proof height (got {}, want {})",
                    proof.height(),
                    k
                ));
            }

            let offset = coset_offset(index, k, coset_size_log2);
            if expected.is_some_and(|expected| proof.coset()[offset] != expected) {
                return Err(anyhow!("leaf value mismatch"));
            }
            proof.verify(index, root_hash)?;

            if round + 1 < num_trees {
                let alpha = H::challenge(*FOLD_DST, &[root_hash]);
                let folded_n = n >> coset_size_log2;
                expected = Some(fold_coset(
                    proof.coset().to_vec(),
                    index % folded_n,
                    n,
                    alpha,
                ));
                n = folded_n;
                k -= coset_size_log2;
                index %= n;
            }
        }

        if !self.folds.last().unwrap().is_constant() {
            return Err(anyhow!("the final folded polynomial is not constant"));
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
    /// The query index must be less than the size of the extended domain, the number of Merkle
    /// proofs must not exceed the number of Merkle trees of the folding argument, every Merkle
    /// proof must contain the whole coset folded by its round, and every opened value must be a
    /// canonical field element, otherwise an error is returned.
    ///
    /// NOTE: this method does not validate the proof, it only deserializes it. The caller must
    /// invoke [`Self::verify`] separately.
    pub fn from_proto(
        degree_bound: usize,
        blowup_log2: usize,
        fold_rate: usize,
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
        if proto.folds.is_empty() {
            return Err(anyhow!("the FRI query has no folds"));
        }
        let schedule = fold_schedule(degree_bound.trailing_zeros() as usize, fold_rate);
        let num_trees = proto.folds.len();
        if num_trees > schedule.len() + 1 {
            return Err(anyhow!("incorrect proof size"));
        }
        Ok(Self {
            degree_bound,
            blowup_log2,
            fold_rate,
            index: proto.index as usize,
            folds: proto
                .folds
                .iter()
                .enumerate()
                .map(|(tree, fold)| {
                    LeafProof::from_proto(
                        fold,
                        coset_size_log2(schedule.as_slice(), tree, num_trees),
                    )
                })
                .collect::<Result<_>>()?,
        })
    }
}

/// A FRI prover.
///
/// The struct contains the main Merkle tree built on the committed polynomial and the Merkle trees
/// of all folded polynomials up to and including the one where the polynomial has been folded into
/// a constant. Note that the final Merkle tree still has more than one leaf due to the low-degree
/// extension.
#[derive(Debug, Clone)]
pub struct Prover<F: Field256, H: Hasher<F>> {
    /// The degree bound of the committed polynomial, always a power of 2.
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// The number of folds per round (see [`fold_schedule`]).
    fold_rate: usize,
    /// The Merkle trees of the folding argument, one for the committed polynomial and then one for
    /// every folding round.
    trees: Vec<Tree<F, H>>,
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

    /// Commits to a polynomial of degree less than `degree_bound`, which must be a power of 2, over
    /// an extended domain `2^blowup_log2` times larger, folding it `fold_rate` times per round
    /// except for the last round, which performs the remaining folds if `log2(degree_bound)` is not
    /// a multiple of `fold_rate`.
    ///
    /// A round with fold rate `r` folds the evaluations `r` times with the challenges `alpha`,
    /// `alpha^2`, `alpha^4`, and so on, with `alpha` derived from the Merkle root of the round.
    /// That is equivalent to folding them once with arity `2^r` and challenge `alpha`.
    pub fn new(
        polynomial: Polynomial<F>,
        degree_bound: usize,
        blowup_log2: usize,
        fold_rate: usize,
    ) -> Self {
        assert!(degree_bound.is_power_of_two());
        assert!(degree_bound >= polynomial.degree_bound());
        assert!(blowup_log2 > 0);
        assert!(fold_rate > 0);

        let n = degree_bound << blowup_log2;
        assert!(n as u64 <= 1u64 << F::BaseField::S);

        let schedule = fold_schedule(degree_bound.trailing_zeros() as usize, fold_rate);
        let mut values = polynomial.shift_domain().lde2(n);
        let mut trees = Vec::with_capacity(schedule.len() + 1);
        trees.push(Tree::new(values.as_slice()));
        for num_folds in schedule {
            let mut alpha = H::challenge(*FOLD_DST, &[trees.last().unwrap().root_hash()]);
            for _ in 0..num_folds {
                values = Self::fold(values.as_slice(), alpha);
                alpha = alpha * alpha;
            }
            trees.push(Tree::new(values.as_slice()));
        }

        Self {
            degree_bound,
            blowup_log2,
            fold_rate,
            trees,
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

    /// Returns the size of the extended domain, equal to `degree_bound * blowup`.
    pub fn extended_domain_size(&self) -> usize {
        self.degree_bound << self.blowup_log2
    }

    /// Alias for [`Self::extended_domain_size`].
    pub fn size(&self) -> usize {
        self.degree_bound << self.blowup_log2
    }

    /// Returns the Merkle root hash of the committed polynomial.
    ///
    /// This is equivalent to the first root stored in the commiment returned by [`Self::commit`].
    pub fn root_hash(&self) -> H256 {
        self.trees[0].root_hash()
    }

    /// Creates the FRI commitment for the committed polynomial.
    pub fn commit(&self) -> Commitment {
        Commitment {
            roots: self.trees.iter().map(Tree::root_hash).collect(),
        }
    }

    /// Builds a FRI [`Query`] for the value at the specified index of the evaluation domain.
    ///
    /// NOTE: `index` is relative to the *extended* evaluation domain, so for example if you
    /// committed to 4 evaluations with a blowup factor of 8 the range for `index` is [0, 32).
    pub fn query(&self, index: usize) -> Query<F, H> {
        let mut n = self.degree_bound << self.blowup_log2;
        assert!(index < n);

        let schedule = fold_schedule(self.degree_bound.trailing_zeros() as usize, self.fold_rate);
        let num_trees = self.trees.len();
        let mut i = index;
        let mut folds = vec![];
        for (round, tree) in self.trees.iter().enumerate() {
            let coset_size_log2 = coset_size_log2(schedule.as_slice(), round, num_trees);
            folds.push(tree.query(i, coset_size_log2));
            n >>= coset_size_log2;
            i %= n;
        }

        assert!(folds.last().unwrap().is_constant());

        Query {
            degree_bound: self.degree_bound,
            blowup_log2: self.blowup_log2,
            fold_rate: self.fold_rate,
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

    fn test_fold_schedule_impl(num_folds: usize, fold_rate: usize, expected: &[usize]) {
        let schedule = fold_schedule(num_folds, fold_rate);
        assert_eq!(schedule, expected);
    }

    #[test]
    fn test_fold_schedule_with_rate_1() {
        test_fold_schedule_impl(0, 1, &[]);
        test_fold_schedule_impl(1, 1, &[1]);
        test_fold_schedule_impl(3, 1, &[1, 1, 1]);
    }

    #[test]
    fn test_fold_schedule_with_full_rounds() {
        test_fold_schedule_impl(4, 2, &[2, 2]);
        test_fold_schedule_impl(6, 3, &[3, 3]);
    }

    #[test]
    fn test_fold_schedule_with_partial_round() {
        test_fold_schedule_impl(7, 3, &[3, 3, 1]);
        test_fold_schedule_impl(8, 3, &[3, 3, 2]);
        test_fold_schedule_impl(14, 3, &[3, 3, 3, 3, 2]);
    }

    #[test]
    fn test_fold_schedule_with_rate_above_folds() {
        test_fold_schedule_impl(0, 3, &[]);
        test_fold_schedule_impl(2, 3, &[2]);
    }

    #[test]
    #[should_panic]
    fn test_fold_schedule_with_rate_0() {
        fold_schedule(3, 0);
    }

    fn test_coset_offset_impl(index: usize, coset_size_log2: usize, expected: usize) {
        let offset = coset_offset(index, 3, coset_size_log2);
        assert_eq!(offset, expected);
    }

    #[test]
    fn test_coset_offset() {
        test_coset_offset_impl(3, 1, 0);
        test_coset_offset_impl(7, 1, 1);
        test_coset_offset_impl(3, 2, 2);
        test_coset_offset_impl(5, 2, 1);
        test_coset_offset_impl(3, 3, 6);
    }

    fn test_coset_size_log2_impl(tree: usize, expected: usize) {
        let result = coset_size_log2(&[3, 2], tree, 3);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_coset_size_log2() {
        test_coset_size_log2_impl(0, 3);
        test_coset_size_log2_impl(1, 2);
        test_coset_size_log2_impl(2, 1);
    }

    fn test_fold_coset_impl(index: usize, coset_size_log2: usize) {
        let values = sample_values();
        let alpha = BS::from_const(5);
        let mut folded = values.clone();
        let mut challenge = alpha;
        for _ in 0..coset_size_log2 {
            folded = Prover::<BS, Sha2Hash<BS>>::fold(folded.as_slice(), challenge);
            challenge = challenge * challenge;
        }
        let base_index = index % folded.len();
        let proof = Tree::<BS, Sha2Hash<BS>>::new(values.as_slice()).query(index, coset_size_log2);
        let result = fold_coset(proof.coset().to_vec(), base_index, 8, alpha);
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

    fn test_all_coset_proofs_verify_impl(coset_size_log2: usize) {
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(sample_values().as_slice());
        let results: Vec<bool> = (0..8)
            .map(|index| {
                let proof = tree.query(index, coset_size_log2);
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
    fn test_leaf_proof_is_not_constant() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(sample_values().as_slice());
        let proof = tree.query(3, 1);
        assert!(!proof.is_constant());
    }

    #[test]
    fn test_constant_tree_proofs_are_constant() {
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(&[BS::from_const(42); 8]);
        let results: Vec<bool> = (0..8)
            .map(|index| tree.query(index, 1).is_constant())
            .collect();
        assert_eq!(results, [true; 8]);
    }

    #[test]
    fn test_almost_constant_tree_proofs_are_not_constant() {
        let mut values = [BS::from_const(42); 8];
        values[5] = BS::from_const(43);
        let tree = Tree::<BS, Sha2Hash<BS>>::new(&values);
        let results: Vec<bool> = (0..8)
            .map(|index| tree.query(index, 1).is_constant())
            .collect();
        assert_eq!(results, [false; 8]);
    }

    #[test]
    fn test_constant_tree_coset_proofs_are_constant() {
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(&[BS::from_const(42); 8]);
        let results: Vec<bool> = (0..8)
            .map(|index| tree.query(index, 2).is_constant())
            .collect();
        assert_eq!(results, [true; 8]);
    }

    #[test]
    fn test_almost_constant_tree_coset_proofs_are_not_constant() {
        let mut values = [BS::from_const(42); 8];
        values[5] = BS::from_const(43);
        let tree = Tree::<BS, Sha2Hash<BS>>::new(&values);
        let results: Vec<bool> = (0..8)
            .map(|index| tree.query(index, 2).is_constant())
            .collect();
        assert_eq!(results, [false; 8]);
    }

    #[test]
    fn test_single_leaf_proof_is_constant() {
        let tree = Tree::<BS, Sha2Hash<BS>>::new(&[BS::from_const(42)]);
        let proof = tree.query(0, 0);
        assert!(proof.is_constant());
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
    const QUERY_INDEX: usize = 3;

    /// Builds a polynomial from its coefficients.
    fn make_polynomial<F: Field256>(coefficients: &[u64]) -> Polynomial<F> {
        Polynomial::with_coefficients(coefficients.iter().copied().map(F::from).collect())
    }

    /// The polynomial committed by the prover in most of the tests.
    fn honest_polynomial() -> Polynomial<BS> {
        make_polynomial(&[12, 34, 56, 78, 90, 12, 34, 56])
    }

    /// Same as [`honest_polynomial`] but with different coefficients.
    fn foreign_polynomial() -> Polynomial<BS> {
        make_polynomial(&[90, 78, 56, 34, 12, 90, 78, 56])
    }

    #[test]
    fn test_sha2_commitment_roots() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), 8, 2, 1);
        let commitment = prover.commit();
        assert_eq!(
            commitment.roots(),
            [
                parse("0x4dbf05b032faad3a0279e11536170c9dc6c2444b65bfda0917160a556b76ab5b"),
                parse("0x3beefd65cfe8506f9b499e8397100d6d695d97af5742e70b60100f056d55307b"),
                parse("0x7d5697dea024a95e3d383fdc569fb615fa391a96f710201c970bebe336476c9b"),
                parse("0x14d99870760ebadb2f9f3b2689116c8348263851c7f2c5ff67845509618b582d"),
            ]
        );
    }

    #[test]
    fn test_keccak256_commitment_roots() {
        let prover = Prover::<BS, Keccak256Hash<BS>>::new(honest_polynomial(), 8, 2, 1);
        let commitment = prover.commit();
        assert_eq!(
            commitment.roots(),
            [
                parse("0xf62fdcf4604e703d26974310d70bca5ef93f22d316f2083167b3bc4fd4437f8f"),
                parse("0x3e708444a31aed3a38ef8d8074873506efba437be058889e79b8499095e5eefd"),
                parse("0xbb62bc0c84acc6aa08f2bac9f342426dc3e9c284630031807c3f848f35b15dcc"),
                parse("0xcdfce461c1f186f69101a452d08021e6b6ed6613637327a5173234bafd9c3018"),
            ]
        );
    }

    #[test]
    fn test_sha2_commitment_roots_with_fold_rate_2() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), 8, 2, 2);
        let commitment = prover.commit();
        assert_eq!(
            commitment.roots(),
            [
                parse("0x4dbf05b032faad3a0279e11536170c9dc6c2444b65bfda0917160a556b76ab5b"),
                parse("0x8618c5dd108b2087b698239dbffc481fc30f13bf66f858deab837d3dc4a80604"),
                parse("0xa644b9842c17a9a101d7d094b300a1456106e3ffe9bd6804440dcad9d30884cf"),
            ]
        );
    }

    #[test]
    fn test_keccak256_commitment_roots_with_fold_rate_2() {
        let prover = Prover::<BS, Keccak256Hash<BS>>::new(honest_polynomial(), 8, 2, 2);
        let commitment = prover.commit();
        assert_eq!(
            commitment.roots(),
            [
                parse("0xf62fdcf4604e703d26974310d70bca5ef93f22d316f2083167b3bc4fd4437f8f"),
                parse("0x0c2c56c72ca0eed397b43a565a38271159e7a31f2c50ddbe2ce8dc147f8bdd71"),
                parse("0x48322f2ed7ccbddc9bacf6cd849e635bf984ec94295b4b038c17372ee06ca995"),
            ]
        );
    }

    #[test]
    fn test_sha2_commitment_roots_with_fold_rate_3() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), 8, 2, 3);
        let commitment = prover.commit();
        assert_eq!(
            commitment.roots(),
            [
                parse("0x4dbf05b032faad3a0279e11536170c9dc6c2444b65bfda0917160a556b76ab5b"),
                parse("0x4deaaee88e0e8dc38e5f3fdaf4214946e0c0142191747fd45c9114c167a8393a"),
            ]
        );
    }

    #[test]
    fn test_keccak256_commitment_roots_with_fold_rate_3() {
        let prover = Prover::<BS, Keccak256Hash<BS>>::new(honest_polynomial(), 8, 2, 3);
        let commitment = prover.commit();
        assert_eq!(
            commitment.roots(),
            [
                parse("0xf62fdcf4604e703d26974310d70bca5ef93f22d316f2083167b3bc4fd4437f8f"),
                parse("0x9da7676f5d2c2209357f2c53e8d3e511c4171368116970718e518d0e49f87613"),
            ]
        );
    }

    #[test]
    fn test_prover_parameters() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        assert_eq!(prover.degree_bound(), DEGREE_BOUND);
        assert_eq!(prover.extended_domain_size(), DEGREE_BOUND << BLOWUP_LOG2);
        assert_eq!(prover.size(), DEGREE_BOUND << BLOWUP_LOG2);
    }

    #[test]
    fn test_prover_root_hash() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let root_hash = prover.root_hash();
        assert_eq!(root_hash, prover.commit().root());
    }

    fn test_queries_impl<F: Field256, H: Hasher<F>>(
        coefficients: &[u64],
        degree_bound: usize,
        blowup_log2: usize,
        fold_rate: usize,
        num_trees: usize,
    ) {
        let polynomial = make_polynomial::<F>(coefficients);
        let prover = Prover::<F, H>::new(polynomial.clone(), degree_bound, blowup_log2, fold_rate);
        let commitment = prover.commit();
        let n = degree_bound << blowup_log2;
        let queries: Vec<Query<F, H>> = (0..n).map(|index| prover.query(index)).collect();
        assert_eq!(commitment.len(), num_trees);
        for (index, query) in queries.iter().enumerate() {
            assert_eq!(query.indices(), (index, (index + n / 2) % n));
            assert_eq!(query.len(), num_trees);
            let x = query.x();
            assert_eq!(
                query.values(),
                (polynomial.evaluate(x), polynomial.evaluate(-x))
            );
            assert!(query.verify(&commitment).is_ok());
        }
    }

    /// Runs [`test_queries_impl`] with every combination of field and hash backend, with a fold
    /// rate of 1.
    fn test_queries(coefficients: &[u64], degree_bound: usize, blowup_log2: usize) {
        let num_trees = degree_bound.trailing_zeros() as usize + 1;
        test_queries_with_fold_rate(coefficients, degree_bound, blowup_log2, 1, num_trees);
    }

    /// Runs [`test_queries_impl`] with every combination of field and hash backend.
    fn test_queries_with_fold_rate(
        coefficients: &[u64],
        degree_bound: usize,
        blowup_log2: usize,
        fold_rate: usize,
        num_trees: usize,
    ) {
        test_queries_impl::<BS, Sha2Hash<BS>>(
            coefficients,
            degree_bound,
            blowup_log2,
            fold_rate,
            num_trees,
        );
        test_queries_impl::<BS, Keccak256Hash<BS>>(
            coefficients,
            degree_bound,
            blowup_log2,
            fold_rate,
            num_trees,
        );
        test_queries_impl::<GL4, Sha2Hash<GL4>>(
            coefficients,
            degree_bound,
            blowup_log2,
            fold_rate,
            num_trees,
        );
        test_queries_impl::<GL4, Keccak256Hash<GL4>>(
            coefficients,
            degree_bound,
            blowup_log2,
            fold_rate,
            num_trees,
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
        test_queries_with_fold_rate(&[12], 1, 2, 3, 1);
    }

    #[test]
    fn test_polynomial_degree_one_fold_rate_3() {
        test_queries_with_fold_rate(&[12, 34], 2, 2, 3, 2);
    }

    #[test]
    fn test_polynomial_degree_three_fold_rate_2() {
        test_queries_with_fold_rate(&[12, 34, 56, 78], 4, 1, 2, 2);
    }

    #[test]
    fn test_polynomial_degree_seven_fold_rate_2() {
        test_queries_with_fold_rate(&[12, 34, 56, 78, 90, 12, 34, 56], 8, 2, 2, 3);
    }

    #[test]
    fn test_polynomial_degree_seven_fold_rate_3() {
        test_queries_with_fold_rate(&[42, 43, 44, 45, 46, 47, 48, 49], 8, 1, 3, 2);
    }

    #[test]
    fn test_polynomial_degree_seven_fold_rate_4() {
        test_queries_with_fold_rate(&[56, 34, 12, 90, 78, 56, 34, 12], 8, 2, 4, 2);
    }

    #[test]
    fn test_polynomial_degree_fifteen_fold_rate_3() {
        test_queries_with_fold_rate(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            16,
            1,
            3,
            3,
        );
    }

    #[test]
    fn test_polynomial_below_degree_bound_fold_rate_2() {
        test_queries_with_fold_rate(&[90, 12], 8, 2, 2, 3);
    }

    #[test]
    fn test_accept_untampered_proof() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let query = prover.query(QUERY_INDEX);
        let result = query.verify(&prover.commit());
        assert!(result.is_ok());
    }

    #[test]
    fn test_reject_foreign_commitment() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let query = prover.query(QUERY_INDEX);
        let foreign = Prover::<BS, Sha2Hash<BS>>::new(foreign_polynomial(), 8, 2, 1).commit();
        let result = query.verify(&foreign);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_corrupted_root() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let mut commitment = prover.commit();
        commitment.roots[2] = H256::repeat_byte(0xAA);
        let result = prover.query(QUERY_INDEX).verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_truncated_folds() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        query.folds.truncate(3);
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_truncated_commitment() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let mut commitment = prover.commit();
        commitment.roots.truncate(3);
        let result = prover.query(QUERY_INDEX).verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_extra_folds() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        let mut donor = prover.query(QUERY_INDEX);
        query.folds.push(donor.folds.pop().unwrap());
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_folding_stopped_early() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let mut commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        commitment.roots.truncate(3);
        query.folds.truncate(3);
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_wrong_proof_height() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        let mut donor = prover.query(QUERY_INDEX);
        query.folds[0] = donor.folds.remove(1);
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_swapped_partners() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        query.folds[0].coset.swap(0, 1);
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_tampered_partner() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        query.folds[1].coset[1] += BS::ONE;
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_foreign_leaf_proof() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        let mut other = prover.query(QUERY_INDEX + 4);
        query.folds[1] = other.folds.remove(1);
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_extra_path_hash() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        query.folds[0].path.push(H256::zero());
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_out_of_range_query_index() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.index = (QUERY_INDEX + (DEGREE_BOUND << BLOWUP_LOG2)) as u64;
        let result =
            Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, FOLD_RATE, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_huge_query_index() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.index = u64::MAX;
        let result =
            Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, FOLD_RATE, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_short_merkle_path() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        query.folds[1].path.pop();
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_non_canonical_partner() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        let modulus: primitive_types::U256 = BS::MODULUS.parse().unwrap();
        proto.folds[2].coset_values[1] = modulus.to_little_endian().to_vec();
        let result =
            Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, FOLD_RATE, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_leaf_without_values() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.folds[2].coset_values.clear();
        let result =
            Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, FOLD_RATE, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_leaf_with_three_values() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        let coset_values = &mut proto.folds[0].coset_values;
        coset_values.push(coset_values[0].clone());
        let result =
            Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, FOLD_RATE, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_long_value() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.folds[1].coset_values[0].resize(33, 0);
        let result =
            Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, FOLD_RATE, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_non_canonical_value() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        let modulus: primitive_types::U256 = BS::MODULUS.parse().unwrap();
        proto.folds[1].coset_values[0] = modulus.to_little_endian().to_vec();
        let result =
            Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, FOLD_RATE, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_short_path_hash() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.folds[0].merkle_path[1].pop();
        let result =
            Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, FOLD_RATE, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_tampered_coset_leaf() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2, 3);
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        let proof = &mut query.folds[0];
        proof.coset[4] += BS::ONE;
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_swapped_coset_leaves() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2, 3);
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        query.folds[0].coset.swap(1, 2);
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_wrong_coset_size() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2, 3);
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        query.fold_rate = 2;
        let result = query.verify(&commitment);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_query_with_wrong_fold_rate() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2, 3);
        let proto = prover.query(QUERY_INDEX).to_proto();
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, 1, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_too_many_folds_for_fold_rate() {
        let prover = Prover::<BS, Sha2Hash<BS>>::new(
            honest_polynomial(),
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
        );
        let proto = prover.query(QUERY_INDEX).to_proto();
        let result = Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, 3, &proto);
        assert!(result.is_err());
    }

    #[test]
    fn test_coset_serialization() {
        let prover =
            Prover::<BS, Sha2Hash<BS>>::new(honest_polynomial(), DEGREE_BOUND, BLOWUP_LOG2, 3);
        let query = prover.query(QUERY_INDEX);
        let decoded =
            Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, 3, &query.to_proto())
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
        );
        let commitment = prover.commit();
        let query = prover.query(QUERY_INDEX);
        let decoded = Query::<BS, Sha2Hash<BS>>::from_proto(
            DEGREE_BOUND,
            BLOWUP_LOG2,
            FOLD_RATE,
            &query.to_proto(),
        )
        .unwrap();
        assert_eq!(decoded.indices(), query.indices());
        assert_eq!(decoded.len(), query.len());
        assert_eq!(decoded.values(), query.values());
        assert_eq!(decoded.x(), query.x());
        assert!(decoded.verify(&commitment).is_ok());
    }
}
