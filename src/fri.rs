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

/// A Merkle proof for a single FRI leaf.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LeafProof<F: Field256, H: Hasher<F>> {
    /// The opened evaluation.
    value: F,
    /// The Merkle path from the leaf up to, but excluding, the root. The first element is the
    /// sibling leaf, that is another evaluation encoded with [`utils::encode_scalar`], and the
    /// following elements are the siblings of the ancestors of the leaf.
    path: Vec<H256>,
    _data: PhantomData<H>,
}

impl<F: Field256, H: Hasher<F>> LeafProof<F, H> {
    /// Returns the length of the Merkle path, which equals the height of the tree minus 1 (the root
    /// hash is not included in this count).
    fn len(&self) -> usize {
        self.path.len()
    }

    /// Verifies the proof against the given root hash.
    fn verify(&self, index: usize, root_hash: H256) -> Result<()> {
        merkle::verify_path::<H>(
            utils::encode_scalar(self.value),
            self.path.as_slice(),
            index,
            root_hash,
        )
    }

    /// Returns true iff the proof witnesses a tree whose leaves are all equal, i.e. the evaluations
    /// of a constant polynomial.
    ///
    /// That's the case if and only if every sibling equals the hash computed so far, starting from
    /// the leaf itself.
    fn is_constant(&self) -> bool {
        let mut hash = utils::encode_scalar(self.value);
        for &sibling in &self.path {
            if sibling != hash {
                return false;
            }
            hash = H::hash_binary(hash, hash);
        }
        true
    }

    /// Serializes this proof to a [`MerkleProof`](`proto::MerkleProof`) protobuf with a single leaf
    /// value.
    fn to_proto(&self) -> proto::MerkleProof {
        proto::MerkleProof {
            leaf_values: vec![self.value.to_le_bytes().to_vec()],
            merkle_path: self
                .path
                .iter()
                .map(|hash| hash.as_bytes().to_vec())
                .collect(),
        }
    }

    /// Deserializes a proof from a [`MerkleProof`](`proto::MerkleProof`) protobuf.
    ///
    /// Returns an error if the leaf doesn't have exactly one value, if the value is not a canonical
    /// field element, or if any hash is invalid.
    fn from_proto(proto: &proto::MerkleProof) -> Result<Self> {
        Ok(Self {
            value: match proto.leaf_values.as_slice() {
                [value] => utils::load_scalar(value.as_slice()),
                values => Err(anyhow!(
                    "invalid FRI leaf size (got {}, want 1)",
                    values.len()
                )),
            }?,
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
        let mut data = Vec::with_capacity(n * 2 - 1);
        data.extend(values.iter().map(|&value| utils::encode_scalar(value)));
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

    /// Returns the evaluation stored in the i-th leaf.
    fn leaf(&self, index: usize) -> F {
        utils::decode_scalar(self.data[index]).unwrap()
    }

    /// Returns a Merkle proof for the leaf at `index`.
    fn query(&self, index: usize) -> LeafProof<F, H> {
        LeafProof {
            value: self.leaf(index),
            path: merkle::merkle_path(self.data.as_slice(), self.num_leaves(), index),
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
    /// rounds equals the log2 of the degree bound of the original polynomial. For example, if the
    /// user commits a degree<8 polynomial encoding 8 values, `len()` will return
    /// log2(8)+1 = 3+1 = 4 regardless of the blowup factor.
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
    /// The index of the element we're opening (the partner index is inferred automatically).
    index: usize,
    /// Proves a pair of "partner" values at each folding round with one [`LeafProof`] pair for
    /// every round. The pair at `folds[0]` proves the opened values.
    folds: Vec<(LeafProof<F, H>, LeafProof<F, H>)>,
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
        (self.folds[0].0.value, self.folds[0].1.value)
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

        if self.index >= n {
            return Err(anyhow!(
                "invalid query index {}: the domain size is {n}",
                self.index
            ));
        }

        let mut k = n.trailing_zeros() as usize;

        let num_folds = self.folds.len();
        if num_folds > self.degree_bound.trailing_zeros() as usize + 1 {
            return Err(anyhow!("incorrect proof size"));
        }
        if commitment.len() != num_folds {
            return Err(anyhow!("wrong number of folding rounds"));
        }

        let mut index = self.index;
        let mut pos = self.folds[0].0.value;
        let mut step = Polynomial::<F>::domain_element2(1, n).invert_unwrap();

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

            if left.value != pos {
                return Err(anyhow!("leaf value mismatch"));
            }
            left.verify(index, root_hash)?;
            right.verify((index + n / 2) % n, root_hash)?;

            let omega_inv = step.pow_small(index);
            n /= 2;
            k -= 1;
            index %= n;

            let neg = right.value;
            let alpha = H::challenge(*FOLD_DST, &[root_hash]);
            pos = (pos + neg + alpha * omega_inv * (pos - neg)) * F::BaseField::TWO_INV;
            step = step.square();
        }

        let (left, right) = self.folds.last().unwrap();
        if !left.is_constant() || !right.is_constant() {
            return Err(anyhow!("the final folded polynomial is not constant"));
        }

        Ok(())
    }

    /// Serializes this query to a [`Query`](`proto::Query`) protobuf.
    pub fn to_proto(&self) -> proto::Query {
        proto::Query {
            index: self.index as u64,
            folds: self
                .folds
                .iter()
                .map(|(left, right)| proto::query::Fold {
                    left: Some(left.to_proto()),
                    right: Some(right.to_proto()),
                })
                .collect(),
        }
    }

    /// Deserializes a query from a [`Query`](`proto::Query`) protobuf.
    ///
    /// REQUIRES: `degree_bound` must be a power of 2, `blowup_log2` must be greater than 0, and the
    /// size of the extended domain must fit within the 2-adicity of `F::BaseField`.
    ///
    /// Every opened value must be a canonical field element, otherwise an error is returned.
    ///
    /// NOTE: this method does not validate the proof, it only deserializes it. The caller must
    /// invoke [`Self::verify`] separately.
    pub fn from_proto(
        degree_bound: usize,
        blowup_log2: usize,
        proto: &proto::Query,
    ) -> Result<Self> {
        assert!(degree_bound.is_power_of_two());
        assert!(blowup_log2 > 0);
        assert!(degree_bound.trailing_zeros() as usize + blowup_log2 <= F::BaseField::S);
        if proto.folds.is_empty() {
            return Err(anyhow!("the FRI query has no folds"));
        }
        Ok(Self {
            degree_bound,
            blowup_log2,
            index: proto.index as usize,
            folds: proto
                .folds
                .iter()
                .map(|fold| match (&fold.left, &fold.right) {
                    (Some(left), Some(right)) => {
                        Ok((LeafProof::from_proto(left)?, LeafProof::from_proto(right)?))
                    }
                    _ => Err(anyhow!("missing FRI folds")),
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
    /// The Merkle trees of the folding argument, one for the committed polynomial and then one for
    /// every folding round.
    trees: Vec<Tree<F, H>>,
}

impl<F: Field256, H: Hasher<F>> Prover<F, H> {
    /// Performs one FRI folding round, returning the evaluations of the folded polynomial (half as
    /// many as `values`).
    ///
    /// `root_hash` is the Merkle root of `values`, from which the folding challenge is derived.
    fn fold(values: &[F], root_hash: H256) -> Vec<F> {
        let n = values.len();
        assert!(n.is_power_of_two());

        let alpha = H::challenge(*FOLD_DST, &[root_hash]);

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

    pub fn new(polynomial: Polynomial<F>, degree_bound: usize, blowup_log2: usize) -> Self {
        assert!(degree_bound.is_power_of_two());
        assert!(degree_bound >= polynomial.degree_bound());
        assert!(blowup_log2 > 0);

        let n = degree_bound << blowup_log2;
        assert!(n as u64 <= 1u64 << F::BaseField::S);

        let num_folds = degree_bound.trailing_zeros() as usize;
        let mut values = polynomial.shift_domain().lde2(n);
        let mut trees = Vec::with_capacity(num_folds + 1);
        trees.push(Tree::new(values.as_slice()));
        for _ in 0..num_folds {
            values = Self::fold(values.as_slice(), trees.last().unwrap().root_hash());
            trees.push(Tree::new(values.as_slice()));
        }

        Self {
            degree_bound,
            blowup_log2,
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
            assert!(left.is_constant());
            assert!(right.is_constant());
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

    fn parse<V: std::str::FromStr<Err: std::fmt::Debug>>(s: &'static str) -> V {
        s.parse().unwrap()
    }

    fn keccak(left: &[u8], right: &[u8]) -> H256 {
        use sha3::Digest;
        let mut hasher = sha3::Keccak256::new();
        hasher.update(left);
        hasher.update(right);
        H256::from_slice(hasher.finalize().as_slice())
    }

    #[test]
    fn test_tree_layout() {
        let values: Vec<BS> = (0..8u64).map(|i| BS::from_const(i * i + 7)).collect();
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(values.as_slice());
        assert_eq!(tree.data.len(), 15);
        assert_eq!(tree.num_leaves(), 8);
        // The leaves are the values themselves, as 32-byte big-endian integers.
        assert_eq!(tree.data[2], H256::from_low_u64_be(11));
        let mut level: Vec<H256> = values
            .iter()
            .map(|value| H256::from_slice(&value.to_u256().to_big_endian()))
            .collect();
        assert_eq!(&tree.data[0..8], level.as_slice());
        // Recompute the internal nodes the way an EVM verifier does:
        // `keccak256(abi.encodePacked(left, right))`.
        let mut offset = 8;
        while level.len() > 1 {
            level = level
                .chunks(2)
                .map(|pair| keccak(pair[0].as_bytes(), pair[1].as_bytes()))
                .collect();
            assert_eq!(&tree.data[offset..(offset + level.len())], level.as_slice());
            offset += level.len();
        }
        assert_eq!(tree.root_hash(), level[0]);
    }

    #[test]
    fn test_tree_leaves() {
        let values: Vec<BS> = (0..8u64).map(|i| BS::from_const(i * i + 7)).collect();
        let tree = Tree::<BS, Sha2Hash<BS>>::new(values.as_slice());
        for (index, &value) in values.iter().enumerate() {
            assert_eq!(tree.leaf(index), value);
        }
        let single = Tree::<BS, Sha2Hash<BS>>::new(&values[0..1]);
        assert_eq!(single.num_leaves(), 1);
        assert_eq!(single.data.len(), 1);
        assert_eq!(single.leaf(0), values[0]);
        assert_eq!(single.root_hash(), single.data[0]);
    }

    #[test]
    fn test_tree_query() {
        let values: Vec<BS> = (0..8u64).map(|i| BS::from_const(i * i + 7)).collect();
        let tree = Tree::<BS, Sha2Hash<BS>>::new(values.as_slice());
        let proof = tree.query(5);
        assert_eq!(proof.value, values[5]);
        assert_eq!(proof.path, [tree.data[4], tree.data[8 + 3], tree.data[12]]);
        assert_eq!(proof.len(), 3);
        // The first element of the Merkle path is the sibling leaf, so it's a value too.
        assert_eq!(
            utils::decode_scalar::<BS>(proof.path[0]).unwrap(),
            values[4]
        );
        assert!(proof.verify(5, tree.root_hash()).is_ok());
        assert!(proof.verify(4, tree.root_hash()).is_err());
        assert!(!proof.is_constant());
        for index in 0..8 {
            assert!(tree.query(index).verify(index, tree.root_hash()).is_ok());
        }
        let single = Tree::<BS, Sha2Hash<BS>>::new(&values[0..1]);
        let proof = single.query(0);
        assert_eq!(proof.value, values[0]);
        assert_eq!(proof.len(), 0);
        assert!(proof.verify(0, single.root_hash()).is_ok());
        assert!(proof.is_constant());
    }

    #[test]
    fn test_almost_constant_tree_query() {
        let mut values = [BS::from_const(42); 8];
        values[5] = BS::from_const(43);
        let tree = Tree::<BS, Sha2Hash<BS>>::new(&values);
        for index in 0..8 {
            assert!(!tree.query(index).is_constant());
        }
    }

    #[test]
    fn test_constant_tree_query() {
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(&[BS::from_const(42); 8]);
        for index in 0..8 {
            let proof = tree.query(index);
            assert!(proof.is_constant());
            assert!(proof.verify(index, tree.root_hash()).is_ok());
        }
    }

    #[test]
    fn test_leaf_proof_serialization() {
        let values: Vec<BS> = (0..8u64).map(|i| BS::from_const(i * i + 7)).collect();
        let tree = Tree::<BS, Keccak256Hash<BS>>::new(values.as_slice());
        let proof = tree.query(3);
        let proto = proof.to_proto();
        assert_eq!(proto.leaf_values, [values[3].to_le_bytes().to_vec()]);
        assert_eq!(proto.merkle_path.len(), 3);
        assert_eq!(LeafProof::from_proto(&proto).unwrap(), proof);
    }

    #[test]
    fn test_commitment_roots() {
        // Pinned roots: changing them changes the protocol.
        let polynomial = Polynomial::<BS>::with_coefficients(
            [12, 34, 56, 78, 90, 12, 34, 56]
                .into_iter()
                .map(BS::from_const)
                .collect(),
        );
        assert_eq!(
            Prover::<BS, Sha2Hash<BS>>::new(polynomial.clone(), 8, 2)
                .commit()
                .roots(),
            [
                parse("0xf243320c355c40b383d412c7e1964911d8b0158e9074dcb4c1d3f8b3d47fbe40"),
                parse("0x6ec4f9a41a3d96df76e26c26acebf9b59832c42b5705595f8d9a32bea19e2592"),
                parse("0xc0da902ce501419871c92091cf5a1dc3a18345334e3dc17c5a60ce4b3b45aaeb"),
                parse("0x51f2f631f31206f27d63c68c70623c1e2bb93acd17d125b551dac8444b290eb7"),
            ]
        );
        assert_eq!(
            Prover::<BS, Keccak256Hash<BS>>::new(polynomial, 8, 2)
                .commit()
                .roots(),
            [
                parse("0x9b4906d8f66d186d59009303c5caef59460de8a06b17c5bf09082778709f5201"),
                parse("0xe4bc0886fb6edf122ba603607a60ac1c4feb73bf2ffebf306e229a490a55d796"),
                parse("0x87b4a1f5b10087b68593b197901f40f857ad5777892fb6cc58fce34d8bc08e15"),
                parse("0x7bb73139b685ac249fc83972820d410a8be43dab72764445e5679c9a182f39f4"),
            ]
        );
    }

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
        polynomial: &Polynomial<F>,
        degree_bound: usize,
        blowup_log2: usize,
    ) {
        let prover = Prover::<F, H>::new(polynomial.clone(), degree_bound, blowup_log2);
        assert_eq!(prover.degree_bound(), degree_bound);
        let n = degree_bound << blowup_log2;
        assert_eq!(prover.extended_domain_size(), n);
        let commitment = prover.commit();
        for i in 0..n {
            let query = prover.query(i);
            assert_eq!(query.indices(), (i, (i + n / 2) % n));
            assert_eq!(query.len(), degree_bound.trailing_zeros() as usize + 1);
            let x = query.x();
            assert_eq!(
                query.values(),
                (polynomial.evaluate(x), polynomial.evaluate(-x))
            );
            assert!(query.verify(&commitment).is_ok());
        }
    }

    fn test_prover(coefficients: Vec<u64>, degree_bound: usize) {
        let bluesky_polynomial = Polynomial::<BS>::with_coefficients(
            coefficients.iter().copied().map(BS::from_const).collect(),
        );
        let goldilocks_polynomial = Polynomial::<GL4>::with_coefficients(
            coefficients.iter().copied().map(GL4::from_const).collect(),
        );
        for blowup_log2 in 1..=3 {
            test_prover_impl::<BS, Sha2Hash<BS>>(&bluesky_polynomial, degree_bound, blowup_log2);
            test_prover_impl::<GL4, Sha2Hash<GL4>>(
                &goldilocks_polynomial,
                degree_bound,
                blowup_log2,
            );
            test_prover_impl::<BS, Keccak256Hash<BS>>(
                &bluesky_polynomial,
                degree_bound,
                blowup_log2,
            );
            test_prover_impl::<GL4, Keccak256Hash<GL4>>(
                &goldilocks_polynomial,
                degree_bound,
                blowup_log2,
            );
        }
    }

    #[test]
    fn test_constant_polynomial() {
        test_prover(vec![12], 1);
        test_prover(vec![34], 1);
    }

    #[test]
    fn test_polynomial_degree_one() {
        test_prover(vec![12, 34], 2);
        test_prover(vec![56, 78], 2);
    }

    #[test]
    fn test_polynomial_degree_three() {
        test_prover(vec![12, 34, 56, 78], 4);
        test_prover(vec![42, 43, 44, 45], 4);
    }

    #[test]
    fn test_polynomial_degree_seven() {
        test_prover(vec![12, 34, 56, 78, 90, 12, 34], 8);
        test_prover(vec![42, 43, 44, 45, 46, 47, 48], 8);
    }

    #[test]
    fn test_polynomial_below_degree_bound() {
        test_prover(vec![12, 34], 8);
    }

    const DEGREE_BOUND: usize = 8;
    const BLOWUP_LOG2: usize = 2;
    const QUERY_INDEX: usize = 3;

    fn make_prover(coefficients: [u64; 8]) -> Prover<BS, Sha2Hash<BS>> {
        Prover::new(
            Polynomial::with_coefficients(coefficients.into_iter().map(BS::from_const).collect()),
            DEGREE_BOUND,
            BLOWUP_LOG2,
        )
    }

    fn honest_prover() -> Prover<BS, Sha2Hash<BS>> {
        make_prover([12, 34, 56, 78, 90, 12, 34, 56])
    }

    fn assert_rejected(result: Result<()>, expected: &str) {
        let error = result
            .expect_err("the tampered proof was accepted")
            .to_string();
        assert!(error.contains(expected), "unexpected error: {error}");
    }

    #[test]
    fn test_accept_untampered_proof() {
        let prover = honest_prover();
        assert!(prover.query(QUERY_INDEX).verify(&prover.commit()).is_ok());
    }

    #[test]
    fn test_reject_foreign_commitment() {
        let query = honest_prover().query(QUERY_INDEX);
        let foreign = make_prover([90, 78, 56, 34, 12, 90, 78, 56]).commit();
        assert_rejected(query.verify(&foreign), "root hash mismatch");
    }

    #[test]
    fn test_reject_corrupted_root() {
        let prover = honest_prover();
        let mut commitment = prover.commit();
        commitment.roots[2] = H256::repeat_byte(0xAA);
        assert_rejected(
            prover.query(QUERY_INDEX).verify(&commitment),
            "root hash mismatch",
        );
    }

    #[test]
    fn test_reject_truncated_folds() {
        let prover = honest_prover();
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        query.folds.truncate(3);
        assert_rejected(query.verify(&commitment), "wrong number of folding rounds");
    }

    #[test]
    fn test_reject_truncated_commitment() {
        let prover = honest_prover();
        let mut commitment = prover.commit();
        commitment.roots.truncate(3);
        assert_rejected(
            prover.query(QUERY_INDEX).verify(&commitment),
            "wrong number of folding rounds",
        );
    }

    #[test]
    fn test_reject_extra_folds() {
        let prover = honest_prover();
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        let mut donor = prover.query(QUERY_INDEX);
        query.folds.push(donor.folds.pop().unwrap());
        assert_rejected(query.verify(&commitment), "incorrect proof size");
    }

    #[test]
    fn test_reject_folding_stopped_early() {
        let prover = honest_prover();
        let mut commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        commitment.roots.truncate(3);
        query.folds.truncate(3);
        assert_rejected(query.verify(&commitment), "not constant");
    }

    #[test]
    fn test_reject_wrong_left_proof_height() {
        let prover = honest_prover();
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        let mut donor = prover.query(QUERY_INDEX);
        query.folds[0].0 = donor.folds.remove(1).0;
        assert_rejected(
            query.verify(&commitment),
            "invalid left-hand side Merkle proof height",
        );
    }

    #[test]
    fn test_reject_wrong_right_proof_height() {
        let prover = honest_prover();
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        let mut donor = prover.query(QUERY_INDEX);
        query.folds[0].1 = donor.folds.remove(1).1;
        assert_rejected(
            query.verify(&commitment),
            "invalid right-hand side Merkle proof height",
        );
    }

    #[test]
    fn test_reject_swapped_partners() {
        let prover = honest_prover();
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        let pair = &mut query.folds[0];
        std::mem::swap(&mut pair.0, &mut pair.1);
        assert_rejected(query.verify(&commitment), "root hash mismatch");
    }

    #[test]
    fn test_reject_foreign_leaf_proof() {
        let prover = honest_prover();
        let commitment = prover.commit();
        let mut query = prover.query(QUERY_INDEX);
        let mut other = prover.query(QUERY_INDEX + 4);
        query.folds[1].0 = other.folds.remove(1).0;
        assert_rejected(query.verify(&commitment), "leaf value mismatch");
    }

    fn assert_query_rejected(proto: &proto::Query, expected: &str) {
        assert_rejected(
            Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, proto).map(|_| ()),
            expected,
        );
    }

    #[test]
    fn test_reject_missing_partner_proof() {
        let mut proto = honest_prover().query(QUERY_INDEX).to_proto();
        proto.folds[0].right = None;
        assert_query_rejected(&proto, "missing FRI folds");
    }

    #[test]
    fn test_reject_missing_proof() {
        let mut proto = honest_prover().query(QUERY_INDEX).to_proto();
        proto.folds[2].left = None;
        assert_query_rejected(&proto, "missing FRI folds");
    }

    #[test]
    fn test_reject_leaf_without_values() {
        let mut proto = honest_prover().query(QUERY_INDEX).to_proto();
        proto.folds[2].right.as_mut().unwrap().leaf_values.clear();
        assert_query_rejected(&proto, "invalid FRI leaf size");
    }

    #[test]
    fn test_reject_leaf_with_two_values() {
        let mut proto = honest_prover().query(QUERY_INDEX).to_proto();
        let leaf_values = &mut proto.folds[0].left.as_mut().unwrap().leaf_values;
        leaf_values.push(leaf_values[0].clone());
        assert_query_rejected(&proto, "invalid FRI leaf size");
    }

    #[test]
    fn test_reject_long_value() {
        let mut proto = honest_prover().query(QUERY_INDEX).to_proto();
        proto.folds[1].left.as_mut().unwrap().leaf_values[0].resize(33, 0);
        assert_query_rejected(&proto, "invalid scalar length");
    }

    #[test]
    fn test_reject_non_canonical_value() {
        let mut proto = honest_prover().query(QUERY_INDEX).to_proto();
        let modulus: primitive_types::U256 = BS::MODULUS.parse().unwrap();
        proto.folds[1].right.as_mut().unwrap().leaf_values[0] = modulus.to_little_endian().to_vec();
        assert_query_rejected(&proto, "invalid scalar");
    }

    #[test]
    fn test_reject_short_path_hash() {
        let mut proto = honest_prover().query(QUERY_INDEX).to_proto();
        proto.folds[0].left.as_mut().unwrap().merkle_path[1].pop();
        assert_query_rejected(&proto, "invalid hash length");
    }

    #[test]
    fn test_reject_extra_path_hash() {
        let prover = honest_prover();
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.folds[0]
            .left
            .as_mut()
            .unwrap()
            .merkle_path
            .push(vec![0; 32]);
        let query =
            Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, &proto).unwrap();
        assert_rejected(
            query.verify(&prover.commit()),
            "invalid left-hand side Merkle proof height",
        );
    }

    #[test]
    fn test_reject_out_of_range_query_index() {
        let prover = honest_prover();
        let commitment = prover.commit();
        let mut proto = prover.query(QUERY_INDEX).to_proto();
        proto.index = (DEGREE_BOUND << BLOWUP_LOG2) as u64;
        let query =
            Query::<BS, Sha2Hash<BS>>::from_proto(DEGREE_BOUND, BLOWUP_LOG2, &proto).unwrap();
        assert_rejected(query.verify(&commitment), "invalid query index");
    }

    #[test]
    fn test_serialization() {
        let polynomial = Polynomial::<BS>::encode2(
            [12u64, 34, 56, 78]
                .into_iter()
                .map(|value| value.into())
                .collect(),
        );
        let prover = Prover::<BS, Sha2Hash<BS>>::new(polynomial, 4, 2);
        let commitment = prover.commit();
        let query = prover.query(2);
        let values = query.values();
        let proto = query.to_proto();
        let query = Query::<BS, Sha2Hash<BS>>::from_proto(4, 2, &proto).unwrap();
        assert_eq!(query.indices(), (2, 10));
        assert_eq!(query.len(), 3);
        assert_eq!(query.values(), values);
        assert!(query.verify(&commitment).is_ok());
    }
}
