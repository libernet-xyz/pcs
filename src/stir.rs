use crate::hash::Hash;
use crate::utils;
use anyhow::{Result, anyhow};
use starkom_bluesky::Scalar;
use starkom_ff::{Field, PrimeField};
use starkom_poly;
use std::marker::PhantomData;
use std::sync::LazyLock;

type Polynomial = starkom_poly::Polynomial<Scalar>;

/// Domain separator tag used in (internal) Merkle tree hashes.
static TREE_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/stir/tree"));

/// Domain separator tag used when deriving the Fiat-Shamir challenge for STIR folding.
static FOLD_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/stir/fold"));

/// Domain separator tag used when deriving the per-round out-of-domain (OOD) challenge point.
static OOD_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/stir/ood"));

/// The modular inverse of the multiplicative generator, used to correct the coset shift in the STIR
/// fold formula: the standard formula divides by `x = g * omega^i`, so the fold coefficient is
/// `alpha * g^{-1} * omega^{-i}`.
static GENERATOR_INV: LazyLock<Scalar> =
    LazyLock::new(|| Scalar::MULTIPLICATIVE_GENERATOR.invert().unwrap());

/// Computes all Merkle hashes of a vector of values up to the root.
///
/// `n` is the number of values and must be a power of two.
///
/// The full Merkle tree is stored inline in the `values` vector as follows:
///
///   * the first `n` elements are the values of the original vector,
///   * the next `n / 2` elements are the hashes of the second-last layer of the tree,
///   * the next `n / 4` elements are the hashes of the third-last layer of the tree,
///   * ...
///   * the last stored element is the Merkle root.
///
/// It's the caller's responsibility to ensure the `values` array has at least `n * 2 - 1` slots so
/// that the full tree can be stored.
///
/// Note that the Merkle root will be at index `(n - 1) * 2`.
///
/// Note about usage: the Merkle trees we use in this module have scalar *vectors* for leaves, not
/// just scalars.
pub fn merklify<H: Hash<Scalar>>(mut values: &mut [Scalar], mut n: usize) {
    assert!(n.is_power_of_two());
    while n > 1 {
        let m = n / 2;
        for j in 0..m {
            values[n + j] = H::hash_raw(*TREE_DST, values[j * 2], values[j * 2 + 1]);
        }
        values = &mut values[n..];
        n = m;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment {
    /// Merkle roots of all committed oracles: index 0 is the initial oracle, indices 1..=M are the
    /// M successive folds. Has M+1 entries in total where M is the number of folding rounds.
    roots: Vec<Scalar>,
    /// Prover's responses to the per-round OOD challenges, one per folding round. Entry i is the
    /// evaluation of the (i+1)-th oracle at the OOD challenge derived after committing that oracle.
    ood_evaluations: Vec<Scalar>,
}

impl Commitment {
    /// Returns the number of folding rounds M. The total number of committed oracles is M+1.
    pub fn num_rounds(&self) -> usize {
        self.ood_evaluations.len()
    }

    /// Returns the Merkle roots of all committed oracles. Has `num_rounds() + 1` entries.
    pub fn roots(&self) -> &[Scalar] {
        &self.roots
    }

    /// Returns the prover's OOD evaluations, one per folding round.
    pub fn ood_evaluations(&self) -> &[Scalar] {
        &self.ood_evaluations
    }

    /// Returns the Merkle root of the initial (pre-fold) oracle.
    pub fn root(&self) -> Scalar {
        self.roots[0]
    }
}

/// A Merkle proof.
///
/// NOTE: this object only stores the sister hashes of the Merkle path and the opened leaf value, it
/// doesn't store the lookup key and the root hash anywhere because those pieces of information are
/// reconstructed separately during the verification of a whole `Query`. In particular, all root
/// hashes are stored in the `Commitment`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeafProof<H: Hash<Scalar>> {
    leaf: Scalar,
    path: Vec<Scalar>,
    _data: PhantomData<H>,
}

impl<H: Hash<Scalar>> LeafProof<H> {
    /// Returns a reference to the leaf values (one for every committed polynomial).
    pub fn leaf(&self) -> Scalar {
        self.leaf
    }

    /// Checks the leaf of this proof against the provided slice.
    ///
    /// The two must match or an error is returned.
    pub fn check_leaf(&self, expected: Scalar) -> Result<()> {
        if self.leaf != expected {
            return Err(anyhow!("leaf value mismatch"));
        }
        Ok(())
    }

    /// Returns the length of the Merkle path, corresponding to the height of the tree minus 1 (the
    /// root hash is not included in this count).
    pub fn len(&self) -> usize {
        self.path.len()
    }

    /// Verifies the proof against the given root hash.
    pub fn verify(&self, mut index: usize, root_hash: Scalar) -> Result<()> {
        let mut hash = self.leaf;
        for sibling in &self.path {
            hash = if index & 1 != 0 {
                H::hash_raw(*TREE_DST, *sibling, hash)
            } else {
                H::hash_raw(*TREE_DST, hash, *sibling)
            };
            index >>= 1;
        }
        if index != 0 {
            return Err(anyhow!("invalid index"));
        }
        if hash != root_hash {
            return Err(anyhow!(
                "root hash mismatch (got {}, want {})",
                hash,
                root_hash
            ));
        }
        Ok(())
    }

    /// Indicates whether or not the committed polynomial is constant.
    ///
    /// This is used in low degree testing to check when the folding process collapses to a degree-0
    /// polynomial.
    pub fn is_constant(&self) -> bool {
        let mut hash = self.leaf;
        for &sibling in &self.path {
            if sibling != hash {
                return false;
            }
            hash = H::hash_raw(*TREE_DST, hash, hash);
        }
        true
    }
}

/// A Merkle tree of polynomial evaluations.
///
/// The tree has N leaf in total, with N being the size of the extended domain, and each leaf has K
/// polynomial evaluations, with K being the number of committed polynomials.
///
/// The internal nodes are single hashes.
#[derive(Debug, Clone)]
pub struct Tree<H: Hash<Scalar>> {
    /// The nodes of the tree stored as a flat array, using the layout generated by `merklify`. The
    /// array has 2*N-1 elements in total, with N being the number of committed evaluations (always
    /// a power of 2). The root hash of the tree is located at index (N-1)*2.
    data: Vec<Scalar>,
    _data: PhantomData<H>,
}

impl<H: Hash<Scalar>> Tree<H> {
    /// Constructs a Merkle tree from an array of polynomial evaluations.
    ///
    /// REQUIRES: the number of provided `values` must be a power of two.
    pub fn new(mut values: Vec<Scalar>) -> Self {
        let n = values.len();
        assert!(n.is_power_of_two());
        values.resize(n * 2 - 1, Scalar::ZERO);
        merklify::<H>(values.as_mut_slice(), n);
        Self {
            data: values,
            _data: Default::default(),
        }
    }

    /// Returns the number of leaves in the tree, corresponding to the size of the evaluation domain
    /// (always a power of 2).
    pub fn num_leaves(&self) -> usize {
        (self.data.len() + 1) / 2
    }

    /// Returns the root hash of the Merkle tree.
    pub fn root_hash(&self) -> Scalar {
        let n = self.num_leaves();
        self.data[(n - 1) * 2]
    }

    /// Returns the i-th leaf.
    ///
    /// REQUIRES: `index` must be strictly less than [`Self::num_leaves`], although this is not
    /// strictly enforced.
    pub fn leaf(&self, index: usize) -> Scalar {
        self.data[index]
    }

    /// Returns a Merkle proof for the leaf at `index`.
    pub fn query(&self, mut index: usize) -> LeafProof<H> {
        let mut n = self.num_leaves();
        assert!(n.is_power_of_two());
        assert!(index < n);
        let leaf = self.data[index];
        let mut path = Vec::with_capacity(n.trailing_zeros() as usize);
        let mut data = self.data.as_slice();
        while n > 1 {
            path.push(data[index ^ 1]);
            data = &data[n..];
            n /= 2;
            index >>= 1;
        }
        LeafProof {
            leaf,
            path,
            _data: Default::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Query<H: Hash<Scalar>> {
    /// Size of the initial evaluation domain |L_0| = N.
    n: usize,
    /// Query index into the initial oracle f_0. The fold-partner index is (index + n/2) % n.
    index: usize,
    /// Fold chain: one pair of `LeafProof` per oracle, for M+1 oracles total. Pair r proves the
    /// values of f_r at the main query position and its fold-partner in L_r.
    folds: Vec<(LeafProof<H>, LeafProof<H>)>,
    /// Shift proofs: one list per folding round, for M rounds total. List r contains t-1 proofs
    /// opening f_{r+1} at the shift challenge positions z_{r,j} derived for round r.
    shift_proofs: Vec<Vec<LeafProof<H>>>,
    _data: PhantomData<H>,
}

impl<H: Hash<Scalar>> Query<H> {
    /// Returns the main query index and its fold-partner index into L_0.
    pub fn indices(&self) -> (usize, usize) {
        (self.index, (self.index + self.n / 2) % self.n)
    }

    /// Returns the domain element x at the main query index.
    ///
    /// Used by the DEEP layer for the algebraic check q(x) * (x - z) = combined(x) - v.
    pub fn x(&self) -> Scalar {
        Polynomial::coset_element2(self.index, self.n)
    }

    /// Returns the values of f_0 at the main query position and its fold-partner.
    pub fn values(&self) -> (Scalar, Scalar) {
        (self.folds[0].0.leaf(), self.folds[0].1.leaf())
    }

    /// Returns the number of committed oracles, equal to M+1.
    pub fn len(&self) -> usize {
        self.folds.len()
    }

    /// Returns the shift proof values for the given folding round r (0-indexed).
    ///
    /// Each element of the returned vector is the value of f_{r+1} at one shift position.
    pub fn shift_values(&self, round: usize) -> Vec<Scalar> {
        self.shift_proofs[round]
            .iter()
            .map(|proof| proof.leaf())
            .collect()
    }

    /// Verifies this proof against the given commitment.
    ///
    /// NOTE: for low-degree testing you also need to check that `len()` returns
    /// `log2(degree_bound) + 1`. This function only verifies the fold chain consistency and the
    /// OOD / shift consistency; it does not check the number of rounds.
    pub fn verify(&self, commitment: &Commitment) -> Result<()> {
        assert!(self.n.is_power_of_two());
        assert!(self.index < self.n);

        // TODO
        todo!()
    }
}

/// A STIR prover.
#[derive(Debug, Clone)]
pub struct Prover<H: Hash<Scalar>> {
    /// The degree bound of the committed polynomial (always a power of 2).
    degree_bound: usize,
    /// The base-2 logarithm of the blowup factor.
    blowup_log2: usize,
    /// Merkle trees for all committed oracles: trees[0] is f_0, trees[r] is the r-th fold.
    /// Used to answer both fold-chain queries and shift queries.
    trees: Vec<Tree<H>>,
    /// Pre-computed OOD evaluations y_r = f_r(s_r), one per folding round.
    /// Computed during `new()` by evaluating each folded polynomial at the Fiat-Shamir OOD
    /// challenge derived from its tree root. Stored here so that `commit()` can package them into
    /// the `Commitment` without needing to re-evaluate.
    ood_evaluations: Vec<Scalar>,
}

impl<H: Hash<Scalar>> Prover<H> {
    pub fn new(polynomial: Polynomial, degree_bound: usize, blowup_log2: usize) -> Self {
        assert!(degree_bound.is_power_of_two());
        assert!(degree_bound >= polynomial.degree_bound());

        let n = degree_bound << blowup_log2;
        assert!(n as u64 <= 1u64 << Scalar::S);

        let main_tree = Tree::<H>::new(polynomial.shifted_lde2(n));

        // TODO
        todo!()
    }

    // TODO
}
