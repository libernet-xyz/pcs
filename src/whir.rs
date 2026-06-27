use crate::hash::Hash;
use crate::utils;
use anyhow::{Result, anyhow};
use starkom_bluesky::Scalar;
use starkom_ff::Field;
use starkom_poly;
use std::marker::PhantomData;
use std::sync::LazyLock;

type Polynomial = starkom_poly::Polynomial<Scalar>;

/// Domain separator tag used in (internal) Merkle tree hashes.
static TREE_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/tree"));

/// Domain separator tag used when deriving the Fiat-Shamir challenge for STIR folding.
static FOLD_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/fold"));

/// Domain separator tag used when deriving the per-round out-of-domain (OOD) challenge point.
static OOD_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/ood"));

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
pub fn merklify<H: Hash<Scalar>>(mut values: &mut [Scalar], mut n: usize) {
    assert!(n.is_power_of_two());
    while n > 1 {
        let m = n / 2;
        for j in 0..m {
            values[n + j] = H::hash_two(*TREE_DST, values[j * 2], values[j * 2 + 1]);
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
    /// Prover's responses to the per-round OOD challenges, one per folding round. Entry r is the
    /// evaluation of oracle f_r at the OOD point s_r = H(OOD_DST, roots[r]).
    ood_values: Vec<Scalar>,
}

impl Commitment {
    /// Returns the number of folding rounds M. The total number of committed oracles is M+1.
    pub fn num_rounds(&self) -> usize {
        self.ood_values.len()
    }

    /// Returns the Merkle roots of all committed oracles. Has `num_rounds() + 1` entries.
    pub fn roots(&self) -> &[Scalar] {
        &self.roots
    }

    /// Returns the prover's OOD evaluations, one per folding round.
    pub fn ood_values(&self) -> &[Scalar] {
        &self.ood_values
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
    /// Returns the leaf value.
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
                H::hash_two(*TREE_DST, *sibling, hash)
            } else {
                H::hash_two(*TREE_DST, hash, *sibling)
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
            hash = H::hash_two(*TREE_DST, hash, hash);
        }
        true
    }
}

/// A Merkle tree of polynomial evaluations.
///
/// The tree has N leaves in total, with N being the size of the extended domain, and 2*N-1 nodes in
/// total. Internal nodes are hashes of the respective child pairs.
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

// TODO
