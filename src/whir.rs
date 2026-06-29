use crate::hash::Hash;
use crate::utils;
use anyhow::{Result, anyhow};
use starkom_bluesky::Scalar;
use std::marker::PhantomData;
use std::sync::LazyLock;

/// Domain separator tag used in (internal) Merkle tree hashes.
static TREE_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/whir/tree"));

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
fn merklify<H: Hash<Scalar>>(mut values: &mut [Scalar], mut n: usize) {
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

/// A Merkle proof.
///
/// NOTE: this object only stores the opened value and the sister hashes of the Merkle path, it
/// doesn't store the lookup key or the root hash anywhere because those pieces of information are
/// reconstructed separately during the verification of a whole WHIR [`Proof`]. In particular, all
/// root hashes are stored in the [`Commitment`].
#[derive(Debug, Clone, PartialEq, Eq)]
struct LeafProof<H: Hash<Scalar>> {
    value: Scalar,
    path: Vec<Scalar>,
    _data: PhantomData<H>,
}

impl<H: Hash<Scalar>> LeafProof<H> {
    /// Returns the leaf value.
    fn leaf(&self) -> Scalar {
        self.value
    }

    /// Returns the length of the Merkle path, corresponding to the height of the tree minus 1 (the
    /// root hash is not included in this count).
    fn len(&self) -> usize {
        self.path.len()
    }

    /// Verifies the proof against the given root hash.
    fn verify(&self, mut index: usize, root_hash: Scalar) -> Result<()> {
        let mut hash = self.value;
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

    /// Indicates whether or not the committed polynomials are constant.
    ///
    /// This is used in low degree testing to check when the folding process collapses to degree-0
    /// polynomials.
    fn is_constant(&self) -> bool {
        let mut hash = self.value;
        for &sibling in &self.path {
            if sibling != hash {
                return false;
            }
            hash = H::hash_two(*TREE_DST, hash, hash);
        }
        true
    }
}

#[derive(Debug, Default, Clone)]
struct Tree<H: Hash<Scalar>> {
    /// The nodes of the tree. There are 2*N-1 elements in this array, with N = number of leaves.
    /// The nodes of the bottom layer are the leaves, that is the committed polynomial evaluations.
    data: Vec<Scalar>,
    _data: PhantomData<H>,
}

impl<H: Hash<Scalar>> Tree<H> {
    /// Returns the number of leaves in the tree, corresponding to the size of the evaluation domain
    /// (always a power of 2).
    fn num_leaves(&self) -> usize {
        (self.data.len() + 1) / 2
    }

    /// Returns the root hash of the Merkle tree.
    fn root_hash(&self) -> Scalar {
        let n = self.num_leaves();
        self.data[(n - 1) * 2]
    }

    /// Returns a reference to the i-th leaf.
    ///
    /// Note that the leaf contains k elements, one for every committed polynomial.
    fn leaf(&self, index: usize) -> Scalar {
        self.data[index]
    }

    /// Returns a Merkle proof for the leaf at `index`.
    fn query(&self, mut index: usize) -> LeafProof<H> {
        let mut n = self.num_leaves();
        assert!(n.is_power_of_two());
        assert!(index < n);
        let value = self.data[index];
        let mut path = Vec::with_capacity(n.trailing_zeros() as usize);
        let mut data = self.data.as_slice();
        while n > 1 {
            path.push(data[index ^ 1]);
            data = &data[n..];
            n /= 2;
            index >>= 1;
        }
        LeafProof {
            value,
            path,
            _data: Default::default(),
        }
    }
}
