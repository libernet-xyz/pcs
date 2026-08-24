use crate::hash::{Hasher, MerkleHasher};
use primitive_types::H256;
use starkom_ff::{Field, Field256};
use std::marker::PhantomData;

/// Hashes a leaf of a Merkle tree.
///
/// Our Merkle trees have vectors of values as leaves (there's one element for every committed
/// polynomial so that we can commit multiple polynomials into the same tree), so the input `values`
/// parameter is a slice of scalar values.
fn hash_leaf<F: Field + From<usize>, G: Field256 + From<F>, H: Hasher<G>>(values: &[F]) -> H256 {
    H::hash(
        std::iter::once(F::ZERO) // TODO: DST
            .chain(std::iter::once(F::from(values.len())))
            .chain(values.iter().cloned())
            .map(Into::<G>::into),
    )
}

/// Computes the internal nodes of a binary Merkle tree up to the root.
///
/// `n` is the number of leaves and must be a power of two. The specified array already contains `n`
/// leaf hashes plus `n-1` empty slots; `merklify` fills the latter with the internal node hashes.
///
/// The full Merkle tree is stored inline in the `hashes` array as follows:
///
///   * the first `n` elements are the caller-provided leaf hashes,
///   * the next `n / 2` elements are the hashes of the second-last layer of the tree,
///   * the next `n / 4` elements are the hashes of the third-last layer of the tree,
///   * ...
///   * the last stored element is the Merkle root.
///
/// It's the caller's responsibility to ensure that `hashes` has at least `n * 2 - 1` slots so that
/// the full tree can be stored.
///
/// Note that the Merkle root will be at index `(n - 1) * 2`.
pub(crate) fn merklify<H: MerkleHasher>(mut hashes: &mut [H256], mut n: usize) {
    assert!(n.is_power_of_two());
    while n > 1 {
        let m = n / 2;
        for j in 0..m {
            hashes[n + j] = H::hash_binary(hashes[j * 2], hashes[j * 2 + 1]);
        }
        hashes = &mut hashes[n..];
        n = m;
    }
}

/// A Merkle proof.
///
/// A FRI `Query` uses several of these: two from the main Merkle tree and two for each folding
/// round.
///
/// NOTE: this object only stores the sister hashes of the Merkle path and the opened leaf values,
/// it doesn't store the lookup key and the root hash anywhere because those pieces of information
/// are reconstructed separately during the verification of a whole `Query`. In particular, all root
/// hashes are stored in the `Commitment`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Proof<F: Field, G: Field256 + From<F>, H: Hasher<G>> {
    leaf: Vec<F>,
    path: Vec<H256>,
    _data: PhantomData<(G, H)>,
}
