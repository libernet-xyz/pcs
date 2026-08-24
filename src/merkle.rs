use crate::hash::{Hasher, MerkleHasher};
use anyhow::{Result, anyhow};
use primitive_types::{H256, U256, U512};
use sha2::Digest;
use starkom_ff::{Field, Field256};
use std::any::TypeId;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::{LazyLock, Mutex};

fn make_dst(modulus: U512) -> U256 {
    let mut hasher = sha3::Sha3_512::new();
    hasher.update(b"starkom/merkle/leaf");
    let hash: U512 = U512::from_little_endian(hasher.finalize().as_slice());
    let value = hash % modulus;
    U256::from_little_endian(&value.to_little_endian()[0..32])
}

fn get_dst<F: Field>() -> F {
    static DST_CACHE: LazyLock<Mutex<BTreeMap<TypeId, U256>>> = LazyLock::new(|| Mutex::default());
    let value = {
        let mut cache = DST_CACHE.lock().unwrap();
        *cache
            .entry(TypeId::of::<F>())
            .or_insert_with(|| make_dst(F::MODULUS.parse().unwrap()))
    };
    F::try_from_le_bytes(&value.to_little_endian()[0..(F::LEN)]).unwrap()
}

/// Hashes a leaf of a Merkle tree.
///
/// Our Merkle trees have vectors of values as leaves (there's one element for every committed
/// polynomial so that we can commit multiple polynomials into the same tree), so the input `values`
/// parameter is a slice of scalar values.
fn hash_leaf<F: Field, G: Field256 + From<F>, H: Hasher<G>>(values: &[F]) -> H256 {
    H::hash(
        std::iter::once(get_dst::<F>())
            .chain(std::iter::once(F::try_from(values.len()).unwrap()))
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

impl<F: Field, G: Field256 + From<F>, H: Hasher<G>> Proof<F, G, H> {
    /// Returns a reference to the leaf values (one for every committed polynomial).
    pub(crate) fn leaf(&self) -> &[F] {
        self.leaf.as_slice()
    }

    /// Checks the leaf of this proof against the provided slice.
    ///
    /// The two must match or an error is returned.
    pub(crate) fn check_leaf(&self, expected: &[F]) -> Result<()> {
        if expected.len() != self.leaf.len()
            || self
                .leaf
                .iter()
                .zip(expected.iter())
                .any(|(&value1, &value2)| value1 != value2)
        {
            return Err(anyhow!("leaf value mismatch"));
        }
        Ok(())
    }

    /// Returns the length of the Merkle path, corresponding to the height of the tree minus 1 (the
    /// root hash is not included in this count).
    pub(crate) fn len(&self) -> usize {
        self.path.len()
    }

    /// Verifies the proof against the given root hash.
    pub(crate) fn verify(&self, mut index: usize, root_hash: H256) -> Result<()> {
        let mut hash = hash_leaf::<F, G, H>(self.leaf.as_slice());
        for &sibling in &self.path {
            hash = if index & 1 != 0 {
                H::hash_binary(sibling, hash)
            } else {
                H::hash_binary(hash, sibling)
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

    /// Returns a boolean indicating whether or not the committed polynomials are constant.
    ///
    /// An error is returned if one or more hashes in the proof are out of range; see [`Hash<H256>`]
    /// for details.
    ///
    /// This function is used in low degree testing to check when the folding process collapses to
    /// degree-0 polynomials.
    ///
    /// Note that some polynomials may collapse earlier than others, and this function returns false
    /// if one or more haven't collapsed yet. So it returns true if and only if all have collapsed.
    pub(crate) fn is_constant(&self) -> Result<bool> {
        let mut hash = hash_leaf::<F, G, H>(self.leaf.as_slice());
        for &sibling in &self.path {
            if sibling != hash {
                return Ok(false);
            }
            hash = H::hash_binary(hash, hash);
        }
        Ok(true)
    }
}

// TODO

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::{Keccak256Hash, Sha2Hash};
    use starkom_bluesky::Scalar as BS;
    use starkom_goldilocks::{GL, GL4};
    use std::fmt::Debug;
    use std::str::FromStr;

    fn from_const<F: Field>(value: u16) -> F {
        F::from(value)
    }

    fn parse<V: FromStr<Err: Debug>>(s: &'static str) -> V {
        s.parse().unwrap()
    }

    #[test]
    fn test_hash_leaf_sha2_bluesky() {
        assert_eq!(
            hash_leaf::<BS, BS, Sha2Hash<BS>>(&[from_const(12)]),
            parse("0x0bd187bc3deea1ef6c2a9ae254cf4e493f1dbbda32c79a662fc1d8437ab7e7c6")
        );
        assert_eq!(
            hash_leaf::<BS, BS, Sha2Hash<BS>>(&[from_const(34), from_const(56)]),
            parse("0xbcd7235ceb553ca6682fd2df813650cad056a20fd6c76e1e04ae9f6a248c6c02")
        );
        assert_eq!(
            hash_leaf::<BS, BS, Sha2Hash<BS>>(&[from_const(78), from_const(90), from_const(12)]),
            parse("0xeb8b9e0099332552744b9111d5a478dc61b231407d6c776586a50a3fc4513ca4")
        );
    }

    #[test]
    fn test_hash_leaf_sha2_goldilocks() {
        assert_eq!(
            hash_leaf::<GL, GL4, Sha2Hash<GL4>>(&[from_const(12)]),
            parse("0xa76cc2edc0687213f2f6ae0352cdae9bb437a5589d51d863458e3f94fdcf1a1f")
        );
        assert_eq!(
            hash_leaf::<GL, GL4, Sha2Hash<GL4>>(&[from_const(34), from_const(56)]),
            parse("0x2197f79dd32a7b9e974f6724b585178b7350e6fc916db209bc0f8a4d68f2e362")
        );
        assert_eq!(
            hash_leaf::<GL, GL4, Sha2Hash<GL4>>(&[from_const(78), from_const(90), from_const(12)]),
            parse("0xa7045f9cf7303a4cd91fd168624cddc5aa063a26d37c8c554d3afc9400669661")
        );
    }

    #[test]
    fn test_hash_leaf_keccak256_bluesky() {
        assert_eq!(
            hash_leaf::<BS, BS, Keccak256Hash<BS>>(&[from_const(12)]),
            parse("0x5e70242e081756f445b5f3048611464568002e68800238dcfef718504de01782")
        );
        assert_eq!(
            hash_leaf::<BS, BS, Keccak256Hash<BS>>(&[from_const(34), from_const(56)]),
            parse("0x6cbf9a3af9793caffe8c509dfaccf0198e833cafa8eb83a0a7d57ce35a97dbd1")
        );
        assert_eq!(
            hash_leaf::<BS, BS, Keccak256Hash<BS>>(&[
                from_const(78),
                from_const(90),
                from_const(12)
            ]),
            parse("0x05ce2ee03a570540c8a67a9b5a419dc7813922da6a39dff3c293806ed0f88fbb")
        );
    }

    #[test]
    fn test_hash_leaf_keccak256_goldilocks() {
        assert_eq!(
            hash_leaf::<GL, GL4, Keccak256Hash<GL4>>(&[from_const(12)]),
            parse("0xf8766a3ad2ed6d5bd215b3ed0531bde06e7a31e218a17d494a99fa8ba255ba2c")
        );
        assert_eq!(
            hash_leaf::<GL, GL4, Keccak256Hash<GL4>>(&[from_const(34), from_const(56)]),
            parse("0x8b5b64c41c3dd48c62fbf06e5f20641aba3b0b69b97b4a6e82215303a033aa77")
        );
        assert_eq!(
            hash_leaf::<GL, GL4, Keccak256Hash<GL4>>(&[
                from_const(78),
                from_const(90),
                from_const(12)
            ]),
            parse("0x51c62c7131bfcd24c7ab2a1a3d1dda4fcec722cd78a833205370db18f8bd60d3")
        );
    }

    #[test]
    fn test_merklify_one_sha2() {
        let mut hashes = vec![parse(
            "0x1fe9e33b7ff790473e13eb6384d61d6abb32d2e50b30a382c9767b5347f5846b",
        )];
        merklify::<Sha2Hash<BS>>(&mut hashes, 1);
        assert_eq!(
            hashes,
            vec![parse(
                "0x1fe9e33b7ff790473e13eb6384d61d6abb32d2e50b30a382c9767b5347f5846b"
            )]
        );
    }

    #[test]
    fn test_merklify_one_keccak256() {
        let mut hashes = vec![parse(
            "0x1fe9e33b7ff790473e13eb6384d61d6abb32d2e50b30a382c9767b5347f5846b",
        )];
        merklify::<Keccak256Hash<BS>>(&mut hashes, 1);
        assert_eq!(
            hashes,
            vec![parse(
                "0x1fe9e33b7ff790473e13eb6384d61d6abb32d2e50b30a382c9767b5347f5846b"
            )]
        );
    }

    #[test]
    fn test_merklify_two_sha2() {
        let mut hashes = vec![
            parse("0x0cbb1061e55efef40fae3c8e34d301c9940889146f816edd475052fc45caf060"),
            parse("0x2d8257d72c4b6e6bfa3a21a22c053d02d1d52b4b5e6097acfd0ed45f827d6ba4"),
        ];
        hashes.resize(3, H256::default());
        merklify::<Sha2Hash<BS>>(&mut hashes, 2);
        assert_eq!(
            hashes,
            vec![
                parse("0x0cbb1061e55efef40fae3c8e34d301c9940889146f816edd475052fc45caf060"),
                parse("0x2d8257d72c4b6e6bfa3a21a22c053d02d1d52b4b5e6097acfd0ed45f827d6ba4"),
                parse("0x3af448b6612ac9931b7413ecca5d209187cd808ae2d3647099aa99fada16c955")
            ]
        );
    }

    #[test]
    fn test_merklify_two_keccak256() {
        let mut hashes = vec![
            parse("0x0cbb1061e55efef40fae3c8e34d301c9940889146f816edd475052fc45caf060"),
            parse("0x2d8257d72c4b6e6bfa3a21a22c053d02d1d52b4b5e6097acfd0ed45f827d6ba4"),
        ];
        hashes.resize(3, H256::default());
        merklify::<Keccak256Hash<BS>>(&mut hashes, 2);
        assert_eq!(
            hashes,
            vec![
                parse("0x0cbb1061e55efef40fae3c8e34d301c9940889146f816edd475052fc45caf060"),
                parse("0x2d8257d72c4b6e6bfa3a21a22c053d02d1d52b4b5e6097acfd0ed45f827d6ba4"),
                parse("0x6e70b19bd160e181bb367ab730eababc6e3673977f7fdce38934ee197bbcb568")
            ]
        );
    }

    #[test]
    fn test_merklify_four_sha2() {
        let mut hashes = vec![
            parse("0x034acf2cced8e9744784ec9c3c626fa83f6f8ddd83faed9d5caa5ad72c91eb1c"),
            parse("0x39dce95a2271a57f999981eef6917f3df8ad25d116c98d809a3b7da5d54805a3"),
            parse("0x435246f701f1483adcb7037fa64b3fa027c41e13a52d1e6c502e9b71dea3ca81"),
            parse("0x7952ed41c4062f594b53b8c548874b41a7a7f05593d7a5a313ed82c2fe1c62d7"),
        ];
        hashes.resize(7, H256::default());
        merklify::<Sha2Hash<BS>>(&mut hashes, 4);
        assert_eq!(
            hashes,
            vec![
                parse("0x034acf2cced8e9744784ec9c3c626fa83f6f8ddd83faed9d5caa5ad72c91eb1c"),
                parse("0x39dce95a2271a57f999981eef6917f3df8ad25d116c98d809a3b7da5d54805a3"),
                parse("0x435246f701f1483adcb7037fa64b3fa027c41e13a52d1e6c502e9b71dea3ca81"),
                parse("0x7952ed41c4062f594b53b8c548874b41a7a7f05593d7a5a313ed82c2fe1c62d7"),
                parse("0x3d139c65859969e86db444ccc0d36ffd0456b3f92d3dfb34fa10c8c07f5ada06"),
                parse("0xdc3dd313db4b53f2720fcfe6373d286a9c40911546e1c434fa24ab650a75b586"),
                parse("0xed171c6bd6876b93f4fc8914e0b2a8db53d7204048ad9fdff6e7127c077ca072"),
            ]
        );
    }

    #[test]
    fn test_merklify_four_keccak256() {
        let mut hashes = vec![
            parse("0x034acf2cced8e9744784ec9c3c626fa83f6f8ddd83faed9d5caa5ad72c91eb1c"),
            parse("0x39dce95a2271a57f999981eef6917f3df8ad25d116c98d809a3b7da5d54805a3"),
            parse("0x435246f701f1483adcb7037fa64b3fa027c41e13a52d1e6c502e9b71dea3ca81"),
            parse("0x7952ed41c4062f594b53b8c548874b41a7a7f05593d7a5a313ed82c2fe1c62d7"),
        ];
        hashes.resize(7, H256::default());
        merklify::<Keccak256Hash<BS>>(&mut hashes, 4);
        assert_eq!(
            hashes,
            vec![
                parse("0x034acf2cced8e9744784ec9c3c626fa83f6f8ddd83faed9d5caa5ad72c91eb1c"),
                parse("0x39dce95a2271a57f999981eef6917f3df8ad25d116c98d809a3b7da5d54805a3"),
                parse("0x435246f701f1483adcb7037fa64b3fa027c41e13a52d1e6c502e9b71dea3ca81"),
                parse("0x7952ed41c4062f594b53b8c548874b41a7a7f05593d7a5a313ed82c2fe1c62d7"),
                parse("0x8a2dbe0367d89a172a0eb470026f5bb85ddd542e0d93885318c5d6b8a3764f28"),
                parse("0x31354bf6d6ca07dbde80395609890a558ff35587ddc2b5c72ccde2b4e15d114f"),
                parse("0xbd82b7f4f83de57b73e00aa0924a5fc4dfaf959bd252aa3fda8abdaf87713dc0"),
            ]
        );
    }

    // TODO
}
