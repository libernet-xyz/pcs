use crate::starkom::proto::pcs::v12 as proto;
use crate::utils;
use anyhow::{Result, anyhow};
use primitive_types::{H256, U256, U512};
use sha2::Digest;
use starkom_ff::{Field, Field256};
use std::fmt::Debug;
use std::marker::PhantomData;

utils::prost_name!(
    proto::Sha2HashBackend,
    "Sha2HashBackend",
    "starkom.proto.pcs.v12"
);

utils::prost_name!(
    proto::KeccakHashBackend,
    "KeccakHashBackend",
    "starkom.proto.pcs.v12"
);

/// Describes a hash backend for hashing internal Merkle tree nodes.
pub trait MerkleHasher {
    /// Protobuf message type used to identify the hash backend used in [proofs](`proto::Proof`) and
    /// [commitments](`proto::Commitment`).
    type ProtoTag: prost::Message + prost::Name + Default;

    /// Hashes a binary Merkle tree node.
    fn hash_binary(left: H256, right: H256) -> H256;

    /// Hashes a ternary Merkle tree node.
    fn hash_ternary(children: [H256; 3]) -> H256;

    /// Loads a hash from its serialized byte form.
    ///
    /// For the SHA2 and Keccak256 backends this function simply defers to [`H256::from_slice`], but
    /// algebraic hash backends will also need to check that the encoded value is in range. For
    /// example, a Poseidon backend over the BlueSky field will check that the value is strictly
    /// less than `0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000001`.
    fn load_hash(bytes: &[u8]) -> Result<H256>;
}

/// Describes a hash backend for our proof system.
///
/// The functionality is a superset of that provided by [`MerkleHasher`].
pub trait Hasher<F: Field256>: MerkleHasher {
    /// Hashes the provided base field elements.
    ///
    /// NOTE: this is a raw hash, it doesn't automatically prepend any DSTs or element count; the
    /// caller is responsible for those when necessary.
    fn hash(inputs: impl IntoIterator<Item = F::BaseField>) -> H256;

    /// Hashes the provided transcript.
    fn hash_transcript(dst: H256, transcript: &[H256]) -> H256;

    /// Generates a ~256-bit Fiat-Shamir challenge from a transcript.
    ///
    /// The provided implementation hashes the transcript with [`Self::hash_transcript`] and
    /// converts the hash to a field element. To avoid distribution bias, the conversion reduces 512
    /// pseudorandom bits modulo the field order. The most significant 256 bits are the hash of the
    /// transcript with `dst`, and the least significant 256 bits are the hash of the former with
    /// `dst + 1` (as a single-element transcript). That way the transcript, which can be long, is
    /// hashed only once.
    fn challenge(dst: H256, transcript: &[H256]) -> F {
        let dst_hi = U256::from_big_endian(dst.as_bytes());
        let dst_lo = dst_hi + U256::one();
        let hi = Self::hash_transcript(H256::from_slice(&dst_hi.to_big_endian()), transcript);
        let lo = Self::hash_transcript(H256::from_slice(&dst_lo.to_big_endian()), &[hi]);
        let mut bytes = [0u8; 64];
        bytes[0..32].copy_from_slice(hi.as_bytes());
        bytes[32..].copy_from_slice(lo.as_bytes());
        let hash = U512::from_big_endian(&bytes);
        let modulus: U512 = F::MODULUS.parse().unwrap();
        let challenge = hash % modulus;
        F::try_from_be_bytes(&challenge.to_big_endian()[32..]).unwrap()
    }
}

mod internal {
    use super::*;

    pub trait LowLevelHash: Debug + Default {
        type ProtoTag: prost::Message + prost::Name + Default;
        fn update(&mut self, bytes: &[u8]);
        fn finalize(self) -> H256;
    }

    #[derive(Debug, Default)]
    pub struct LowLevelSha2Hash {
        hasher: sha2::Sha256,
    }

    impl LowLevelHash for LowLevelSha2Hash {
        type ProtoTag = proto::Sha2HashBackend;

        fn update(&mut self, bytes: &[u8]) {
            self.hasher.update(bytes);
        }

        fn finalize(self) -> H256 {
            H256::from_slice(self.hasher.finalize().as_slice())
        }
    }

    #[derive(Debug, Default)]
    pub struct LowLevelKeccak256Hash {
        hasher: sha3::Keccak256,
    }

    impl LowLevelHash for LowLevelKeccak256Hash {
        type ProtoTag = proto::KeccakHashBackend;

        fn update(&mut self, bytes: &[u8]) {
            self.hasher.update(bytes);
        }

        fn finalize(self) -> H256 {
            H256::from_slice(self.hasher.finalize().as_slice())
        }
    }

    pub struct HasherImpl<H: LowLevelHash, F: Field256> {
        _data: PhantomData<(H, F)>,
    }

    impl<H: LowLevelHash, F: Field256> Debug for HasherImpl<H, F> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("HasherImpl").finish()
        }
    }

    impl<H: LowLevelHash, F: Field256> Default for HasherImpl<H, F> {
        fn default() -> Self {
            Self { _data: PhantomData }
        }
    }

    impl<H: LowLevelHash, F: Field256> Copy for HasherImpl<H, F> {}

    impl<H: LowLevelHash, F: Field256> Clone for HasherImpl<H, F> {
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<H: LowLevelHash, F: Field256> PartialEq for HasherImpl<H, F> {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }

    impl<H: LowLevelHash, F: Field256> Eq for HasherImpl<H, F> {}

    impl<H: LowLevelHash, F: Field256> PartialOrd for HasherImpl<H, F> {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl<H: LowLevelHash, F: Field256> Ord for HasherImpl<H, F> {
        fn cmp(&self, _other: &Self) -> std::cmp::Ordering {
            std::cmp::Ordering::Equal
        }
    }

    impl<H: LowLevelHash, F: Field256> MerkleHasher for HasherImpl<H, F> {
        type ProtoTag = H::ProtoTag;

        fn hash_binary(left: H256, right: H256) -> H256 {
            let mut hasher = H::default();
            hasher.update(left.as_bytes());
            hasher.update(right.as_bytes());
            hasher.finalize()
        }

        fn hash_ternary(children: [H256; 3]) -> H256 {
            let mut hasher = H::default();
            hasher.update(children[0].as_bytes());
            hasher.update(children[1].as_bytes());
            hasher.update(children[2].as_bytes());
            hasher.finalize()
        }

        fn load_hash(bytes: &[u8]) -> Result<H256> {
            if bytes.len() != 32 {
                return Err(anyhow!(
                    "invalid hash length (expected 32, got {})",
                    bytes.len()
                ));
            }
            Ok(H256::from_slice(bytes))
        }
    }

    impl<H: LowLevelHash, F: Field256> Hasher<F> for HasherImpl<H, F> {
        fn hash(inputs: impl IntoIterator<Item = F::BaseField>) -> H256 {
            let length = F::BaseField::LEN;
            let mut hasher = H::default();
            for input in inputs {
                let bytes = input.to_u256().to_big_endian();
                hasher.update(&bytes[(32 - length)..]);
            }
            hasher.finalize()
        }

        fn hash_transcript(dst: H256, transcript: &[H256]) -> H256 {
            let mut hasher = H::default();
            hasher.update(dst.as_bytes());
            hasher.update(&U256::from(transcript.len() as u64).to_big_endian());
            for element in transcript {
                hasher.update(element.as_bytes());
            }
            hasher.finalize()
        }
    }
}

/// SHA2 hash backend.
pub type Sha2Hash<F> = internal::HasherImpl<internal::LowLevelSha2Hash, F>;

/// Keccak-256 hash backend.
pub type Keccak256Hash<F> = internal::HasherImpl<internal::LowLevelKeccak256Hash, F>;

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Name;
    use starkom_bluesky::Scalar as BS;
    use starkom_goldilocks::GL4;
    use starkom_koalabear::KB8;
    use std::str::FromStr;
    use std::sync::LazyLock;

    static DST: LazyLock<H256> = LazyLock::new(|| {
        let mut hasher = sha3::Sha3_256::new();
        hasher.update(b"starkom/pcs/test");
        H256::from_slice(hasher.finalize().as_slice())
    });

    /// Converts a small integer into an element of the base field of `F`.
    fn base_from_u64<F: Field256>(value: u64) -> F::BaseField {
        F::BaseField::try_from(value as usize).unwrap()
    }

    fn parse<V: FromStr<Err: Debug>>(s: &'static str) -> V {
        s.parse().unwrap()
    }

    #[test]
    fn test_hash_backend_type_urls() {
        assert_eq!(
            proto::Sha2HashBackend::type_url(),
            "type.starkom.io/starkom.proto.pcs.v12.Sha2HashBackend"
        );
        assert_eq!(
            proto::KeccakHashBackend::type_url(),
            "type.starkom.io/starkom.proto.pcs.v12.KeccakHashBackend"
        );
    }

    fn test_sha2_hash_impl<F: Field256>(expected: [&'static str; 3]) {
        assert_eq!(
            Sha2Hash::<F>::hash([base_from_u64::<F>(12)]),
            parse(expected[0])
        );
        assert_eq!(
            Sha2Hash::<F>::hash([base_from_u64::<F>(34), base_from_u64::<F>(56)]),
            parse(expected[1])
        );
        assert_eq!(
            Sha2Hash::<F>::hash([
                base_from_u64::<F>(78),
                base_from_u64::<F>(90),
                base_from_u64::<F>(12),
            ]),
            parse(expected[2])
        );
    }

    #[test]
    fn test_sha2_hash_bluesky() {
        test_sha2_hash_impl::<BS>([
            "0xa82872b96246dac512ddf0515f5da862a92ecebebcb92537b6e3e73199694c45",
            "0xbdcf24876d0b8979976f54ea123b70112da34a5cb4dc381646a3321f0817a5e8",
            "0xce080de3e477a622d8b9711eb599aec6fe9ddda88c47c1175bff2aaadc43c3b4",
        ]);
    }

    #[test]
    fn test_sha2_hash_goldilocks() {
        test_sha2_hash_impl::<GL4>([
            "0xd27a8d9a8d38c7a37c922450a7a1961f138abfa25f5da3df2b972820715fa5ae",
            "0xa7978a35218774c33d029c976f494b3e220f9ed48dd640e0426fb6cdf84873c2",
            "0x668a3637e141226dead35825f67cd915432d1dd3711b0dbb47018e2a6bcda74b",
        ]);
    }

    #[test]
    fn test_sha2_hash_koalabear() {
        test_sha2_hash_impl::<KB8>([
            "0xc22ceee66b8ab104483c83053173b33f992dd4fcb457284c60ef9d699a1c7059",
            "0xfb606c01b68bbcd68240cdded4699855adb5077bb20e833188f1ccca89aa0441",
            "0x88d3ee772a0f65a517336500ef3e3e5fcba85c54ea54e9ee7430e60a156d60a3",
        ]);
    }

    fn test_keccak256_hash_impl<F: Field256>(expected: [&'static str; 3]) {
        assert_eq!(
            Keccak256Hash::<F>::hash([base_from_u64::<F>(12)]),
            parse(expected[0])
        );
        assert_eq!(
            Keccak256Hash::<F>::hash([base_from_u64::<F>(34), base_from_u64::<F>(56)]),
            parse(expected[1])
        );
        assert_eq!(
            Keccak256Hash::<F>::hash([
                base_from_u64::<F>(78),
                base_from_u64::<F>(90),
                base_from_u64::<F>(12),
            ]),
            parse(expected[2])
        );
    }

    #[test]
    fn test_keccak256_hash_bluesky() {
        test_keccak256_hash_impl::<BS>([
            "0xdf6966c971051c3d54ec59162606531493a51404a002842f56009d7e5cf4a8c7",
            "0x72700d0d963d58363ea77095ddabb7ed1a429a1fe618c8ace040205d52391bb9",
            "0x452eb69ea7065787fb9f51a2894ff6a6ff50ae842aaa68b3ad40ed3580117a8a",
        ]);
    }

    #[test]
    fn test_keccak256_hash_goldilocks() {
        test_keccak256_hash_impl::<GL4>([
            "0x626899464858d84aa8a337b4fd305d199dad99f1c20a1129ab98bc233c50c6da",
            "0x8adb2ad7853e6ad763a60b9d7f6990e6e9aac20763774108bed54d2bf97d7efd",
            "0x9a618a2e56f77252d293ab77d88244f941ea0e9f8715ad79a57956d2103d700d",
        ]);
    }

    #[test]
    fn test_keccak256_hash_koalabear() {
        test_keccak256_hash_impl::<KB8>([
            "0x9a5fa99a11a3338d080ebdeaf8b8ced6034f49e60bce1414f85f23b2da0bd83a",
            "0xe636d0cf62b6c803a036491934663ecc32f1d0aecea706d85be4c4088d3c8c70",
            "0x08e548e958310c1a2b82559a4f81ee527dd848142615707e81a0adc661b80b55",
        ]);
    }

    fn test_sha2_hash_binary_impl<F: Field256>() {
        assert_eq!(
            Sha2Hash::<F>::hash_binary(
                parse("0x2c8ea861b3ec34715e3d50053d001e9d9929a774954e93ffa1f1784f94339592"),
                parse("0x57c43629d9178a54c611df8562260309677c1a10a3bd3ca4c9997acbc5f908bb")
            ),
            parse("0xff888724c7e8b1359d4cb4014d8ddcba61b938e5becbfd4f0af454dee20b83be")
        );
        assert_eq!(
            Sha2Hash::<F>::hash_binary(
                parse("0x229045f7089bd1cb6527f1edd7d94a7e91f2dcf607553e2a89fe24d72b2b013c"),
                parse("0x021905a563d6bf385cb3f2eb4bc69d7750812bcb29542ee4bcfcb0fd0e1bc9b8")
            ),
            parse("0x61944633a56b8421a4b1b53555bfa58283989e76fc21b7e6c1634730969f81f4")
        );
    }

    #[test]
    fn test_sha2_hash_binary() {
        test_sha2_hash_binary_impl::<BS>();
        test_sha2_hash_binary_impl::<GL4>();
    }

    fn test_keccak256_hash_binary_impl<F: Field256>() {
        assert_eq!(
            Keccak256Hash::<F>::hash_binary(
                parse("0x2c8ea861b3ec34715e3d50053d001e9d9929a774954e93ffa1f1784f94339592"),
                parse("0x57c43629d9178a54c611df8562260309677c1a10a3bd3ca4c9997acbc5f908bb")
            ),
            parse("0x0887805e67bb6ece5489f58f1863dffec93f52583cd050eca57c33cbd33abb09")
        );
        assert_eq!(
            Keccak256Hash::<F>::hash_binary(
                parse("0x229045f7089bd1cb6527f1edd7d94a7e91f2dcf607553e2a89fe24d72b2b013c"),
                parse("0x021905a563d6bf385cb3f2eb4bc69d7750812bcb29542ee4bcfcb0fd0e1bc9b8")
            ),
            parse("0x3cecf97526033fb79c366cde597b0f66b7774ec26388b71fd591d7275c81dfbe")
        );
    }

    #[test]
    fn test_keccak256_hash_binary() {
        test_keccak256_hash_binary_impl::<BS>();
        test_keccak256_hash_binary_impl::<GL4>();
    }

    fn test_sha2_hash_ternary_impl<F: Field256>() {
        assert_eq!(
            Sha2Hash::<F>::hash_ternary([
                parse("0x7148a9186844710414337e2a454e082f5bc45a823187142bf8519d48153bb3d7"),
                parse("0x6b3cdb310e6ba5e984fb0ff3722a1968d6b22ce0cdad6b1683167ccdb2b3a75a"),
                parse("0x239f4553fa744a1b2a49af348cf9412fe105c3bc0accb23030a40bfe59d7fa2b"),
            ]),
            parse("0xea2a6de526a3b32de298db973d5cbc604d816eeae526beef5a089bd2ae7ca3eb")
        );
        assert_eq!(
            Sha2Hash::<F>::hash_ternary([
                parse("0x73b6d673e9b67cbca809416002ed33ddeb704d58ce0101a6555531e63426383b"),
                parse("0x5b68318ce7baa6d9a0b6e67cee81b1a730dc7f68c34b7f806fd70ec4614f2096"),
                parse("0x5829d1d7b55ee4345afa69b4acf18d722f6962d0e61447a2711b5851b77d1c2b"),
            ]),
            parse("0xcb8497f1c9898641ec978cfe347ff1ccc43cc9a15f9e741f15e18dc0660d50bf")
        );
    }

    #[test]
    fn test_sha2_hash_ternary() {
        test_sha2_hash_ternary_impl::<BS>();
        test_sha2_hash_ternary_impl::<GL4>();
    }

    fn test_keccak256_hash_ternary_impl<F: Field256>() {
        assert_eq!(
            Keccak256Hash::<F>::hash_ternary([
                parse("0x7148a9186844710414337e2a454e082f5bc45a823187142bf8519d48153bb3d7"),
                parse("0x6b3cdb310e6ba5e984fb0ff3722a1968d6b22ce0cdad6b1683167ccdb2b3a75a"),
                parse("0x239f4553fa744a1b2a49af348cf9412fe105c3bc0accb23030a40bfe59d7fa2b"),
            ]),
            parse("0x2153695c5a08dcffb4189075c974db1faddcd9005c493937120b80eeaf3602fb")
        );
        assert_eq!(
            Keccak256Hash::<F>::hash_ternary([
                parse("0x73b6d673e9b67cbca809416002ed33ddeb704d58ce0101a6555531e63426383b"),
                parse("0x5b68318ce7baa6d9a0b6e67cee81b1a730dc7f68c34b7f806fd70ec4614f2096"),
                parse("0x5829d1d7b55ee4345afa69b4acf18d722f6962d0e61447a2711b5851b77d1c2b"),
            ]),
            parse("0x45b7f5b14de8d4ccf714a72bc733123c271484bdfe449b55d892f9ad1a26ea3a")
        );
    }

    #[test]
    fn test_keccak256_hash_ternary() {
        test_keccak256_hash_ternary_impl::<BS>();
        test_keccak256_hash_ternary_impl::<GL4>();
    }

    #[test]
    fn test_load_sha2_hash() {
        let hash1: H256 = "0x8c313b317b2ff13ecf3abea4e59dad76b24a13e7b5b8fa10a939fc2ab1713175"
            .parse()
            .unwrap();
        let hash2: H256 = "0x51c39e4b5287d491be2e02080c39493b77be8fd3d032227815357c5360e3d230"
            .parse()
            .unwrap();
        assert_eq!(Sha2Hash::<BS>::load_hash(hash1.as_bytes()).unwrap(), hash1);
        assert_eq!(Sha2Hash::<BS>::load_hash(hash2.as_bytes()).unwrap(), hash2);
        assert!(Sha2Hash::<BS>::load_hash(&hash1.as_bytes()[0..31]).is_err());
    }

    #[test]
    fn test_load_keccak256_hash() {
        let hash1: H256 = "0x8c313b317b2ff13ecf3abea4e59dad76b24a13e7b5b8fa10a939fc2ab1713175"
            .parse()
            .unwrap();
        let hash2: H256 = "0x51c39e4b5287d491be2e02080c39493b77be8fd3d032227815357c5360e3d230"
            .parse()
            .unwrap();
        assert_eq!(
            Keccak256Hash::<BS>::load_hash(hash1.as_bytes()).unwrap(),
            hash1
        );
        assert_eq!(
            Keccak256Hash::<BS>::load_hash(hash2.as_bytes()).unwrap(),
            hash2
        );
        assert!(Keccak256Hash::<BS>::load_hash(&hash1.as_bytes()[0..31]).is_err());
    }

    #[test]
    fn test_sha2_challenge_bluesky() {
        assert_eq!(
            Sha2Hash::<BS>::challenge(
                *DST,
                &[parse(
                    "0x5b584bf4398b7ef509abeb33ba8521c96a4a497ffba046a492cc43ac34174c16"
                )]
            ),
            parse("0x7c6534149fcf5a86469aa81dc5ed7eb1247df06d53c550ba3a78d94ae8ab350d")
        );
        assert_eq!(
            Sha2Hash::<BS>::challenge(
                *DST,
                &[
                    parse("0x2a1d8bd2a5dc960774c53b77e3e8d677b225bb45ec5d9caaf544e4f229a8afcd"),
                    parse("0x39de8bea57c1ab4082270791ce637189f07d169852f8bba5d05784368b505a12"),
                ]
            ),
            parse("0x21b62448a09d0f47610e8f1fa5afb2c4dde88c4bd71161be74904ae305e95ca2")
        );
        assert_eq!(
            Sha2Hash::<BS>::challenge(
                *DST,
                &[
                    parse("0x2cbe7a924ef4a68b49dc6eac0fcf7c8504cc9ecfcb1628cf2b9686d8597e088e"),
                    parse("0x4f6c036f1eaa65e8b761c7fc972156ca4a8340d47d26cd091c9a63655b415896"),
                    parse("0x32c783c083c0fafa4dba39e2176fc3a14791b83d4f1a51372aa483279a93d687"),
                ]
            ),
            parse("0x3b5e54e02f06cd4d72a0e49b6fef86b76345aa46528cf85ffcddab81a2311c20")
        );
    }

    #[test]
    fn test_sha2_challenge_goldilocks() {
        assert_eq!(
            Sha2Hash::<GL4>::challenge(
                *DST,
                &[parse(
                    "0x5b584bf4398b7ef509abeb33ba8521c96a4a497ffba046a492cc43ac34174c16"
                )]
            ),
            parse("0x49b6d25fcfd1a79d4e7b6709da6a47a48e135bddcac04f00700c350e67699b22")
        );
        assert_eq!(
            Sha2Hash::<GL4>::challenge(
                *DST,
                &[
                    parse("0x2a1d8bd2a5dc960774c53b77e3e8d677b225bb45ec5d9caaf544e4f229a8afcd"),
                    parse("0x39de8bea57c1ab4082270791ce637189f07d169852f8bba5d05784368b505a12"),
                ]
            ),
            parse("0xcfda2227d1ffdd6a14d76c1e55183308df20e1d2e5f144099b124b076c4c8db5")
        );
        assert_eq!(
            Sha2Hash::<GL4>::challenge(
                *DST,
                &[
                    parse("0x2cbe7a924ef4a68b49dc6eac0fcf7c8504cc9ecfcb1628cf2b9686d8597e088e"),
                    parse("0x4f6c036f1eaa65e8b761c7fc972156ca4a8340d47d26cd091c9a63655b415896"),
                    parse("0x32c783c083c0fafa4dba39e2176fc3a14791b83d4f1a51372aa483279a93d687"),
                ]
            ),
            parse("0xaf5d033d19e06a29d46573c6fbe668e521c58054636b3f9814d9fcfe93aa2ac6")
        );
    }

    #[test]
    fn test_keccak256_challenge_bluesky() {
        assert_eq!(
            Keccak256Hash::<BS>::challenge(
                *DST,
                &[parse(
                    "0x5b584bf4398b7ef509abeb33ba8521c96a4a497ffba046a492cc43ac34174c16"
                )]
            ),
            parse("0x6d8cbbf7a48eedea2cd9c827f3caf9ebbcdf18f13d5120c1078472f4a4c51889")
        );
        assert_eq!(
            Keccak256Hash::<BS>::challenge(
                *DST,
                &[
                    parse("0x2a1d8bd2a5dc960774c53b77e3e8d677b225bb45ec5d9caaf544e4f229a8afcd"),
                    parse("0x39de8bea57c1ab4082270791ce637189f07d169852f8bba5d05784368b505a12"),
                ]
            ),
            parse("0x55021224fa461943ba11d2eb8914241499a8d6651496d6bf7c3de325215da41e")
        );
        assert_eq!(
            Keccak256Hash::<BS>::challenge(
                *DST,
                &[
                    parse("0x2cbe7a924ef4a68b49dc6eac0fcf7c8504cc9ecfcb1628cf2b9686d8597e088e"),
                    parse("0x4f6c036f1eaa65e8b761c7fc972156ca4a8340d47d26cd091c9a63655b415896"),
                    parse("0x32c783c083c0fafa4dba39e2176fc3a14791b83d4f1a51372aa483279a93d687"),
                ]
            ),
            parse("0x5fdcf88830fed1cec79eac7d295593258985e884868f15935479ec6d037fa0f0")
        );
    }

    #[test]
    fn test_keccak256_challenge_goldilocks() {
        assert_eq!(
            Keccak256Hash::<GL4>::challenge(
                *DST,
                &[parse(
                    "0x5b584bf4398b7ef509abeb33ba8521c96a4a497ffba046a492cc43ac34174c16"
                )]
            ),
            parse("0x70976bfb022745ede6b644d5ed620d81cea15fefeaa94f285366098409a61878")
        );
        assert_eq!(
            Keccak256Hash::<GL4>::challenge(
                *DST,
                &[
                    parse("0x2a1d8bd2a5dc960774c53b77e3e8d677b225bb45ec5d9caaf544e4f229a8afcd"),
                    parse("0x39de8bea57c1ab4082270791ce637189f07d169852f8bba5d05784368b505a12"),
                ]
            ),
            parse("0x1134b29f1d7a0ed7759a4cce507dc48fb8dbbb3da068c3ae34f650df05e9db4a")
        );
        assert_eq!(
            Keccak256Hash::<GL4>::challenge(
                *DST,
                &[
                    parse("0x2cbe7a924ef4a68b49dc6eac0fcf7c8504cc9ecfcb1628cf2b9686d8597e088e"),
                    parse("0x4f6c036f1eaa65e8b761c7fc972156ca4a8340d47d26cd091c9a63655b415896"),
                    parse("0x32c783c083c0fafa4dba39e2176fc3a14791b83d4f1a51372aa483279a93d687"),
                ]
            ),
            parse("0x31429fe891097e8ff3ef2414039309efc1de6fd301643e5e82abe1c8b1d6ca5a")
        );
    }
}
