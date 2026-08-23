use primitive_types::{H128, H256, U128, U512};
use sha2::Digest;
use starkom_ff::Field256;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::LazyLock;

/// Describes a hash backend for our DEEP-FRI prover.
///
/// In order to warrant sufficient security even under Grover, our proof system works entirely with
/// 256-bit hashes and 256-bit field elements. If the arithmetization process uses a smaller field
/// such as Goldilocks, all scalar values must be embedded into a 256-bit field such as Goldilocks^4
/// before they can be committed.
pub trait Hash<F: Field256> {
    /// Hashes the provided field elements.
    ///
    /// NOTE: this is a raw hash, it doesn't automatically prepend any DSTs or element count; the
    /// caller is responsible for those when necessary.
    fn hash(inputs: impl IntoIterator<Item = F>) -> H256;

    /// Hashes a binary Merkle tree node.
    fn hash_binary(left: H256, right: H256) -> H256;

    /// Hashes a ternary Merkle tree node.
    fn hash_ternary(children: [H256; 3]) -> H256;

    /// Generates a Fiat-Shamir challenge from the provided transcript.
    fn challenge(transcript: &[H256]) -> F;
}

mod internal {
    use super::*;

    pub trait LowLevelHash: Debug + Default {
        fn update(&mut self, bytes: &[u8]);
        fn finalize(self) -> H256;
    }

    #[derive(Debug, Default)]
    pub struct LowLevelSha2Hash {
        hasher: sha2::Sha256,
    }

    impl LowLevelHash for LowLevelSha2Hash {
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
        fn update(&mut self, bytes: &[u8]) {
            self.hasher.update(bytes);
        }

        fn finalize(self) -> H256 {
            H256::from_slice(self.hasher.finalize().as_slice())
        }
    }

    fn make_dst(s: &'static [u8]) -> H128 {
        let mut hasher = sha3::Sha3_256::new();
        hasher.update(s);
        H128::from_slice(&hasher.finalize().as_slice()[0..16])
    }

    pub struct HashImpl<H: LowLevelHash, F: Field256> {
        _data: PhantomData<(H, F)>,
    }

    impl<H: LowLevelHash, F: Field256> HashImpl<H, F> {
        fn hash_transcript(dst: H128, transcript: &[H256]) -> H256 {
            let mut hasher = H::default();
            hasher.update(dst.as_bytes());
            hasher.update(&U128::from(transcript.len() as u64).to_big_endian());
            for element in transcript {
                hasher.update(element.as_bytes());
            }
            hasher.finalize()
        }
    }

    impl<H: LowLevelHash, F: Field256> Hash<F> for HashImpl<H, F> {
        fn hash(inputs: impl IntoIterator<Item = F>) -> H256 {
            let mut hasher = H::default();
            for input in inputs {
                hasher.update(&input.to_be_bytes());
            }
            hasher.finalize()
        }

        fn hash_binary(left: H256, right: H256) -> H256 {
            let mut hasher = H::default();
            hasher.update(&left.to_fixed_bytes());
            hasher.update(&right.to_fixed_bytes());
            hasher.finalize()
        }

        fn hash_ternary(children: [H256; 3]) -> H256 {
            let mut hasher = H::default();
            hasher.update(&children[0].to_fixed_bytes());
            hasher.update(&children[1].to_fixed_bytes());
            hasher.update(&children[2].to_fixed_bytes());
            hasher.finalize()
        }

        fn challenge(transcript: &[H256]) -> F {
            let hi = {
                static DST: LazyLock<H128> = LazyLock::new(|| make_dst(b"starkom/pcs/challenge/0"));
                Self::hash_transcript(*DST, transcript)
            };
            let lo = {
                static DST: LazyLock<H128> = LazyLock::new(|| make_dst(b"starkom/pcs/challenge/1"));
                Self::hash_transcript(*DST, transcript)
            };
            let mut bytes = [0u8; 64];
            bytes[0..32].copy_from_slice(hi.as_bytes());
            bytes[32..].copy_from_slice(lo.as_bytes());
            let hash = U512::from_big_endian(&bytes);
            let modulus: U512 = F::MODULUS.parse().unwrap();
            let challenge = hash % modulus;
            F::try_from_be_bytes(&challenge.to_big_endian()[32..]).unwrap()
        }
    }
}

/// SHA2 hash backend.
pub type Sha2Hash<F> = internal::HashImpl<internal::LowLevelSha2Hash, F>;

/// Keccak-256 hash backend.
pub type Keccak256Hash<F> = internal::HashImpl<internal::LowLevelKeccak256Hash, F>;

#[cfg(test)]
mod tests {
    use super::*;
    use starkom_bluesky::Scalar as BS;
    use starkom_ff::Field;
    use starkom_goldilocks::GL4;
    use std::str::FromStr;

    fn from_u64<F: Field256>(value: u64) -> F {
        F::from(value)
    }

    fn parse<V: FromStr<Err: Debug>>(s: &'static str) -> V {
        s.parse().unwrap()
    }

    fn test_sha2_hash_impl<F: Field256>() {
        assert_eq!(
            Sha2Hash::<F>::hash([from_u64(12)]),
            parse("0xa82872b96246dac512ddf0515f5da862a92ecebebcb92537b6e3e73199694c45")
        );
        assert_eq!(
            Sha2Hash::<F>::hash([from_u64(34), from_u64(56)]),
            parse("0xbdcf24876d0b8979976f54ea123b70112da34a5cb4dc381646a3321f0817a5e8")
        );
        assert_eq!(
            Sha2Hash::<F>::hash([from_u64(78), from_u64(90), from_u64(12)]),
            parse("0xce080de3e477a622d8b9711eb599aec6fe9ddda88c47c1175bff2aaadc43c3b4")
        );
    }

    #[test]
    fn test_sha2_hash() {
        test_sha2_hash_impl::<BS>();
        test_sha2_hash_impl::<GL4>();
    }

    fn test_keccak256_hash_impl<F: Field256>() {
        assert_eq!(
            Keccak256Hash::<F>::hash([from_u64(12)]),
            parse("0xdf6966c971051c3d54ec59162606531493a51404a002842f56009d7e5cf4a8c7")
        );
        assert_eq!(
            Keccak256Hash::<F>::hash([from_u64(34), from_u64(56)]),
            parse("0x72700d0d963d58363ea77095ddabb7ed1a429a1fe618c8ace040205d52391bb9")
        );
        assert_eq!(
            Keccak256Hash::<F>::hash([from_u64(78), from_u64(90), from_u64(12)]),
            parse("0x452eb69ea7065787fb9f51a2894ff6a6ff50ae842aaa68b3ad40ed3580117a8a")
        );
    }

    #[test]
    fn test_keccak256_hash() {
        test_keccak256_hash_impl::<BS>();
        test_keccak256_hash_impl::<GL4>();
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
        eprintln!("{}", BS::random_default());
        eprintln!("{}", BS::random_default());
        eprintln!("{}", BS::random_default());
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
    fn test_sha2_challenge_bluesky() {
        assert_eq!(
            Sha2Hash::<BS>::challenge(&[parse(
                "0x5b584bf4398b7ef509abeb33ba8521c96a4a497ffba046a492cc43ac34174c16"
            )]),
            parse("0x78b05d4cf077c387a38026c25943a8b03440178d8e351dc7140b9479a58efc62")
        );
        assert_eq!(
            Sha2Hash::<BS>::challenge(&[
                parse("0x2a1d8bd2a5dc960774c53b77e3e8d677b225bb45ec5d9caaf544e4f229a8afcd"),
                parse("0x39de8bea57c1ab4082270791ce637189f07d169852f8bba5d05784368b505a12"),
            ]),
            parse("0x6900b583e0659590d5ea91b8c882cba9054d8a1b7929c445c3746d0ec095bc68")
        );
        assert_eq!(
            Sha2Hash::<BS>::challenge(&[
                parse("0x2cbe7a924ef4a68b49dc6eac0fcf7c8504cc9ecfcb1628cf2b9686d8597e088e"),
                parse("0x4f6c036f1eaa65e8b761c7fc972156ca4a8340d47d26cd091c9a63655b415896"),
                parse("0x32c783c083c0fafa4dba39e2176fc3a14791b83d4f1a51372aa483279a93d687"),
            ]),
            parse("0x3f137d653dc81b281abd74125613820759f872831822d305bd203f591f5baf09")
        );
    }

    #[test]
    fn test_sha2_challenge_goldilocks() {
        assert_eq!(
            Sha2Hash::<GL4>::challenge(&[parse(
                "0x5b584bf4398b7ef509abeb33ba8521c96a4a497ffba046a492cc43ac34174c16"
            )]),
            parse("0x8d9ac2b5aad3733563e99865b441c1eb314ebace4103d55988fcc3b8b2911f2d")
        );
        assert_eq!(
            Sha2Hash::<GL4>::challenge(&[
                parse("0x2a1d8bd2a5dc960774c53b77e3e8d677b225bb45ec5d9caaf544e4f229a8afcd"),
                parse("0x39de8bea57c1ab4082270791ce637189f07d169852f8bba5d05784368b505a12"),
            ]),
            parse("0x9fad853c08e40ffafa35e323bca17ac34b52f8a2836116571f0044f556665b72")
        );
        assert_eq!(
            Sha2Hash::<GL4>::challenge(&[
                parse("0x2cbe7a924ef4a68b49dc6eac0fcf7c8504cc9ecfcb1628cf2b9686d8597e088e"),
                parse("0x4f6c036f1eaa65e8b761c7fc972156ca4a8340d47d26cd091c9a63655b415896"),
                parse("0x32c783c083c0fafa4dba39e2176fc3a14791b83d4f1a51372aa483279a93d687"),
            ]),
            parse("0x22ada69eb25d9e0c1654b47c06ca8e6ca7596367a20dd56d966ff4787296bb73")
        );
    }

    #[test]
    fn test_keccak256_challenge_bluesky() {
        assert_eq!(
            Keccak256Hash::<BS>::challenge(&[parse(
                "0x5b584bf4398b7ef509abeb33ba8521c96a4a497ffba046a492cc43ac34174c16"
            )]),
            parse("0x031c4e9c4d002b5609c02156388648c3ffe34d80ed0b17bfd50a95d0baf45aa8")
        );
        assert_eq!(
            Keccak256Hash::<BS>::challenge(&[
                parse("0x2a1d8bd2a5dc960774c53b77e3e8d677b225bb45ec5d9caaf544e4f229a8afcd"),
                parse("0x39de8bea57c1ab4082270791ce637189f07d169852f8bba5d05784368b505a12"),
            ]),
            parse("0x068087bf4a49a0d2e40500e0d2ce69afb2192f042318d53fefeb73c535b2652f")
        );
        assert_eq!(
            Keccak256Hash::<BS>::challenge(&[
                parse("0x2cbe7a924ef4a68b49dc6eac0fcf7c8504cc9ecfcb1628cf2b9686d8597e088e"),
                parse("0x4f6c036f1eaa65e8b761c7fc972156ca4a8340d47d26cd091c9a63655b415896"),
                parse("0x32c783c083c0fafa4dba39e2176fc3a14791b83d4f1a51372aa483279a93d687"),
            ]),
            parse("0x2ceda98a8005aae79c4dafd76c1464d376dacd5e97ce2bd208f33d936a8bddd4")
        );
    }

    #[test]
    fn test_keccak256_challenge_goldilocks() {
        assert_eq!(
            Keccak256Hash::<GL4>::challenge(&[parse(
                "0x5b584bf4398b7ef509abeb33ba8521c96a4a497ffba046a492cc43ac34174c16"
            )]),
            parse("0x8f8f79a93bf43c083b2334b4893247860192c59a0576b6cecd9425968f5217c8")
        );
        assert_eq!(
            Keccak256Hash::<GL4>::challenge(&[
                parse("0x2a1d8bd2a5dc960774c53b77e3e8d677b225bb45ec5d9caaf544e4f229a8afcd"),
                parse("0x39de8bea57c1ab4082270791ce637189f07d169852f8bba5d05784368b505a12"),
            ]),
            parse("0x2d93d1e7a6ca805e782ede54b65dd3b1d8b74b11dc08f2537a82bcbabddb6780")
        );
        assert_eq!(
            Keccak256Hash::<GL4>::challenge(&[
                parse("0x2cbe7a924ef4a68b49dc6eac0fcf7c8504cc9ecfcb1628cf2b9686d8597e088e"),
                parse("0x4f6c036f1eaa65e8b761c7fc972156ca4a8340d47d26cd091c9a63655b415896"),
                parse("0x32c783c083c0fafa4dba39e2176fc3a14791b83d4f1a51372aa483279a93d687"),
            ]),
            parse("0x0bbf9a8a1b46deed8e1ab544ad73abe8a7c96cfb0c57e7ef4a7b6dd52997814d")
        );
    }
}
