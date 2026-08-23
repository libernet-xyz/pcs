use primitive_types::{H128, H256, U128, U512};
use sha2::Digest;
use starkom_ff::Field256;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::LazyLock;

/// Describes a hash backend for our DEEP-FRI prover.
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
    ///
    /// The generated challenge is a ~256-bit field element because the challenge must have
    /// sufficient unpredictability. For 256-bit implementations of the [`Hash`] trait this field
    /// can be the same as `F`, while for smaller fields it needs to be an extension field (for
    /// example, when `F` is Goldilocks `G` can be Goldilocks^4).
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

    fn from_u64<F: Field256>(value: u64) -> F {
        F::from(value)
    }

    fn parse_hash(s: &'static str) -> H256 {
        s.parse().unwrap()
    }

    fn test_sha2_hash_impl<F: Field256>() {
        assert_eq!(
            Sha2Hash::<F>::hash([from_u64(12)]),
            parse_hash("0xa82872b96246dac512ddf0515f5da862a92ecebebcb92537b6e3e73199694c45")
        );
        assert_eq!(
            Sha2Hash::<F>::hash([from_u64(34), from_u64(56)]),
            parse_hash("0xbdcf24876d0b8979976f54ea123b70112da34a5cb4dc381646a3321f0817a5e8")
        );
        assert_eq!(
            Sha2Hash::<F>::hash([from_u64(78), from_u64(90), from_u64(12)]),
            parse_hash("0xce080de3e477a622d8b9711eb599aec6fe9ddda88c47c1175bff2aaadc43c3b4")
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
            parse_hash("0xdf6966c971051c3d54ec59162606531493a51404a002842f56009d7e5cf4a8c7")
        );
        assert_eq!(
            Keccak256Hash::<F>::hash([from_u64(34), from_u64(56)]),
            parse_hash("0x72700d0d963d58363ea77095ddabb7ed1a429a1fe618c8ace040205d52391bb9")
        );
        assert_eq!(
            Keccak256Hash::<F>::hash([from_u64(78), from_u64(90), from_u64(12)]),
            parse_hash("0x452eb69ea7065787fb9f51a2894ff6a6ff50ae842aaa68b3ad40ed3580117a8a")
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
                parse_hash("0x2c8ea861b3ec34715e3d50053d001e9d9929a774954e93ffa1f1784f94339592"),
                parse_hash("0x57c43629d9178a54c611df8562260309677c1a10a3bd3ca4c9997acbc5f908bb")
            ),
            parse_hash("0xff888724c7e8b1359d4cb4014d8ddcba61b938e5becbfd4f0af454dee20b83be")
        );
        assert_eq!(
            Sha2Hash::<F>::hash_binary(
                parse_hash("0x229045f7089bd1cb6527f1edd7d94a7e91f2dcf607553e2a89fe24d72b2b013c"),
                parse_hash("0x021905a563d6bf385cb3f2eb4bc69d7750812bcb29542ee4bcfcb0fd0e1bc9b8")
            ),
            parse_hash("0x61944633a56b8421a4b1b53555bfa58283989e76fc21b7e6c1634730969f81f4")
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
                parse_hash("0x2c8ea861b3ec34715e3d50053d001e9d9929a774954e93ffa1f1784f94339592"),
                parse_hash("0x57c43629d9178a54c611df8562260309677c1a10a3bd3ca4c9997acbc5f908bb")
            ),
            parse_hash("0x0887805e67bb6ece5489f58f1863dffec93f52583cd050eca57c33cbd33abb09")
        );
        assert_eq!(
            Keccak256Hash::<F>::hash_binary(
                parse_hash("0x229045f7089bd1cb6527f1edd7d94a7e91f2dcf607553e2a89fe24d72b2b013c"),
                parse_hash("0x021905a563d6bf385cb3f2eb4bc69d7750812bcb29542ee4bcfcb0fd0e1bc9b8")
            ),
            parse_hash("0x3cecf97526033fb79c366cde597b0f66b7774ec26388b71fd591d7275c81dfbe")
        );
    }

    #[test]
    fn test_keccak256_hash_binary() {
        test_keccak256_hash_binary_impl::<BS>();
        test_keccak256_hash_binary_impl::<GL4>();
    }

    // TODO
}
