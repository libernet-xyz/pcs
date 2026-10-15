use anyhow::{Context, Result, anyhow};
use primitive_types::H256;
use sha2::Digest;
use starkom_ff::Field;

/// Authority used in the type URLs of all protobuf messages that go into `google.protobuf.Any`
/// fields.
pub(crate) const TYPE_URL_AUTHORITY: &str = "type.starkom.io";

/// Implements [`prost::Name`] for a protobuf message type, overriding
/// [`type_url`](`prost::Name::type_url`) so that it's prefixed with [`TYPE_URL_AUTHORITY`] rather
/// than a bare slash.
///
/// The arguments are the message type, its simple name as it appears in the `.proto` file, and the
/// name of its package.
macro_rules! prost_name {
    ($type:ty, $name:literal, $package:literal) => {
        impl ::prost::Name for $type {
            const NAME: &'static str = $name;
            const PACKAGE: &'static str = $package;

            fn type_url() -> String {
                format!(
                    "{}/{}",
                    $crate::utils::TYPE_URL_AUTHORITY,
                    <Self as ::prost::Name>::full_name()
                )
            }
        }
    };
}

pub(crate) use prost_name;

pub(crate) fn make_dst(s: &'static [u8]) -> H256 {
    let mut hasher = sha3::Sha3_256::new();
    hasher.update(s);
    H256::from_slice(hasher.finalize().as_slice())
}

/// Encodes a `u64` into a `H256` for use in various transcript hashes.
///
/// The value is stored in big-endian byte order in the last 8 bytes, and the first 24 bytes are
/// zero, so the result is the same as the 32 byte big-endian encoding of the value as a 256-bit
/// integer.
pub(crate) fn encode_u64(value: u64) -> H256 {
    let mut bytes = [0u8; 32];
    bytes[24..32].copy_from_slice(&value.to_be_bytes());
    H256::from_slice(&bytes)
}

/// Encodes a `usize` into a `H256` for use in various transcript hashes.
///
/// Same as [`encode_u64`], so the encoding doesn't depend on the platform's pointer width.
pub(crate) fn encode_usize(value: usize) -> H256 {
    encode_u64(value as u64)
}

pub(crate) fn load_scalar<F: Field>(bytes: &[u8]) -> Result<F> {
    if bytes.len() > F::LEN {
        return Err(anyhow!(
            "invalid scalar length (expected {}, got {})",
            F::LEN,
            bytes.len()
        ));
    }
    let mut padded_bytes = [0u8; 32];
    padded_bytes[0..bytes.len()].copy_from_slice(bytes);
    F::try_from_le_bytes(&padded_bytes)
        .into_option()
        .context(format!(
            "invalid scalar (exceeds the [0, {}) range)",
            F::MODULUS
        ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use primitive_types::U256;
    use starkom_ff::{Field256, bls12_381::Scalar};

    fn parse_hash(s: &'static str) -> H256 {
        s.parse().unwrap()
    }

    #[test]
    fn test_encode_u64() {
        assert_eq!(encode_u64(0), H256::zero());
        assert_eq!(
            encode_u64(1),
            parse_hash("0x0000000000000000000000000000000000000000000000000000000000000001")
        );
        assert_eq!(
            encode_u64(0x0102030405060708),
            parse_hash("0x0000000000000000000000000000000000000000000000000102030405060708")
        );
        assert_eq!(
            encode_u64(u64::MAX),
            parse_hash("0x000000000000000000000000000000000000000000000000ffffffffffffffff")
        );
    }

    #[test]
    fn test_encode_u64_is_big_endian_u256() {
        for value in [0, 1, 255, 256, 0x0102030405060708, u64::MAX - 1, u64::MAX] {
            assert_eq!(
                encode_u64(value).as_bytes(),
                U256::from(value).to_big_endian().as_slice()
            );
        }
    }

    #[test]
    fn test_encode_usize() {
        assert_eq!(encode_usize(0), H256::zero());
        assert_eq!(
            encode_usize(42),
            parse_hash("0x000000000000000000000000000000000000000000000000000000000000002a")
        );
        for value in [0, 1, 42, 0x01020304, usize::MAX - 1, usize::MAX] {
            assert_eq!(encode_usize(value), encode_u64(value as u64));
        }
    }

    #[test]
    fn test_load_scalar() {
        let value1: Scalar = "0x1fcc49bb3722a3a6ac41aa0aac6ed68f218a7240ca2a9befd4e3243251f7ecaf"
            .parse()
            .unwrap();
        let value2: Scalar = "0x39b3bad9cc2a94bf8f5f3c5277885fa778893898702be5ec79eaf59e08053a23"
            .parse()
            .unwrap();
        assert_eq!(
            load_scalar::<Scalar>(&value1.to_le_bytes()).unwrap(),
            value1
        );
        assert_eq!(
            load_scalar::<Scalar>(&value2.to_le_bytes()).unwrap(),
            value2
        );
    }

    fn test_load_truncated_scalar_impl(value: Scalar) {
        let bytes = value.to_le_bytes();
        let length = bytes
            .iter()
            .rposition(|&byte| byte != 0)
            .map_or(0, |index| index + 1);
        assert_eq!(load_scalar::<Scalar>(&bytes[0..length]).unwrap(), value);
    }

    #[test]
    fn test_load_truncated_scalar() {
        test_load_truncated_scalar_impl(
            "0x00cc49bb3722a3a6ac41aa0aac6ed68f218a7240ca2a9befd4e3243251f7ecaf"
                .parse()
                .unwrap(),
        );
        test_load_truncated_scalar_impl(
            "0x000049bb3722a3a6ac41aa0aac6ed68f218a7240ca2a9befd4e3243251f7ecaf"
                .parse()
                .unwrap(),
        );
        test_load_truncated_scalar_impl(
            "0x000000000000000000000000000000000000000000000000000000000000ecaf"
                .parse()
                .unwrap(),
        );
        test_load_truncated_scalar_impl(
            "0x00000000000000000000000000000000000000000000000000000000000000af"
                .parse()
                .unwrap(),
        );
        test_load_truncated_scalar_impl(
            "0x0000000000000000000000000000000000000000000000000000000000000000"
                .parse()
                .unwrap(),
        );
    }
}
