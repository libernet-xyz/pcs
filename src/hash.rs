use ff::{Field, PrimeField};
use sha2::{self, Digest};
use starkom_bluesky::Scalar;
use starkom_poseidon2 as poseidon;
use std::marker::PhantomData;

/// A generic cryptographic hash backend for use with FRI.
///
/// This trait is used for both binary Merkle trees and Fiat-Shamir challenges.
pub trait Hash<F: PrimeField> {
    /// Hashes two input scalars with a DST.
    fn hash_raw(dst: F, input1: F, input2: F) -> F;

    /// Hashes many input scalars.
    fn hash_many(inputs: &[F]) -> F;
}

/// SHA-256 hash backend.
///
/// This hashing backend is designed to be EVM-friendly so that our FRI proofs can be easily
/// verified in the EVM.
///
/// In order to convert the result to a BlueSky scalar without any significant distribution skew we
/// actually need a 512-bit hash, so that we can perform the conversion via modular reduction.
/// However we want to use the 256-bit version of SHA2 for compatibility with the EVM. So we obtain
/// a 512-bit hash as follows:
///
///   * the low 256 bits of the hash are Sha2_256(0 || input1 || input2),
///   * the high 256 bits of the hash are Sha2_256(1 || input1 || input2),
///   * the 512 bits are converted to a scalar via modular reduction.
#[derive(Debug, Default, Copy, Clone)]
pub struct Sha2Hash<F: PrimeField> {
    _data: PhantomData<F>,
}

impl Sha2Hash<Scalar> {
    fn hash_internal(inputs: impl IntoIterator<Item = Scalar>) -> [u8; 32] {
        let mut hasher = sha2::Sha256::default();
        for input in inputs {
            hasher.update(input.to_big_endian());
        }
        let mut bytes: [u8; 32] = hasher.finalize().into();
        bytes.reverse();
        bytes
    }
}

impl Hash<Scalar> for Sha2Hash<Scalar> {
    fn hash_raw(dst: Scalar, input1: Scalar, input2: Scalar) -> Scalar {
        let lo = Self::hash_internal([Scalar::ZERO, dst, input1, input2]);
        let hi = Self::hash_internal([Scalar::ONE, dst, input1, input2]);
        let mut bytes = [0u8; 64];
        bytes[0..32].copy_from_slice(&lo);
        bytes[32..64].copy_from_slice(&hi);
        Scalar::from_repr_wide(&bytes)
    }

    fn hash_many(inputs: &[Scalar]) -> Scalar {
        let lo = Self::hash_internal(std::iter::once(Scalar::ZERO).chain(inputs.iter().cloned()));
        let hi = Self::hash_internal(std::iter::once(Scalar::ONE).chain(inputs.iter().cloned()));
        let mut bytes = [0u8; 64];
        bytes[0..32].copy_from_slice(&lo);
        bytes[32..64].copy_from_slice(&hi);
        Scalar::from_repr_wide(&bytes)
    }
}

/// Poseidon2 hash backend.
///
/// This Poseidon2 instance is optimized for hashing binary Merkle tree nodes with a DST, so it uses
/// state vector size T=4 (rate=3, capacity=1).
#[derive(Debug, Default, Copy, Clone)]
pub struct Poseidon2Hash<F: PrimeField> {
    _data: PhantomData<F>,
}

impl Poseidon2Hash<Scalar> {
    fn hash_internal(inputs: &[Scalar]) -> Scalar {
        poseidon::hash::<poseidon::BlueSkyConfig<4>, Scalar, 4>(inputs)
    }
}

impl Hash<Scalar> for Poseidon2Hash<Scalar> {
    fn hash_raw(dst: Scalar, input1: Scalar, input2: Scalar) -> Scalar {
        Self::hash_internal(&[dst, input1, input2])
    }

    fn hash_many(inputs: &[Scalar]) -> Scalar {
        Self::hash_internal(inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::parse_scalar;

    #[test]
    fn test_sha2_hash_raw() {
        assert_eq!(
            Sha2Hash::<Scalar>::hash_raw(12.into(), 34.into(), 56.into()),
            parse_scalar("0x6ba46ed6f29f6a4f8e1a9d8f93b51f5e902143d00356b65b867dde9cc879a8d1")
        );
        assert_eq!(
            Sha2Hash::<Scalar>::hash_raw(56.into(), 78.into(), 90.into()),
            parse_scalar("0x1d6f14df7d976189975985d9427a9afc683c573adff33c4e77f0f0a577ac6062")
        );
    }

    #[test]
    fn test_sha2_hash_three() {
        assert_eq!(
            Sha2Hash::<Scalar>::hash_many(&[12.into(), 34.into(), 56.into()]),
            parse_scalar("0x6ba46ed6f29f6a4f8e1a9d8f93b51f5e902143d00356b65b867dde9cc879a8d1")
        );
        assert_eq!(
            Sha2Hash::<Scalar>::hash_many(&[56.into(), 78.into(), 90.into()]),
            parse_scalar("0x1d6f14df7d976189975985d9427a9afc683c573adff33c4e77f0f0a577ac6062")
        );
    }

    #[test]
    fn test_sha2_hash_many() {
        assert_eq!(
            Sha2Hash::<Scalar>::hash_many(&[12.into()]),
            parse_scalar("0x70f2261a2a020a7ae1fe4cd2a47244115f5243bebb22dc002adfc597e24a436a"),
        );
        assert_eq!(
            Sha2Hash::<Scalar>::hash_many(&[12.into(), 34.into()]),
            parse_scalar("0x614eaeb45d6c697d7cf720c4c7c604efe3e2d7ee733caa3a67a951975bcfd1c7"),
        );
        assert_eq!(
            Sha2Hash::<Scalar>::hash_many(&[56.into(), 78.into(), 90.into(), 12.into(), 34.into()]),
            parse_scalar("0x76493ab96fdfde689a3e4d9c5576e4006cbaabd96170b2207e8f837cc798266c"),
        );
    }

    #[test]
    fn test_poseidon2_hash_raw() {
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_raw(12.into(), 34.into(), 56.into()),
            parse_scalar("0x236092ebefc7e6565e0e75414d8fdce1ce2e19bb59002d36b794b9c3111bb9cd")
        );
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_raw(56.into(), 78.into(), 90.into()),
            parse_scalar("0x2fa39a3a76d0cf8220bd6f9899b209110ad1cca7b0bdc2b340661fa7063f2ba0")
        );
    }

    #[test]
    fn test_poseidon2_hash_three() {
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_many(&[12.into(), 34.into(), 56.into()]),
            parse_scalar("0x236092ebefc7e6565e0e75414d8fdce1ce2e19bb59002d36b794b9c3111bb9cd")
        );
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_many(&[56.into(), 78.into(), 90.into()]),
            parse_scalar("0x2fa39a3a76d0cf8220bd6f9899b209110ad1cca7b0bdc2b340661fa7063f2ba0")
        );
    }

    #[test]
    fn test_poseidon2_hash_many() {
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_many(&[12.into()]),
            parse_scalar("0x45782306ba3302ebe2f07eacbf5d0c36a5f307dc1cde4f3f9e8196ef498eddf2"),
        );
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_many(&[12.into(), 34.into()]),
            parse_scalar("0x08802dc1d5eaf75680808adb1d19bb420f34f5e786f09e05ffa5d41fb2bdfe6d"),
        );
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_many(&[
                56.into(),
                78.into(),
                90.into(),
                12.into(),
                34.into()
            ]),
            parse_scalar("0x7499d072269d7c32ad0477050bacd7cc84009b845c016280d679bc7849ed845a"),
        );
    }
}
