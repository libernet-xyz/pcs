use ff::PrimeField;
use primitive_types::{H512, U512};
use sha3::Digest;
use starkom_bluesky::Scalar;
use std::any::TypeId;
use std::collections::BTreeMap;
use std::sync::{LazyLock, Mutex};

fn get_modulus<F: PrimeField<Repr = [u8; 32]>>() -> U512 {
    static MODULI: LazyLock<Mutex<BTreeMap<TypeId, U512>>> = LazyLock::new(|| Mutex::default());
    let type_id = TypeId::of::<F>();
    let mut values = MODULI.lock().unwrap();
    match values.get(&type_id) {
        Some(value) => value.clone(),
        None => {
            let value = F::MODULUS.parse().unwrap();
            values.insert(type_id, value);
            value
        }
    }
}

/// Interprets the provided 64 bytes in little-endian order and converts them to a BlueSky scalar
/// via modular reduction.
pub(crate) fn h512_to_scalar<F: PrimeField<Repr = [u8; 32]>>(h512: H512) -> F {
    let dividend = U512::from_little_endian(h512.as_bytes());
    let remainder = dividend % get_modulus::<F>();
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&remainder.to_little_endian()[0..32]);
    F::from_repr_vartime(bytes).unwrap()
}

/// Hashes an arbitrary text string into a uniformly distributed BlueSky scalar.
///
/// Under the hood this function works by hashing the string with SHA3-512 and converting the
/// resulting 64 bytes to a BlueSky scalar via modular reduction.
pub(crate) fn hash_to_scalar<F: PrimeField<Repr = [u8; 32]>>(message: &[u8]) -> F {
    let mut hasher = sha3::Sha3_512::new();
    hasher.update(message);
    h512_to_scalar::<F>(H512::from_slice(hasher.finalize().as_slice()))
}

/// Parses a BlueSky scalar, panicking if parsing fails.
///
/// This functionality is provided only for static strings because user inputs must always be
/// validated.
pub(crate) fn parse_scalar(s: &'static str) -> Scalar {
    s.parse().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use blstrs::Scalar as BlsScalar;
    use primitive_types::U256;

    fn parse_bls_scalar(s: &'static str) -> BlsScalar {
        let u256: U256 = s.parse().unwrap();
        BlsScalar::from_bytes_le(&u256.to_little_endian())
            .into_option()
            .unwrap()
    }

    #[test]
    fn test_hash_to_bluesky_scalar() {
        assert_eq!(
            hash_to_scalar::<Scalar>(b"lorem ipsum dolor sit amet"),
            parse_scalar("0x69c562c4b39c86fc322322c86cfe5be83fbd472c6a38862bdd2f362bfa442ad6")
        );
        assert_eq!(
            hash_to_scalar::<Scalar>(b"sator arepo tenet opera rotas"),
            parse_scalar("0x027880d47636bf77d55804a6cf2d5ec8f09427cdf678e2ed3d74c432cc2efa7a")
        );
    }

    #[test]
    fn test_hash_to_bls_scalar() {
        assert_eq!(
            hash_to_scalar::<BlsScalar>(b"lorem ipsum dolor sit amet"),
            parse_bls_scalar("0x2e153f5d15640c364521222297d2406f547ac0976a9cc5c1f4c42c0b20ff6d30")
        );
        assert_eq!(
            hash_to_scalar::<BlsScalar>(b"sator arepo tenet opera rotas"),
            parse_bls_scalar("0x1fdb6bf666ad555ca1a740f680d59736b736aa58ccd139eafe8889370196aa24")
        );
    }
}
