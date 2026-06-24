use primitive_types::H512;
use sha3::Digest;
use starkom_ff::PrimeField256;

/// Hashes an arbitrary text string into a uniformly distributed scalar.
///
/// Under the hood this function works by hashing the string with SHA3-512 and converting the
/// resulting 64 bytes to a scalar via modular reduction.
pub(crate) fn hash_to_scalar<F: PrimeField256>(message: &[u8]) -> F {
    let mut hasher = sha3::Sha3_512::new();
    hasher.update(message);
    F::from_h512(H512::from_slice(hasher.finalize().as_slice()))
}

#[cfg(test)]
pub(crate) mod testing {
    use starkom_bluesky::Scalar;

    /// TEST ONLY: parses a BlueSky scalar, panicking if parsing fails.
    pub(crate) fn parse_scalar(s: &'static str) -> Scalar {
        s.parse().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::testing::parse_scalar;
    use super::*;
    use primitive_types::U256;
    use starkom_bluesky::Scalar;
    use starkom_ff::{Field, bls12_381::Scalar as BlsScalar};

    fn parse_bls_scalar(s: &'static str) -> BlsScalar {
        let u256: U256 = s.parse().unwrap();
        BlsScalar::try_from_le_bytes(&u256.to_little_endian())
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
