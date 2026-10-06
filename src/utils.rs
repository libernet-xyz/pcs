use anyhow::{Context, Result, anyhow};
use primitive_types::H256;
use sha2::Digest;
use starkom_ff::Field;

pub(crate) fn make_dst(s: &'static [u8]) -> H256 {
    let mut hasher = sha3::Sha3_256::new();
    hasher.update(s);
    H256::from_slice(hasher.finalize().as_slice())
}

pub(crate) fn load_hash(bytes: &[u8]) -> Result<H256> {
    if bytes.len() != 32 {
        return Err(anyhow!(
            "invalid hash length (expected 32, got {})",
            bytes.len()
        ));
    }
    Ok(H256::from_slice(bytes))
}

pub(crate) fn load_scalar<F: Field>(bytes: &[u8]) -> Result<F> {
    if bytes.len() != F::LEN {
        return Err(anyhow!(
            "invalid scalar length (expected {}, got {})",
            F::LEN,
            bytes.len()
        ));
    }
    F::try_from_le_bytes(bytes)
        .into_option()
        .context(format!("invalid scalar (exceeds the {} range)", F::MODULUS))
}

#[cfg(test)]
mod tests {
    use super::*;
    use starkom_ff::{Field256, bls12_381::Scalar};

    #[test]
    fn test_load_hash() {
        let hash1: H256 = "0x8c313b317b2ff13ecf3abea4e59dad76b24a13e7b5b8fa10a939fc2ab1713175"
            .parse()
            .unwrap();
        let hash2: H256 = "0x51c39e4b5287d491be2e02080c39493b77be8fd3d032227815357c5360e3d230"
            .parse()
            .unwrap();
        assert_eq!(load_hash(hash1.as_bytes()).unwrap(), hash1);
        assert_eq!(load_hash(hash2.as_bytes()).unwrap(), hash2);
        assert!(load_hash(&hash1.as_bytes()[0..31]).is_err());
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
        assert!(load_hash(&value1.to_le_bytes()[0..31]).is_err());
    }
}
