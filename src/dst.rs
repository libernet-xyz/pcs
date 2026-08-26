use primitive_types::{U256, U512};
use sha2::Digest;
use starkom_ff::Field;
use std::any::TypeId;
use std::collections::BTreeMap;
use std::sync::{LazyLock, Mutex};

fn make_dst(s: &'static [u8], modulus: U512) -> U256 {
    let mut hasher = sha3::Sha3_512::new();
    hasher.update(s);
    let hash: U512 = U512::from_little_endian(hasher.finalize().as_slice());
    let value = hash % modulus;
    U256::from_little_endian(&value.to_little_endian()[0..32])
}

pub(crate) fn get_dst<F: Field>(s: &'static [u8]) -> F {
    static DST_CACHE: LazyLock<Mutex<BTreeMap<(TypeId, &'static [u8]), U256>>> =
        LazyLock::new(|| Mutex::default());
    let value = {
        let mut cache = DST_CACHE.lock().unwrap();
        *cache
            .entry((TypeId::of::<F>(), s))
            .or_insert_with(|| make_dst(s, F::MODULUS.parse().unwrap()))
    };
    F::try_from_le_bytes(&value.to_little_endian()[0..(F::LEN)]).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use starkom_bluesky::Scalar as BS;
    use starkom_goldilocks::GL;
    use std::fmt::Debug;
    use std::str::FromStr;

    fn parse<V: FromStr<Err: Debug>>(s: &'static str) -> V {
        s.parse().unwrap()
    }

    #[test]
    fn test_dsts() {
        assert_eq!(
            get_dst::<BS>(b"starkom/merkle/leaf"),
            parse("0x08cb2652a56289bd316cfcb356f5d2be485538e04a601fb14fc2c98f03077fcb")
        );
        assert_eq!(
            get_dst::<GL>(b"starkom/merkle/leaf"),
            parse("0x9e8c852f6e39922a")
        );
    }
}
