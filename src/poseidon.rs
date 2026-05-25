use starkom_bluesky::Scalar;
use starkom_poseidon2 as poseidon;

pub(crate) fn hash_t3(inputs: &[Scalar]) -> Scalar {
    poseidon::hash::<poseidon::BlueSkyConfig<3>, Scalar, 3>(inputs)
}

pub(crate) fn hash_t4(inputs: &[Scalar]) -> Scalar {
    poseidon::hash::<poseidon::BlueSkyConfig<4>, Scalar, 4>(inputs)
}
