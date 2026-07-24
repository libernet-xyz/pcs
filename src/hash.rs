use primitive_types::{H256, U512};
use sha2::{self, Digest};
use starkom_bluesky::Scalar;
use starkom_ff::{Field, Field256, PrimeField, PrimeField64, PrimeField256};
use starkom_poseidon as poseidon1;
use starkom_poseidon2 as poseidon2;
use std::marker::PhantomData;

/// Generic cryptographic hash functions.
pub trait Hash<V> {
    /// Hashes two input values.
    fn hash_two(input1: V, input2: V) -> H256;

    /// Hashes three input values.
    fn hash_three(input1: V, input2: V, input3: V) -> H256;

    /// Hashes many input values.
    ///
    /// WARNING: if the input slice doesn't have a fixed length predetermined by the protocol the
    /// caller is responsible for including the number of inputs at the beginning, because
    /// implementors of this trait are not guaranteed to guard against length-extension attacks or
    /// collisions on the padding area. Note that SHA2 is vulnerable to length-extension attacks,
    /// and Starkom's Poseidon implementations automatically pad with zeros making e.g.
    /// `hash_many(x1, x2)` identical to `hash_many(x1, x2, 0)` if the `Hash` implementor uses T=4.
    fn hash_many<I: IntoIterator<Item = V>>(inputs: I) -> H256;
}

/// A generic cryptographic hash backend for use with DEEP-FRI.
///
/// This trait inherits both `Hash<F>` and `Hash<H256>`, and can be used for all of the following
/// use cases:
///
/// * hashing Merkle leaves made up of polynomial evaluations,
/// * hashing internal Merkle tree nodes (which are themselves 256-bit hashes),
/// * deriving Fiat-Shamir challenges.
///
/// NOTE: it is guaranteed that all `H256` hashes flowing through the program are always generated
/// by the `HashBackend` in use, so implementors are allowed to make assumptions about the content
/// of any `H256` input parameters. For example, a Poseidon implementatiopn over the BlueSky field
/// can assume that all `H256` values wrap BlueSky scalars with a certain endianness, while a
/// Poseidon implementation based on Goldilocks can assume that all `H256`s wrap four Goldilocks
/// scalars with a certain layout.
pub trait HashBackend<F: PrimeField>: Hash<F> + Hash<H256> {
    /// Encodes a field value into an `H256` so that it can be used as an input to `Hash<H256>`
    /// functions.
    ///
    /// This version is for 64-bit fields such as Goldilocks.
    fn encode_scalar64(value: F) -> H256
    where
        F: PrimeField64,
    {
        H256::from_slice(&value.to_be_bytes())
    }

    /// Encodes a field value into an `H256` so that it can be used as an input to `Hash<H256>`
    /// functions.
    ///
    /// This version is for 256-bit fields such as BlueSky.
    fn encode_scalar256(value: F) -> H256
    where
        F: PrimeField256,
    {
        H256::from_slice(&value.to_be_bytes())
    }

    /// Encodes a `usize` into an `H256` so that it can be used as an input to `Hash<H256>`
    /// functions.
    fn encode_usize(value: usize) -> H256 {
        H256::from_slice(&Scalar::from(value as u64).to_be_bytes())
    }

    /// Derives a Fiat-Shamir challenge from a set of transcript hashes.
    ///
    /// `dst` is a domain separator tag.
    fn challenge<I: IntoIterator<Item = H256>>(dst: F, transcript_hashes: I) -> F;
}

/// SHA-256 hash backend.
///
/// This hashing backend is designed to be EVM-friendly so that our FRI proofs can be easily
/// verified in the EVM.
#[derive(Debug, Default, Copy, Clone)]
pub struct Sha2Hash<F: PrimeField> {
    _data: PhantomData<F>,
}

impl Hash<Scalar> for Sha2Hash<Scalar> {
    fn hash_two(input1: Scalar, input2: Scalar) -> H256 {
        let mut hasher = sha2::Sha256::default();
        hasher.update(&input1.to_be_bytes());
        hasher.update(&input2.to_be_bytes());
        H256::from_slice(hasher.finalize().as_slice())
    }

    fn hash_three(input1: Scalar, input2: Scalar, input3: Scalar) -> H256 {
        let mut hasher = sha2::Sha256::default();
        hasher.update(&input1.to_be_bytes());
        hasher.update(&input2.to_be_bytes());
        hasher.update(&input3.to_be_bytes());
        H256::from_slice(hasher.finalize().as_slice())
    }

    fn hash_many<I: IntoIterator<Item = Scalar>>(inputs: I) -> H256 {
        let mut hasher = sha2::Sha256::default();
        for input in inputs.into_iter() {
            hasher.update(&input.to_be_bytes());
        }
        H256::from_slice(hasher.finalize().as_slice())
    }
}

impl Hash<H256> for Sha2Hash<Scalar> {
    fn hash_two(input1: H256, input2: H256) -> H256 {
        let mut hasher = sha2::Sha256::default();
        hasher.update(input1.as_bytes());
        hasher.update(input2.as_bytes());
        H256::from_slice(hasher.finalize().as_slice())
    }

    fn hash_three(input1: H256, input2: H256, input3: H256) -> H256 {
        let mut hasher = sha2::Sha256::default();
        hasher.update(input1.as_bytes());
        hasher.update(input2.as_bytes());
        hasher.update(input3.as_bytes());
        H256::from_slice(hasher.finalize().as_slice())
    }

    fn hash_many<I: IntoIterator<Item = H256>>(inputs: I) -> H256 {
        let mut hasher = sha2::Sha256::default();
        for input in inputs.into_iter() {
            hasher.update(input.as_bytes());
        }
        H256::from_slice(hasher.finalize().as_slice())
    }
}

impl HashBackend<Scalar> for Sha2Hash<Scalar> {
    fn challenge<I: IntoIterator<Item = H256>>(dst: Scalar, transcript_hashes: I) -> Scalar {
        let transcript_hash = {
            let transcript_hashes: Vec<H256> = transcript_hashes.into_iter().collect();
            Self::hash_many(
                std::iter::once(Self::encode_scalar256(dst))
                    .chain(std::iter::once(Self::encode_usize(transcript_hashes.len())))
                    .chain(transcript_hashes.into_iter()),
            )
        };
        let dst1 = dst;
        let dst2 = dst + Scalar::ONE;
        let hi = {
            let mut hasher = sha2::Sha256::default();
            hasher.update(&dst1.to_be_bytes());
            hasher.update(transcript_hash.as_bytes());
            H256::from_slice(hasher.finalize().as_slice())
        };
        let lo = {
            let mut hasher = sha2::Sha256::default();
            hasher.update(&dst2.to_be_bytes());
            hasher.update(transcript_hash.as_bytes());
            H256::from_slice(hasher.finalize().as_slice())
        };
        let mut bytes = [0u8; 64];
        bytes[0..32].copy_from_slice(hi.as_bytes());
        bytes[32..64].copy_from_slice(lo.as_bytes());
        Scalar::from_u512_mod_n(U512::from_big_endian(&bytes))
    }
}

/// Raw T=3 and T=4 Poseidon instances, for internal use.
///
/// Do not use this trait directly, refer to [`Poseidon1Hash`] and [`Poseidon2Hash`] instead.
pub trait PoseidonHasher<F: PrimeField> {
    fn hash_t3<I: IntoIterator<Item = F>>(inputs: I) -> F;
    fn hash_t4<I: IntoIterator<Item = F>>(inputs: I) -> F;
}

/// Implements [`PoseidonHasher`] with the Poseidon1 permutation.
///
/// For internal use. Do not use directly, refer to [`Poseidon1Hash`] instead.
#[derive(Debug, Default, Copy, Clone)]
pub struct Poseidon1Hasher<F: PrimeField> {
    _data: PhantomData<F>,
}

impl PoseidonHasher<Scalar> for Poseidon1Hasher<Scalar> {
    fn hash_t3<I: IntoIterator<Item = Scalar>>(inputs: I) -> Scalar {
        poseidon1::hash0::<poseidon1::BlueSkyConfig3, Scalar, 3, 2, 1>(inputs)
    }

    fn hash_t4<I: IntoIterator<Item = Scalar>>(inputs: I) -> Scalar {
        poseidon1::hash0::<poseidon1::BlueSkyConfig4, Scalar, 4, 3, 1>(inputs)
    }
}

/// Implements [`PoseidonHasher`] with the Poseidon2 permutation.
///
/// For internal use. Do not use directly, refer to [`Poseidon2Hash`] instead.
#[derive(Debug, Default, Copy, Clone)]
pub struct Poseidon2Hasher<F: PrimeField> {
    _data: PhantomData<F>,
}

impl PoseidonHasher<Scalar> for Poseidon2Hasher<Scalar> {
    fn hash_t3<I: IntoIterator<Item = Scalar>>(inputs: I) -> Scalar {
        poseidon2::hash0::<poseidon2::BlueSkyConfig3, Scalar, 3, 2, 1>(inputs)
    }

    fn hash_t4<I: IntoIterator<Item = Scalar>>(inputs: I) -> Scalar {
        poseidon2::hash0::<poseidon2::BlueSkyConfig4, Scalar, 4, 3, 1>(inputs)
    }
}

/// Generic Poseidon hash backend.
///
/// Do not use this directly, refer to the [`Poseidon1Hash`] and [`Poseidon2Hash`] instantiations
/// instead.
#[derive(Debug, Default, Copy, Clone)]
pub struct PoseidonHash<F: PrimeField, H: PoseidonHasher<F>> {
    _data: PhantomData<(F, H)>,
}

impl<H: PoseidonHasher<Scalar>> PoseidonHash<Scalar, H> {
    fn extract_scalar(h256: H256) -> Scalar {
        Scalar::try_from_be_bytes(h256.as_bytes()).unwrap()
    }
}

impl<H: PoseidonHasher<Scalar>> Hash<Scalar> for PoseidonHash<Scalar, H> {
    fn hash_two(input1: Scalar, input2: Scalar) -> H256 {
        let hash = H::hash_t3([input1, input2]);
        H256::from_slice(&hash.to_be_bytes())
    }

    fn hash_three(input1: Scalar, input2: Scalar, input3: Scalar) -> H256 {
        let hash = H::hash_t4([input1, input2, input3]);
        H256::from_slice(&hash.to_be_bytes())
    }

    fn hash_many<I: IntoIterator<Item = Scalar>>(inputs: I) -> H256 {
        let hash = H::hash_t4(inputs);
        H256::from_slice(&hash.to_be_bytes())
    }
}

impl<H: PoseidonHasher<Scalar>> Hash<H256> for PoseidonHash<Scalar, H> {
    fn hash_two(input1: H256, input2: H256) -> H256 {
        let hash = H::hash_t3([Self::extract_scalar(input1), Self::extract_scalar(input2)]);
        H256::from_slice(&hash.to_be_bytes())
    }

    fn hash_three(input1: H256, input2: H256, input3: H256) -> H256 {
        let hash = H::hash_t4([
            Self::extract_scalar(input1),
            Self::extract_scalar(input2),
            Self::extract_scalar(input3),
        ]);
        H256::from_slice(&hash.to_be_bytes())
    }

    fn hash_many<I: IntoIterator<Item = H256>>(inputs: I) -> H256 {
        let inputs: Vec<Scalar> = inputs.into_iter().map(Self::extract_scalar).collect();
        let hash = H::hash_t4(inputs);
        H256::from_slice(&hash.to_be_bytes())
    }
}

impl<H: PoseidonHasher<Scalar>> HashBackend<Scalar> for PoseidonHash<Scalar, H> {
    fn challenge<I: IntoIterator<Item = H256>>(dst: Scalar, transcript_hashes: I) -> Scalar {
        let transcript_hashes: Vec<Scalar> = transcript_hashes
            .into_iter()
            .map(Self::extract_scalar)
            .collect();
        let transcript_hash = Self::hash_many(
            std::iter::once(dst)
                .chain(std::iter::once(
                    Scalar::from(transcript_hashes.len() as u64),
                ))
                .chain(transcript_hashes),
        );
        H::hash_t3([dst, Self::extract_scalar(transcript_hash)])
    }
}

/// Poseidon1 hash backend.
pub type Poseidon1Hash<F> = PoseidonHash<F, Poseidon1Hasher<F>>;

/// Poseidon2 hash backend.
pub type Poseidon2Hash<F> = PoseidonHash<F, Poseidon2Hasher<F>>;

#[cfg(test)]
mod tests {
    use super::*;
    use starkom_bluesky::from_const;

    fn parse_hash(s: &'static str) -> H256 {
        s.parse().unwrap()
    }

    fn h256(value: usize) -> H256 {
        H256::from_slice(&Scalar::from(value as u64).to_be_bytes())
    }

    #[test]
    fn test_sha2_hash_two_scalars() {
        assert_eq!(
            Sha2Hash::<Scalar>::hash_two(from_const(12), from_const(34)),
            parse_hash("0xe7148fa58812b9dc9041188cf631e333ed79b8e203992fbfcf7beb201c7c13f7")
        );
        assert_eq!(
            Sha2Hash::<Scalar>::hash_two(from_const(12), from_const(56)),
            parse_hash("0x36f7f77cbfe046d965d34cc3bd42d86b2e0ac42fc902196cdd3ac8810f02e5f0")
        );
        assert_eq!(
            Sha2Hash::<Scalar>::hash_two(from_const(56), from_const(78)),
            parse_hash("0x7a83016495e114903621dc328b4bcfbf65aee3df18968ba278f8fc96283a2d92")
        );
    }

    #[test]
    fn test_sha2_hash_two_words() {
        assert_eq!(
            Sha2Hash::<Scalar>::hash_two(h256(12), h256(34)),
            parse_hash("0xe7148fa58812b9dc9041188cf631e333ed79b8e203992fbfcf7beb201c7c13f7")
        );
        assert_eq!(
            Sha2Hash::<Scalar>::hash_two(h256(12), h256(56)),
            parse_hash("0x36f7f77cbfe046d965d34cc3bd42d86b2e0ac42fc902196cdd3ac8810f02e5f0")
        );
        assert_eq!(
            Sha2Hash::<Scalar>::hash_two(h256(56), h256(78)),
            parse_hash("0x7a83016495e114903621dc328b4bcfbf65aee3df18968ba278f8fc96283a2d92")
        );
    }

    #[test]
    fn test_sha2_hash_three_scalars() {
        assert_eq!(
            Sha2Hash::<Scalar>::hash_three(from_const(12), from_const(34), from_const(56)),
            parse_hash("0xbd08c4c9e298576ab0486eec6348a32b299de885ae958e995a4b517f2697b842")
        );
        assert_eq!(
            Sha2Hash::<Scalar>::hash_three(from_const(12), from_const(56), from_const(34)),
            parse_hash("0xba4e56fdf167d335b75d22469b2e24407a8efbdb4c8112546aa1519d9175b7f9")
        );
        assert_eq!(
            Sha2Hash::<Scalar>::hash_three(from_const(56), from_const(78), from_const(90)),
            parse_hash("0x83b5d2c73f322fe3c806514e7cbb7009796e4b9ed15b4766ac6dc04a24533647")
        );
    }

    #[test]
    fn test_sha2_hash_three_words() {
        assert_eq!(
            Sha2Hash::<Scalar>::hash_three(h256(12), h256(34), h256(56)),
            parse_hash("0xbd08c4c9e298576ab0486eec6348a32b299de885ae958e995a4b517f2697b842")
        );
        assert_eq!(
            Sha2Hash::<Scalar>::hash_three(h256(12), h256(56), h256(34)),
            parse_hash("0xba4e56fdf167d335b75d22469b2e24407a8efbdb4c8112546aa1519d9175b7f9")
        );
        assert_eq!(
            Sha2Hash::<Scalar>::hash_three(h256(56), h256(78), h256(90)),
            parse_hash("0x83b5d2c73f322fe3c806514e7cbb7009796e4b9ed15b4766ac6dc04a24533647")
        );
    }

    #[test]
    fn test_sha2_hash_many_scalars() {
        assert_eq!(
            Sha2Hash::<Scalar>::hash_many([from_const(12)]),
            parse_hash("0xa82872b96246dac512ddf0515f5da862a92ecebebcb92537b6e3e73199694c45"),
        );
        assert_eq!(
            Sha2Hash::<Scalar>::hash_many([from_const(12), from_const(34)]),
            parse_hash("0xe7148fa58812b9dc9041188cf631e333ed79b8e203992fbfcf7beb201c7c13f7"),
        );
        assert_eq!(
            Sha2Hash::<Scalar>::hash_many([from_const(12), from_const(34), from_const(56)]),
            parse_hash("0xbd08c4c9e298576ab0486eec6348a32b299de885ae958e995a4b517f2697b842"),
        );
        assert_eq!(
            Sha2Hash::<Scalar>::hash_many([
                from_const(56),
                from_const(78),
                from_const(90),
                from_const(12),
                from_const(34)
            ]),
            parse_hash("0x4e8b5c5fe411c82f0449cead9ef8ae232d5dbb6bf91fefa0686f18062cc8f50e"),
        );
    }

    #[test]
    fn test_sha2_hash_many_words() {
        assert_eq!(
            Sha2Hash::<Scalar>::hash_many([h256(12)]),
            parse_hash("0xa82872b96246dac512ddf0515f5da862a92ecebebcb92537b6e3e73199694c45"),
        );
        assert_eq!(
            Sha2Hash::<Scalar>::hash_many([h256(12), h256(34)]),
            parse_hash("0xe7148fa58812b9dc9041188cf631e333ed79b8e203992fbfcf7beb201c7c13f7"),
        );
        assert_eq!(
            Sha2Hash::<Scalar>::hash_many([h256(12), h256(34), h256(56)]),
            parse_hash("0xbd08c4c9e298576ab0486eec6348a32b299de885ae958e995a4b517f2697b842"),
        );
        assert_eq!(
            Sha2Hash::<Scalar>::hash_many([h256(56), h256(78), h256(90), h256(12), h256(34)]),
            parse_hash("0x4e8b5c5fe411c82f0449cead9ef8ae232d5dbb6bf91fefa0686f18062cc8f50e"),
        );
    }

    #[test]
    fn test_poseidon1_hash_two_scalars() {
        assert_eq!(
            Poseidon1Hash::<Scalar>::hash_two(from_const(12), from_const(34)),
            parse_hash("0x45470d74563e5e49fe3bd2a161b36116e3c6a6a2f9c105bfe8c2599ff6116b06")
        );
        assert_eq!(
            Poseidon1Hash::<Scalar>::hash_two(from_const(12), from_const(56)),
            parse_hash("0x517c153fb7badd9aa6ac42a9528e58b48fb550f0d57d5aecdb2c09d355deb80d")
        );
        assert_eq!(
            Poseidon1Hash::<Scalar>::hash_two(from_const(56), from_const(78)),
            parse_hash("0x1ba4c686a3529d3bfc13890b2e1438b7adf780e2978cb2cabdd47653f402e8fe")
        );
    }

    #[test]
    fn test_poseidon1_hash_two_words() {
        assert_eq!(
            Poseidon1Hash::<Scalar>::hash_two(h256(12), h256(34)),
            parse_hash("0x45470d74563e5e49fe3bd2a161b36116e3c6a6a2f9c105bfe8c2599ff6116b06")
        );
        assert_eq!(
            Poseidon1Hash::<Scalar>::hash_two(h256(12), h256(56)),
            parse_hash("0x517c153fb7badd9aa6ac42a9528e58b48fb550f0d57d5aecdb2c09d355deb80d")
        );
        assert_eq!(
            Poseidon1Hash::<Scalar>::hash_two(h256(56), h256(78)),
            parse_hash("0x1ba4c686a3529d3bfc13890b2e1438b7adf780e2978cb2cabdd47653f402e8fe")
        );
    }

    #[test]
    fn test_poseidon1_hash_three_scalars() {
        assert_eq!(
            Poseidon1Hash::<Scalar>::hash_three(from_const(12), from_const(34), from_const(56)),
            parse_hash("0x1125d1d7bcc64d065695f306f08db087abc90d214fd982461296e607de7d4d49")
        );
        assert_eq!(
            Poseidon1Hash::<Scalar>::hash_three(from_const(12), from_const(56), from_const(34)),
            parse_hash("0x71c3d9b1e8a0e5102b7c56a20d1bc7a838f90881529e4132798df25f18980546")
        );
        assert_eq!(
            Poseidon1Hash::<Scalar>::hash_three(from_const(56), from_const(78), from_const(90)),
            parse_hash("0x516b43041b6e111a7be5670972354589d8686593fbd2a994e14c53e55bb803cd")
        );
    }

    #[test]
    fn test_poseidon1_hash_three_words() {
        assert_eq!(
            Poseidon1Hash::<Scalar>::hash_three(h256(12), h256(34), h256(56)),
            parse_hash("0x1125d1d7bcc64d065695f306f08db087abc90d214fd982461296e607de7d4d49")
        );
        assert_eq!(
            Poseidon1Hash::<Scalar>::hash_three(h256(12), h256(56), h256(34)),
            parse_hash("0x71c3d9b1e8a0e5102b7c56a20d1bc7a838f90881529e4132798df25f18980546")
        );
        assert_eq!(
            Poseidon1Hash::<Scalar>::hash_three(h256(56), h256(78), h256(90)),
            parse_hash("0x516b43041b6e111a7be5670972354589d8686593fbd2a994e14c53e55bb803cd")
        );
    }

    #[test]
    fn test_poseidon1_hash_many_scalars() {
        assert_eq!(
            Poseidon1Hash::<Scalar>::hash_many([from_const(12)]),
            parse_hash("0x7c8412cec3a535ff1ceb3ed699e7a26f4bfdb11008ab52acdda94d8908e94232"),
        );
        assert_eq!(
            Poseidon1Hash::<Scalar>::hash_many([from_const(12), from_const(34)]),
            parse_hash("0x6fda95845c8b151a4ec3317c5023a03da1e4d8404d1d8442897978cf7597cedb"),
        );
        assert_eq!(
            Poseidon1Hash::<Scalar>::hash_many([from_const(12), from_const(34), from_const(56)]),
            parse_hash("0x1125d1d7bcc64d065695f306f08db087abc90d214fd982461296e607de7d4d49"),
        );
        assert_eq!(
            Poseidon1Hash::<Scalar>::hash_many([
                from_const(56),
                from_const(78),
                from_const(90),
                from_const(12),
                from_const(34)
            ]),
            parse_hash("0x7f11ea73b86da117580a88152e63526cce28e51a7b2f4603307d2404a19dc663"),
        );
    }

    #[test]
    fn test_poseidon1_hash_many_words() {
        assert_eq!(
            Poseidon1Hash::<Scalar>::hash_many([h256(12)]),
            parse_hash("0x7c8412cec3a535ff1ceb3ed699e7a26f4bfdb11008ab52acdda94d8908e94232"),
        );
        assert_eq!(
            Poseidon1Hash::<Scalar>::hash_many([h256(12), h256(34)]),
            parse_hash("0x6fda95845c8b151a4ec3317c5023a03da1e4d8404d1d8442897978cf7597cedb"),
        );
        assert_eq!(
            Poseidon1Hash::<Scalar>::hash_many([h256(12), h256(34), h256(56)]),
            parse_hash("0x1125d1d7bcc64d065695f306f08db087abc90d214fd982461296e607de7d4d49"),
        );
        assert_eq!(
            Poseidon1Hash::<Scalar>::hash_many([h256(56), h256(78), h256(90), h256(12), h256(34)]),
            parse_hash("0x7f11ea73b86da117580a88152e63526cce28e51a7b2f4603307d2404a19dc663"),
        );
    }

    #[test]
    fn test_poseidon2_hash_two_scalars() {
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_two(from_const(12), from_const(34)),
            parse_hash("0x165e74be18ef4be6de5e232cd3480dcc38176807ac918b904576964612c5b6de")
        );
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_two(from_const(12), from_const(56)),
            parse_hash("0x5ad01fd39362490ad40fa8a2692c0f90ffed1c29bdc394366d9b54489d9df2d3")
        );
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_two(from_const(56), from_const(78)),
            parse_hash("0x38e7bb7b6ccae0c74031423877db058f4ab3a284964e2d91bf97497851eca5db")
        );
    }

    #[test]
    fn test_poseidon2_hash_two_words() {
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_two(h256(12), h256(34)),
            parse_hash("0x165e74be18ef4be6de5e232cd3480dcc38176807ac918b904576964612c5b6de")
        );
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_two(h256(12), h256(56)),
            parse_hash("0x5ad01fd39362490ad40fa8a2692c0f90ffed1c29bdc394366d9b54489d9df2d3")
        );
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_two(h256(56), h256(78)),
            parse_hash("0x38e7bb7b6ccae0c74031423877db058f4ab3a284964e2d91bf97497851eca5db")
        );
    }

    #[test]
    fn test_poseidon2_hash_three_scalars() {
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_three(from_const(12), from_const(34), from_const(56)),
            parse_hash("0x236092ebefc7e6565e0e75414d8fdce1ce2e19bb59002d36b794b9c3111bb9cd")
        );
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_three(from_const(12), from_const(56), from_const(34)),
            parse_hash("0x0ae96711384158467002a664233e687ce6f3c0f27ac41ea97ac378c72ad4077e")
        );
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_three(from_const(56), from_const(78), from_const(90)),
            parse_hash("0x2fa39a3a76d0cf8220bd6f9899b209110ad1cca7b0bdc2b340661fa7063f2ba0")
        );
    }

    #[test]
    fn test_poseidon2_hash_three_words() {
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_three(h256(12), h256(34), h256(56)),
            parse_hash("0x236092ebefc7e6565e0e75414d8fdce1ce2e19bb59002d36b794b9c3111bb9cd")
        );
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_three(h256(12), h256(56), h256(34)),
            parse_hash("0x0ae96711384158467002a664233e687ce6f3c0f27ac41ea97ac378c72ad4077e")
        );
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_three(h256(56), h256(78), h256(90)),
            parse_hash("0x2fa39a3a76d0cf8220bd6f9899b209110ad1cca7b0bdc2b340661fa7063f2ba0")
        );
    }

    #[test]
    fn test_poseidon2_hash_many_scalars() {
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_many([from_const(12)]),
            parse_hash("0x45782306ba3302ebe2f07eacbf5d0c36a5f307dc1cde4f3f9e8196ef498eddf2"),
        );
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_many([from_const(12), from_const(34)]),
            parse_hash("0x08802dc1d5eaf75680808adb1d19bb420f34f5e786f09e05ffa5d41fb2bdfe6d"),
        );
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_many([from_const(12), from_const(34), from_const(56)]),
            parse_hash("0x236092ebefc7e6565e0e75414d8fdce1ce2e19bb59002d36b794b9c3111bb9cd"),
        );
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_many([
                from_const(56),
                from_const(78),
                from_const(90),
                from_const(12),
                from_const(34)
            ]),
            parse_hash("0x7499d072269d7c32ad0477050bacd7cc84009b845c016280d679bc7849ed845a"),
        );
    }

    #[test]
    fn test_poseidon2_hash_many_words() {
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_many([h256(12)]),
            parse_hash("0x45782306ba3302ebe2f07eacbf5d0c36a5f307dc1cde4f3f9e8196ef498eddf2"),
        );
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_many([h256(12), h256(34)]),
            parse_hash("0x08802dc1d5eaf75680808adb1d19bb420f34f5e786f09e05ffa5d41fb2bdfe6d"),
        );
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_many([h256(12), h256(34), h256(56)]),
            parse_hash("0x236092ebefc7e6565e0e75414d8fdce1ce2e19bb59002d36b794b9c3111bb9cd"),
        );
        assert_eq!(
            Poseidon2Hash::<Scalar>::hash_many([h256(56), h256(78), h256(90), h256(12), h256(34)]),
            parse_hash("0x7499d072269d7c32ad0477050bacd7cc84009b845c016280d679bc7849ed845a"),
        );
    }
}
