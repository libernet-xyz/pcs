//! Encoding of commitments and proofs in the Starkom Compact Proof Format (SCPF).
//!
//! SCPF is the binary format of the proofs sent to EVM verifiers, and it's specified in
//! [`FORMAT.md`](https://github.com/libernet-xyz/scpf/blob/main/FORMAT.md). SCPF data is made of a
//! 64-byte header, a pool of distinct 32-byte words, and a stream of items:
//!
//! - **varints**, i.e. unsigned integers of up to 64 bits encoded in 1 to 9 bytes (see
//!   [`encode_varint`]);
//! - **inline words**, i.e. 32-byte words stored directly in the stream, which hold field elements
//!   as big-endian integers;
//! - **pool references**, i.e. the varint positions of 32-byte words in the pool, which hold
//!   hashes.
//!
//! The stream has no tags or lengths other than those that the decoder can't derive, so the layout
//! of the stream of every encoded message is defined by its encoder (see
//! [`Commitment::to_scpf`](`crate::Commitment::to_scpf`) and
//! [`Proof::to_scpf`](`crate::Proof::to_scpf`)).

use crate::utils;
use primitive_types::H256;
use sha3::Digest;
use starkom_ff::PrimeField256;
use std::cmp::Reverse;
use std::collections::BTreeMap;

/// The signature at the start of the header: the ASCII characters `SCPF`.
const SIGNATURE: [u8; 4] = *b"SCPF";

/// Parses a part of the version number of the crate at compile time.
///
/// Fails the build if the part doesn't fit in a byte, which is the size of its header field.
const fn parse_version_part(digits: &str) -> u8 {
    match u8::from_str_radix(digits, 10) {
        Ok(part) => part,
        Err(_) => panic!("a part of the crate version doesn't fit in the SCPF header"),
    }
}

/// The version of the header: the major, minor, and patch version numbers of this crate.
///
/// SCPF is specific to this crate, so there's no separate version for the format.
pub(crate) const VERSION: [u8; 3] = [
    parse_version_part(env!("CARGO_PKG_VERSION_MAJOR")),
    parse_version_part(env!("CARGO_PKG_VERSION_MINOR")),
    parse_version_part(env!("CARGO_PKG_VERSION_PATCH")),
];

/// The size of the header in bytes.
const HEADER_SIZE: usize = 64;

/// Returns the schema identifier of SCPF data whose root message is the protobuf message called
/// `full_name`, e.g. `starkom.proto.pcs.v10.Proof`: the Keccak-256 hash of the full name.
///
/// This is the hash of Ethereum's `KECCAK256` instruction rather than SHA3-256 (FIPS 202), so a
/// Solidity decoder can derive the identifier at compile time with `keccak256("...")`.
pub(crate) fn schema_identifier(full_name: &str) -> H256 {
    let mut hasher = sha3::Keccak256::new();
    hasher.update(full_name.as_bytes());
    H256::from_slice(hasher.finalize().as_slice())
}

/// Appends the varint encoding of `value` to `output`.
///
/// The number of leading one bits of the first byte is the number of bytes that follow it, and the
/// value is stored in big-endian order in the remaining bits: `n` bytes hold `7 * n` bits for `n`
/// up to 8, and values of 57 bits or more take 9 bytes, `0xff` followed by the 64-bit value. This
/// function always uses the shortest encoding.
fn encode_varint(value: u64, output: &mut Vec<u8>) {
    match (1..=8).find(|length| value >> (7 * length) == 0) {
        Some(length) => {
            let prefix = ((1u64 << (length - 1)) - 1) << (7 * length + 1);
            output.extend_from_slice(&(prefix | value).to_be_bytes()[(8 - length)..]);
        }
        None => {
            output.push(0xff);
            output.extend_from_slice(&value.to_be_bytes());
        }
    }
}

/// An item of the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Item {
    /// An unsigned integer, encoded as a varint.
    Varint(u64),
    /// A 32-byte word stored in the stream.
    InlineWord(H256),
    /// A 32-byte word stored in the pool, encoded in the stream as a pool reference.
    PoolWord(H256),
}

/// Encodes SCPF data.
///
/// Callers push the items of the stream in order and then call [`Self::finish`], which builds the
/// pool and returns the data.
#[derive(Debug, Default)]
pub(crate) struct Encoder {
    items: Vec<Item>,
}

impl Encoder {
    /// Pushes an unsigned integer, encoded as a varint.
    pub(crate) fn push_varint(&mut self, value: u64) {
        self.items.push(Item::Varint(value));
    }

    /// Pushes a `usize`, encoded as a varint.
    pub(crate) fn push_usize(&mut self, value: usize) {
        self.push_varint(value as u64);
    }

    /// Pushes a boolean, encoded as a varint: 1 for true and 0 for false.
    pub(crate) fn push_bool(&mut self, value: bool) {
        self.push_varint(value.into());
    }

    /// Pushes a field element, encoded as an inline word holding its integer value in big-endian
    /// byte order (see [`utils::encode_scalar`]).
    pub(crate) fn push_scalar<F: PrimeField256>(&mut self, value: F) {
        self.items
            .push(Item::InlineWord(utils::encode_scalar(value)));
    }

    /// Pushes field elements in order, each one encoded as by [`Self::push_scalar`].
    pub(crate) fn push_scalars<F: PrimeField256>(&mut self, values: &[F]) {
        for &value in values {
            self.push_scalar(value);
        }
    }

    /// Pushes a hash, stored in the pool and encoded as a pool reference.
    pub(crate) fn push_hash(&mut self, hash: H256) {
        self.items.push(Item::PoolWord(hash));
    }

    /// Pushes hashes in order, each one encoded as by [`Self::push_hash`].
    pub(crate) fn push_hashes(&mut self, hashes: &[H256]) {
        for &hash in hashes {
            self.push_hash(hash);
        }
    }

    /// Returns the SCPF data, with the given schema identifier in the header.
    ///
    /// The pool holds every distinct word referenced by the stream once, sorted by decreasing
    /// number of references so that the most referenced words get the shortest pool references.
    /// Words with the same number of references are sorted by increasing value.
    ///
    /// Panics if the pool has more than `2^32 - 1` words, which wouldn't fit in the header.
    pub(crate) fn finish(self, schema_identifier: H256) -> Vec<u8> {
        let mut counts: BTreeMap<H256, usize> = BTreeMap::new();
        for item in &self.items {
            if let Item::PoolWord(word) = item {
                *counts.entry(*word).or_default() += 1;
            }
        }
        // The keys are sorted by value and the sort is stable, so equal counts keep that order.
        let mut pool: Vec<H256> = counts.keys().copied().collect();
        pool.sort_by_key(|word| Reverse(counts[word]));
        let positions: BTreeMap<H256, u64> = pool
            .iter()
            .enumerate()
            .map(|(position, word)| (*word, position as u64))
            .collect();
        let pool_size = u32::try_from(pool.len()).expect("the SCPF pool is too large");

        let mut data = Vec::with_capacity(HEADER_SIZE + pool.len() * 32);
        data.extend_from_slice(&SIGNATURE);
        data.extend_from_slice(&[0; 4]);
        data.extend_from_slice(&VERSION);
        data.extend_from_slice(&[0; 5]);
        data.extend_from_slice(schema_identifier.as_bytes());
        data.extend_from_slice(&pool_size.to_be_bytes());
        data.extend_from_slice(&[0; 12]);
        for word in &pool {
            data.extend_from_slice(word.as_bytes());
        }
        for item in &self.items {
            match item {
                Item::Varint(value) => encode_varint(*value, &mut data),
                Item::InlineWord(word) => data.extend_from_slice(word.as_bytes()),
                Item::PoolWord(word) => encode_varint(positions[word], &mut data),
            }
        }
        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use starkom_bluesky::Scalar as BS;
    use starkom_ff::Field;

    fn parse(s: &'static str) -> H256 {
        s.parse().unwrap()
    }

    /// Returns the header of SCPF data with the given schema identifier and pool size.
    fn header(schema_identifier: H256, pool_size: u32) -> Vec<u8> {
        [
            &b"SCPF"[..],
            &[0, 0, 0, 0][..],
            &VERSION[..],
            &[0, 0, 0, 0, 0][..],
            schema_identifier.as_bytes(),
            &pool_size.to_be_bytes()[..],
            &[0; 12][..],
        ]
        .concat()
    }

    #[test]
    fn test_version() {
        assert_eq!(VERSION, [10, 1, 0]);
    }

    #[test]
    fn test_schema_identifier_of_empty_name() {
        assert_eq!(
            schema_identifier(""),
            parse("0xc5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470")
        );
    }

    #[test]
    fn test_schema_identifier() {
        assert_eq!(
            schema_identifier("starkom.proto.pcs.v10.Proof"),
            parse("0xb7fcbc946ada95241d640499d14b4c6824a850187a7cd259c954bb722be8c086")
        );
    }

    fn test_encode_varint_impl(value: u64, expected: &[u8]) {
        let mut output = vec![];
        encode_varint(value, &mut output);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_encode_varint() {
        test_encode_varint_impl(0, &[0x00]);
        test_encode_varint_impl(1, &[0x01]);
        test_encode_varint_impl(127, &[0x7f]);
        test_encode_varint_impl(128, &[0x80, 0x80]);
        test_encode_varint_impl(300, &[0x81, 0x2c]);
        test_encode_varint_impl(16383, &[0xbf, 0xff]);
        test_encode_varint_impl(16384, &[0xc0, 0x40, 0x00]);
        test_encode_varint_impl((1 << 21) - 1, &[0xdf, 0xff, 0xff]);
        test_encode_varint_impl(1 << 21, &[0xe0, 0x20, 0x00, 0x00]);
        test_encode_varint_impl((1 << 28) - 1, &[0xef, 0xff, 0xff, 0xff]);
        test_encode_varint_impl(1 << 28, &[0xf0, 0x10, 0x00, 0x00, 0x00]);
        test_encode_varint_impl((1 << 35) - 1, &[0xf7, 0xff, 0xff, 0xff, 0xff]);
        test_encode_varint_impl(1 << 35, &[0xf8, 0x08, 0x00, 0x00, 0x00, 0x00]);
        test_encode_varint_impl((1 << 42) - 1, &[0xfb, 0xff, 0xff, 0xff, 0xff, 0xff]);
        test_encode_varint_impl(1 << 42, &[0xfc, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00]);
        test_encode_varint_impl((1 << 49) - 1, &[0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
        test_encode_varint_impl(1 << 49, &[0xfe, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        test_encode_varint_impl(
            (1 << 56) - 1,
            &[0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
        );
        test_encode_varint_impl(1 << 56, &[0xff, 0x01, 0, 0, 0, 0, 0, 0, 0]);
        test_encode_varint_impl(u64::MAX, &[0xff; 9]);
    }

    #[test]
    fn test_empty_data() {
        let schema_identifier =
            parse("0x0102030405060708091011121314151617181920212223242526272829303132");
        let encoder = Encoder::default();
        let data = encoder.finish(schema_identifier);
        assert_eq!(data, header(schema_identifier, 0));
    }

    #[test]
    fn test_varints() {
        let schema_identifier = H256::repeat_byte(0x42);
        let mut encoder = Encoder::default();
        encoder.push_varint(300);
        encoder.push_usize(7);
        encoder.push_bool(true);
        encoder.push_bool(false);
        encoder.push_varint(u64::MAX);
        let data = encoder.finish(schema_identifier);
        assert_eq!(
            data,
            [
                &header(schema_identifier, 0)[..],
                &[0x81, 0x2c, 0x07, 0x01, 0x00][..],
                &[0xff; 9][..],
            ]
            .concat()
        );
    }

    #[test]
    fn test_scalars() {
        let schema_identifier = H256::repeat_byte(0x42);
        let mut encoder = Encoder::default();
        encoder.push_scalar(BS::from_const(0x1234));
        encoder.push_scalars(&[BS::ZERO, -BS::ONE]);
        let data = encoder.finish(schema_identifier);
        assert_eq!(
            data,
            [
                &header(schema_identifier, 0)[..],
                parse("0x0000000000000000000000000000000000000000000000000000000000001234")
                    .as_bytes(),
                parse("0x0000000000000000000000000000000000000000000000000000000000000000")
                    .as_bytes(),
                parse("0x7ffffffffffffffffffffffffffffffe0673ddf29e9b5547c000000000000000")
                    .as_bytes(),
            ]
            .concat()
        );
    }

    #[test]
    fn test_hashes() {
        let schema_identifier = H256::repeat_byte(0x42);
        let mut encoder = Encoder::default();
        encoder.push_hash(H256::repeat_byte(0xaa));
        encoder.push_hashes(&[H256::repeat_byte(0xbb), H256::repeat_byte(0xcc)]);
        let data = encoder.finish(schema_identifier);
        assert_eq!(
            data,
            [
                &header(schema_identifier, 3)[..],
                H256::repeat_byte(0xaa).as_bytes(),
                H256::repeat_byte(0xbb).as_bytes(),
                H256::repeat_byte(0xcc).as_bytes(),
                &[0, 1, 2][..],
            ]
            .concat()
        );
    }

    #[test]
    fn test_pool_order() {
        let schema_identifier = H256::repeat_byte(0x42);
        let mut encoder = Encoder::default();
        encoder.push_hashes(&[
            H256::repeat_byte(0xdd),
            H256::repeat_byte(0xbb),
            H256::repeat_byte(0xcc),
            H256::repeat_byte(0xaa),
            H256::repeat_byte(0xcc),
            H256::repeat_byte(0xbb),
            H256::repeat_byte(0xcc),
        ]);
        let data = encoder.finish(schema_identifier);
        // 0xcc has three references, 0xbb two, and 0xdd and 0xaa one each (0xaa is smaller).
        assert_eq!(
            data,
            [
                &header(schema_identifier, 4)[..],
                H256::repeat_byte(0xcc).as_bytes(),
                H256::repeat_byte(0xbb).as_bytes(),
                H256::repeat_byte(0xaa).as_bytes(),
                H256::repeat_byte(0xdd).as_bytes(),
                &[3, 1, 0, 2, 0, 1, 0][..],
            ]
            .concat()
        );
    }

    #[test]
    fn test_long_pool_reference() {
        let schema_identifier = H256::repeat_byte(0x42);
        let mut encoder = Encoder::default();
        let hashes: Vec<H256> = (0..129u64).map(|i| utils::encode_u64(i + 1)).collect();
        encoder.push_hashes(&hashes);
        let data = encoder.finish(schema_identifier);
        assert_eq!(data.len(), 64 + 129 * 32 + 128 + 2);
        assert_eq!(&data[48..52], &[0, 0, 0, 129]);
        assert_eq!(
            &data[(64 + 128 * 32)..(64 + 129 * 32)],
            hashes[128].as_bytes()
        );
        assert_eq!(&data[(64 + 129 * 32 + 127)..], &[0x7f, 0x80, 0x80]);
    }

    #[test]
    fn test_mixed_items() {
        let schema_identifier = H256::repeat_byte(0x42);
        let mut encoder = Encoder::default();
        encoder.push_usize(2);
        encoder.push_hash(H256::repeat_byte(0xaa));
        encoder.push_scalar(BS::from_const(5));
        encoder.push_hash(H256::repeat_byte(0xaa));
        let data = encoder.finish(schema_identifier);
        assert_eq!(
            data,
            [
                &header(schema_identifier, 1)[..],
                H256::repeat_byte(0xaa).as_bytes(),
                &[2, 0][..],
                parse("0x0000000000000000000000000000000000000000000000000000000000000005")
                    .as_bytes(),
                &[0][..],
            ]
            .concat()
        );
    }
}
