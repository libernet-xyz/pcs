use crate::utils;
use starkom_bluesky::Scalar;
use std::sync::LazyLock;

/// Target security level in bits.
pub const LAMBDA: usize = 128;

/// Domain separator tag for the Fiat-Shamir challenge used to derive query indices.
static QUERY_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/pcs/query"));

/// Domain separator tag for the Fiat-Shamir challenge used to build the random linear combination.
static RLC_DST: LazyLock<Scalar> = LazyLock::new(|| utils::hash_to_scalar(b"starkom/pcs/rlc"));

/// Returns the number of FRI queries required to achieve 128-bit security using a blowup factor of
/// `2^blowup_log2` when opening `num_points` evaluation points.
fn num_queries(blowup_log2: usize, num_points: usize) -> usize {
    let extra = num_points.next_power_of_two().trailing_zeros() as usize;
    (LAMBDA + extra).div_ceil(blowup_log2)
}

// TODO

#[cfg(test)]
mod tests {
    use super::*;

    // TODO
}
