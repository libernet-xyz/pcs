# DEEP-FRI

[![CI](https://img.shields.io/github/actions/workflow/status/libernet-xyz/pcs/ci.yml?label=CI)](https://github.com/libernet-xyz/pcs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/starkom-pcs)](https://crates.io/crates/starkom-pcs)
[![license](https://img.shields.io/crates/l/starkom-pcs)](https://github.com/libernet-xyz/pcs/blob/main/LICENSE)

## Overview

This crate contains Starkom's quantum-resistant polynomial commitment scheme, a DEEP-FRI
implementation that works with any prime field with sufficient 2-adicity.

Starkom's zkSTARK suite currently provides three fields and all work correctly with this PCS: the [BLS12-381 scalar field][bls12-381], [BlueSky][bluesky], and [Goldilocks][goldilocks].

Two hash backends are provided, one using SHA2-256 and one using Keccak-256, and both are
implemented in the most EVM-friendly possible way. Check out Starkom's [EVM verifier][evm-verifier].

[bls12-381]: https://docs.rs/starkom-ff/latest/starkom_ff/bls12_381/struct.Scalar.html
[bluesky]: https://docs.rs/starkom-bluesky
[evm-verifier]: https://github.com/libernet-xyz/evm-verifier
[goldilocks]: https://docs.rs/starkom-goldilocks
