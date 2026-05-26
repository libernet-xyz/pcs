# Overview

This crate contains Starkom's quantum-resistant polynomial commitment scheme, a DEEP-FRI
implementation.

It currently works on the [BlueSky](https://docs.rs/starkom-bluesky) field only.

Two different hash backends are provided: one based on SHA-256 for fast verification on the EVM and
one based on [Poseidon2](https://docs.rs/starkom-poseidon2) for efficient recursion.

The SHA-256 hash backend is compatible with our
[EVM verifier](https://github.com/libernet-xyz/evm-verifier).
