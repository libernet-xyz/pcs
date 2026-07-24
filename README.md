# DEEP-FRI

[![CI](https://img.shields.io/github/actions/workflow/status/libernet-xyz/pcs/ci.yml?label=CI)](https://github.com/libernet-xyz/pcs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/starkom-pcs)](https://crates.io/crates/starkom-pcs)
[![license](https://img.shields.io/crates/l/starkom-pcs)](https://github.com/libernet-xyz/pcs/blob/main/LICENSE)

## Overview

This crate contains Starkom's quantum-resistant polynomial commitment scheme, a DEEP-FRI
implementation.

It currently works on the [BlueSky](https://docs.rs/starkom-bluesky) field only.

Three different hash backends are provided: one based on SHA-256 for fast verification on the EVM,
one based on [Poseidon](https://docs.rs/starkom-poseidon), and one based on
[Poseidon2](https://docs.rs/starkom-poseidon2) for efficient recursion.

The SHA-256 hash backend is compatible with our
[EVM verifier](https://github.com/libernet-xyz/evm-verifier).
