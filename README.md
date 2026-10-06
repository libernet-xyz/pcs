# DEEP-FRI

[![CI](https://img.shields.io/github/actions/workflow/status/libernet-xyz/pcs/ci.yml?label=CI)](https://github.com/libernet-xyz/pcs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/starkom-pcs)](https://crates.io/crates/starkom-pcs)
[![license](https://img.shields.io/crates/l/starkom-pcs)](https://github.com/libernet-xyz/pcs/blob/main/LICENSE)

## Overview

This crate contains Starkom's quantum-resistant polynomial commitment scheme, a DEEP-FRI
implementation that works with any field with sufficient 2-adicity.

> [!NOTE]
> This PCS is **not** zero-knowledge: even though all committed polynomials are automatically
> shifted to a coset of the evaluation domain so that none of its Merkle queries reveal any live
> domain locations, the queries can still leak enough information for an external observer to
> interpolate the original polynomials.
>
> To make it fully zero-knowledge you need to:
>
> 1. add at least [`num_queries`][num_queries] + 1 uniformly random elements to every polynomial to
>    compensate for the information leak,
> 2. commit one extra uniformly random polynomial to mask the FRI query siblings in the folding
>    argument.
>
> The [`starkom-plonk`][starkom-plonk] crate does both when its
> [blinding system][plonk-options-blind] is enabled.

Starkom's zkSTARK suite currently provides five fields and all work correctly with this PCS: the
[BLS12-381 scalar field][bls12-381], [BlueSky][bluesky], [Schraderbrau][schraderbrau],
[Goldilocks][goldilocks], and [KoalaBear][koalabear].

Two hash backends are provided, one using SHA2-256 and one using Keccak-256, and both are
implemented in the most EVM-friendly possible way. Check out Starkom's [EVM verifier][evm-verifier].

[bls12-381]: https://docs.rs/starkom-ff/latest/starkom_ff/bls12_381/struct.Scalar.html
[bluesky]: https://docs.rs/starkom-bluesky
[evm-verifier]: https://github.com/libernet-xyz/evm-verifier
[goldilocks]: https://docs.rs/starkom-goldilocks
[koalabear]: https://docs.rs/starkom-koalabear
[num_queries]: https://docs.rs/starkom-pcs/latest/starkom_pcs/fn.num_queries.html
[plonk-options-blind]: https://docs.rs/starkom-plonk/latest/starkom_plonk/struct.Options.html#structfield.blind
[schraderbrau]: https://docs.rs/starkom-schraderbrau
[starkom-plonk]: https://docs.rs/starkom-plonk
