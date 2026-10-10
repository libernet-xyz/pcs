# DEEP-FRI

[![CI](https://img.shields.io/github/actions/workflow/status/libernet-xyz/pcs/ci.yml?label=CI)](https://github.com/libernet-xyz/pcs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/starkom-pcs)](https://crates.io/crates/starkom-pcs)
[![license](https://img.shields.io/crates/l/starkom-pcs)](https://github.com/libernet-xyz/pcs/blob/main/LICENSE)

## Overview

This crate contains Starkom's quantum-resistant polynomial commitment scheme, a DEEP-FRI
implementation that works with any field with sufficient 2-adicity.

> [!NOTE]
> This PCS is **not** perfectly zero-knowledge out of the box: even though all committed polynomials
> are automatically shifted to a coset of the evaluation domain so that none of its Merkle queries
> reveal any live domain locations, the queries can still leak enough information for an external
> observer to interpolate the original polynomials.
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

> [!NOTE]
> Honest provers commit polynomials of degree strictly less than the degree bound, but a valid proof
> only guarantees a degree at most equal to it. This slack is common to DEEP-FRI implementations and
> harmless for PLONK and AIR ([`starkom-plonk`][starkom-plonk] is unaffected), but protocols whose
> soundness depends on the exact degree bound must take it into account. See
> [`Proof::verify`][proof-verify] for details.

This PCS is field-agnostic, the only requirement is that the field implements the
[`Field256` trait from the `starkom-ff` crate][field256]. Starkom's zkSTARK suite currently provides
five fields and all work correctly with this PCS: the [BLS12-381 scalar field][bls12-381],
[BlueSky][bluesky], [Schraderbrau][schraderbrau], [Goldilocks][goldilocks], and
[KoalaBear][koalabear]. Note that Goldilocks and KoalaBear are small fields but you can use their
256-bit extensions, respectively [`GL4`][gl4] and [`KB8`][kb8].

Two hash backends are provided, one using SHA2-256 and one using Keccak-256, and both are
implemented in the most EVM-friendly possible way. Check out Starkom's [EVM verifier][evm-verifier].

[bls12-381]: https://docs.rs/starkom-ff/latest/starkom_ff/bls12_381/struct.Scalar.html
[bluesky]: https://docs.rs/starkom-bluesky
[evm-verifier]: https://github.com/libernet-xyz/evm-verifier
[field256]: https://docs.rs/starkom-ff/latest/starkom_ff/trait.Field256.html
[gl4]: https://docs.rs/starkom-goldilocks/latest/starkom_goldilocks/gl4/struct.Scalar.html
[goldilocks]: https://docs.rs/starkom-goldilocks
[kb8]: https://docs.rs/starkom-koalabear/latest/starkom_koalabear/kb8/struct.Scalar.html
[koalabear]: https://docs.rs/starkom-koalabear
[num_queries]: https://docs.rs/starkom-pcs/latest/starkom_pcs/fn.num_queries.html
[plonk-options-blind]: https://docs.rs/starkom-plonk/latest/starkom_plonk/struct.Options.html#structfield.blind
[proof-verify]: https://docs.rs/starkom-pcs/latest/starkom_pcs/struct.Proof.html#method.verify
[schraderbrau]: https://docs.rs/starkom-schraderbrau
[starkom-plonk]: https://docs.rs/starkom-plonk
