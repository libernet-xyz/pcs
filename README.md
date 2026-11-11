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
> To make it fully zero-knowledge you need to do the following.
>
> 1. Add at least [`num_queries`][num_queries] + T \* (num_opened_points + 1) uniformly random
>    elements to every polynomial, where T is the [`EXTENSION_DEGREE`][extension-degree] of the
>    field you're using. Every query reveals one element of every polynomial, and every opened point
>    reveals T elements because it's in the 256-bit field. The remaining T elements add ~256 bits of
>    excess entropy: the Merkle roots and paths are deterministic hashes of all evaluations, so with
>    less excess entropy an observer could guess a low-entropy polynomial and check the guess
>    against them by brute force.
> 2. Make sure that every Merkle leaf has at least ~256 bits of entropy, because the Merkle paths
>    reveal the hashes of unopened leaves. A polynomial over the 256-bit field is enough on its own,
>    but a batch of polynomials over the base field needs at least T of them. If a batch has fewer,
>    add uniformly random polynomials to it until it has T. Their coefficients must be uniformly
>    random up to the degree bound (e.g. random evaluations at every point of the evaluation
>    domain), because zero or otherwise predictable polynomials would leave the leaf hashes open to
>    brute force.
> 3. Commit one extra uniformly random polynomial to mask the FRI query siblings in the folding
>    argument.
>
> All these measures only account for a single proof. Every proof reveals more evaluations of the
> committed polynomials, so a zero-knowledge commitment must be opened only once.
>
> The [`starkom-plonk`][starkom-plonk] crate does all of the above when its
> [blinding system][plonk-options-blind] is enabled. A PLONK proof carries both a PCS commitment and
> a PCS proof, both fresh, so PLONK opens every commitment only once.

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

All Fiat-Shamir challenges and opened points are elements of the 256-bit field, but the committed
polynomials can be defined either over the 256-bit field or over its base field, one batch at a
time (see [`Committer::add_batch`][committer-add-batch] and
[`Committer::add_base_batch`][committer-add-base-batch]). Polynomials over the base field are
cheaper to commit and open, and a valid proof guarantees that they're actually over the base field.

Two hash backends are provided, one using SHA2-256 and one using Keccak-256, and both are
implemented in the most EVM-friendly possible way. Check out Starkom's [EVM verifier][evm-verifier].

[bls12-381]: https://docs.rs/starkom-ff/latest/starkom_ff/bls12_381/struct.Scalar.html
[bluesky]: https://docs.rs/starkom-bluesky
[committer-add-base-batch]: https://docs.rs/starkom-pcs/latest/starkom_pcs/struct.Committer.html#method.add_base_batch
[committer-add-batch]: https://docs.rs/starkom-pcs/latest/starkom_pcs/struct.Committer.html#method.add_batch
[evm-verifier]: https://github.com/libernet-xyz/evm-verifier
[extension-degree]: https://docs.rs/starkom-ff/latest/starkom_ff/trait.Field.html#associatedconstant.EXTENSION_DEGREE
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
