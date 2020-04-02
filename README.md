![Rust](https://github.com/d6y/oner_induction/workflows/Rust/badge.svg)

# A 1R implementation in Rust

Re-implementing the 1R algorithm described in [Holte (1993)](https://link.springer.com/article/10.1023%2FA%3A1022631118932).

1R learns a rule (`IF...THEN...ELSE`) based on one attribute (feature) of the database. This gives a baseline performance for comparing with other algorithms.

This crate is a complement to <https://crates.io/crates/oner_quantize>, a 1R rule induction implementation.

# Documentation and examples

- [API reference and usage](https://docs.rs/oner_induction)
- An example application: <https://github.com/d6y/oner>


# License

Copyright 2020 Richard Dallaway

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at <https://mozilla.org/MPL/2.0/>.
