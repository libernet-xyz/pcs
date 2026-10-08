// Copyright 2026 The Libernet Team
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../README.md")]

pub mod starkom {
    pub mod proto {
        pub mod pcs {
            pub mod v7 {
                include!(concat!(env!("OUT_DIR"), "/starkom.proto.pcs.v6.rs"));
            }
        }
    }
}

mod deep;
mod merkle;
mod utils;

pub mod fri;
pub mod hash;

pub use deep::*;
