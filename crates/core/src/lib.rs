//! Core library for rust-simulation-tools.
//!
//! Pure Rust implementations with no Python dependencies.
//! Provides AMBER file parsers, trajectory readers, SASA calculations,
//! alignment, unwrapping, and fingerprinting engines.

pub mod amber;
pub mod fingerprint;
pub mod kabsch;
pub mod sasa;
pub mod trajectory;
pub mod util;
pub mod selection;
pub mod wrapping;
