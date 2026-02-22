//! # rustml-scipy
//!
//! Scientific computing in Rust — a port of Python's scipy.
//!
//! ## Modules
//!
//! - `stats` — Statistical functions (distributions, tests, descriptive stats)
//! - `linalg` — Linear algebra (beyond ndarray basics)
//! - `optimize` — Optimization algorithms
//!
//! ## Example
//!
//! ```rust
//! use rustml_scipy::stats;
//!
//! let data = vec![2.3, 3.1, 2.8, 3.5, 2.9, 3.2, 2.7];
//!
//! let mean = stats::mean(&data);
//! let std = stats::std(&data);
//! let (t_stat, p_value) = stats::ttest_1samp(&data, 3.0);
//!
//! println!("mean={mean:.3}, std={std:.3}, t={t_stat:.3}, p={p_value:.3}");
//! ```

pub mod stats;

// Coming soon:
// pub mod linalg;
// pub mod optimize;
// pub mod signal;
