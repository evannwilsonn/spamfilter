//! spamfilter: an SMS spam classifier built from scratch in Rust.
//!
//! Modules follow the project loop:
//! - `data`: load the corpus and split it (Data)
//! - `metrics`: majority baseline and precision/recall/F1 (Baseline, Evaluation)
//! - `tokenize` + `nb`: the model (Model, Improvement)
//! - `linfa_cmp`: a library implementation to check against
//! - `serve`: the HTTP API (Deployment)

pub mod data;
pub mod linfa_cmp;
pub mod metrics;
pub mod nb;
pub mod serve;
pub mod tokenize;
