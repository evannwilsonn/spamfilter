//! Loading the SMS Spam Collection and splitting it for evaluation.
//!
//! Dataset: Almeida, Hidalgo and Yamakami, "SMS Spam Collection v.1",
//! UCI Machine Learning Repository, CC BY 4.0. One message per line,
//! tab separated: `label<TAB>text`, label is `ham` or `spam`.

use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Label {
    Ham,
    Spam,
}

impl Label {
    pub fn parse(s: &str) -> Option<Label> {
        match s.trim() {
            "ham" => Some(Label::Ham),
            "spam" => Some(Label::Spam),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Label::Ham => "ham",
            Label::Spam => "spam",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Example {
    pub label: Label,
    pub text: String,
}

/// Parse the UCI TSV. Lines that do not match `label<TAB>text` are skipped
/// and counted so the caller can report them instead of silently losing data.
pub fn parse_tsv(contents: &str) -> (Vec<Example>, usize) {
    let mut out = Vec::new();
    let mut skipped = 0;
    for line in contents.lines() {
        let Some((label, text)) = line.split_once('\t') else {
            if !line.trim().is_empty() {
                skipped += 1;
            }
            continue;
        };
        match Label::parse(label) {
            Some(label) => out.push(Example {
                label,
                text: text.to_string(),
            }),
            None => skipped += 1,
        }
    }
    (out, skipped)
}

pub fn load_tsv(path: impl AsRef<Path>) -> std::io::Result<(Vec<Example>, usize)> {
    let contents = fs::read_to_string(path)?;
    Ok(parse_tsv(&contents))
}

/// Stratified shuffle split: each class is shuffled and split separately so
/// the test set keeps the same ham/spam ratio as the full corpus. With 13.4%
/// spam, a plain random split could land a test set with noticeably fewer
/// spam messages and make recall numbers noisy.
pub fn stratified_split(
    examples: &[Example],
    test_fraction: f64,
    seed: u64,
) -> (Vec<Example>, Vec<Example>) {
    assert!(
        (0.0..1.0).contains(&test_fraction),
        "test_fraction must be in [0, 1)"
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut train = Vec::new();
    let mut test = Vec::new();
    for label in [Label::Ham, Label::Spam] {
        let mut group: Vec<&Example> = examples.iter().filter(|e| e.label == label).collect();
        group.shuffle(&mut rng);
        let n_test = (group.len() as f64 * test_fraction).round() as usize;
        for (i, ex) in group.into_iter().enumerate() {
            if i < n_test {
                test.push(ex.clone());
            } else {
                train.push(ex.clone());
            }
        }
    }
    train.shuffle(&mut rng);
    test.shuffle(&mut rng);
    (train, test)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_lines_and_counts_bad_ones() {
        let raw = "ham\thello there\nspam\tWIN NOW\n\nbogus\tnope\nno tab here\n";
        let (ex, skipped) = parse_tsv(raw);
        assert_eq!(ex.len(), 2);
        assert_eq!(ex[0].label, Label::Ham);
        assert_eq!(ex[1].text, "WIN NOW");
        assert_eq!(skipped, 2);
    }

    #[test]
    fn split_is_stratified_and_deterministic() {
        let mut ex = Vec::new();
        for i in 0..90 {
            ex.push(Example {
                label: Label::Ham,
                text: format!("h{i}"),
            });
        }
        for i in 0..10 {
            ex.push(Example {
                label: Label::Spam,
                text: format!("s{i}"),
            });
        }
        let (train, test) = stratified_split(&ex, 0.2, 7);
        assert_eq!(test.len(), 20);
        assert_eq!(train.len(), 80);
        assert_eq!(test.iter().filter(|e| e.label == Label::Spam).count(), 2);
        let (train2, _) = stratified_split(&ex, 0.2, 7);
        let a: Vec<_> = train.iter().map(|e| &e.text).collect();
        let b: Vec<_> = train2.iter().map(|e| &e.text).collect();
        assert_eq!(a, b);
    }
}
