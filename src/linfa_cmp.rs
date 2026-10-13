//! Comparison against a library implementation.
//!
//! Same tokenizer, same split, but the model comes from `linfa-bayes`
//! (`MultinomialNb`) instead of `nb.rs`. If the two agree closely, the
//! from-scratch version is doing the right maths. The library needs a dense
//! document-term matrix, so this builds one: rows are messages, columns are
//! vocabulary tokens, cells are raw counts.

use crate::data::{Example, Label};
use crate::tokenize::{tokenize, TokenizeOptions};
use linfa::prelude::*;
use linfa_bayes::MultinomialNb;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

pub struct LinfaModel {
    vocab: HashMap<String, usize>,
    model: MultinomialNb<f64, usize>,
    opts: TokenizeOptions,
}

fn label_to_usize(l: Label) -> usize {
    match l {
        Label::Ham => 0,
        Label::Spam => 1,
    }
}

fn usize_to_label(u: usize) -> Label {
    if u == 1 {
        Label::Spam
    } else {
        Label::Ham
    }
}

impl LinfaModel {
    pub fn fit(alpha: f64, opts: TokenizeOptions, train: &[Example]) -> Self {
        let mut vocab: HashMap<String, usize> = HashMap::new();
        let tokenized: Vec<Vec<String>> = train.iter().map(|e| tokenize(&e.text, opts)).collect();
        for toks in &tokenized {
            for t in toks {
                let n = vocab.len();
                vocab.entry(t.clone()).or_insert(n);
            }
        }
        let x = Self::matrix(&vocab, &tokenized);
        let y: Array1<usize> = train.iter().map(|e| label_to_usize(e.label)).collect();
        let ds = Dataset::new(x, y);
        let model = MultinomialNb::params()
            .alpha(alpha)
            .fit(&ds)
            .expect("linfa MultinomialNb fit failed");
        Self { vocab, model, opts }
    }

    fn matrix(vocab: &HashMap<String, usize>, docs: &[Vec<String>]) -> Array2<f64> {
        let mut x = Array2::<f64>::zeros((docs.len(), vocab.len()));
        for (i, toks) in docs.iter().enumerate() {
            for t in toks {
                if let Some(&j) = vocab.get(t) {
                    x[[i, j]] += 1.0;
                }
            }
        }
        x
    }

    pub fn predict_all(&self, texts: &[&str]) -> Vec<Label> {
        let docs: Vec<Vec<String>> = texts.iter().map(|t| tokenize(t, self.opts)).collect();
        let x = Self::matrix(&self.vocab, &docs);
        self.model
            .predict(&x)
            .iter()
            .map(|&u| usize_to_label(u))
            .collect()
    }
}
