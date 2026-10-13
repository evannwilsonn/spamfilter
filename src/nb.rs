//! Multinomial naive Bayes, written from scratch.
//!
//! The model is Bayes' rule with one simplifying assumption: given the class,
//! every token is independent of every other token.
//!
//! ```text
//! P(spam | tokens) ∝ P(spam) · Π_i P(token_i | spam)
//! ```
//!
//! Training is counting. For each class we record how many training messages
//! it has (the prior) and how often each token appears in it. Prediction sums
//! log-probabilities instead of multiplying probabilities so that a 40-token
//! message does not underflow to zero.
//!
//! Laplace (add-`alpha`) smoothing keeps a token the model has never seen in
//! one class from zeroing out that class entirely. Without it, a single
//! unseen word makes P(class) = 0, which is wrong.
//!
//! `spam_threshold` is a decision knob, not part of the model: by default a
//! message is spam when P(spam) > 0.5, but for a mail filter a false positive
//! (real message in the spam folder) is worse than a false negative, so the
//! threshold can be raised to trade recall for precision.

use crate::data::{Example, Label};
use crate::tokenize::{tokenize, TokenizeOptions};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaiveBayes {
    pub alpha: f64,
    pub tokenize: TokenizeOptionsSer,
    pub spam_threshold: f64,
    /// Number of training documents per class.
    doc_count: [u64; 2],
    /// Total token count per class (sum of all entries in `token_count`).
    total_tokens: [u64; 2],
    /// Token → [count in ham, count in spam].
    token_count: HashMap<String, [u64; 2]>,
}

/// Mirror of `TokenizeOptions` that derives serde so a trained model can be
/// saved to disk with its tokenizer settings and reloaded consistently.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenizeOptionsSer {
    pub bigrams: bool,
}

impl From<TokenizeOptions> for TokenizeOptionsSer {
    fn from(o: TokenizeOptions) -> Self {
        Self { bigrams: o.bigrams }
    }
}
impl From<TokenizeOptionsSer> for TokenizeOptions {
    fn from(o: TokenizeOptionsSer) -> Self {
        Self { bigrams: o.bigrams }
    }
}

fn idx(label: Label) -> usize {
    match label {
        Label::Ham => 0,
        Label::Spam => 1,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Prediction {
    pub label: Label,
    /// P(spam | text), in [0, 1].
    pub p_spam: f64,
    /// Tokens that pushed the decision hardest, with their log-odds
    /// contribution: positive means "toward spam". Useful for the CLI and
    /// API so a prediction can be explained rather than just asserted.
    pub top_evidence: Vec<(String, f64)>,
}

impl NaiveBayes {
    pub fn new(alpha: f64, tokenize: TokenizeOptions) -> Self {
        assert!(alpha > 0.0, "alpha must be positive");
        Self {
            alpha,
            tokenize: tokenize.into(),
            spam_threshold: 0.5,
            doc_count: [0, 0],
            total_tokens: [0, 0],
            token_count: HashMap::new(),
        }
    }

    pub fn with_threshold(mut self, t: f64) -> Self {
        assert!((0.0..=1.0).contains(&t));
        self.spam_threshold = t;
        self
    }

    pub fn train(&mut self, examples: &[Example]) {
        for ex in examples {
            let c = idx(ex.label);
            self.doc_count[c] += 1;
            for tok in tokenize(&ex.text, self.tokenize.into()) {
                self.total_tokens[c] += 1;
                self.token_count.entry(tok).or_insert([0, 0])[c] += 1;
            }
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.token_count.len()
    }

    pub fn fit(alpha: f64, tokenize: TokenizeOptions, examples: &[Example]) -> Self {
        let mut m = Self::new(alpha, tokenize);
        m.train(examples);
        m
    }

    fn log_prior(&self, c: usize) -> f64 {
        let n: u64 = self.doc_count.iter().sum();
        (self.doc_count[c] as f64 / n as f64).ln()
    }

    /// log P(token | class) with add-alpha smoothing over the vocabulary.
    fn log_likelihood(&self, token: &str, c: usize) -> f64 {
        let count = self.token_count.get(token).map(|v| v[c]).unwrap_or(0) as f64;
        let v = self.vocab_size() as f64;
        ((count + self.alpha) / (self.total_tokens[c] as f64 + self.alpha * v)).ln()
    }

    pub fn predict(&self, text: &str) -> Prediction {
        assert!(
            self.doc_count.iter().sum::<u64>() > 0,
            "predict called on an untrained model"
        );
        let tokens = tokenize(text, self.tokenize.into());

        let mut log_ham = self.log_prior(idx(Label::Ham));
        let mut log_spam = self.log_prior(idx(Label::Spam));
        let mut evidence: Vec<(String, f64)> = Vec::with_capacity(tokens.len());

        for tok in &tokens {
            // Out-of-vocabulary tokens carry no information about either
            // class, so skip them. Scoring them anyway is subtly wrong: the
            // smoothed probability alpha / (N_c + alpha·V) is larger for the
            // class with fewer total tokens (spam), so every unseen word
            // would tilt the verdict toward spam. Measured on the held-out
            // split, that mistake cost 6 extra false positives.
            if !self.token_count.contains_key(tok) {
                continue;
            }
            let lh = self.log_likelihood(tok, idx(Label::Ham));
            let ls = self.log_likelihood(tok, idx(Label::Spam));
            log_ham += lh;
            log_spam += ls;
            evidence.push((tok.clone(), ls - lh));
        }

        // Normalise with log-sum-exp so p_spam is a real probability.
        let max = log_ham.max(log_spam);
        let z = (log_ham - max).exp() + (log_spam - max).exp();
        let p_spam = (log_spam - max).exp() / z;

        evidence.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        evidence.dedup_by(|a, b| a.0 == b.0);
        evidence.truncate(5);

        let label = if p_spam > self.spam_threshold {
            Label::Spam
        } else {
            Label::Ham
        };
        Prediction {
            label,
            p_spam,
            top_evidence: evidence,
        }
    }

    pub fn predict_label(&self, text: &str) -> Label {
        self.predict(text).label
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn ex(label: Label, text: &str) -> Example {
        Example {
            label,
            text: text.to_string(),
        }
    }

    fn tiny_model() -> NaiveBayes {
        NaiveBayes::fit(
            1.0,
            TokenizeOptions::default(),
            &[
                ex(Label::Ham, "hey are we still on for lunch"),
                ex(Label::Ham, "running late see you soon"),
                ex(Label::Ham, "lunch was great thanks"),
                ex(Label::Spam, "WIN a free prize call now"),
                ex(Label::Spam, "free entry win cash call"),
            ],
        )
    }

    #[test]
    fn prior_matches_class_frequencies() {
        let m = tiny_model();
        assert_relative_eq!(m.log_prior(0), (3.0f64 / 5.0).ln());
        assert_relative_eq!(m.log_prior(1), (2.0f64 / 5.0).ln());
    }

    #[test]
    fn likelihood_uses_laplace_smoothing() {
        let m = tiny_model();
        // "free" appears twice in spam; spam has 11 tokens; vocab is 20.
        let v = m.vocab_size() as f64;
        let spam_tokens = m.total_tokens[1] as f64;
        let expected = ((2.0 + 1.0) / (spam_tokens + v)).ln();
        assert_relative_eq!(m.log_likelihood("free", 1), expected);
        // Unseen token: count 0 but still a finite probability.
        let unseen = m.log_likelihood("zebra", 0);
        assert!(unseen.is_finite());
        assert_relative_eq!(unseen, (1.0 / (m.total_tokens[0] as f64 + v)).ln());
    }

    #[test]
    fn classifies_obvious_cases() {
        let m = tiny_model();
        assert_eq!(m.predict_label("free prize call now"), Label::Spam);
        assert_eq!(m.predict_label("see you at lunch"), Label::Ham);
    }

    #[test]
    fn probabilities_are_normalised() {
        let m = tiny_model();
        let p = m.predict("win cash").p_spam;
        assert!(p > 0.5 && p <= 1.0);
        let q = m.predict("lunch soon").p_spam;
        assert!((0.0..0.5).contains(&q));
    }

    #[test]
    fn unseen_words_fall_back_to_the_prior() {
        let m = tiny_model();
        // Every token out of vocabulary: ignored, so P(spam) is the prior 2/5.
        let p = m.predict("zebra quokka").p_spam;
        assert_relative_eq!(p, 0.4, epsilon = 1e-12);
        // Mixed: unseen words must not change the verdict for seen ones.
        let a = m.predict("free prize").p_spam;
        let b = m.predict("free zebra prize quokka").p_spam;
        assert_relative_eq!(a, b, epsilon = 1e-12);
    }

    #[test]
    fn threshold_changes_decision_but_not_probability() {
        let m = tiny_model();
        let p0 = m.predict("win cash now");
        let strict = m.clone().with_threshold(0.999);
        let p1 = strict.predict("win cash now");
        assert_relative_eq!(p0.p_spam, p1.p_spam);
        assert_eq!(p0.label, Label::Spam);
        assert_eq!(p1.label, Label::Ham);
    }

    #[test]
    fn evidence_points_the_right_way() {
        let m = tiny_model();
        let p = m.predict("free prize");
        assert!(p.top_evidence.iter().all(|(_, w)| *w > 0.0));
        let h = m.predict("lunch thanks");
        assert!(h.top_evidence.iter().all(|(_, w)| *w < 0.0));
    }
}
