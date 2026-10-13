//! Evaluation. Spam is the positive class.
//!
//! Accuracy alone is misleading here: 86.6% of messages are ham, so a model
//! that always says "ham" scores 86.6% while catching zero spam. Precision,
//! recall and F1 on the spam class are what actually describe a filter.

use crate::data::Label;
use serde::Serialize;

#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq)]
pub struct Confusion {
    pub tp: u64,
    pub tn: u64,
    pub fp: u64,
    pub fn_: u64,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct Metrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub confusion: Confusion,
}

pub fn confusion(pairs: impl IntoIterator<Item = (Label, Label)>) -> Confusion {
    let mut c = Confusion::default();
    for (truth, pred) in pairs {
        match (truth, pred) {
            (Label::Spam, Label::Spam) => c.tp += 1,
            (Label::Ham, Label::Ham) => c.tn += 1,
            (Label::Ham, Label::Spam) => c.fp += 1,
            (Label::Spam, Label::Ham) => c.fn_ += 1,
        }
    }
    c
}

fn safe_div(a: u64, b: u64) -> f64 {
    if b == 0 {
        0.0
    } else {
        a as f64 / b as f64
    }
}

impl Metrics {
    pub fn from_confusion(c: Confusion) -> Self {
        let total = c.tp + c.tn + c.fp + c.fn_;
        let precision = safe_div(c.tp, c.tp + c.fp);
        let recall = safe_div(c.tp, c.tp + c.fn_);
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        Self {
            accuracy: safe_div(c.tp + c.tn, total),
            precision,
            recall,
            f1,
            confusion: c,
        }
    }

    pub fn compute(pairs: impl IntoIterator<Item = (Label, Label)>) -> Self {
        Self::from_confusion(confusion(pairs))
    }
}

/// The baseline every model has to beat: predict the majority class.
pub struct MajorityBaseline {
    pub label: Label,
}

impl MajorityBaseline {
    pub fn fit(labels: impl IntoIterator<Item = Label>) -> Self {
        let mut spam = 0i64;
        let mut ham = 0i64;
        for l in labels {
            match l {
                Label::Spam => spam += 1,
                Label::Ham => ham += 1,
            }
        }
        Self {
            label: if spam > ham { Label::Spam } else { Label::Ham },
        }
    }

    pub fn predict(&self, _text: &str) -> Label {
        self.label
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn metrics_on_a_known_confusion() {
        let c = Confusion {
            tp: 8,
            tn: 80,
            fp: 2,
            fn_: 10,
        };
        let m = Metrics::from_confusion(c);
        assert_relative_eq!(m.accuracy, 0.88);
        assert_relative_eq!(m.precision, 0.8);
        assert_relative_eq!(m.recall, 8.0 / 18.0);
        assert_relative_eq!(m.f1, 2.0 * 0.8 * (8.0 / 18.0) / (0.8 + 8.0 / 18.0));
    }

    #[test]
    fn majority_baseline_has_zero_recall_on_imbalanced_data() {
        let labels = [Label::Ham, Label::Ham, Label::Ham, Label::Spam];
        let b = MajorityBaseline::fit(labels);
        let m = Metrics::compute(labels.iter().map(|&l| (l, b.predict(""))));
        assert_relative_eq!(m.accuracy, 0.75);
        assert_relative_eq!(m.recall, 0.0);
        assert_relative_eq!(m.f1, 0.0);
    }

    #[test]
    fn no_division_by_zero() {
        let m = Metrics::from_confusion(Confusion::default());
        assert_eq!(m.accuracy, 0.0);
        assert_eq!(m.f1, 0.0);
    }
}
