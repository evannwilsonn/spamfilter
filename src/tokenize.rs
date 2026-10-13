//! Turning an SMS into features.
//!
//! Two deliberate choices, both of which matter for spam:
//!
//! * Runs of digits are collapsed to a single `<num>` token. Spam is full of
//!   phone numbers and prize amounts that never repeat exactly, so the raw
//!   digits would be useless; the *presence* of a number is the signal.
//! * Currency symbols and `!` are kept as their own tokens. Ham rarely says
//!   "£1000" or "WIN!!!".
//!
//! `bigrams` optionally appends word pairs ("call now", "txt stop") so the
//! model can pick up short phrases. That is the "Improvement" step of the
//! loop; the baseline model runs with unigrams only.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TokenizeOptions {
    pub bigrams: bool,
}

pub fn tokenize(text: &str, opts: TokenizeOptions) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut in_digits = false;

    let flush = |cur: &mut String, in_digits: &mut bool, tokens: &mut Vec<String>| {
        if *in_digits {
            tokens.push("<num>".to_string());
        } else if !cur.is_empty() {
            tokens.push(std::mem::take(cur));
        }
        cur.clear();
        *in_digits = false;
    };

    for ch in text.chars() {
        if ch.is_ascii_digit() {
            if !in_digits {
                flush(&mut cur, &mut in_digits, &mut tokens);
                in_digits = true;
            }
        } else if ch.is_alphabetic() || ch == '\'' {
            if in_digits {
                flush(&mut cur, &mut in_digits, &mut tokens);
            }
            cur.extend(ch.to_lowercase());
        } else if matches!(ch, '£' | '$' | '€' | '!') {
            flush(&mut cur, &mut in_digits, &mut tokens);
            tokens.push(ch.to_string());
        } else {
            flush(&mut cur, &mut in_digits, &mut tokens);
        }
    }
    flush(&mut cur, &mut in_digits, &mut tokens);

    // Apostrophe-only fragments ("'") are noise.
    tokens.retain(|t| t != "'");

    if opts.bigrams {
        let pairs: Vec<String> = tokens
            .windows(2)
            .map(|w| format!("{}_{}", w[0], w[1]))
            .collect();
        tokens.extend(pairs);
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowercases_and_collapses_numbers() {
        let t = tokenize("Call 08452810075 NOW!", TokenizeOptions::default());
        assert_eq!(t, vec!["call", "<num>", "now", "!"]);
    }

    #[test]
    fn keeps_currency_and_handles_mixed_runs() {
        let t = tokenize("win £1000 cash 2day", TokenizeOptions::default());
        assert_eq!(t, vec!["win", "£", "<num>", "cash", "<num>", "day"]);
    }

    #[test]
    fn bigrams_are_appended() {
        let t = tokenize("call now", TokenizeOptions { bigrams: true });
        assert_eq!(t, vec!["call", "now", "call_now"]);
    }

    #[test]
    fn empty_input_gives_no_tokens() {
        assert!(tokenize("   ...  ", TokenizeOptions::default()).is_empty());
    }
}
