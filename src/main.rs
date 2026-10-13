use clap::{Parser, Subcommand};
use spamfilter::data::{load_tsv, stratified_split, Example, Label};
use spamfilter::linfa_cmp::LinfaModel;
use spamfilter::metrics::{MajorityBaseline, Metrics};
use spamfilter::nb::NaiveBayes;
use spamfilter::tokenize::TokenizeOptions;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "spamfilter",
    about = "SMS spam classifier (naive Bayes from scratch)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Train on the TSV, evaluate on a held-out split, print all the numbers.
    Eval {
        #[arg(long, default_value = "data/SMSSpamCollection.tsv")]
        data: PathBuf,
        #[arg(long, default_value_t = 0.2)]
        test_fraction: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 1.0)]
        alpha: f64,
        /// Also train and score the linfa MultinomialNb for comparison.
        #[arg(long)]
        linfa: bool,
        /// Print JSON instead of a table.
        #[arg(long)]
        json: bool,
    },
    /// Train on the whole TSV and save the model as JSON.
    Train {
        #[arg(long, default_value = "data/SMSSpamCollection.tsv")]
        data: PathBuf,
        #[arg(long, default_value = "model.json")]
        out: PathBuf,
        #[arg(long, default_value_t = 1.0)]
        alpha: f64,
        #[arg(long)]
        bigrams: bool,
        #[arg(long, default_value_t = 0.5)]
        threshold: f64,
    },
    /// Classify one message with a saved model.
    Predict {
        #[arg(long, default_value = "model.json")]
        model: PathBuf,
        text: String,
    },
    /// Serve a saved model over HTTP.
    Serve {
        #[arg(long, default_value = "model.json")]
        model: PathBuf,
        #[arg(long, default_value = "127.0.0.1:3000")]
        addr: String,
    },
}

fn load_model(path: &PathBuf) -> NaiveBayes {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| die(&format!("cannot read {}: {e}", path.display())));
    serde_json::from_str(&raw).unwrap_or_else(|e| die(&format!("bad model file: {e}")))
}

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1)
}

fn load_data(path: &PathBuf) -> Vec<Example> {
    let (examples, skipped) =
        load_tsv(path).unwrap_or_else(|e| die(&format!("cannot read {}: {e}", path.display())));
    if skipped > 0 {
        eprintln!("warning: skipped {skipped} malformed lines");
    }
    examples
}

struct Row {
    name: String,
    m: Metrics,
}

fn score(name: &str, test: &[Example], preds: impl Iterator<Item = Label>) -> Row {
    let m = Metrics::compute(test.iter().map(|e| e.label).zip(preds));
    Row {
        name: name.to_string(),
        m,
    }
}

fn print_table(rows: &[Row]) {
    println!(
        "{:<36} {:>8} {:>9} {:>8} {:>8}   {:>4} {:>4} {:>4} {:>4}",
        "model", "acc", "prec", "recall", "f1", "tp", "fp", "fn", "tn"
    );
    for r in rows {
        let c = r.m.confusion;
        println!(
            "{:<36} {:>8.4} {:>9.4} {:>8.4} {:>8.4}   {:>4} {:>4} {:>4} {:>4}",
            r.name, r.m.accuracy, r.m.precision, r.m.recall, r.m.f1, c.tp, c.fp, c.fn_, c.tn
        );
    }
}

fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Eval {
            data,
            test_fraction,
            seed,
            alpha,
            linfa,
            json,
        } => {
            let examples = load_data(&data);
            let (train, test) = stratified_split(&examples, test_fraction, seed);
            let spam_test = test.iter().filter(|e| e.label == Label::Spam).count();
            eprintln!(
                "train={} test={} (spam in test: {}), seed={seed}, alpha={alpha}",
                train.len(),
                test.len(),
                spam_test
            );

            let mut rows = Vec::new();

            let base = MajorityBaseline::fit(train.iter().map(|e| e.label));
            rows.push(score(
                "majority baseline (always ham)",
                &test,
                test.iter().map(|e| base.predict(&e.text)),
            ));

            let uni = NaiveBayes::fit(alpha, TokenizeOptions { bigrams: false }, &train);
            rows.push(score(
                "naive bayes, unigrams",
                &test,
                test.iter().map(|e| uni.predict_label(&e.text)),
            ));

            let bi = NaiveBayes::fit(alpha, TokenizeOptions { bigrams: true }, &train);
            rows.push(score(
                "naive bayes, unigrams+bigrams",
                &test,
                test.iter().map(|e| bi.predict_label(&e.text)),
            ));

            for t in [0.9, 0.99] {
                let strict = bi.clone().with_threshold(t);
                rows.push(score(
                    &format!("naive bayes, bigrams, threshold {t}"),
                    &test,
                    test.iter().map(|e| strict.predict_label(&e.text)),
                ));
            }

            if linfa {
                let lm = LinfaModel::fit(alpha, TokenizeOptions { bigrams: false }, &train);
                let texts: Vec<&str> = test.iter().map(|e| e.text.as_str()).collect();
                rows.push(score(
                    "linfa MultinomialNb, unigrams",
                    &test,
                    lm.predict_all(&texts).into_iter(),
                ));
                let lmb = LinfaModel::fit(alpha, TokenizeOptions { bigrams: true }, &train);
                rows.push(score(
                    "linfa MultinomialNb, unigrams+bigrams",
                    &test,
                    lmb.predict_all(&texts).into_iter(),
                ));
            }

            if json {
                let v: Vec<serde_json::Value> = rows
                    .iter()
                    .map(|r| serde_json::json!({ "model": r.name, "metrics": r.m }))
                    .collect();
                println!("{}", serde_json::to_string_pretty(&v).unwrap());
            } else {
                print_table(&rows);
            }
        }
        Cmd::Train {
            data,
            out,
            alpha,
            bigrams,
            threshold,
        } => {
            let examples = load_data(&data);
            let model = NaiveBayes::fit(alpha, TokenizeOptions { bigrams }, &examples)
                .with_threshold(threshold);
            let json = serde_json::to_string(&model).unwrap();
            std::fs::write(&out, json)
                .unwrap_or_else(|e| die(&format!("cannot write {}: {e}", out.display())));
            eprintln!(
                "trained on {} messages, vocab {}, saved to {}",
                examples.len(),
                model.vocab_size(),
                out.display()
            );
        }
        Cmd::Predict { model, text } => {
            let m = load_model(&model);
            let p = m.predict(&text);
            println!("{}  (p_spam={:.4})", p.label.as_str(), p.p_spam);
            for (tok, w) in &p.top_evidence {
                println!("  {:<20} {:+.3}", tok, w);
            }
        }
        Cmd::Serve { model, addr } => {
            let m = load_model(&model);
            tracing_subscriber::fmt::init();
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async move {
                let listener = tokio::net::TcpListener::bind(&addr)
                    .await
                    .unwrap_or_else(|e| die(&format!("cannot bind {addr}: {e}")));
                eprintln!("listening on http://{addr}");
                axum::serve(listener, spamfilter::serve::router(m))
                    .await
                    .unwrap();
            });
        }
    }
}
