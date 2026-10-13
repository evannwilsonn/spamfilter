//! Axum HTTP API around a trained model.
//!
//! `POST /predict { "text": "..." }` → `{ label, p_spam, top_evidence }`
//! `GET  /health`                   → `{ "status": "ok", "vocab_size": n }`
//! `GET  /`                         → a small test page that calls /predict

use crate::nb::NaiveBayes;
use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::CorsLayer;

#[derive(Clone)]
pub struct AppState {
    pub model: Arc<NaiveBayes>,
}

#[derive(Deserialize)]
pub struct PredictRequest {
    pub text: String,
}

#[derive(Serialize)]
pub struct PredictResponse {
    pub label: &'static str,
    pub p_spam: f64,
    pub top_evidence: Vec<Evidence>,
}

#[derive(Serialize)]
pub struct Evidence {
    pub token: String,
    pub log_odds: f64,
}

const MAX_TEXT_BYTES: usize = 4096;

async fn predict(
    State(state): State<AppState>,
    Json(req): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, (StatusCode, String)> {
    if req.text.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "text is empty".into()));
    }
    if req.text.len() > MAX_TEXT_BYTES {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("text longer than {MAX_TEXT_BYTES} bytes"),
        ));
    }
    let p = state.model.predict(&req.text);
    Ok(Json(PredictResponse {
        label: p.label.as_str(),
        p_spam: p.p_spam,
        top_evidence: p
            .top_evidence
            .into_iter()
            .map(|(token, log_odds)| Evidence { token, log_odds })
            .collect(),
    }))
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "vocab_size": state.model.vocab_size(),
        "spam_threshold": state.model.spam_threshold,
    }))
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

pub fn router(model: NaiveBayes) -> Router {
    let state = AppState {
        model: Arc::new(model),
    };
    Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/predict", post(predict))
        .layer(CorsLayer::permissive())
        .with_state(state)
}
