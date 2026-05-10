//! Shim manifest types — sidecar metadata cortex needs to wire a shim
//! into the forward pass.
//!
//! Mirrors the JSON spec in `project_cortex_v1_shim_api.md`:
//!
//! ```json
//! {
//!   "id": "should_respond",
//!   "version": "0.3.1",
//!   "phase": "gate",
//!   "attachment": {"layer": "final", "pooling": "last_token"},
//!   "input_shape":  {"hidden_dim": 4096},
//!   "output_shape": {"kind": "scalar"},
//!   "description": "Does this prompt warrant a response from Bob"
//! }
//! ```

use serde::{Deserialize, Serialize};

/// Three-phase shim activation per the cortex API contract.
///
/// - **Injection** shims fire on every forward pass at a hidden-layer
///   entrance, adding a residual delta.
/// - **Gate** shims fire once at end-of-prefill, pooling the prompt's
///   final hidden state into a scalar (or category) decision.
/// - **Steer** shims fire per-token during decode, shaping the final
///   hidden state before logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShimPhase {
    Injection,
    Gate,
    Steer,
}

/// Where in the model the shim attaches.
///
/// `layer`: `"final"` or `"entrance:N"` or `"entrance:all"`.
/// `pooling`: how the hidden state is reduced before the shim runs —
/// `"last_token"`, `"mean"`, `"attention"`, or `"none"`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Attachment {
    pub layer: String,
    pub pooling: String,
}

/// Input tensor shape declaration. v1 supports a single named dimension
/// (the model's hidden size); future revisions may add more.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InputShape {
    pub hidden_dim: u32,
}

/// Output shape — the kind of decision a shim emits.
///
/// - `"scalar"` — single-value gate (e.g. should_respond confidence).
/// - `"category:N"` — discrete N-way classification.
/// - `"hidden_delta"` — residual to add (steer/inject phase).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OutputShape {
    /// Encoded as a tagged string in the cortex spec; we keep it
    /// opaque on the wire and let callers parse the discriminant.
    pub kind: String,
}

/// One shim's full manifest. Cortex needs every field to wire the
/// shim correctly.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShimManifest {
    pub id: String,
    pub version: String,
    pub phase: ShimPhase,
    pub attachment: Attachment,
    pub input_shape: InputShape,
    pub output_shape: OutputShape,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Compact summary returned by `GET /v1/shims/`. Cortex may return
/// either an array of full manifests or these summaries; the client
/// asks for the full manifest list because it's strictly more useful.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShimSummary {
    pub id: String,
    pub version: String,
    pub phase: ShimPhase,
}

/// One decision returned by `/v1/shims/infer`.
///
/// `decision` is opaque on the wire (matches the manifest's
/// `output_shape.kind`). Callers cast to scalar / category index /
/// vector based on the shim they're invoking.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShimDecision {
    pub decision: serde_json::Value,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_round_trip() {
        let m = ShimManifest {
            id: "should_respond".into(),
            version: "0.3.1".into(),
            phase: ShimPhase::Gate,
            attachment: Attachment {
                layer: "final".into(),
                pooling: "last_token".into(),
            },
            input_shape: InputShape { hidden_dim: 4096 },
            output_shape: OutputShape {
                kind: "scalar".into(),
            },
            description: Some("Does this prompt warrant a response".into()),
        };

        let json = serde_json::to_string(&m).unwrap();
        let back: ShimManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn phase_serializes_snake_case() {
        let json = serde_json::to_string(&ShimPhase::Injection).unwrap();
        assert_eq!(json, "\"injection\"");
        let json = serde_json::to_string(&ShimPhase::Gate).unwrap();
        assert_eq!(json, "\"gate\"");
        let json = serde_json::to_string(&ShimPhase::Steer).unwrap();
        assert_eq!(json, "\"steer\"");
    }

    #[test]
    fn manifest_omits_description_when_none() {
        let m = ShimManifest {
            id: "t".into(),
            version: "0".into(),
            phase: ShimPhase::Steer,
            attachment: Attachment {
                layer: "final".into(),
                pooling: "none".into(),
            },
            input_shape: InputShape { hidden_dim: 1024 },
            output_shape: OutputShape {
                kind: "hidden_delta".into(),
            },
            description: None,
        };
        let json = serde_json::to_string(&m).unwrap();
        assert!(!json.contains("description"));
    }
}
