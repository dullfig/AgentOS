> **Cross-session coordination:** Before building anything described here, read `C:\Users\danu\.claude\projects\C--src-ringhub-integration\memory\MEMORY.md` — shared brain across all Claude sessions on this project. This design doc is self-contained but assumes the broader context lives in that folder.

# Shim Management Subsystem — AgentOS Design Doc

**Status:** Design — not yet implemented.
**Created:** 2026-04-19 by the integration-folder session, to hand off to the AgentOS-folder session.
**Primary audience:** the Claude session that will implement this.

## What shims are (in this project's context)

Shims are small (~28K-parameter) FFN modules that sit between a transformer's final hidden-state output and its token emission step, providing **gating and steering signals that the base model itself does not produce.** They are trained once, loaded at inference time, and composable.

In this project, shims are the "control plane" for a concierge agent named Bob. Bob runs on a single shared base model (Qwen 7B base is the v1 target). Shims decide, per-context:
- Should Bob respond at all? (`should_respond` — binary gate)
- What's his retrieval confidence per corpus? (`memex_confidence_per_corpus` — scalar per memex)
- Which corpora should receive the query? (`which_memex` — routing distribution)
- Which voice register applies? (`context_register` — category: chat / ambient / interview / comment)
- Is this a crisis / moderation case? (`is_crisis_or_moderation` — binary gate)
- Is this query about a notable community member? (`is_about_notable_entity` — category)
- Should this content be flagged for archive ingestion priority? (`is_archive_worthy` — binary gate)

Without shims, context-appropriate Bob would require multiple fine-tunes of the base model (chat-bubble Bob vs. ambient-listener Bob vs. interview-mode Bob). The shim architecture replaces fine-tune proliferation with cheap, composable, swappable gating — one base model, many contexts.

**Why this matters for Bob specifically:** Bob is designed to respect silence and say "I don't know" when appropriate — behaviors that existing LLMs do not reliably produce. Shim-gated abstention makes these structural, not prompt-engineered. See `project_silence_as_first_class.md` in the integration memory for the broader rationale.

## Why AgentOS owns the management subsystem

Cortex's responsibility: inference runtime. Loads a model, executes it, loads an ONNX shim, executes it. Constrained runtime duties.

AgentOS's responsibility: agent lifecycle, tools, triggers, WAL, per-user state. Administrative. Shim management (registry, training orchestration, deployment, rollback, observability, evaluation) is administrative and belongs one level above the inference runtime. AgentOS's call-chain dispatch also opens a future door — per-user shim configurations, if that ever becomes useful.

Cortex loads what AgentOS tells it to load. AgentOS owns the decisions.

The training pipeline itself extends what `C:\src\classifiers` already industrialized (small CNN → ONNX export). AgentOS's shim-management subsystem orchestrates that pipeline for shim purposes rather than sheet-music-symbol purposes.

## v1 scope — narrow, deliberately

**Build one shim end-to-end before abstracting anything.**

- One shim: `should_respond` (binary gate — does Bob respond in this context, yes or no)
- Hand-labeled training data (a few hundred examples, tedious but tractable in an afternoon)
- Training pipeline extending the `C:\src\classifiers` recipe
- ONNX export to a known path
- Cortex loads at startup; static deployment
- No hot reload, no version management, no A/B, no canary
- Logs decisions with context for later analysis

Prove the end-to-end loop: *user message → context features → shim → decision → Bob's behavior reflects the decision → we can observe whether the decision was right.*

Everything fancier waits until this loop is visibly working.

## Full system capabilities (for v2+ planning, NOT v1)

| Capability | What it does |
|---|---|
| **Registry** | Metadata per shim type: purpose, input shape (which layer's hidden state), output shape, training-data source, deployment status |
| **Training pipeline** | Generalized wrapper around the classifiers recipe, per-shim-type configuration |
| **Training-data pipeline** | Per-shim: label sources (hand, synthetic, active-learning, community signal). Hardest part. |
| **Model store** | Versioned ONNX files in B2 (existing storage). Metadata DB tracks active version per shim purpose. |
| **Runtime loader** | Cortex picks up active version per shim at startup. Hot reload optional later. |
| **Observability** | Every shim decision logged with context. Needed for debugging and active learning. |
| **Swap / rollback / A/B** | Standard ML-deployment pattern. |
| **Evaluation harness** | Held-out test set per shim; automated regression checks; drift detection. |

## First shim specification — `should_respond`

**Purpose:** given a conversation context (message, prior turns, platform surface, user relationship), decide whether Bob should produce a response at all, or stay silent.

**Input:** hidden-state vector from the base model's final layer, computed over the input context + any retrieval results. The exact layer and pooling strategy is an implementation choice — pick what matches the classifiers repo's existing pattern.

**Output:** scalar in [0, 1]. Threshold at 0.5 for binary decision; allow configurable threshold per surface (chat bubble → low threshold so Bob responds easily; ambient post stream → high threshold so Bob stays quiet by default).

**Training data sources (bootstrap path):**

1. **Hand-labeled baseline (first week of work):** produce ~300-500 examples of `(context, should_respond_label)`. Mix of:
   - Direct questions to Bob (label: respond)
   - Passive posts with no direct question (label: stay silent)
   - Thread replies where Bob is not addressed (label: stay silent unless highly relevant)
   - Crisis / safety contexts (label: route to moderator, not Bob — though this is a separate shim later)
   - Content where Bob has retrieval signal vs. no signal

2. **Synthetic augmentation (week two):** use Claude Opus / Sonnet to generate ~1000-2000 additional labeled examples with rationales. High-quality synthetic data with known provenance.

3. **Active learning (post-deployment):** deploy with uncertainty threshold; flag borderline decisions; hand-label; retrain. This is how the shim improves from real deployment traffic.

4. **Community signal (future):** implicit labels from member behavior ("thanks Bob" → positive; "why are you responding" → negative; thread ignores Bob's response → neutral-negative).

**Evaluation criteria:**
- Accuracy on held-out test set (target: >85% on hand-labeled test split)
- Qualitative review of borderline decisions — do they feel right?
- Real-deployment regret monitoring — how often does a human reviewer think Bob should have / shouldn't have spoken?

## Integration points

**How cortex receives shim outputs:**

Option 1 (simple): cortex loads the shim as a separate ONNX model, invokes it with the hidden state, receives the decision, and the AgentOS agent loop reads the decision and acts accordingly.

Option 2 (tighter coupling): shim is loaded as part of cortex's forward pass, decision is emitted alongside logits. More efficient, harder to swap shims.

**Recommendation: Option 1 for v1.** Simpler, cleaner separation, easier to iterate on the shim without touching cortex's generation path. Revisit if latency becomes a bottleneck.

**How AgentOS consumes shim outputs:**

- `should_respond` < threshold → AgentOS's agent loop decides not to invoke the generator. Bob stays silent. If this happens on a direct-invocation surface (chat bubble), fall back to a structured "I'm not sure how to respond to that" utterance rather than empty output.
- `memex_confidence_per_corpus` scores feed into the fan-out response composition (see `project_memex_per_corpus.md`).
- Other shims gate specific downstream actions (routing, queue flagging, register selection).

**Where ONNX files live:**

- Training artifacts: local disk during development
- Deployed: B2 bucket (RingHub already uses B2 for user content), standardized path like `/shims/{shim_name}/{version}/model.onnx`
- Metadata: small registry file at `/shims/{shim_name}/{version}/metadata.json` (training data hash, evaluation metrics, promotion timestamp)
- Active version pointer: `/shims/{shim_name}/active.txt` — plain-text file with the version string currently in production

Cortex reads the active pointer at startup, pulls the referenced ONNX, caches locally, executes.

## What NOT to build up front

- Full management subsystem before two shims working end-to-end
- A/B testing until one shim deployed and iterating
- Drift detection until drift is actually observed
- Multi-tenant shim profiles until a user explicitly needs different gating
- Hot-reload until startup-reload is painful in practice
- Admin GUI — CLI suffices until it doesn't
- Per-user shim weights until you have a user requesting them

## Shim roadmap (after v1)

In rough priority order:

1. **`should_respond`** (v1 — foundation)
2. **`memex_confidence_per_corpus`** — per-memex retrieval confidence; enables honest "I don't know" and weighted fan-out
3. **`which_memex`** — routing dispatcher; optimization once fan-out patterns are understood
4. **`context_register`** — chat / ambient / interview / comment — which voice register applies
5. **`is_crisis_or_moderation`** — routes sensitive content to human moderators
6. **`is_about_notable_entity`** — tags queries touching Columbo tier-1 targets
7. **`is_archive_worthy`** — flags content for ingestion priority

Each shim beyond the first is primarily a data + training task rather than infrastructure. Infrastructure gets built incrementally as needs emerge.

## References

All paths are on Daniel's Windows filesystem. These are authoritative for the design context:

- `C:\Users\danu\.claude\projects\C--src-ringhub-integration\memory\project_agentos_shim_management.md` — this design doc's canonical memory-system source
- `C:\Users\danu\.claude\projects\C--src-ringhub-integration\memory\project_cortex_ffn_shims.md` — the shim concept and taxonomy
- `C:\Users\danu\.claude\projects\C--src-ringhub-integration\memory\project_silence_as_first_class.md` — why abstention matters architecturally
- `C:\Users\danu\.claude\projects\C--src-ringhub-integration\memory\project_memex_per_corpus.md` — memex fleet and how shims interact with it
- `C:\Users\danu\.claude\projects\C--src-ringhub-integration\memory\project_classifiers_success.md` — the existing training pipeline to extend
- `C:\src\classifiers\ROADMAP.md` — concrete training-pipeline examples (accidentals, clefs, etc.)

Read the integration memory files before starting work. They contain design decisions that are not repeated here but are load-bearing for this subsystem.

## First step for the implementing session

1. Read the integration memory (start with `MEMORY.md` → `project_agentos_shim_management.md`)
2. Inspect the classifiers repo's recipe pattern (`C:\src\classifiers\accidentals\` has `extract_fast.py`, `train.py`, `eval.py` — the canonical example)
3. Decide: does the shim management subsystem become its own crate in the AgentOS workspace (e.g., `agentos-shims`), or a module within `agentos-kernel`?
4. Sketch the v1 end-to-end loop in pseudocode before writing actual code. Confirm the loop is right before committing to it.
5. Produce 20 hand-labeled `(context, should_respond)` examples as a sanity check on what training data for this shim actually looks like. That exercise will clarify the data pipeline design faster than any amount of planning.
