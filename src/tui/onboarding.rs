//! Onboarding script engine — walks a decision tree defined in the organism YAML.
//!
//! The engine is a state machine driven by the TUI runner loop. It injects
//! `ChatEntry` messages into Bob's tab (creating the illusion of a conversation),
//! triggers TUI actions via `open:` targets, and blocks on `wait:` conditions.
//!
//! When the script completes, all injected messages remain in Bob's thread —
//! the real LLM picks up exactly where the script left off.

use std::collections::HashMap;

use crate::organism::{OnboardingChoice, OnboardingStep};

/// Actions the runner must execute after calling `advance()`.
pub enum OnboardingAction {
    /// Inject a message into Bob's chat tab.
    InjectChat { role: String, text: String },
    /// Trigger a TUI action (e.g., open provider wizard).
    OpenTarget(String),
    /// Present a numbered choice to the user.
    PresentChoice {
        prompt: String,
        options: Vec<(String, String)>, // (label, value)
    },
    /// Script is done — transition to normal mode.
    Complete,
}

/// The onboarding script engine.
pub struct OnboardingEngine {
    /// Flattened step queue. Choice branches are spliced in when chosen.
    steps: Vec<OnboardingStep>,
    /// Current position in the step queue.
    cursor: usize,
    /// Whether we're blocked waiting for user choice input.
    awaiting_choice: bool,
    /// Current choice options (valid when awaiting_choice is true).
    current_options: Vec<OnboardingChoice>,
    /// Variables for interpolation: {provider}, {model}, etc.
    vars: HashMap<String, String>,
    /// Whether the script has completed.
    pub finished: bool,
}

impl OnboardingEngine {
    /// Create from parsed onboarding steps. Returns None if steps are empty.
    pub fn new(steps: Vec<OnboardingStep>) -> Option<Self> {
        if steps.is_empty() {
            return None;
        }
        Some(Self {
            steps,
            cursor: 0,
            awaiting_choice: false,
            current_options: Vec::new(),
            vars: HashMap::new(),
            finished: false,
        })
    }

    /// Advance the engine. Returns actions for the runner to execute.
    ///
    /// Processes steps eagerly until hitting a blocking step (choice or wait)
    /// or reaching the end. Multiple `say` + `open` steps execute in one call.
    pub fn advance(&mut self, has_pool: bool) -> Vec<OnboardingAction> {
        if self.finished || self.awaiting_choice {
            return Vec::new();
        }

        let mut actions = Vec::new();

        while self.cursor < self.steps.len() {
            let step = self.steps[self.cursor].clone();
            match step {
                OnboardingStep::Say(text) => {
                    let interpolated = self.interpolate(&text);
                    actions.push(OnboardingAction::InjectChat {
                        role: "agent".into(),
                        text: interpolated,
                    });
                    self.cursor += 1;
                }
                OnboardingStep::Open(target) => {
                    actions.push(OnboardingAction::OpenTarget(target));
                    self.cursor += 1;
                }
                OnboardingStep::Wait(ref condition) => {
                    if self.check_condition(condition, has_pool) {
                        self.cursor += 1;
                        // Condition met — continue processing
                    } else {
                        // Block — will retry on next advance() call
                        break;
                    }
                }
                OnboardingStep::Choice { prompt, options } => {
                    let opts: Vec<(String, String)> = options
                        .iter()
                        .map(|o| (o.label.clone(), o.value.clone()))
                        .collect();
                    self.current_options = options;
                    self.awaiting_choice = true;
                    actions.push(OnboardingAction::PresentChoice {
                        prompt,
                        options: opts,
                    });
                    // Don't advance cursor — choice will splice steps
                    break;
                }
            }
        }

        // Check if we've reached the end
        if self.cursor >= self.steps.len() && !self.awaiting_choice {
            self.finished = true;
            actions.push(OnboardingAction::Complete);
        }

        actions
    }

    /// Feed a user choice (1-based index). Splices the chosen branch's
    /// steps into the queue at the current cursor position.
    pub fn submit_choice(&mut self, choice_index: usize) {
        if !self.awaiting_choice || choice_index == 0 || choice_index > self.current_options.len() {
            return;
        }

        let chosen = &self.current_options[choice_index - 1];
        let value = chosen.value.clone();
        let branch_steps = chosen.steps.clone();

        // Store the choice value for interpolation
        self.vars.insert("choice".into(), value);

        // Remove the Choice step at cursor and splice in the branch steps
        self.steps.remove(self.cursor);
        for (i, step) in branch_steps.into_iter().enumerate() {
            self.steps.insert(self.cursor + i, step);
        }

        self.awaiting_choice = false;
        self.current_options.clear();
    }

    /// Check if a wait condition is satisfied.
    fn check_condition(&self, condition: &str, has_pool: bool) -> bool {
        match condition {
            "provider-ready" => has_pool,
            "model-loaded" => false, // TODO: wire to actual model state
            _ => false,
        }
    }

    /// Interpolate {var} placeholders in text.
    fn interpolate(&self, text: &str) -> String {
        let mut result = text.to_string();
        for (key, value) in &self.vars {
            result = result.replace(&format!("{{{key}}}"), value);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::organism::{OnboardingChoice, OnboardingStep};

    #[test]
    fn empty_steps_returns_none() {
        assert!(OnboardingEngine::new(vec![]).is_none());
    }

    #[test]
    fn say_steps_advance_eagerly() {
        let steps = vec![
            OnboardingStep::Say("Hello".into()),
            OnboardingStep::Say("World".into()),
        ];
        let mut engine = OnboardingEngine::new(steps).unwrap();
        let actions = engine.advance(false);
        // Both says + Complete
        assert_eq!(actions.len(), 3);
        assert!(engine.finished);
        assert!(matches!(actions[0], OnboardingAction::InjectChat { .. }));
        assert!(matches!(actions[1], OnboardingAction::InjectChat { .. }));
        assert!(matches!(actions[2], OnboardingAction::Complete));
    }

    #[test]
    fn choice_blocks_until_submitted() {
        let steps = vec![
            OnboardingStep::Choice {
                prompt: "Pick one:".into(),
                options: vec![
                    OnboardingChoice {
                        label: "Option A".into(),
                        value: "a".into(),
                        steps: vec![OnboardingStep::Say("Chose A".into())],
                    },
                    OnboardingChoice {
                        label: "Option B".into(),
                        value: "b".into(),
                        steps: vec![OnboardingStep::Say("Chose B".into())],
                    },
                ],
            },
            OnboardingStep::Say("Done".into()),
        ];
        let mut engine = OnboardingEngine::new(steps).unwrap();

        // First advance presents the choice
        let actions = engine.advance(false);
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], OnboardingAction::PresentChoice { .. }));
        assert!(!engine.finished);

        // Still blocked
        let actions = engine.advance(false);
        assert!(actions.is_empty());

        // Submit choice 2 (Option B)
        engine.submit_choice(2);

        // Now advance processes the branch + trailing step
        let actions = engine.advance(false);
        assert_eq!(actions.len(), 3); // "Chose B" + "Done" + Complete
        assert!(engine.finished);
    }

    #[test]
    fn wait_blocks_until_condition_met() {
        let steps = vec![
            OnboardingStep::Wait("provider-ready".into()),
            OnboardingStep::Say("Connected!".into()),
        ];
        let mut engine = OnboardingEngine::new(steps).unwrap();

        // No pool — blocked
        let actions = engine.advance(false);
        assert!(actions.is_empty());
        assert!(!engine.finished);

        // Pool ready
        let actions = engine.advance(true);
        assert_eq!(actions.len(), 2); // "Connected!" + Complete
        assert!(engine.finished);
    }

    #[test]
    fn open_is_silent_and_non_blocking() {
        let steps = vec![
            OnboardingStep::Say("Opening...".into()),
            OnboardingStep::Open("providers/add-key/anthropic".into()),
            OnboardingStep::Wait("provider-ready".into()),
        ];
        let mut engine = OnboardingEngine::new(steps).unwrap();
        let actions = engine.advance(false);
        // Say + Open, then blocked on Wait
        assert_eq!(actions.len(), 2);
        assert!(matches!(actions[0], OnboardingAction::InjectChat { .. }));
        assert!(matches!(actions[1], OnboardingAction::OpenTarget(..)));
        assert!(!engine.finished);
    }

    #[test]
    fn interpolation_works() {
        let steps = vec![
            OnboardingStep::Choice {
                prompt: "Pick:".into(),
                options: vec![OnboardingChoice {
                    label: "Anthropic".into(),
                    value: "anthropic".into(),
                    steps: vec![OnboardingStep::Say("You chose {choice}".into())],
                }],
            },
        ];
        let mut engine = OnboardingEngine::new(steps).unwrap();
        engine.advance(false); // presents choice
        engine.submit_choice(1);
        let actions = engine.advance(false);
        if let OnboardingAction::InjectChat { text, .. } = &actions[0] {
            assert_eq!(text, "You chose anthropic");
        } else {
            panic!("expected InjectChat");
        }
    }
}
