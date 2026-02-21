use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::{CompiledMdp, MdpError, compiled::PROB_TOLERANCE};

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Serializable MDP schema used for YAML IO and validation.
pub struct MdpSpec {
    /// Schema version for future compatibility checks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<u32>,
    /// String id of the start state.
    pub start: String,
    /// All state declarations in the model.
    pub states: Vec<StateSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// A single state declaration in the MDP schema.
pub struct StateSpec {
    /// Unique state id.
    pub id: String,
    /// Whether this state is terminal (defaults to `false` if omitted).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal: Option<bool>,
    /// Available actions from this state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actions: Option<Vec<ActionSpec>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// A named action and its stochastic outcomes.
pub struct ActionSpec {
    pub id: String,
    pub outcomes: Vec<OutcomeSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// One probabilistic transition for an action.
pub struct OutcomeSpec {
    pub next: String,
    pub prob: f64,
    pub reward: f64,
}

impl MdpSpec {
    /// Validate schema invariants using the crate default tolerance.
    pub fn validate(&self) -> Result<(), MdpError> {
        self.validate_with_tolerance(PROB_TOLERANCE)
    }

    /// Validate ids, transitions, and probability constraints.
    pub fn validate_with_tolerance(&self, tolerance: f64) -> Result<(), MdpError> {
        // Start state id must be present and non-empty.
        if self.start.trim().is_empty() {
            return Err(MdpError::MissingStart);
        }

        // State ids must be unique.
        let mut ids = HashSet::with_capacity(self.states.len());
        for state in &self.states {
            if !ids.insert(state.id.clone()) {
                return Err(MdpError::DuplicateStateId {
                    id: state.id.clone(),
                });
            }
        }

        // Start state must resolve to a known state id.
        if !ids.contains(&self.start) {
            return Err(MdpError::UnknownStartState {
                start: self.start.clone(),
            });
        }

        // Fast membership map for outcome target validation.
        let known_state_ids: HashMap<_, _> = self.states.iter().map(|s| (&s.id, true)).collect();

        for state in &self.states {
            let terminal = state.terminal.unwrap_or(false);
            let actions = state.actions.as_deref().unwrap_or(&[]);

            if terminal && !actions.is_empty() {
                return Err(MdpError::TerminalStateHasActions {
                    state: state.id.clone(),
                });
            }

            let mut action_ids = HashSet::with_capacity(actions.len());
            for action in actions {
                if !action_ids.insert(action.id.clone()) {
                    return Err(MdpError::DuplicateActionId {
                        state: state.id.clone(),
                        action: action.id.clone(),
                    });
                }

                if action.outcomes.is_empty() {
                    return Err(MdpError::EmptyOutcomes {
                        state: state.id.clone(),
                        action: action.id.clone(),
                    });
                }

                let mut sum = 0.0_f64;
                for (i, outcome) in action.outcomes.iter().enumerate() {
                    if outcome.prob.is_nan() || !outcome.prob.is_finite() || outcome.prob < 0.0 {
                        return Err(MdpError::InvalidProbability {
                            state: state.id.clone(),
                            action: action.id.clone(),
                            outcome_index: i,
                            value: outcome.prob,
                        });
                    }

                    if !outcome.reward.is_finite() {
                        return Err(MdpError::InvalidReward {
                            state: state.id.clone(),
                            action: action.id.clone(),
                            outcome_index: i,
                            value: outcome.reward,
                        });
                    }

                    if !known_state_ids.contains_key(&outcome.next) {
                        return Err(MdpError::UnknownNextState {
                            state: state.id.clone(),
                            action: action.id.clone(),
                            next: outcome.next.clone(),
                        });
                    }

                    sum += outcome.prob;
                }

                // Outcome probabilities for an action must sum to 1 within tolerance.
                if (sum - 1.0).abs() > tolerance {
                    return Err(MdpError::ProbabilitySum {
                        state: state.id.clone(),
                        action: action.id.clone(),
                        sum,
                        tolerance,
                    });
                }
            }
        }

        Ok(())
    }

    /// Compile this spec into the runtime representation.
    pub fn compile(&self) -> Result<CompiledMdp, MdpError> {
        CompiledMdp::from_spec(self)
    }
}
