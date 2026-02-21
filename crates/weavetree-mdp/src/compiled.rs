use std::collections::HashMap;

use crate::{MdpError, MdpSpec};

/// Floating point tolerance used when validating probability sums.
pub(crate) const PROB_TOLERANCE: f64 = 1e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Dense index for states in a compiled MDP.
pub struct StateKey(usize);

impl StateKey {
    /// Return the underlying state index.
    pub fn index(self) -> usize {
        self.0
    }
}

impl From<usize> for StateKey {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone)]
/// Runtime form of an MDP with resolved state references and precomputed CDFs.
pub struct CompiledMdp {
    start: StateKey,
    states: Vec<StateRec>,
    state_ids: Vec<String>,
    state_id_to_key: HashMap<String, StateKey>,
}

#[derive(Debug, Clone)]
struct StateRec {
    terminal: bool,
    actions: Vec<ActionRec>,
}

#[derive(Debug, Clone)]
struct ActionRec {
    outcomes: Vec<OutcomeRec>,
    cdf: Vec<f64>,
}

#[derive(Debug, Clone)]
struct OutcomeRec {
    next: StateKey,
    reward: f64,
}

impl CompiledMdp {
    /// Compile and validate a spec into a fast runtime representation.
    pub(crate) fn from_spec(spec: &MdpSpec) -> Result<Self, MdpError> {
        spec.validate_with_tolerance(PROB_TOLERANCE)?;

        let mut state_id_to_key = HashMap::with_capacity(spec.states.len());
        let mut state_ids = Vec::with_capacity(spec.states.len());

        for (idx, state) in spec.states.iter().enumerate() {
            let key = StateKey::from(idx);
            state_id_to_key.insert(state.id.clone(), key);
            state_ids.push(state.id.clone());
        }

        let start = state_id_to_key.get(&spec.start).copied().ok_or_else(|| {
            MdpError::UnknownStartState {
                start: spec.start.clone(),
            }
        })?;

        let mut states = Vec::with_capacity(spec.states.len());
        for state in &spec.states {
            let terminal = state.terminal.unwrap_or(false);
            let mut actions = Vec::new();

            for action in state.actions.as_deref().unwrap_or(&[]) {
                let mut outcomes = Vec::with_capacity(action.outcomes.len());
                let mut cdf = Vec::with_capacity(action.outcomes.len());
                let mut cumulative = 0.0_f64;

                for outcome in &action.outcomes {
                    cumulative += outcome.prob;
                    cdf.push(cumulative);
                    let next = state_id_to_key.get(&outcome.next).copied().ok_or_else(|| {
                        MdpError::UnknownNextState {
                            state: state.id.clone(),
                            action: action.id.clone(),
                            next: outcome.next.clone(),
                        }
                    })?;

                    outcomes.push(OutcomeRec {
                        next,
                        reward: outcome.reward,
                    });
                }

                actions.push(ActionRec { outcomes, cdf });
            }

            states.push(StateRec { terminal, actions });
        }

        Ok(Self {
            start,
            states,
            state_ids,
            state_id_to_key,
        })
    }

    /// Return the start state key.
    pub fn start(&self) -> StateKey {
        self.start
    }

    /// Return the number of compiled states.
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// Check whether a state is terminal.
    pub fn is_terminal(&self, key: StateKey) -> Option<bool> {
        self.states.get(key.index()).map(|state| state.terminal)
    }

    /// Return the number of actions available from a state.
    pub fn num_actions(&self, key: StateKey) -> Option<usize> {
        self.states
            .get(key.index())
            .map(|state| state.actions.len())
    }

    /// Convert a state key back to its original string id.
    pub fn state_id(&self, key: StateKey) -> Option<&str> {
        self.state_ids.get(key.index()).map(String::as_str)
    }

    /// Convert a string id into a compiled state key.
    pub fn state_key(&self, id: &str) -> Option<StateKey> {
        self.state_id_to_key.get(id).copied()
    }

    /// Sample one transition for `(state_key, action_id)` using a uniform sample in `[0, 1)`.
    pub(crate) fn sample_transition(
        &self,
        state_key: StateKey,
        action_id: usize,
        sample: f64,
    ) -> Option<(StateKey, f64, bool)> {
        let state = self.states.get(state_key.index())?;
        if state.terminal {
            return Some((state_key, 0.0, true));
        }

        let action = state.actions.get(action_id)?;
        if action.outcomes.is_empty() {
            return None;
        }

        let mut chosen_idx = action.cdf.partition_point(|p| *p < sample);
        if chosen_idx >= action.outcomes.len() {
            chosen_idx = action.outcomes.len() - 1;
        }

        let outcome = &action.outcomes[chosen_idx];
        let next_terminal = self.states.get(outcome.next.index())?.terminal;
        Some((outcome.next, outcome.reward, next_terminal))
    }
}
