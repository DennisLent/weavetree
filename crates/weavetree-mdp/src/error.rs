use thiserror::Error;

#[derive(Debug, Error)]
/// Error type for MDP loading, validation, compilation, and builder operations.
pub enum MdpError {
    #[error("failed to read YAML file: {0}")]
    Io(#[from] std::io::Error),

    #[error("failed to parse YAML: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("missing start state")]
    MissingStart,

    #[error("start state '{start}' does not exist")]
    UnknownStartState { start: String },

    #[error("duplicate state id '{id}'")]
    DuplicateStateId { id: String },

    #[error("duplicate action id '{action}' in state '{state}'")]
    DuplicateActionId { state: String, action: String },

    #[error("state '{state}' is terminal and cannot declare actions")]
    TerminalStateHasActions { state: String },

    #[error("outcome in state '{state}', action '{action}' references unknown next state '{next}'")]
    UnknownNextState {
        state: String,
        action: String,
        next: String,
    },

    #[error(
        "invalid probability in state '{state}', action '{action}', outcome {outcome_index}: {value}"
    )]
    InvalidProbability {
        state: String,
        action: String,
        outcome_index: usize,
        value: f64,
    },

    #[error(
        "invalid reward in state '{state}', action '{action}', outcome {outcome_index}: {value}"
    )]
    InvalidReward {
        state: String,
        action: String,
        outcome_index: usize,
        value: f64,
    },

    #[error(
        "probability sum for state '{state}', action '{action}' must be within {tolerance} of 1.0, got {sum}"
    )]
    ProbabilitySum {
        state: String,
        action: String,
        sum: f64,
        tolerance: f64,
    },

    #[error("state '{state}' action '{action}' must contain at least one outcome")]
    EmptyOutcomes { state: String, action: String },

    #[error("builder referenced unknown state '{state}'")]
    BuilderUnknownState { state: String },

    #[error("builder referenced unknown action '{action}' in state '{state}'")]
    BuilderUnknownAction { state: String, action: String },
}
