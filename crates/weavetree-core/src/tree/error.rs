use std::fmt;

use crate::tree::ids::{ActionId, NodeId, StateKey};

/// Error type for MCTS tree construction and search operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TreeError {
    /// Attempted to access a node id that does not exist in the arena.
    MissingNode { node_id: NodeId },
    /// Attempted to access an action edge that does not exist for a node.
    MissingEdge {
        node_id: NodeId,
        action_id: ActionId,
    },
    /// Tree policy could not select an action from a node.
    ActionSelectionFailed { node_id: NodeId },
    /// Attempted to insert an already-observed outcome as new.
    OutcomeInsertFailed {
        node_id: NodeId,
        action_id: ActionId,
    },
    /// Rollout policy returned an action outside `[0, num_actions)`.
    InvalidRolloutAction {
        state_key: StateKey,
        action_id: ActionId,
        num_actions: usize,
    },
}

impl fmt::Display for TreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TreeError::MissingNode { node_id } => {
                write!(f, "missing node with id {}", node_id.index())
            }
            TreeError::MissingEdge { node_id, action_id } => write!(
                f,
                "missing edge {} on node {}",
                action_id.index(),
                node_id.index()
            ),
            TreeError::ActionSelectionFailed { node_id } => {
                write!(f, "failed to select action on node {}", node_id.index())
            }
            TreeError::OutcomeInsertFailed { node_id, action_id } => write!(
                f,
                "failed to insert new outcome for edge {} on node {}",
                action_id.index(),
                node_id.index()
            ),
            TreeError::InvalidRolloutAction {
                state_key,
                action_id,
                num_actions,
            } => write!(
                f,
                "rollout policy selected invalid action {} for state {} with {} actions",
                action_id.index(),
                state_key.value(),
                num_actions
            ),
        }
    }
}

impl std::error::Error for TreeError {}
