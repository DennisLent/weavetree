use core::f64;

use crate::tree::{
    ids::{ActionId, NodeId, StateKey},
    outcomes::OutcomeSet,
    stats::EdgeStats,
};

#[derive(Debug, Clone)]
/// represents “taking a particular action from this node.”
/// Allows holding the stats of the edge and the outcomes associated with it
pub struct ActionEdge {
    action: ActionId,
    edge_stats: EdgeStats,
    outcomes: OutcomeSet,
}

impl ActionEdge {
    /// Create a new action edge
    pub fn new(action: ActionId) -> Self {
        ActionEdge {
            edge_stats: EdgeStats::new(),
            outcomes: OutcomeSet::new(),
            action,
        }
    }

    /// Getter for the actionId
    pub fn action(&self) -> ActionId {
        self.action
    }

    /// Function to be used for backpropagation.
    /// Immediately records the rollout return and increments the visits.
    pub fn record(&mut self, rollout_return: f64) {
        self.edge_stats.record(rollout_return);
    }

    /// Calculate UCB score for this given edge
    pub fn ucb_score(&self, n_parent: u64, c: f64) -> f64 {
        if self.edge_stats.is_unvisited() {
            f64::INFINITY
        } else {
            self.edge_stats.q()
                + c * f64::sqrt(f64::ln(n_parent as f64) / self.edge_stats.visits() as f64)
        }
    }

    /// Find the next node associated to this state key
    /// If found returns `Some(NodeId)` else None
    pub fn get_child_for(&self, next_state_key: StateKey) -> Option<NodeId> {
        self.outcomes.get_child_for(next_state_key)
    }

    /// Insert an outcome to the OutcomeSet
    /// We also make sure the Statekey has not been inserted yet
    /// Returns Option<NodeId>, with Some(child_id) in case the insert worked
    pub fn insert_outcome(&mut self, next_state_key: StateKey, child_id: NodeId) -> Option<NodeId> {
        self.outcomes.insert_outcome(next_state_key, child_id)
    }

    /// Icrement the count on a single occurence
    /// Returns Option<NodeId>, with Some(child_id) in case the incrementing worked
    pub fn increment_outcome(&mut self, next_state_key: StateKey) -> Option<NodeId> {
        self.outcomes.increment_outcome(next_state_key)
    }

    /// Return the amount of times this edge has been visited
    pub fn visits(&self) -> u64 {
        self.edge_stats.visits()
    }

    /// Return the mean value estimate for this edge.
    pub fn q(&self) -> f64 {
        self.edge_stats.q()
    }

    /// Return the amount of distinct outcomes observed under this edge.
    pub fn outcomes_len(&self) -> usize {
        self.outcomes.len()
    }

    /// Return the count for a given observed next state key.
    pub fn outcome_count_for(&self, next_state_key: StateKey) -> Option<u64> {
        self.outcomes.count_for(next_state_key)
    }
}
