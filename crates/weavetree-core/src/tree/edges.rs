use crate::tree::{outcomes::OutcomeSet, stats::EdgeStats};

#[derive(Debug, Clone, PartialEq)]
/// represents “taking a particular action from this node.”
/// Allows holding the stats of the edge and the outcomes associated with it
pub struct ActionEdge {
    edge_stats: EdgeStats,
    outcomes: OutcomeSet
}

impl ActionEdge {

    /// Create a new action edge
    pub fn new() -> Self {
        ActionEdge { edge_stats: EdgeStats::new(), outcomes: OutcomeSet::new() }
    }

    /// Function to be used for backpropagation.
    /// Immediately records the rollout return and increments the visits.
    pub fn record(&mut self, rollout_return: f64) {
        self.edge_stats.record(rollout_return);
    }
}