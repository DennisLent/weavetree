/// TODO: Potential for memeory optimization to use u32 and f32 instead.
/// Stores the numbers MCTS updates constantly
#[derive(Debug, Clone, Copy)]
pub struct EdgeStats {
    visits: u64,
    value_sum: f64,
}

impl EdgeStats {
    pub fn new() -> Self {
        EdgeStats {
            visits: 0,
            value_sum: 0.0,
        }
    }

    /// Retrieve the amount of visits to a certain edge
    pub fn visits(&self) -> u64 {
        self.visits
    }

    /// Increase the visit counter by 1.
    /// Typical during backpropagation.
    fn record_visit(&mut self) {
        self.visits += 1;
    }

    /// Retrieve the value sum of a certain edge.
    pub fn value_sum(&self) -> f64 {
        self.value_sum
    }

    /// Increase the value sum of an edge by a certain value.
    fn record_value(&mut self, rollout_return: f64) {
        self.value_sum += rollout_return
    }

    /// Function to be used for backpropagation.
    /// Immediately records the rollout return and increments the visits.
    pub fn record(&mut self, rollout_return: f64) {
        self.record_visit();
        self.record_value(rollout_return);
    }

    /// Helper function just to check if the edge has been visisted or not
    pub fn is_unvisited(&self) -> bool {
        self.visits == 0
    }

    /// Determine the Q value of the edge
    pub fn q(&self) -> f64 {
        if self.is_unvisited() {
            0.0
        } else {
            self.value_sum / self.visits as f64
        }
    }
}
