
/// Stores the numbers MCTS updates constantly
pub struct EdgeStats {
    visits: u64,
    value_sum: f64
}

impl EdgeStats {

    /// Retrieve the amount of visits to a certain edge
    pub fn visits(&self) -> u64 {
        self.visits
    }

    /// Increase the visit counter by 1.
    /// Typical during backpropagation.
    pub fn record_visit(&mut self) {
        self.visits += 1;
    }

    /// Retrieve the value sum of a certain edge.
    pub fn value(&self) -> f64 {
        self.value_sum
    }

    /// Increase the value sum of an edge by a certain value.
    pub fn record_value(&mut self, value: f64) {
        self.value_sum += value
    }

    /// Determine the Q value of the edge
    pub fn q(&self) -> f64 {
        self.value_sum / self.visits as f64
    }
}