use std::hash::Hash;

/// Generic interface for user-defined MDP domains with arbitrary state types.
pub trait MdpDomain {
    type State: Clone + Eq + Hash;

    /// Return the initial state of the domain.
    fn start_state(&self) -> Self::State;

    /// Return whether a state is terminal.
    fn is_terminal(&self, state: &Self::State) -> bool;

    /// Return the number of available actions for a state.
    fn num_actions(&self, state: &Self::State) -> usize;

    /// Sample one transition using a uniform random sample in `[0, 1)`.
    fn step(&self, state: &Self::State, action_id: usize, sample: f64) -> (Self::State, f64, bool);
}
