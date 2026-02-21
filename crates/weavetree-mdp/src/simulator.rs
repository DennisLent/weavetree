use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{CompiledMdp, StateKey};

#[derive(Debug, Clone)]
/// Seeded simulator over a compiled MDP.
pub struct MdpSimulator {
    mdp: CompiledMdp,
    rng: ChaCha8Rng,
}

impl MdpSimulator {
    /// Create a simulator with deterministic RNG seed.
    pub fn new(mdp: CompiledMdp, seed: u64) -> Self {
        Self {
            mdp,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Borrow the underlying compiled MDP.
    pub fn mdp(&self) -> &CompiledMdp {
        &self.mdp
    }

    /// Return how many actions are available for a state.
    pub fn num_actions(&self, state_key: StateKey) -> usize {
        self.mdp.num_actions(state_key).unwrap_or(0)
    }

    /// Sample one `(next_state, reward, terminal)` transition.
    /// Invalid state/action inputs are treated as a no-op terminal transition.
    pub fn step(&mut self, state_key: StateKey, action_id: usize) -> (StateKey, f64, bool) {
        let sample = (self.rng.next_u64() as f64) / ((u64::MAX as f64) + 1.0);
        self.mdp
            .sample_transition(state_key, action_id, sample)
            .unwrap_or((state_key, 0.0, true))
    }
}
