use std::{cell::RefCell, rc::Rc};

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use weavetree_core::{ActionId, StateKey as CoreStateKey};

use crate::{CompiledMdp, MdpDomain, StateInterner, StateKey};

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

#[derive(Debug, Clone)]
/// Seeded simulator over a user-defined MDP domain with arbitrary state encoding.
pub struct DomainSimulator<D>
where
    D: MdpDomain,
{
    domain: D,
    state_interner: StateInterner<D::State>,
    rng: ChaCha8Rng,
}

impl<D> DomainSimulator<D>
where
    D: MdpDomain,
{
    /// Create a domain simulator with deterministic RNG seed.
    pub fn new(domain: D, seed: u64) -> Self {
        let mut state_interner = StateInterner::new();
        let _ = state_interner.intern(domain.start_state());
        Self {
            domain,
            state_interner,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Return the key of the domain start state.
    pub fn start_state_key(&self) -> u64 {
        0
    }

    /// Borrow the underlying domain implementation.
    pub fn domain(&self) -> &D {
        &self.domain
    }

    /// Resolve a key back into its decoded state.
    pub fn state_for_key(&self, key: u64) -> Option<&D::State> {
        self.state_interner.get(key)
    }

    /// Return whether an interned state key is terminal.
    pub fn is_terminal_by_key(&self, state_key: u64) -> bool {
        self.state_interner
            .get(state_key)
            .map(|state| self.domain.is_terminal(state))
            .unwrap_or(true)
    }

    /// Return how many actions are available for an interned state key.
    pub fn num_actions_by_key(&self, state_key: u64) -> usize {
        self.state_interner
            .get(state_key)
            .map(|state| self.domain.num_actions(state))
            .unwrap_or(0)
    }

    /// Sample one `(next_state_key, reward, terminal)` transition.
    /// Invalid state/action inputs are treated as a no-op terminal transition.
    pub fn step_by_key(&mut self, state_key: u64, action_id: usize) -> (u64, f64, bool) {
        let Some(state) = self.state_interner.get(state_key).cloned() else {
            return (state_key, 0.0, true);
        };

        let sample = (self.rng.next_u64() as f64) / ((u64::MAX as f64) + 1.0);
        let (next_state, reward, terminal) = self.domain.step(&state, action_id, sample);
        let next_key = self.state_interner.intern(next_state);
        (next_key, reward, terminal)
    }

    /// Wrap this simulator in shared interior mutability for MCTS callback wiring.
    pub fn into_shared(self) -> SharedDomainSimulator<D> {
        SharedDomainSimulator::new(self)
    }
}

/// Shared wrapper that offers direct callback adapters for `weavetree_core::Tree::run`.
#[derive(Clone)]
pub struct SharedDomainSimulator<D>
where
    D: MdpDomain,
{
    inner: Rc<RefCell<DomainSimulator<D>>>,
}

impl<D> SharedDomainSimulator<D>
where
    D: MdpDomain,
{
    pub fn new(simulator: DomainSimulator<D>) -> Self {
        Self {
            inner: Rc::new(RefCell::new(simulator)),
        }
    }

    /// Return the encoded start state key for tree initialization.
    pub fn start_state_key(&self) -> CoreStateKey {
        let key = self.inner.borrow().start_state_key();
        CoreStateKey::from(key)
    }

    /// Return whether the root state is terminal.
    pub fn root_is_terminal(&self) -> bool {
        let key = self.inner.borrow().start_state_key();
        self.inner.borrow().is_terminal_by_key(key)
    }

    /// Build a callback compatible with `Tree::run` `num_actions`.
    pub fn num_actions_fn(&self) -> impl FnMut(CoreStateKey) -> usize + '_ {
        let inner = Rc::clone(&self.inner);
        move |state| inner.borrow().num_actions_by_key(state.value())
    }

    /// Build a callback compatible with `Tree::run` `step`.
    pub fn step_fn(&self) -> impl FnMut(CoreStateKey, ActionId) -> (CoreStateKey, f64, bool) + '_ {
        let inner = Rc::clone(&self.inner);
        move |state, action| {
            let (next, reward, terminal) = inner
                .borrow_mut()
                .step_by_key(state.value(), action.index());
            (CoreStateKey::from(next), reward, terminal)
        }
    }
}
