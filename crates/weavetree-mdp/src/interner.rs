use std::collections::HashMap;
use std::hash::Hash;

/// Stable key interner for arbitrary states.
#[derive(Debug, Clone)]
pub struct StateInterner<S>
where
    S: Clone + Eq + Hash,
{
    states: Vec<S>,
    state_to_key: HashMap<S, u64>,
}

impl<S> Default for StateInterner<S>
where
    S: Clone + Eq + Hash,
{
    fn default() -> Self {
        Self {
            states: Vec::new(),
            state_to_key: HashMap::new(),
        }
    }
}

impl<S> StateInterner<S>
where
    S: Clone + Eq + Hash,
{
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert the state if needed and return a stable dense key.
    pub fn intern(&mut self, state: S) -> u64 {
        if let Some(key) = self.state_to_key.get(&state) {
            return *key;
        }

        let key = self.states.len() as u64;
        self.states.push(state.clone());
        self.state_to_key.insert(state, key);
        key
    }

    pub fn get(&self, key: u64) -> Option<&S> {
        self.states.get(key as usize)
    }

    pub fn key_of(&self, state: &S) -> Option<u64> {
        self.state_to_key.get(state).copied()
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}
