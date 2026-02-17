use crate::tree::ids::{NodeId, StateKey};

//TODO: Potentially need to switch the set to a hashmap, lets see about that later

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// represents one observed next state under a given `(s,a)` edge.
/// Conceptually it holds `(next_state_key, child_node_id, count)`
struct Outcome {
    next_state_key: StateKey,
    child: NodeId,
    count: u64,
}

impl Outcome {
    /// Create a new outcome
    /// By default the count is set to 1 as we have just observed it
    fn new(next_state_key: StateKey, child: NodeId) -> Self {
        Outcome {
            next_state_key,
            child,
            count: 1,
        }
    }

    /// Increment the count of an outcome by 1
    fn increment_count(&mut self) {
        self.count += 1
    }

    fn child(&self) -> NodeId {
        self.child
    }

    fn count(&self) -> u64 {
        self.count
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// holds all outcomes observed for one action edge.
/// Stores all observed outcomes for a single action edge.
/// That’s how the tree “discovers” stochastic branches naturally.
pub struct OutcomeSet {
    outcomes: Vec<Outcome>,
}

impl OutcomeSet {
    /// Create a new empty OutcomeSet
    pub fn new() -> Self {
        OutcomeSet {
            outcomes: Vec::new(),
        }
    }

    /// Find the next node associated to this state key
    /// If found returns `Some(NodeId)` else None
    pub fn get_child_for(&self, next_state_key: StateKey) -> Option<NodeId> {
        let outcome = self
            .outcomes
            .iter()
            .find(|outcome| outcome.next_state_key == next_state_key);
        outcome.map(|outcome| outcome.child())
    }

    /// Insert an outcome to the set
    /// We also make sure the Statekey has not been inserted yet
    /// Returns Option<NodeId>, with Some(child_id) in case the insert worked
    pub fn insert_outcome(&mut self, next_state_key: StateKey, child_id: NodeId) -> Option<NodeId> {
        if !self
            .outcomes
            .iter()
            .any(|outcome| outcome.next_state_key == next_state_key)
        {
            self.outcomes.push(Outcome::new(next_state_key, child_id));
            Some(child_id)
        } else {
            None
        }
    }

    /// Icrement the count on a single occurence
    /// Returns Option<NodeId>, with Some(child_id) in case the incrementing worked
    pub fn increment_outcome(&mut self, next_state_key: StateKey) -> Option<NodeId> {
        let outcome = self
            .outcomes
            .iter_mut()
            .find(|outcome| outcome.next_state_key == next_state_key);
        match outcome {
            Some(outcome) => {
                outcome.increment_count();
                Some(outcome.child())
            }
            None => None,
        }
    }

    /// Return the amount of distinct outcomes seen for this edge.
    pub fn len(&self) -> usize {
        self.outcomes.len()
    }

    /// Return how many times a specific next state has been observed.
    pub fn count_for(&self, next_state_key: StateKey) -> Option<u64> {
        self.outcomes
            .iter()
            .find(|outcome| outcome.next_state_key == next_state_key)
            .map(|outcome| outcome.count())
    }
}
