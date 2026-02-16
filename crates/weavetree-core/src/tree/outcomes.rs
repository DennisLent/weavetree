use crate::tree::ids::{NodeId, StateKey};


//TODO: Potentially need to switch the set to a hashmap, lets see about that later
//TODO: for insert: return the child id (or Option<NodeId>) 
// for “observe outcome”: return (child_id, is_new) in one call
// might be more convenient for the rollout loops

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// represents one observed next state under a given `(s,a)` edge.
/// Conceptually it holds `(next_state_key, child_node_id, count)`
pub struct Outcome {
    next_state_key: StateKey,
    child: NodeId,
    count: u64
}

impl Outcome {

    /// Create a new outcome
    /// By default the count is set to 1 as we have just observed it
    fn new(next_state_key: StateKey, child: NodeId) -> Self {
        Outcome { next_state_key, child, count: 1 }
    }

    /// Increment the count of an outcome by 1
    fn increment_count(&mut self) {
        self.count += 1
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// holds all outcomes observed for one action edge.
/// Stores all observed outcomes for a single action edge.
/// That’s how the tree “discovers” stochastic branches naturally.
pub struct OutcomeSet {
    outcomes: Vec<Outcome>
}

impl OutcomeSet {


    /// Create a new empty OutcomeSet
    pub fn new() -> Self {
        
        OutcomeSet { outcomes: Vec::new() }
    }

    /// Find the next node associated to this state key
    /// If found returns `Some(NodeId)` else None
    pub fn get_child_for(&self, next_state_key: StateKey) -> Option<NodeId> {

        let outcome = self.outcomes.iter().find(|outcome| outcome.next_state_key == next_state_key);
        match outcome {
            Some(outcome) => Some(outcome.child),
            None => None
        }
    }

    /// Insert an outcome to the set
    /// We also make sure the Statekey has not been inserted yet
    /// On success return `true` else return `false`
    pub fn insert_outcome(&mut self, next_state_key: StateKey, child_id: NodeId) -> bool {

        if !self.outcomes.iter().any(|outcome| outcome.next_state_key == next_state_key) {
            self.outcomes.push(Outcome::new(next_state_key, child_id));
            true
        }
        else {
            false
        }
    }

    /// Icrement the count on a single occurence
    /// On success return `true` else return `false`
    pub fn increment_count(&mut self, next_state_key: StateKey) -> bool {

        let outcome = self.outcomes.iter_mut().find(|outcome| outcome.next_state_key == next_state_key); 
        match outcome {
            Some(outcome) => {
                outcome.increment_count();
                true
            }
            None => false

        }
    }
}