use crate::tree::{edges::ActionEdge, ids::{ActionId, NodeId, StateKey}};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Enum to help determine if a node has been expanded or not
/// Useful for parallel expansion

pub enum ExpansionState{
    Unexpanded,
    Expanding,
    Expanded
}


#[derive(Debug, Clone)]
/// represents a decision state in the search tree.
/// This class iterates edges to select best UCB action and accesses an edge by action index

pub struct Node{
    state_key: StateKey,
    depth: u64,
    parent: Option<(NodeId, ActionId)>,
    edges: Vec<ActionEdge>,
    is_terminal: bool,
    expansion_state: ExpansionState
}

impl Node {

    /// Create a new Node instance
    pub fn new(state_key: StateKey, depth: u64, parent: Option<(NodeId, ActionId)>, is_terminal: bool) -> Self {

        Node { state_key, depth, parent, edges: Vec::new(), is_terminal, expansion_state: ExpansionState::Unexpanded }
    }

    /// Select an edge based on the UCB formula
    pub fn select_edge(&self, c: f64) -> Option<ActionId> {
        let n_parent = self.edges.iter().map(|edge| edge.visits()).sum();
        let scores: Vec<f64> = self.edges.iter().map(|edge| edge.ucb_score(n_parent, c)).collect();
        let index_of_max = scores.iter().enumerate().max_by(|(_, a), (_, b)|a.total_cmp(b)).map(|(index, _)| index);
        match index_of_max {
            Some(index) => Some(self.edges[index].action()),
            None => None
        }

    }

    pub fn depth(&self) -> u64 {
        self.depth
    }

    pub fn state_key(&self) -> StateKey {
        self.state_key
    }

    pub fn is_terminal(&self) -> bool {
        self.is_terminal
    }

    pub fn get_expansion_state(&self) -> ExpansionState {
        self.expansion_state
    }

    pub fn set_expansion_state(&mut self, state: ExpansionState) {
        self.expansion_state = state
    }

    pub fn parent(&self) -> Option<(NodeId, ActionId)>{
        self.parent
    }
}