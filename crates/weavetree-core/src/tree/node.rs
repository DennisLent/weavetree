use crate::tree::{
    edges::ActionEdge,
    ids::{ActionId, NodeId, StateKey},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Enum to help determine if a node has been expanded or not
/// Useful for parallel expansion

pub enum ExpansionState {
    Unexpanded,
    Expanding,
    Expanded,
}

#[derive(Debug, Clone)]
/// represents a decision state in the search tree.
/// This class iterates edges to select best UCB action and accesses an edge by action index

pub struct Node {
    state_key: StateKey,
    depth: u64,
    parent: Option<(NodeId, ActionId)>,
    edges: Vec<ActionEdge>,
    is_terminal: bool,
    expansion_state: ExpansionState,
}

impl Node {
    /// Create a new Node instance
    pub fn new(
        state_key: StateKey,
        depth: u64,
        parent: Option<(NodeId, ActionId)>,
        is_terminal: bool,
    ) -> Self {
        Node {
            state_key,
            depth,
            parent,
            edges: Vec::new(),
            is_terminal,
            expansion_state: ExpansionState::Unexpanded,
        }
    }

    /// Expand this node by creating an edge per legal action.
    /// The search loop determines `num_actions` from the environment.
    pub fn expand(&mut self, num_actions: usize) {
        if self.expansion_state == ExpansionState::Expanded {
            return;
        }

        self.edges = (0..num_actions)
            .map(|i| ActionEdge::new(ActionId::from(i)))
            .collect();

        self.expansion_state = ExpansionState::Expanded;
    }

    /// Select an edge based on UCB. 
    /// Returns the chosen `ActionId` (index in `edges`).
    pub fn select_edge(&self, c: f64) -> Option<ActionId> {
        if self.edges.is_empty() {
            return None;
        }

        // Parent visit count: sum of child edge visits
        let n_parent: u64 = self.edges.iter().map(|e| e.visits()).sum::<u64>().max(1);

        // track best score + best index.
        let mut best_idx: usize = 0;
        let mut best_score: f64 = f64::NEG_INFINITY;

        for (i, edge) in self.edges.iter().enumerate() {
            let score = edge.ucb_score(n_parent, c);

            // tie breaker in case of similar scores prefer smaller index.
            if score > best_score || (score == best_score && i < best_idx) {
                best_score = score;
                best_idx = i;
            }
        }

        Some(ActionId::from(best_idx))
    }

    /// Using an action id, return the corresponding action edge
    pub fn edge(&self, action_id: ActionId) -> Option<&ActionEdge> {
        self.edges.get(action_id.index())
    }

    /// Using an action id, return the corresponding action edge as mutable
    pub fn edge_mut(&mut self, action_id: ActionId) -> Option<&mut ActionEdge> {
        self.edges.get_mut(action_id.index())
    }

    /// Return the depth of a specific node
    pub fn depth(&self) -> u64 {
        self.depth
    }

    /// Return the state key of a specific node
    pub fn state_key(&self) -> StateKey {
        self.state_key
    }

    /// Check function to see if a node is terminal
    pub fn is_terminal(&self) -> bool {
        self.is_terminal
    }

    /// Returns the expansion state of a node
    pub fn expansion_state(&self) -> ExpansionState {
        self.expansion_state
    }

    /// Sets the expansion state of a node
    pub fn set_expansion_state(&mut self, state: ExpansionState) {
        self.expansion_state = state
    }

    /// Helper to be called to see if a node is expanded
    pub fn is_expanded(&self) -> bool {
        self.expansion_state == ExpansionState::Expanded
    }

    /// Return the parent of a given node
    pub fn parent(&self) -> Option<(NodeId, ActionId)> {
        self.parent
    }
}
