use crate::tree::{
    arena::Arena,
    error::TreeError,
    ids::{ActionId, NodeId, StateKey},
    node::Node,
};

#[derive(Debug, Clone)]
/// Tree policy that keeps the core generic
/// The tree doesn’t know the simulator type, only queries via closures.
pub struct TreePolicyResult {
    pub path: Vec<(NodeId, ActionId)>, // edges taken from root to leaf
    pub leaf: NodeId,                  // node where rollout should start (often newly created)
    pub leaf_is_new: bool,             // whether we just created this node
    pub reward: f64,                   // reward accumulated along the selected path
}

#[derive(Debug, Clone)]
/// owns the arena (root is always at index 0)
/// provides the tree search and operations
pub struct Tree {
    arena: Arena<Node>,
}

impl Tree {
    /// Create a tree with a single root node.
    pub fn new(root_state_key: StateKey, root_is_terminal: bool) -> Self {
        let mut arena = Arena::new();
        let root = Node::new(root_state_key, 0, None, root_is_terminal);
        let _ = arena.allocate(root);
        Tree { arena }
    }

    /// Return the root node id.
    pub fn root_id(&self) -> NodeId {
        NodeId::from(0)
    }

    /// Return how many nodes exist in the tree arena.
    pub fn node_count(&self) -> usize {
        self.arena.len()
    }

    /// Return an immutable node handle.
    pub(crate) fn node(&self, node_id: NodeId) -> Result<&Node, TreeError> {
        self.arena
            .get(node_id)
            .ok_or(TreeError::MissingNode { node_id })
    }

    /// Return a mutable node handle.
    pub(crate) fn node_mut(&mut self, node_id: NodeId) -> Result<&mut Node, TreeError> {
        self.arena
            .get_mut(node_id)
            .ok_or(TreeError::MissingNode { node_id })
    }

    /// Pick the root action with the highest visit count.
    pub fn best_root_action_by_visits(&self) -> Result<Option<ActionId>, TreeError> {
        let root = self.node(self.root_id())?;
        let mut best: Option<(ActionId, u64)> = None;

        for edge in root.edges() {
            let candidate = (edge.action(), edge.visits());
            best = match best {
                Some((best_action, best_visits))
                    if best_visits > candidate.1
                        || (best_visits == candidate.1
                            && best_action.index() < candidate.0.index()) =>
                {
                    Some((best_action, best_visits))
                }
                _ => Some(candidate),
            };
        }

        Ok(best.map(|(action, _)| action))
    }

    /// Pick the root action with the highest mean value estimate.
    pub fn best_root_action_by_value(&self) -> Result<Option<ActionId>, TreeError> {
        let root = self.node(self.root_id())?;
        let mut best: Option<(ActionId, f64)> = None;

        for edge in root.edges() {
            let candidate = (edge.action(), edge.q());
            best = match best {
                Some((best_action, best_q))
                    if best_q > candidate.1
                        || (best_q == candidate.1 && best_action.index() < candidate.0.index()) =>
                {
                    Some((best_action, best_q))
                }
                _ => Some(candidate),
            };
        }

        Ok(best.map(|(action, _)| action))
    }

    pub fn tree_policy<FNum, FStep>(
        &mut self,
        c: f64,
        mut num_actions: FNum,
        mut step: FStep,
    ) -> Result<TreePolicyResult, TreeError>
    where
        FNum: FnMut(StateKey) -> usize,
        FStep: FnMut(StateKey, ActionId) -> (StateKey, f64, bool),
    {
        let mut current = self.root_id();
        let mut path: Vec<(NodeId, ActionId)> = Vec::new();
        let mut reward: f64 = 0.0;

        loop {
            let (state_key, depth, is_terminal) = {
                let node = self.node(current)?;
                (node.state_key(), node.depth(), node.is_terminal())
            };

            if is_terminal {
                return Ok(TreePolicyResult {
                    path,
                    leaf: current,
                    leaf_is_new: false,
                    reward,
                });
            }

            // Expand action edges if needed
            {
                let node = self.node_mut(current)?;
                if !node.is_expanded() {
                    let n = num_actions(state_key);

                    // If no actions, treat as leaf/terminal-like stop
                    if n == 0 {
                        return Ok(TreePolicyResult {
                            path,
                            leaf: current,
                            leaf_is_new: false,
                            reward,
                        });
                    }

                    node.expand(n);
                }
            }

            // Pick action by UCB
            let action = {
                let node = self.node(current)?;
                node.select_edge(c)
                    .ok_or(TreeError::ActionSelectionFailed { node_id: current })?
            };

            path.push((current, action));

            // Sample environment outcome (chance)
            let (next_key, r, next_terminal) = step(state_key, action);
            reward += r;

            // Update outcome counts / route to child
            let existing_child = {
                let node = self.node_mut(current)?;
                let edge = node.edge_mut(action).ok_or(TreeError::MissingEdge {
                    node_id: current,
                    action_id: action,
                })?;

                // if observed before, increment count and get child
                edge.increment_outcome(next_key)
            };

            if let Some(child) = existing_child {
                current = child;
                continue;
            }

            // New outcome: allocate child node
            let child_id = {
                let child_node =
                    Node::new(next_key, depth + 1, Some((current, action)), next_terminal);
                self.arena.allocate(child_node)
            };

            // Register new outcome (count starts at 1)
            {
                let node = self.node_mut(current)?;
                let edge = node.edge_mut(action).ok_or(TreeError::MissingEdge {
                    node_id: current,
                    action_id: action,
                })?;
                edge.insert_outcome(next_key, child_id)
                    .ok_or(TreeError::OutcomeInsertFailed {
                        node_id: current,
                        action_id: action,
                    })?;
            }

            return Ok(TreePolicyResult {
                path,
                leaf: child_id,
                leaf_is_new: true,
                reward,
            });
        }
    }
}
