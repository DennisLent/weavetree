use crate::tree::{arena::Arena, ids::{ActionId, NodeId, StateKey}, node::Node};

#[derive(Debug, Clone)]
/// Tree policy that keeps the core generic
/// The tree doesn’t know the simulator type, only queries via closures.
pub struct TreePolicyResult {
    pub path: Vec<(NodeId, ActionId)>, // edges taken from root to leaf
    pub leaf: NodeId,                  // node where rollout should start (often newly created)
    pub leaf_is_new: bool,             // whether we just created this node
    pub reward: f64                    // reward accumulated along the slected path
}


#[derive(Debug, Clone)]
/// owns the arena (root is always at index 0)
/// provides the tree search and operations
pub struct Tree{
    arena: Arena<Node>
}

impl Tree {
    pub fn tree_policy<FNum, FStep>(
        &mut self,
        c: f64,
        mut num_actions: FNum,
        mut step: FStep,
    ) -> Option<TreePolicyResult>
    where
        FNum: FnMut(StateKey) -> usize,
        FStep: FnMut(StateKey, ActionId) -> (StateKey, f64, bool),
    {
        let mut current = NodeId::from(0);
        let mut path: Vec<(NodeId, ActionId)> = Vec::new();
        let mut reward: f64 = 0.0;

        loop {
            let (state_key, depth, is_terminal) = {
                let node = self.arena.get(current)?;
                (node.state_key(), node.depth(), node.is_terminal())
            };

            if is_terminal {
                return Some(TreePolicyResult {
                    path,
                    leaf: current,
                    leaf_is_new: false,
                    reward,
                });
            }

            // Expand action edges if needed
            {
                let node = self.arena.get_mut(current)?;
                if !node.is_expanded() {
                    let n = num_actions(state_key);

                    // If no actions, treat as leaf/terminal-like stop
                    if n == 0 {
                        return Some(TreePolicyResult {
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
                let node = self.arena.get(current)?;
                node.select_edge(c)?
            };

            path.push((current, action));

            // Sample environment outcome (chance)
            let (next_key, r, next_terminal) = step(state_key, action);
            reward += r;

            // Update outcome counts / route to child
            let existing_child = {
                let node = self.arena.get_mut(current)?;
                let edge = node.edge_mut(action)?;

                // if observed before, increment count and get child
                edge.increment_outcome(next_key)
            };

            if let Some(child) = existing_child {
                current = child;
                continue;
            }

            // New outcome: allocate child node
            let child_id = {
                let child_node = Node::new(next_key, depth + 1, Some((current, action)), next_terminal);
                self.arena.allocate(child_node)
            };

            // Register new outcome (count starts at 1)
            {
                let node = self.arena.get_mut(current)?;
                let edge = node.edge_mut(action)?;
                let _ = edge.insert_outcome(next_key, child_id);
            }

            return Some(TreePolicyResult {
                path,
                leaf: child_id,
                leaf_is_new: true,
                reward,
            });
        }
    }
}



