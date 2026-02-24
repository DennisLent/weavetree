use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct TreeSnapshot {
    pub schema_version: u32,
    pub root_node_id: usize,
    pub node_count: usize,
    pub nodes: Vec<NodeSnapshot>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeSnapshot {
    pub node_id: usize,
    pub state_key: u64,
    pub depth: u64,
    pub is_terminal: bool,
    pub parent_node_id: Option<usize>,
    pub parent_action_id: Option<usize>,
    pub edges: Vec<ActionEdgeSnapshot>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ActionEdgeSnapshot {
    pub action_id: usize,
    pub visits: u64,
    pub value_sum: f64,
    pub q: f64,
    pub outcomes: Vec<OutcomeSnapshot>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OutcomeSnapshot {
    pub next_state_key: u64,
    pub child_node_id: usize,
    pub count: u64,
}
