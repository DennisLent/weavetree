/// A wraper for an integer index used to index nodes
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

impl NodeId {
    /// Get the value of the actual node without having to access and risk overiding the internal value
    pub fn index(&self) -> usize {
        self.0
    }
}

impl From<usize> for NodeId {
    /// Allow for explicit conversion from usize to NodeId
    fn from(value: usize) -> Self {
        NodeId(value)
    }
}

/// Representation of the state to avoid storing the full state and heavy cloning.
/// This needs to be deterministic, collision-resistant, and must not depend on rollout/search metadata.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct StateKey(u64);

impl StateKey {
    /// Return the internal numeric representation of this key.
    pub fn value(&self) -> u64 {
        self.0
    }
}

impl From<u64> for StateKey {
    /// Allow for explicit conversion from u64 to StateKey.
    fn from(value: u64) -> Self {
        StateKey(value)
    }
}

/// A wraper for an integer index used to determine the node's action list
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ActionId(usize);

impl ActionId {
    /// Get the value of the actual action without having to access and risk overiding the internal value
    pub fn index(&self) -> usize {
        self.0
    }
}

impl From<usize> for ActionId {
    /// Allow for explicit conversion from usize to ActionId
    fn from(value: usize) -> Self {
        ActionId(value)
    }
}
