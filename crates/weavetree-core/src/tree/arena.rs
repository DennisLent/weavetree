use std::{slice::{Iter, IterMut}, vec::IntoIter};

use crate::tree::ids::NodeId;

/// Holds all items and allows for fast allocation and is cache friendly.
struct Arena<T>{
    storage: Vec<T>,
}

impl<T> Arena<T> {

    /// Create a new empty storage
    pub fn new() -> Self{
        Arena { storage: Vec::new() }
    }

    /// Allocate a new item to the storage and return the associated NodeId
    pub fn allocate(&mut self, item: T) -> NodeId {
        let id = NodeId::from(self.storage.len());
        self.storage.push(item);
        id
    }

    /// Retrieve an associated item from the Area
    pub fn get(&self, node_id: NodeId) -> Option<&T> {
        self.storage.get(node_id.index())
    }

    /// Retrieve an associated item from the Arena as a mutable borrow
    pub fn get_mut(&mut self, node_id: NodeId) -> Option<&mut T> {
        self.storage.get_mut(node_id.index())
    }

    /// Check the length of the Arena
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Check if the Arena is empty
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Iteration helper for the Arena
    pub fn iter(&self) -> Iter<'_, T> {
        self.storage.iter()
    }

    /// Mutable iteration helper for the Arena
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.storage.iter_mut()
    }
}

/// Iteration support for Arena.
/// - `for x in arena` (moves items out)
impl<T> IntoIterator for Arena<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.storage.into_iter()
    }
}

/// Iteration support for Arena.
/// - `for x in &arena` (borrows items)
impl<'a, T> IntoIterator for &'a Arena<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.storage.iter()
    }
}

/// Iteration support for Arena.
/// - `for x in &mut arena` (mutably borrows items)
impl<'a, T> IntoIterator for &'a mut Arena<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.storage.iter_mut()
    }
}
