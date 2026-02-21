# Introduction

Weavetree is a project focused on fast and reproducible Monte Carlo Tree Search (MCTS) in Rust.

You can build MDPs directly in the browser using [Weavetree Studio](./studio/), which is hosted with the docs.

At the moment, this repository contains one crate: `weavetree-core`. That crate provides the core search functionality, including tree traversal, rollout handling, and backpropagation.

The long-term structure can expand, but today `weavetree-core` is the main implementation and the right place to start.
