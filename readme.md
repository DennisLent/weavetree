[![codecov](https://codecov.io/gh/DennisLent/weavetree/graph/badge.svg?token=ltemm3sPff)](https://codecov.io/gh/DennisLent/weavetree)
[![docs](https://img.shields.io/badge/docs-github_pages-2ea44f?logo=github)](https://dennislent.github.io/weavetree/)

# Weavetree

A fast, reproducible abstraction-aware Monte Carlo Tree Search engine written in Rust, with optional Python bindings.

## Todo

- weavetree-cli for toolchain purposes
- reproducibility
- abstraction algorithms (partitioning, bisimulation, approximate bisimulation)
- concurrency during simulation for faster sims
- MCTS and uMCTS
- Publish-ready crate metadata (description, license, repo, docs, crate-level readme, keywords, categories)
- root README
- integration level testing

## Issues

- Core uses StateKey(u64) | compiled MDP uses StateKey(usize)1`