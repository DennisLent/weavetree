# Introduction

Weavetree is a Rust project for fast, reproducible Monte Carlo Tree Search (MCTS).

This repository currently provides two crates that work together:

- `weavetree-core`: the generic MCTS engine (`Tree`, selection/expansion, rollout, backpropagation).
- `weavetree-mdp`: tooling to define environments as MDPs and plug them into `weavetree-core`.

If your environment is simple and table-like, `weavetree-mdp` lets you describe it with YAML and compile it into a runtime simulator.
If your environment needs richer state encoding (structs/enums/boards), `weavetree-mdp` also supports custom typed simulators through `MdpDomain`.

You can also build and edit YAML MDPs in the browser via [Weavetree Studio](./studio/).
