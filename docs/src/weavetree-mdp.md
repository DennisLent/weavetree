# Weavetree MDP

`weavetree-mdp` exists to solve a practical gap between environment modeling and search execution.

`weavetree-core` intentionally stays environment-agnostic: it only needs callbacks for action count, transition dynamics, and rollout action selection.
That design keeps search generic and fast, but it means *you* need a clean way to model states, actions, transitions, and rewards.

`weavetree-mdp` provides that modeling layer in two forms:

1. A declarative YAML workflow for explicit finite MDPs.
2. A typed-domain workflow for richer Rust-native state representations.

In both cases, you end up with simulator callbacks that plug directly into `Tree::run`.

## What issue is being solved

Without `weavetree-mdp`, you would need to repeat the same glue code:

- ad-hoc model schema parsing,
- id/key conversion,
- transition sampling,
- validation for probability correctness,
- wiring simulator functions to MCTS.

`weavetree-mdp` centralizes these concerns so you can focus on domain behavior instead of infrastructure.

## Which path should you choose?

- Use **YAML MDP** if your environment is tabular and easy to enumerate.
- Use **Typed Domain** if your state is structured (boards, game objects, nested structs, generated states).

Read the next pages in order:

- [YAML MDP Workflow](./weavetree-mdp/yaml-workflow.md)
- [Typed Domain Workflow](./weavetree-mdp/typed-domain.md)
