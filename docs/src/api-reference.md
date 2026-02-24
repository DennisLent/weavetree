# Public API Reference (`weavetree-core`)

This page documents the core search crate (`weavetree-core`).
For environment modeling and simulator APIs, see [Using weavetree-mdp](./weavetree-mdp.md).

The crate root re-exports the primary types used by consumers.

## Re-exported types

- `Tree` and `TreePolicyResult`
- `TreeSnapshot`, `NodeSnapshot`, `ActionEdgeSnapshot`, `OutcomeSnapshot`
- `SearchConfig`, `SearchConfigError`
- `IterationMetrics`, `RunMetrics`
- `ReturnType`
- `TreeError`
- `NodeId`, `ActionId`, `StateKey`

## `Tree`

Main constructor and query methods:

```rust
use weavetree_core::{StateKey, Tree};

let tree = Tree::new(StateKey::from(0), false);
let root = tree.root_id();
let count = tree.node_count();
```

Search entry points:

- `iterate(...)` executes one iteration.
- `run(...)` executes many iterations.
- `run_with_hook(...)` executes many iterations with per-iteration callback.

Export entry points:

- `snapshot()` exports a structured tree snapshot.
- `snapshot_json_pretty()` exports the same snapshot as pretty JSON text.

Decision extraction:

- `best_root_action_by_visits()` picks root edge with highest visit count.
- `best_root_action_by_value()` picks root edge with highest mean value `q`.

Both methods return `Result<Option<ActionId>, TreeError>`. `None` means the root has no edges yet.

## Metrics

`IterationMetrics` is emitted per iteration and includes:

- `leaf`
- `leaf_is_new`
- `path_len`
- `reward_prefix`
- `rollout_return`
- `total_return`
- `node_count`

`RunMetrics` aggregates:

- `iterations_requested`
- `iterations_completed`
- `total_return_sum`
- `average_total_return`

Standardized detailed logging events are available via `RunLogEvent`:

- `run_started`
- `iteration_completed`
- `run_completed`

## `ReturnType`

Rollout return behavior:

- `Discounted`: `r0 + gamma*r1 + gamma^2*r2 + ...`
- `EpisodicUndiscounted`: sum of rewards with no discount.
- `FixedHorizon`: undiscounted sum with strict horizon cap.

## ID wrappers

ID types expose `.index()` for `NodeId` and `ActionId`, and `.value()` for `StateKey`:

```rust
use weavetree_core::{ActionId, StateKey};

let action = ActionId::from(3);
let state = StateKey::from(42);

assert_eq!(action.index(), 3);
assert_eq!(state.value(), 42);
```
