# YAML MDP Workflow

This workflow is for environments where you can explicitly list states, actions, outcomes, and transition probabilities.

`weavetree-mdp` gives you:

- `MdpSpec` for schema-level modeling
- validation (duplicate ids, unknown transitions, probability sums)
- `CompiledMdp` for fast runtime transitions
- `MdpSimulator` for seeded sampling

## Step 1: Create a YAML model

Create a file such as `model.yaml`:

```yaml
version: 1
start: s0
states:
  - id: s0
    actions:
      - id: a0
        outcomes:
          - next: s1
            prob: 0.7
            reward: 1.0
          - next: s0
            prob: 0.3
            reward: 0.0
      - id: a1
        outcomes:
          - next: s2
            prob: 1.0
            reward: -0.2
  - id: s1
    terminal: true
  - id: s2
    terminal: false
    actions: []
```

How to read this format:

- `start`: id of the initial state.
- each `state` has unique `id`
- each action contains probabilistic `outcomes`
- each outcome defines `next`, `prob`, and `reward`
- `terminal: true` means the state has no actions.

## Step 2: Use Weavetree Studio for visual editing/compilation

If you prefer not to hand-edit YAML, open [Weavetree Studio](../studio.md).

Studio provides a visual interface to build and edit MDPs, and the resulting model can be used by this same YAML workflow.

## Step 3: Compile and validate YAML

```rust
use weavetree_mdp::compile_yaml;

let compiled = compile_yaml("path-to-your-yaml/model.yaml")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Compilation includes validation. Common failures include:

- missing start state
- duplicate state/action ids
- unknown `next` targets
- action probabilities that do not sum to `1.0` (within a minor tolerance)

## Step 4: Build a simulator

```rust
use weavetree_mdp::MdpSimulator;

let mut simulator = MdpSimulator::new(compiled, 12345);
```

Seeding the simulator is required so that we can have:

- deterministic replay for debugging
- reproducible experiments
- stable test behavior

## Step 5: Wire simulator into `weavetree-core::Tree`

```rust
use std::cell::RefCell;
use weavetree_core::{ActionId, ReturnType, SearchConfig, StateKey as CoreStateKey, Tree};
use weavetree_mdp::{MdpSimulator, StateKey, compile_yaml};

let compiled = compile_yaml("model.yaml")?;
let start = compiled.start();
let simulator = RefCell::new(MdpSimulator::new(compiled, 12345));

let mut tree = Tree::new(CoreStateKey::from(start.index() as u64), false);

let config = SearchConfig {
    iterations: 100,
    c: 0.0,
    gamma: 1.0,
    max_steps: 4,
    return_type: ReturnType::Discounted,
    fixed_horizon_steps: 4,
};

tree.run(
    &config,
    |state| simulator.borrow().num_actions(StateKey::from(state.value() as usize)),
    |state, action| {
        let (next, reward, terminal) = simulator
            .borrow_mut()
            .step(StateKey::from(state.value() as usize), action.index());
        (CoreStateKey::from(next.index() as u64), reward, terminal)
    },
    |_state, _num_actions| ActionId::from(0),
)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

What is happening under the hood:

- `Tree` uses `u64`-based `StateKey`
- `MdpSimulator` uses dense state indices
- the closures convert between those key types while preserving the same state identity

## Step 6: Extract decisions

After `run`, query the root recommendation:

```rust
let best = tree.best_root_action_by_value()?;
println!("best root action: {:?}", best.map(|a| a.index()));
# Ok::<(), Box<dyn std::error::Error>>(())
```

For a runnable end-to-end file, see `crates/weavetree-mdp/examples/core_mcts.rs`.
