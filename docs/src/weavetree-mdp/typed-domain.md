# Typed Domain Workflow

Use this workflow when your state cannot be expressed cleanly as a fixed list of string ids.
Examples: board games, inventories, dynamic objects, nested world state.

`weavetree-mdp` supports this via `MdpDomain` + `DomainSimulator`.

## Step 1: Define your state type

State must be:

- `Clone`
- `Eq`
- `Hash`

Example:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MyState {
    position: i32,
    goal: i32,
}
```

## Step 2: Implement `MdpDomain`

The trait describes environment dynamics:

- `start_state`: initial state
- `is_terminal`: stop condition
- `num_actions`: dense action count for current state
- `step`: transition function `(state, action_id, sample) -> (next, reward, terminal)`

```rust
use weavetree_mdp::MdpDomain;

struct MyDomain;

impl MdpDomain for MyDomain {
    type State = MyState;

    fn start_state(&self) -> Self::State {
        MyState { position: 0, goal: 5 }
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        state.position == state.goal
    }

    fn num_actions(&self, state: &Self::State) -> usize {
        if self.is_terminal(state) { 0 } else { 2 }
    }

    fn step(&self, state: &Self::State, action_id: usize, _sample: f64) -> (Self::State, f64, bool) {
        if self.is_terminal(state) {
            return (state.clone(), 0.0, true);
        }

        let mut next = state.clone();
        match action_id {
            0 => next.position += 1, // move toward goal
            1 => next.position -= 1, // move away
            _ => return (state.clone(), 0.0, true),
        }

        let terminal = next.position == next.goal;
        let reward = if terminal { 1.0 } else { -0.01 };
        (next, reward, terminal)
    }
}
```

## Step 3: Build `DomainSimulator`

```rust
use weavetree_mdp::DomainSimulator;

let shared = DomainSimulator::new(MyDomain, 7).into_shared();
```

What `DomainSimulator` handles for you:

- deterministic RNG from seed
- state interning from typed state to stable `u64` keys
- callback adapters for `Tree::run`

## Step 4: Initialize the tree and search config

```rust
use weavetree_core::{ReturnType, SearchConfig, Tree};

let mut tree = Tree::new(shared.start_state_key(), shared.root_is_terminal());

let config = SearchConfig {
    iterations: 200,
    c: 1.0,
    gamma: 1.0,
    max_steps: 16,
    return_type: ReturnType::Discounted,
    fixed_horizon_steps: 16,
};
```

## Step 5: Run MCTS with simulator-provided closures

```rust
use weavetree_core::ActionId;

tree.run(
    &config,
    shared.num_actions_fn(),
    shared.step_fn(),
    |_state, _num_actions| ActionId::from(0),
)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Why this is useful:

- you keep full control over domain logic
- no need to flatten complex state manually
- migration path from simple to complex environments without having to adapt

## Step 6: Inspect chosen action

```rust
let best = tree.best_root_action_by_value()?;
println!("best action index: {:?}", best.map(|a| a.index()));
# Ok::<(), Box<dyn std::error::Error>>(())
```

If your action indices map to richer actions (e.g., board coordinates), decode them in your domain layer.

For a complete runnable typed-domain example, see `crates/weavetree-mdp/examples/tic_tac_toe.rs`.
