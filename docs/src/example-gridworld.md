# Example: Tiny Gridworld

This example uses a tiny deterministic environment with five labeled states: `0`, `1`, `2`, `3`, and `4`. State `4` is terminal and yields a reward of `1.0` when entered. All other transitions yield `0.0`.

We encode each gridworld state as a `StateKey` value and use two actions:

- Action `0`: move forward (`s -> s + 1`).
- Action `1`: stay in place (`s -> s`).

The search should learn that moving forward from state `0` is better than staying.

```rust
use weavetree_core::{ActionId, ReturnType, SearchConfig, StateKey, Tree};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut tree = Tree::new(StateKey::from(0), false);

    // Basic configuration
    let config = SearchConfig {
        iterations: 200,
        c: 1.4,
        gamma: 1.0,
        max_steps: 8,
        return_type: ReturnType::Discounted,
        fixed_horizon_steps: 8,
    };

    // 2 actions
    // Forward or stay in place (except in state 4 which is terminal)
    let num_actions = |state: StateKey| {
        if state.value() == 4 { 0 } else { 2 }
    };

    let step = |state: StateKey, action: ActionId| {
        let s = state.value();
        if s == 4 {
            return (state, 0.0, true);
        }

        let next = match action.index() {
            0 => (s + 1).min(4),
            1 => s,
            _ => s,
        };

        let reward = if next == 4 { 1.0 } else { 0.0 };
        let terminal = next == 4;
        (StateKey::from(next), reward, terminal)
    };

    let rollout_policy = |_state: StateKey, _num_actions: usize| ActionId::from(0);

    let run = tree.run(&config, num_actions, step, rollout_policy)?;
    println!("completed: {}", run.iterations_completed);

    if let Some(best) = tree.best_root_action_by_value()? {
        println!("best root action by value: {}", best.index());
    }

    Ok(())
}
```

In this setup, action `0` from the root should become preferred because it is the only action that can eventually reach terminal state `4` and collect the reward.
