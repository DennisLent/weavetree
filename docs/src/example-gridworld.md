# Example: Tiny Gridworld

This example uses a tiny deterministic environment with five labeled states: `0`, `1`, `2`, `3`, and `4`. State `4` is terminal and yields a reward of `1.0` when entered. All other transitions yield `0.0`.

We encode each gridworld state as a `StateKey` value and use two actions:

- Action `0`: move forward (`s -> s + 1`).
- Action `1`: stay in place (`s -> s`).

The search should learn that moving forward from state `0` is better than staying.

```rust
use std::fs;

use weavetree_core::{ActionId, ReturnType, RunLogEvent, SearchConfig, StateKey, Tree};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut tree = Tree::new(StateKey::from(0), false);

    let config = SearchConfig {
        iterations: 6,
        c: 1.4,
        gamma: 1.0,
        max_steps: 8,
        return_type: ReturnType::Discounted,
        fixed_horizon_steps: 8,
    };

    let num_actions = |state: StateKey| if state.value() == 4 { 0 } else { 2 };

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

    println!("{}", RunLogEvent::run_started(&config).to_text_line());
    let mut iteration = 0usize;
    let run = tree.run_with_hook(&config, num_actions, step, rollout_policy, |metrics| {
        let event = RunLogEvent::iteration_completed(iteration, metrics);
        println!("{}", event.to_text_line());
        iteration += 1;
    })?;
    println!("{}", RunLogEvent::run_completed(&run).to_text_line());

    let out_dir = "/tmp/weavetree-doc-runs";
    fs::create_dir_all(out_dir)?;

    let snapshot_json = tree.snapshot_json_pretty()?;
    fs::write(format!("{out_dir}/rust_tree_snapshot.json"), snapshot_json)?;

    println!("completed: {}", run.iterations_completed);
    println!(
        "best root action by value: {}",
        tree.best_root_action_by_value()?.unwrap().index()
    );
    println!("json_path: {out_dir}/rust_tree_snapshot.json");

    Ok(())
}
```

Captured output from running this example:

```text
run_started iterations_requested=6 c=1.400000 gamma=1.000000 max_steps=8 return_type=discounted fixed_horizon_steps=8
iteration_completed iteration=0 leaf_node_id=1 leaf_is_new=true path_len=1 reward_prefix=0.000000 rollout_return=1.000000 total_return=1.000000 node_count=2
iteration_completed iteration=1 leaf_node_id=2 leaf_is_new=true path_len=1 reward_prefix=0.000000 rollout_return=1.000000 total_return=1.000000 node_count=3
iteration_completed iteration=2 leaf_node_id=3 leaf_is_new=true path_len=2 reward_prefix=0.000000 rollout_return=1.000000 total_return=1.000000 node_count=4
iteration_completed iteration=3 leaf_node_id=4 leaf_is_new=true path_len=2 reward_prefix=0.000000 rollout_return=1.000000 total_return=1.000000 node_count=5
iteration_completed iteration=4 leaf_node_id=5 leaf_is_new=true path_len=2 reward_prefix=0.000000 rollout_return=1.000000 total_return=1.000000 node_count=6
iteration_completed iteration=5 leaf_node_id=6 leaf_is_new=true path_len=2 reward_prefix=0.000000 rollout_return=1.000000 total_return=1.000000 node_count=7
run_completed iterations_requested=6 iterations_completed=6 total_return_sum=6.000000 average_total_return=1.000000
completed: 6
best root action by value: 0
json_path: /tmp/weavetree-doc-runs/rust_tree_snapshot.json
```

Exported `tree_snapshot.json` from that same run:

```json
{
  "schema_version": 1,
  "root_node_id": 0,
  "node_count": 7,
  "nodes": [
    {
      "node_id": 0,
      "state_key": 0,
      "depth": 0,
      "is_terminal": false,
      "parent_node_id": null,
      "parent_action_id": null,
      "edges": [
        {
          "action_id": 0,
          "visits": 3,
          "value_sum": 3.0,
          "q": 1.0,
          "outcomes": [
            {
              "next_state_key": 1,
              "child_node_id": 1,
              "count": 3
            }
          ]
        },
        {
          "action_id": 1,
          "visits": 3,
          "value_sum": 3.0,
          "q": 1.0,
          "outcomes": [
            {
              "next_state_key": 0,
              "child_node_id": 2,
              "count": 3
            }
          ]
        }
      ]
    },
    {
      "node_id": 1,
      "state_key": 1,
      "depth": 1,
      "is_terminal": false,
      "parent_node_id": 0,
      "parent_action_id": 0,
      "edges": [
        {
          "action_id": 0,
          "visits": 1,
          "value_sum": 1.0,
          "q": 1.0,
          "outcomes": [
            {
              "next_state_key": 2,
              "child_node_id": 3,
              "count": 1
            }
          ]
        },
        {
          "action_id": 1,
          "visits": 1,
          "value_sum": 1.0,
          "q": 1.0,
          "outcomes": [
            {
              "next_state_key": 1,
              "child_node_id": 5,
              "count": 1
            }
          ]
        }
      ]
    },
    {
      "node_id": 2,
      "state_key": 0,
      "depth": 1,
      "is_terminal": false,
      "parent_node_id": 0,
      "parent_action_id": 1,
      "edges": [
        {
          "action_id": 0,
          "visits": 1,
          "value_sum": 1.0,
          "q": 1.0,
          "outcomes": [
            {
              "next_state_key": 1,
              "child_node_id": 4,
              "count": 1
            }
          ]
        },
        {
          "action_id": 1,
          "visits": 1,
          "value_sum": 1.0,
          "q": 1.0,
          "outcomes": [
            {
              "next_state_key": 0,
              "child_node_id": 6,
              "count": 1
            }
          ]
        }
      ]
    },
    {
      "node_id": 3,
      "state_key": 2,
      "depth": 2,
      "is_terminal": false,
      "parent_node_id": 1,
      "parent_action_id": 0,
      "edges": []
    },
    {
      "node_id": 4,
      "state_key": 1,
      "depth": 2,
      "is_terminal": false,
      "parent_node_id": 2,
      "parent_action_id": 0,
      "edges": []
    },
    {
      "node_id": 5,
      "state_key": 1,
      "depth": 2,
      "is_terminal": false,
      "parent_node_id": 1,
      "parent_action_id": 1,
      "edges": []
    },
    {
      "node_id": 6,
      "state_key": 0,
      "depth": 2,
      "is_terminal": false,
      "parent_node_id": 2,
      "parent_action_id": 1,
      "edges": []
    }
  ]
}
```

In this setup, action `0` from the root should become preferred because it is the only action that can eventually reach terminal state `4` and collect the reward.
