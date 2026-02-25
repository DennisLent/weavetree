# Python Bindings

Python bindings are exposed under one roof via `weavetree`, with two submodules:

- `weavetree.mdp`: YAML compilation and MDP simulator APIs
- `weavetree.mcts`: core MCTS tree and search APIs

## Install for local development

From this repository:

```bash
cd crates/weavetree-py
maturin develop
```

## Imports

Recommended imports:

```python
import weavetree as wt
from weavetree.mdp import compile_yaml_str, MdpSimulator
from weavetree.mcts import SearchConfig, tree
```

## YAML API in Python

### Step 1: Compile from a YAML string

```python
yaml_text = """
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
"""

compiled = compile_yaml_str(yaml_text)
print("state_count:", compiled.state_count())
print("start_state_key:", compiled.start_state_key())
```

### Step 2: Inspect compiled state metadata

```python
start_key = compiled.start_state_key()
print("start id:", compiled.state_id(start_key))
print("start terminal?:", compiled.is_terminal(start_key))
print("start actions:", compiled.num_actions(start_key))

s0_key = compiled.state_key("s0")
print("s0 key:", s0_key)
```

## MdpSimulator API in Python

### Step 1: Create a seeded simulator

```python
sim = MdpSimulator(compiled, 12345)
```

### Step 2: Sample transitions

```python
state_key = compiled.start_state_key()
num_actions = sim.num_actions(state_key)
print("num_actions at start:", num_actions)

next_state_key, reward, terminal = sim.step(state_key, 0)
print("next id:", compiled.state_id(next_state_key))
print("reward:", reward)
print("terminal:", terminal)
```

## MCTS API in Python

Use `weavetree.mcts` to run search directly against `MdpSimulator`.

### Step 1: Construct tree and config

```python
root_key = compiled.start_state_key()
root_terminal = compiled.is_terminal(root_key)

t = tree(root_key, root_terminal)
config = SearchConfig(
    iterations=300,
    c=1.0,
    gamma=1.0,
    max_steps=16,
    return_type="discounted",
    fixed_horizon_steps=16,
)
```

### Step 2: Run search

```python
metrics = t.run(sim, config, rollout_action=0)
print("iterations:", metrics.iterations_completed)
print("avg return:", metrics.average_total_return)
```

Detailed logging is opt-in:

```python
metrics = t.run(
    sim,
    config,
    detailed_logging=True,     # print per-iteration diagnostics
    log_format="jsonl",        # "text" or "jsonl"
    log_path="run.jsonl",      # optional file export
    export_tree_path="tree_snapshot.json",  # optional final tree export
)
```

You can also provide a custom rollout policy callback:

```python
def rollout_policy(state_key: int, num_actions: int) -> int:
    return 0

metrics = t.run(sim, config, rollout_policy=rollout_policy)
```

Callback errors are fail-fast: if `rollout_policy` (or typed-domain callbacks) raises,
`Tree.run` stops immediately and propagates that Python exception.

### Step 3: Read recommended root action

```python
best_by_value = t.best_root_action_by_value()
best_by_visits = t.best_root_action_by_visits()

print("best action (value):", best_by_value)
print("best action (visits):", best_by_visits)
```

## Typed Domain in Python

When YAML is too restrictive, define the domain directly in Python and use
`weavetree.mdp.TypedSimulator`.

Your domain object must implement:

- `start_state() -> Any`
- `state_token(state) -> str | bytes`
- `is_terminal(state) -> bool`
- `num_actions(state) -> int`
- `step(state, action_id, sample) -> (next_state, reward, terminal)`

`state_token` is required so the simulator can intern typed states into stable keys.
Tokens must be unique for semantically distinct states.
Use `check_token_collisions=True` in `TypedSimulator(...)` to enable a runtime
collision check (debug safety mode).

### Example typed domain

```python
from dataclasses import dataclass

from weavetree.mdp import TypedSimulator
from weavetree.mcts import SearchConfig, tree


@dataclass(frozen=True)
class CounterState:
    value: int
    goal: int


class CounterDomain:
    def start_state(self):
        return CounterState(value=0, goal=5)

    def state_token(self, state):
        # Must be deterministic and canonical for equivalent states.
        return f"{state.value}:{state.goal}"

    def is_terminal(self, state):
        return state.value >= state.goal

    def num_actions(self, state):
        return 0 if self.is_terminal(state) else 2

    def step(self, state, action_id, sample):
        if self.is_terminal(state):
            return state, 0.0, True

        if action_id == 0:
            nxt = CounterState(state.value + 1, state.goal)
        elif action_id == 1:
            # stochastic backward move
            delta = 1 if sample < 0.5 else 2
            nxt = CounterState(max(0, state.value - delta), state.goal)
        else:
            return state, 0.0, True

        terminal = nxt.value >= nxt.goal
        reward = 1.0 if terminal else -0.01
        return nxt, reward, terminal
```

### Run MCTS with `TypedSimulator`

```python
domain = CounterDomain()
sim = TypedSimulator(domain, seed=11, check_token_collisions=True)

root_key = sim.start_state_key()
root_terminal = sim.is_terminal_by_key(root_key)

t = tree(root_key, root_terminal)
config = SearchConfig(iterations=400, c=1.0, gamma=1.0, max_steps=16)

metrics = t.run(sim, config, rollout_action=0)
print("iterations:", metrics.iterations_completed)
print("best root action:", t.best_root_action_by_value())
```

### Decode interned states

```python
state_key = sim.start_state_key()
state_obj = sim.state_for_key(state_key)
print("decoded start state:", state_obj)
```

## End-to-end example (with detailed logs + tree snapshot)

```python
from pathlib import Path

from weavetree.mdp import compile_yaml_str, MdpSimulator
from weavetree.mcts import SearchConfig, tree

yaml_text = """
version: 1
start: s0
states:
  - id: s0
    actions:
      - id: a0
        outcomes:
          - next: s1
            prob: 1.0
            reward: 1.0
      - id: a1
        outcomes:
          - next: s2
            prob: 1.0
            reward: 0.2
  - id: s1
    terminal: true
  - id: s2
    terminal: true
"""

compiled = compile_yaml_str(yaml_text)
sim = MdpSimulator(compiled, 7)

root_key = compiled.start_state_key()
t = tree(root_key, compiled.is_terminal(root_key))

config = SearchConfig(iterations=6, c=0.5, gamma=1.0, max_steps=4)

out_dir = Path("/tmp/weavetree-doc-runs")
out_dir.mkdir(parents=True, exist_ok=True)
log_path = out_dir / "python_run.jsonl"
json_path = out_dir / "python_tree_snapshot.json"

metrics = t.run(
    sim,
    config,
    rollout_action=0,
    detailed_logging=True,
    log_format="text",
    log_path=str(log_path),
    export_tree_path=str(json_path),
)

print("completed:", metrics.iterations_completed)
print("best action:", t.best_root_action_by_value())
print("log_path:", log_path)
print("json_path:", json_path)
```

Captured output from running this example:

```text
run_started iterations_requested=6 c=0.500000 gamma=1.000000 max_steps=4 return_type=discounted fixed_horizon_steps=32
iteration_completed iteration=0 leaf_node_id=1 leaf_is_new=true path_len=1 reward_prefix=1.000000 rollout_return=0.000000 total_return=1.000000 node_count=2
iteration_completed iteration=1 leaf_node_id=2 leaf_is_new=true path_len=1 reward_prefix=0.200000 rollout_return=0.000000 total_return=0.200000 node_count=3
iteration_completed iteration=2 leaf_node_id=1 leaf_is_new=false path_len=1 reward_prefix=1.000000 rollout_return=0.000000 total_return=1.000000 node_count=3
iteration_completed iteration=3 leaf_node_id=1 leaf_is_new=false path_len=1 reward_prefix=1.000000 rollout_return=0.000000 total_return=1.000000 node_count=3
iteration_completed iteration=4 leaf_node_id=1 leaf_is_new=false path_len=1 reward_prefix=1.000000 rollout_return=0.000000 total_return=1.000000 node_count=3
iteration_completed iteration=5 leaf_node_id=1 leaf_is_new=false path_len=1 reward_prefix=1.000000 rollout_return=0.000000 total_return=1.000000 node_count=3
run_completed iterations_requested=6 iterations_completed=6 total_return_sum=5.200000 average_total_return=0.866667
completed: 6
best action: 0
log_path: /tmp/weavetree-doc-runs/python_run.jsonl
json_path: /tmp/weavetree-doc-runs/python_tree_snapshot.json
```

Exported `tree_snapshot.json` from that same run:

```json
{
  "schema_version": 1,
  "root_node_id": 0,
  "node_count": 3,
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
          "visits": 5,
          "value_sum": 5.0,
          "q": 1.0,
          "outcomes": [
            {
              "next_state_key": 1,
              "child_node_id": 1,
              "count": 5
            }
          ]
        },
        {
          "action_id": 1,
          "visits": 1,
          "value_sum": 0.2,
          "q": 0.2,
          "outcomes": [
            {
              "next_state_key": 2,
              "child_node_id": 2,
              "count": 1
            }
          ]
        }
      ]
    },
    {
      "node_id": 1,
      "state_key": 1,
      "depth": 1,
      "is_terminal": true,
      "parent_node_id": 0,
      "parent_action_id": 0,
      "edges": []
    },
    {
      "node_id": 2,
      "state_key": 2,
      "depth": 1,
      "is_terminal": true,
      "parent_node_id": 0,
      "parent_action_id": 1,
      "edges": []
    }
  ]
}
```

## Error behavior

- `compile_yaml_file` / `compile_yaml_str` raise `ValueError` on parse or validation failures.
- `CompiledMdp.is_terminal`, `CompiledMdp.num_actions`, `CompiledMdp.state_id`, and `CompiledMdp.state_key` raise `KeyError` for unknown keys/ids.
- `TypedSimulator` raises `TypeError` if `state_token` does not return `str` or `bytes`.
- `TypedSimulator` can raise `ValueError` if `check_token_collisions=True` and two different states return the same token.
- `TypedSimulator` deep-copies states internally to keep interning stable.
- `Tree.run` raises:
  - original Python callback exceptions (typed domain / rollout policy)
  - `ValueError` for core search errors (for example invalid rollout action indices)
