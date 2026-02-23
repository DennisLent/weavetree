# weavetree Python bindings

This crate exposes `weavetree-core` and `weavetree-mdp` in one Python module: `weavetree`.

## Build/install locally

```bash
cd crates/weavetree-py
maturin develop
```

## Imports

```python
import weavetree as wt
from weavetree.mdp import compile_yaml_str
from weavetree.mcts import tree
```

## Exposed API

- `weavetree.mdp`
  - `compile_yaml_file(path: str) -> CompiledMdp`
  - `compile_yaml_str(yaml: str) -> CompiledMdp`
  - `CompiledMdp`
  - `MdpSimulator`
  - `TypedSimulator`
- `weavetree.mcts`
  - `SearchConfig`
  - `RunMetrics`
  - `Tree`
  - `tree(root_state_key: int, root_is_terminal: bool) -> Tree`
