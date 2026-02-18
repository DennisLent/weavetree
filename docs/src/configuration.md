# Configuration

Search configuration is represented by `SearchConfig` in `tree/mcts.rs`.

## Fields

- `iterations`: number of MCTS iterations to execute (i.e. simulation limit)
- `c`: exploration constant in UCB.
- `gamma`: discount factor for discounted rollouts.
- `max_steps`: global rollout cap (i.e. simulation depth)
- `return_type`: one of `discounted`, `episodic_undiscounted`, or `fixed_horizon`.
- `fixed_horizon_steps`: secondary cap used when return type is fixed horizon.

## Default values

The crate default is:

```yaml
iterations: 256
c: 1.4
gamma: 1.0
max_steps: 128
return_type: discounted
fixed_horizon_steps: 32
```

This YAML is embedded into the crate as `search.default.yaml`.

## Loading from YAML

You can parse config from a string, file path, or the embedded default.

```rust
use weavetree_core::SearchConfig;

let from_default = SearchConfig::from_default_yaml()?;
let from_str = SearchConfig::from_yaml_str(
    r#"
iterations: 512
c: 1.2
gamma: 0.99
max_steps: 200
return_type: discounted
fixed_horizon_steps: 64
"#,
)?;
let from_path = SearchConfig::from_yaml_path("config/search.yaml")?;
```

## Validation rules

`SearchConfig` rejects invalid values before search starts:

- `iterations > 0`
- `max_steps > 0`
- `fixed_horizon_steps > 0`
- `c` is finite and `>= 0`
- `gamma` is finite and `>= 0`

Invalid configuration returns `SearchConfigError::Invalid` with a clear message, while parse and file errors map to `Yaml` and `Io`.
