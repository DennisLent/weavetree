# Errors and Edge Cases

The library exposes two error types:

- `TreeError` for runtime search issues.
- `SearchConfigError` for config parsing and validation.

`TreeError` includes variants like missing nodes or edges, failed action selection, and invalid rollout actions. In normal usage, the most common integration issue is `InvalidRolloutAction`, which means your rollout policy returned an action outside the valid range.

`SearchConfigError` covers file I/O errors, YAML parse errors, and invalid values such as non-positive iteration counts.

A few important behaviors are intentionally stable: terminal roots end iterations immediately, zero-action states are handled without panics, and invalid rollout actions return typed errors instead of being silently adjusted.
