# Getting Started

This section covers two practical steps: cloning this repository and using `weavetree-core` from another Rust project.

## Clone the repository

Clone the project and enter the repository root:

```bash
git clone https://github.com/DennisLent/weavetree.git
cd weavetree
```

The Rust crate lives in `crates/weavetree-core`. If you want to work directly in this repo, you can use it immediately as a workspace member.

## Add `weavetree-core` to your own project

If you want to consume `weavetree-core` from another repository before publishing, add it as a git dependency:

```rust
[dependencies]
weavetree-core = { git = "https://github.com/DennisLent/weavetree", package = "weavetree-core" }
```

If you have a local checkout and want a path dependency during development:

```toml
[dependencies]
weavetree-core = { path = "../weavetree/crates/weavetree-core" }
```

After adding the dependency, you can import public types like this:

```rust
use weavetree_core::{ActionId, SearchConfig, StateKey, Tree};
```
