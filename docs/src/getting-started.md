# Getting Started

This section covers cloning the repository and adding dependencies to use Weavetree in another project.

If you want to design an MDP in the browser first, launch [Weavetree Studio](./studio/).

## Clone the repository

```bash
git clone https://github.com/DennisLent/weavetree.git
cd weavetree
```

## Add dependencies in your own project

For local development with a path dependency:

```toml
[dependencies]
weavetree-core = { path = "../weavetree/crates/weavetree-core" }
weavetree-mdp = { path = "../weavetree/crates/weavetree-mdp" }
```

For git dependencies before publishing:

```toml
[dependencies]
weavetree-core = { git = "https://github.com/DennisLent/weavetree", package = "weavetree-core" }
weavetree-mdp = { git = "https://github.com/DennisLent/weavetree", package = "weavetree-mdp" }
```

## Pick an integration style

Use `weavetree-mdp` in one of two ways:

1. YAML/spec flow: define states/actions/outcomes in YAML, compile to `CompiledMdp`, and run with `MdpSimulator`.
2. Custom domain flow: implement `MdpDomain` for your own state type and run with `DomainSimulator`.
