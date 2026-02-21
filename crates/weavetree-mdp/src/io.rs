use std::{fs, path::Path};

use crate::{CompiledMdp, MdpError, MdpSpec};

/// Load an MDP spec from YAML on disk.
pub fn load_yaml(path: impl AsRef<Path>) -> Result<MdpSpec, MdpError> {
    let yaml = fs::read_to_string(path)?;
    let spec: MdpSpec = serde_yaml::from_str(&yaml)?;
    Ok(spec)
}

/// Load and compile an MDP from a YAML file.
pub fn compile_yaml(path: impl AsRef<Path>) -> Result<CompiledMdp, MdpError> {
    let spec = load_yaml(path)?;
    spec.compile()
}

/// Serialize and write an MDP spec to YAML.
pub fn save_yaml(path: impl AsRef<Path>, spec: &MdpSpec) -> Result<(), MdpError> {
    let yaml = serde_yaml::to_string(spec)?;
    fs::write(path, yaml)?;
    Ok(())
}
