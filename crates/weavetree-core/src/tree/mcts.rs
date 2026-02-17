use std::{fmt, fs, path::Path};

use serde::{Deserialize, Serialize};

use crate::tree::{
    error::TreeError,
    ids::{ActionId, NodeId},
    rollout::{ReturnType, RolloutParams, rollout},
    search_tree::Tree,
};

const DEFAULT_SEARCH_CONFIG_YAML: &str = include_str!("../../config/search.default.yaml");

/// Search configuration for MCTS iterations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    pub iterations: usize,
    pub c: f64,
    pub gamma: f64,
    pub max_steps: usize,
    pub return_type: ReturnType,
    pub fixed_horizon_steps: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            iterations: 256,
            c: 1.4,
            gamma: 1.0,
            max_steps: 128,
            return_type: ReturnType::Discounted,
            fixed_horizon_steps: 32,
        }
    }
}

impl SearchConfig {
    /// Parse a search config from YAML text.
    pub fn from_yaml_str(yaml: &str) -> Result<Self, SearchConfigError> {
        let config: SearchConfig = serde_yaml::from_str(yaml).map_err(SearchConfigError::Yaml)?;
        config.validate()?;
        Ok(config)
    }

    /// Parse a search config from a YAML file path.
    pub fn from_yaml_path(path: impl AsRef<Path>) -> Result<Self, SearchConfigError> {
        let yaml = fs::read_to_string(path).map_err(SearchConfigError::Io)?;
        Self::from_yaml_str(&yaml)
    }

    /// Return the default YAML config included with this crate.
    pub fn default_yaml() -> &'static str {
        DEFAULT_SEARCH_CONFIG_YAML
    }

    /// Parse the default YAML config included with this crate.
    pub fn from_default_yaml() -> Result<Self, SearchConfigError> {
        Self::from_yaml_str(Self::default_yaml())
    }

    fn validate(&self) -> Result<(), SearchConfigError> {
        if self.iterations == 0 {
            return Err(SearchConfigError::Invalid(
                "iterations must be greater than 0".to_string(),
            ));
        }
        if !self.c.is_finite() || self.c < 0.0 {
            return Err(SearchConfigError::Invalid(
                "c must be finite and >= 0".to_string(),
            ));
        }
        if !self.gamma.is_finite() || self.gamma < 0.0 {
            return Err(SearchConfigError::Invalid(
                "gamma must be finite and >= 0".to_string(),
            ));
        }
        if self.max_steps == 0 {
            return Err(SearchConfigError::Invalid(
                "max_steps must be greater than 0".to_string(),
            ));
        }
        if self.fixed_horizon_steps == 0 {
            return Err(SearchConfigError::Invalid(
                "fixed_horizon_steps must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }

    fn rollout_params(&self) -> RolloutParams {
        RolloutParams {
            return_type: self.return_type,
            gamma: self.gamma,
            max_steps: self.max_steps,
            fixed_horizon_steps: self.fixed_horizon_steps,
        }
    }
}

/// Error type for loading and validating `SearchConfig`.
#[derive(Debug)]
pub enum SearchConfigError {
    Io(std::io::Error),
    Yaml(serde_yaml::Error),
    Invalid(String),
}

impl fmt::Display for SearchConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchConfigError::Io(err) => write!(f, "failed to read config file: {err}"),
            SearchConfigError::Yaml(err) => write!(f, "failed to parse config YAML: {err}"),
            SearchConfigError::Invalid(err) => write!(f, "invalid search config: {err}"),
        }
    }
}

impl std::error::Error for SearchConfigError {}

/// Per-iteration metrics emitted by MCTS.
#[derive(Debug, Clone, Copy)]
pub struct IterationMetrics {
    pub leaf: NodeId,
    pub path_len: usize,
    pub reward_prefix: f64,
    pub rollout_return: f64,
    pub total_return: f64,
}

/// Aggregate metrics for a complete search run.
#[derive(Debug, Clone)]
pub struct RunMetrics {
    pub iterations_requested: usize,
    pub iterations_completed: usize,
    pub total_return_sum: f64,
    pub average_total_return: f64,
}

impl RunMetrics {
    fn new(iterations_requested: usize) -> Self {
        RunMetrics {
            iterations_requested,
            iterations_completed: 0,
            total_return_sum: 0.0,
            average_total_return: 0.0,
        }
    }

    fn record(&mut self, metrics: IterationMetrics) {
        self.iterations_completed += 1;
        self.total_return_sum += metrics.total_return;
        self.average_total_return = self.total_return_sum / self.iterations_completed as f64;
    }
}

impl Tree {
    /// Backpropagate one return across all edges traversed by tree policy.
    pub fn backpropagate(
        &mut self,
        path: &[(NodeId, ActionId)],
        total_return: f64,
    ) -> Result<(), TreeError> {
        for (node_id, action_id) in path {
            let node = self.node_mut(*node_id)?;
            let edge = node.edge_mut(*action_id).ok_or(TreeError::MissingEdge {
                node_id: *node_id,
                action_id: *action_id,
            })?;
            edge.record(total_return);
        }
        Ok(())
    }

    /// Execute one complete MCTS iteration: selection/expansion, rollout, backpropagation.
    pub fn iterate<FNum, FStep, FPolicy>(
        &mut self,
        config: &SearchConfig,
        num_actions: &mut FNum,
        step: &mut FStep,
        rollout_policy: &mut FPolicy,
    ) -> Result<IterationMetrics, TreeError>
    where
        FNum: FnMut(crate::tree::ids::StateKey) -> usize,
        FStep:
            FnMut(crate::tree::ids::StateKey, ActionId) -> (crate::tree::ids::StateKey, f64, bool),
        FPolicy: FnMut(crate::tree::ids::StateKey, usize) -> ActionId,
    {
        let policy_result = self.tree_policy(config.c, &mut *num_actions, &mut *step)?;
        let leaf = self.node(policy_result.leaf)?;
        let leaf_state_key = leaf.state_key();
        let rollout_return = if leaf.is_terminal() {
            0.0
        } else {
            rollout(
                leaf_state_key,
                &mut *num_actions,
                &mut *step,
                &mut *rollout_policy,
                config.rollout_params(),
            )?
        };
        let total_return = policy_result.reward + rollout_return;

        self.backpropagate(&policy_result.path, total_return)?;

        Ok(IterationMetrics {
            leaf: policy_result.leaf,
            path_len: policy_result.path.len(),
            reward_prefix: policy_result.reward,
            rollout_return,
            total_return,
        })
    }

    /// Run MCTS for `config.iterations`, collecting aggregate metrics.
    pub fn run<FNum, FStep, FPolicy>(
        &mut self,
        config: &SearchConfig,
        num_actions: FNum,
        step: FStep,
        rollout_policy: FPolicy,
    ) -> Result<RunMetrics, TreeError>
    where
        FNum: FnMut(crate::tree::ids::StateKey) -> usize,
        FStep:
            FnMut(crate::tree::ids::StateKey, ActionId) -> (crate::tree::ids::StateKey, f64, bool),
        FPolicy: FnMut(crate::tree::ids::StateKey, usize) -> ActionId,
    {
        self.run_with_hook(config, num_actions, step, rollout_policy, |_| {})
    }

    /// Run MCTS and invoke a callback after each completed iteration.
    pub fn run_with_hook<FNum, FStep, FPolicy, FHook>(
        &mut self,
        config: &SearchConfig,
        mut num_actions: FNum,
        mut step: FStep,
        mut rollout_policy: FPolicy,
        mut on_iteration: FHook,
    ) -> Result<RunMetrics, TreeError>
    where
        FNum: FnMut(crate::tree::ids::StateKey) -> usize,
        FStep:
            FnMut(crate::tree::ids::StateKey, ActionId) -> (crate::tree::ids::StateKey, f64, bool),
        FPolicy: FnMut(crate::tree::ids::StateKey, usize) -> ActionId,
        FHook: FnMut(&IterationMetrics),
    {
        let mut metrics = RunMetrics::new(config.iterations);

        for _ in 0..config.iterations {
            let iteration_metrics =
                self.iterate(config, &mut num_actions, &mut step, &mut rollout_policy)?;

            on_iteration(&iteration_metrics);
            metrics.record(iteration_metrics);
        }

        Ok(metrics)
    }
}
