use serde::{Deserialize, Serialize};

use crate::tree::{
    error::TreeError,
    ids::{ActionId, StateKey},
};

/// Controls how rollout rewards are aggregated into a return.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReturnType {
    /// Sum raw rewards until terminal or rollout limit.
    EpisodicUndiscounted,
    /// Sum discounted rewards using `gamma`.
    #[default]
    Discounted,
    /// Sum rewards but clamp to a strict fixed step horizon.
    FixedHorizon,
}

/// Rollout parameters controlling return shape and stopping criteria.
#[derive(Debug, Clone, Copy)]
pub struct RolloutParams {
    pub return_type: ReturnType,
    pub gamma: f64,
    pub max_steps: usize,
    pub fixed_horizon_steps: usize,
}

impl RolloutParams {
    /// Resolve the actual rollout step limit for the current return mode.
    pub fn step_limit(&self) -> usize {
        match self.return_type {
            ReturnType::FixedHorizon => self.max_steps.min(self.fixed_horizon_steps),
            ReturnType::Discounted | ReturnType::EpisodicUndiscounted => self.max_steps,
        }
    }
}

/// Run a default-policy rollout from `start_state_key`.
///
/// The environment interface remains generic and only needs:
/// - `num_actions(state_key) -> usize`
/// - `step(state_key, action_id) -> (next_state_key, reward, is_terminal)`
/// - `rollout_policy(state_key, num_actions) -> action_id`
#[allow(dead_code)]
pub fn rollout<FNum, FStep, FPolicy>(
    start_state_key: StateKey,
    mut num_actions: FNum,
    mut step: FStep,
    mut rollout_policy: FPolicy,
    params: RolloutParams,
) -> Result<f64, TreeError>
where
    FNum: FnMut(StateKey) -> usize,
    FStep: FnMut(StateKey, ActionId) -> (StateKey, f64, bool),
    FPolicy: FnMut(StateKey, usize) -> ActionId,
{
    rollout_fallible(
        start_state_key,
        |state| Ok::<usize, TreeError>(num_actions(state)),
        |state, action| Ok::<(StateKey, f64, bool), TreeError>(step(state, action)),
        |state, n| Ok::<ActionId, TreeError>(rollout_policy(state, n)),
        params,
    )
}

/// Fallible rollout variant where environment/policy callbacks may fail.
pub fn rollout_fallible<FNum, FStep, FPolicy, E>(
    start_state_key: StateKey,
    mut num_actions: FNum,
    mut step: FStep,
    mut rollout_policy: FPolicy,
    params: RolloutParams,
) -> Result<f64, E>
where
    FNum: FnMut(StateKey) -> Result<usize, E>,
    FStep: FnMut(StateKey, ActionId) -> Result<(StateKey, f64, bool), E>,
    FPolicy: FnMut(StateKey, usize) -> Result<ActionId, E>,
    E: From<TreeError>,
{
    let mut state_key = start_state_key;
    let mut total_return = 0.0;
    let mut discount = 1.0;

    for _ in 0..params.step_limit() {
        let action_count = num_actions(state_key)?;
        if action_count == 0 {
            break;
        }

        let action_id = rollout_policy(state_key, action_count)?;
        if action_id.index() >= action_count {
            return Err(TreeError::InvalidRolloutAction {
                state_key,
                action_id,
                num_actions: action_count,
            }
            .into());
        }
        let (next_state_key, reward, is_terminal) = step(state_key, action_id)?;

        match params.return_type {
            ReturnType::Discounted => {
                total_return += discount * reward;
                discount *= params.gamma;
            }
            ReturnType::EpisodicUndiscounted | ReturnType::FixedHorizon => {
                total_return += reward;
            }
        }

        state_key = next_state_key;

        if is_terminal {
            break;
        }
    }

    Ok(total_return)
}
