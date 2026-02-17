use crate::tree::{
    ids::{ActionId, StateKey},
    rollout::{ReturnType, RolloutParams, rollout},
};

#[test]
fn return_modes_are_applied_correctly() {
    let num_actions = |state: StateKey| match state.value() {
        0 | 1 => 1,
        _ => 0,
    };
    let step = |state: StateKey, _action: ActionId| match state.value() {
        0 => (StateKey::from(1), 2.0, false),
        1 => (StateKey::from(2), 4.0, true),
        _ => (state, 0.0, true),
    };
    let rollout_policy = |_state: StateKey, _num_actions: usize| ActionId::from(0);

    let discounted = rollout(
        StateKey::from(0),
        num_actions,
        step,
        rollout_policy,
        RolloutParams {
            return_type: ReturnType::Discounted,
            gamma: 0.5,
            max_steps: 8,
            fixed_horizon_steps: 8,
        },
    )
    .expect("discounted rollout should succeed");
    assert!((discounted - 4.0).abs() < f64::EPSILON);

    let episodic = rollout(
        StateKey::from(0),
        num_actions,
        step,
        rollout_policy,
        RolloutParams {
            return_type: ReturnType::EpisodicUndiscounted,
            gamma: 0.5,
            max_steps: 8,
            fixed_horizon_steps: 8,
        },
    )
    .expect("episodic rollout should succeed");
    assert!((episodic - 6.0).abs() < f64::EPSILON);

    let fixed_horizon = rollout(
        StateKey::from(0),
        num_actions,
        step,
        rollout_policy,
        RolloutParams {
            return_type: ReturnType::FixedHorizon,
            gamma: 0.5,
            max_steps: 8,
            fixed_horizon_steps: 1,
        },
    )
    .expect("fixed horizon rollout should succeed");
    assert!((fixed_horizon - 2.0).abs() < f64::EPSILON);
}
