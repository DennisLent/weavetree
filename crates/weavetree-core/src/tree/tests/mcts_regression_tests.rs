use crate::{ActionId, ReturnType, SearchConfig, StateKey, Tree, TreeError};

#[test]
fn terminal_root_iteration_has_empty_path_and_zero_return() {
    let mut tree = Tree::new(StateKey::from(42), true);
    let config = SearchConfig {
        iterations: 1,
        c: 1.4,
        gamma: 1.0,
        max_steps: 8,
        return_type: ReturnType::Discounted,
        fixed_horizon_steps: 8,
    };

    let mut num_actions = |_state: StateKey| 0;
    let mut step = |state: StateKey, _action: ActionId| (state, 0.0, true);
    let mut rollout_policy = |_state: StateKey, _num_actions: usize| ActionId::from(0);

    let metrics = tree
        .iterate(&config, &mut num_actions, &mut step, &mut rollout_policy)
        .expect("terminal root iteration should succeed");

    assert_eq!(metrics.path_len, 0);
    assert_eq!(metrics.total_return, 0.0);
}

#[test]
fn zero_action_state_stops_safely() {
    let mut tree = Tree::new(StateKey::from(1), false);
    let config = SearchConfig {
        iterations: 3,
        c: 1.4,
        gamma: 1.0,
        max_steps: 8,
        return_type: ReturnType::Discounted,
        fixed_horizon_steps: 8,
    };

    let mut num_actions = |_state: StateKey| 0;
    let mut step = |state: StateKey, _action: ActionId| (state, 0.0, false);
    let mut rollout_policy = |_state: StateKey, _num_actions: usize| ActionId::from(0);

    let metrics = tree
        .run_with_hook(
            &config,
            &mut num_actions,
            &mut step,
            &mut rollout_policy,
            |_iter| {},
        )
        .expect("run should succeed on zero-action state");

    assert_eq!(metrics.iterations_completed, 3);
    assert_eq!(metrics.total_return_sum, 0.0);
}

#[test]
fn invalid_rollout_policy_action_index_returns_error() {
    let mut tree = Tree::new(StateKey::from(0), false);
    let config = SearchConfig {
        iterations: 1,
        c: 1.4,
        gamma: 1.0,
        max_steps: 8,
        return_type: ReturnType::Discounted,
        fixed_horizon_steps: 8,
    };

    let mut num_actions = |_state: StateKey| 1;
    let mut step = |_state: StateKey, _action: ActionId| (StateKey::from(1), 0.0, false);
    let mut rollout_policy = |_state: StateKey, _num_actions: usize| ActionId::from(99);

    let err = tree
        .iterate(&config, &mut num_actions, &mut step, &mut rollout_policy)
        .expect_err("invalid rollout action should error");

    assert!(matches!(
        err,
        TreeError::InvalidRolloutAction {
            action_id,
            num_actions: 1,
            ..
        } if action_id.index() == 99
    ));
}
