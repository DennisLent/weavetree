use weavetree_core::{ActionId, ReturnType, SearchConfig, StateKey, Tree, TreeError};

#[test]
fn public_terminal_root_iteration_is_stable() {
    let mut tree = Tree::new(StateKey::from(10), true);
    let config = SearchConfig {
        iterations: 1,
        c: 1.4,
        gamma: 1.0,
        max_steps: 4,
        return_type: ReturnType::Discounted,
        fixed_horizon_steps: 4,
    };

    let mut num_actions = |_state: StateKey| 0;
    let mut step = |state: StateKey, _action: ActionId| (state, 0.0, true);
    let mut rollout_policy = |_state: StateKey, _num_actions: usize| ActionId::from(0);

    let metrics = tree
        .iterate(&config, &mut num_actions, &mut step, &mut rollout_policy)
        .expect("iteration should succeed");

    assert_eq!(metrics.path_len, 0);
    assert_eq!(metrics.total_return, 0.0);
}

#[test]
fn public_zero_action_run_completes_without_error() {
    let mut tree = Tree::new(StateKey::from(7), false);
    let config = SearchConfig {
        iterations: 5,
        c: 1.4,
        gamma: 1.0,
        max_steps: 4,
        return_type: ReturnType::Discounted,
        fixed_horizon_steps: 4,
    };

    let num_actions = |_state: StateKey| 0;
    let step = |state: StateKey, _action: ActionId| (state, 0.0, false);
    let rollout_policy = |_state: StateKey, _num_actions: usize| ActionId::from(0);

    let run = tree
        .run(&config, num_actions, step, rollout_policy)
        .expect("run should succeed");
    assert_eq!(run.iterations_completed, 5);
}

#[test]
fn public_invalid_rollout_action_returns_error() {
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
    let mut rollout_policy = |_state: StateKey, _num_actions: usize| ActionId::from(10);

    let err = tree
        .iterate(&config, &mut num_actions, &mut step, &mut rollout_policy)
        .expect_err("invalid rollout action should fail");

    assert!(matches!(
        err,
        TreeError::InvalidRolloutAction {
            action_id,
            num_actions: 1,
            ..
        } if action_id.index() == 10
    ));
}
