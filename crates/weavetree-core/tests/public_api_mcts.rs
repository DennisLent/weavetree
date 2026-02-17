use weavetree_core::{ActionId, ReturnType, SearchConfig, StateKey, Tree};

#[test]
fn public_run_prefers_higher_value_root_action() {
    let mut tree = Tree::new(StateKey::from(0), false);
    let config = SearchConfig {
        iterations: 20,
        c: 0.0,
        gamma: 1.0,
        max_steps: 4,
        return_type: ReturnType::Discounted,
        fixed_horizon_steps: 4,
    };

    let num_actions = |state: StateKey| if state.value() == 0 { 2 } else { 0 };

    let step = |state: StateKey, action: ActionId| {
        if state.value() != 0 {
            return (state, 0.0, true);
        }

        if action.index() == 0 {
            (StateKey::from(1), 1.0, true)
        } else {
            (StateKey::from(2), 5.0, true)
        }
    };

    let rollout_policy = |_state: StateKey, _num_actions: usize| ActionId::from(0);

    let run = tree
        .run(&config, num_actions, step, rollout_policy)
        .expect("run should succeed");

    assert_eq!(run.iterations_completed, config.iterations);

    let best_by_visits = tree
        .best_root_action_by_visits()
        .expect("root action lookup should succeed")
        .expect("an action should be available");
    let best_by_value = tree
        .best_root_action_by_value()
        .expect("root action lookup should succeed")
        .expect("an action should be available");

    assert_eq!(best_by_visits.index(), 1);
    assert_eq!(best_by_value.index(), 1);
}

#[test]
fn public_default_yaml_config_parses() {
    let config = SearchConfig::from_default_yaml().expect("default yaml should parse");
    assert_eq!(config.return_type, ReturnType::Discounted);
    assert!(config.iterations > 0);
}
