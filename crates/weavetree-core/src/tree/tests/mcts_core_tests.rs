use crate::{ActionId, ReturnType, SearchConfig, StateKey, Tree};

#[test]
fn deterministic_iterations_backpropagate_visits() {
    let mut tree = Tree::new(StateKey::from(0), false);
    let config = SearchConfig {
        iterations: 2,
        c: 1.4,
        gamma: 1.0,
        max_steps: 8,
        return_type: ReturnType::Discounted,
        fixed_horizon_steps: 8,
    };

    let mut num_actions = |state: StateKey| match state.value() {
        0 | 1 => 1,
        _ => 0,
    };

    let mut step = |state: StateKey, _action: ActionId| match state.value() {
        0 => (StateKey::from(1), 1.0, false),
        1 => (StateKey::from(2), 2.0, true),
        _ => (state, 0.0, true),
    };

    let mut rollout_policy = |_state: StateKey, _num_actions: usize| ActionId::from(0);

    let _ = tree
        .iterate(&config, &mut num_actions, &mut step, &mut rollout_policy)
        .expect("first iteration should succeed");
    let _ = tree
        .iterate(&config, &mut num_actions, &mut step, &mut rollout_policy)
        .expect("second iteration should succeed");

    let root = tree.node(tree.root_id()).expect("root exists");
    let root_edge = root.edge(ActionId::from(0)).expect("root action exists");
    assert_eq!(root_edge.visits(), 2);

    let child_id = root_edge
        .get_child_for(StateKey::from(1))
        .expect("child should exist for state 1");
    let child = tree.node(child_id).expect("child exists");
    let child_edge = child.edge(ActionId::from(0)).expect("child action exists");
    assert_eq!(child_edge.visits(), 1);
}

#[test]
fn stochastic_transitions_create_distinct_outcomes_and_count_occurrences() {
    let mut tree = Tree::new(StateKey::from(0), false);
    let config = SearchConfig {
        iterations: 3,
        c: 1.4,
        gamma: 1.0,
        max_steps: 4,
        return_type: ReturnType::Discounted,
        fixed_horizon_steps: 4,
    };

    let mut sequence = vec![1_u64, 2_u64, 1_u64].into_iter();

    let mut num_actions = |state: StateKey| match state.value() {
        0 => 1,
        _ => 0,
    };

    let mut step = move |state: StateKey, _action: ActionId| {
        if state.value() == 0 {
            let next_state = sequence.next().expect("enough stochastic outcomes");
            (StateKey::from(next_state), 0.0, true)
        } else {
            (state, 0.0, true)
        }
    };

    let mut rollout_policy = |_state: StateKey, _num_actions: usize| ActionId::from(0);

    for _ in 0..3 {
        let _ = tree
            .iterate(&config, &mut num_actions, &mut step, &mut rollout_policy)
            .expect("iteration should succeed");
    }

    let root = tree.node(tree.root_id()).expect("root exists");
    let edge = root.edge(ActionId::from(0)).expect("action 0 exists");

    assert_eq!(edge.outcomes_len(), 2);
    assert_eq!(edge.outcome_count_for(StateKey::from(1)), Some(2));
    assert_eq!(edge.outcome_count_for(StateKey::from(2)), Some(1));
    assert_eq!(tree.node_count(), 3);
}

#[test]
fn default_config_yaml_parses() {
    let config = SearchConfig::from_default_yaml().expect("default yaml should parse");
    assert_eq!(config.return_type, ReturnType::Discounted);
    assert!(config.iterations > 0);
}
