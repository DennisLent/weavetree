use crate::{ActionId, ReturnType, RunLogEvent, SearchConfig, StateKey, Tree, TreeError};

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

#[test]
fn run_log_event_jsonl_contains_event_tag() {
    let config = SearchConfig {
        iterations: 2,
        c: 1.4,
        gamma: 1.0,
        max_steps: 8,
        return_type: ReturnType::Discounted,
        fixed_horizon_steps: 8,
    };

    let event = RunLogEvent::run_started(&config);
    let line = event.to_json_line().expect("serialize run log event");

    assert!(line.contains("\"event\":\"run_started\""));
    assert!(line.contains("\"iterations_requested\":2"));
}

#[test]
fn tree_snapshot_exports_nodes_edges_and_outcomes() {
    let mut tree = Tree::new(StateKey::from(0), false);
    let config = SearchConfig {
        iterations: 2,
        c: 0.0,
        gamma: 1.0,
        max_steps: 4,
        return_type: ReturnType::Discounted,
        fixed_horizon_steps: 4,
    };

    let mut num_actions = |_state: StateKey| 1;
    let mut step = |_state: StateKey, _action: ActionId| (StateKey::from(1), 1.0, true);
    let mut rollout_policy = |_state: StateKey, _num_actions: usize| ActionId::from(0);

    tree.run(&config, &mut num_actions, &mut step, &mut rollout_policy)
        .expect("run should succeed");

    let snapshot = tree.snapshot();
    assert_eq!(snapshot.schema_version, 1);
    assert_eq!(snapshot.node_count, tree.node_count());
    assert_eq!(snapshot.nodes.len(), tree.node_count());

    let root = &snapshot.nodes[0];
    assert_eq!(root.node_id, 0);
    assert_eq!(root.state_key, 0);
    assert_eq!(root.edges.len(), 1);

    let edge = &root.edges[0];
    assert_eq!(edge.action_id, 0);
    assert_eq!(edge.visits, 2);
    assert_eq!(edge.outcomes.len(), 1);
    assert_eq!(edge.outcomes[0].next_state_key, 1);
    assert_eq!(edge.outcomes[0].count, 2);

    let json = tree
        .snapshot_json_pretty()
        .expect("snapshot json serialization should succeed");
    assert!(json.contains("\"schema_version\": 1"));
}
