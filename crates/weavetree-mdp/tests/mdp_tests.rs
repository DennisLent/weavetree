use std::cell::RefCell;

use weavetree_core::{ActionId, ReturnType, SearchConfig, StateKey as CoreStateKey, Tree};
use weavetree_mdp::{MdpError, MdpSimulator, MdpSpec, StateKey};

const VALID_MDP_YAML: &str = r#"
version: 1
start: s0
states:
  - id: s0
    terminal: false
    actions:
      - id: a0
        outcomes:
          - next: s1
            prob: 0.7
            reward: 1.0
          - next: s0
            prob: 0.3
            reward: 0.0
      - id: a1
        outcomes:
          - next: s2
            prob: 1.0
            reward: -0.2
  - id: s1
    terminal: true
  - id: s2
    terminal: false
    actions: []
"#;

#[test]
fn yaml_parse_and_compile_success() {
    let spec: MdpSpec = serde_yaml::from_str(VALID_MDP_YAML).expect("valid yaml");
    let compiled = spec.compile().expect("compile should succeed");
    let start = compiled.start();

    assert_eq!(compiled.state_count(), 3);
    assert_eq!(start.index(), 0);
    assert_eq!(compiled.state_id(start), Some("s0"));
}

#[test]
fn validation_fails_for_probability_sum() {
    let yaml = r#"
start: s0
states:
  - id: s0
    actions:
      - id: a0
        outcomes:
          - next: s0
            prob: 0.9
            reward: 1.0
"#;

    let spec: MdpSpec = serde_yaml::from_str(yaml).expect("valid syntax");
    let err = spec.compile().expect_err("compile should fail");

    assert!(matches!(err, MdpError::ProbabilitySum { .. }));
}

#[test]
fn validation_fails_for_unknown_state_reference() {
    let yaml = r#"
start: s0
states:
  - id: s0
    actions:
      - id: a0
        outcomes:
          - next: missing
            prob: 1.0
            reward: 1.0
"#;

    let spec: MdpSpec = serde_yaml::from_str(yaml).expect("valid syntax");
    let err = spec.compile().expect_err("compile should fail");

    assert!(matches!(err, MdpError::UnknownNextState { .. }));
}

#[test]
fn sampling_is_deterministic_for_fixed_seed() {
    let yaml = r#"
start: s0
states:
  - id: s0
    actions:
      - id: a0
        outcomes:
          - next: s0
            prob: 0.6
            reward: 0.0
          - next: s1
            prob: 0.4
            reward: 1.0
  - id: s1
    terminal: true
"#;

    let spec: MdpSpec = serde_yaml::from_str(yaml).expect("valid syntax");
    let compiled = spec.compile().expect("compile should succeed");

    let mut sim_a = MdpSimulator::new(compiled.clone(), 42);
    let mut sim_b = MdpSimulator::new(compiled, 42);

    let mut trace_a = Vec::new();
    let mut trace_b = Vec::new();

    for _ in 0..20 {
        trace_a.push(sim_a.step(StateKey::from(0), 0));
        trace_b.push(sim_b.step(StateKey::from(0), 0));
    }

    assert_eq!(trace_a, trace_b);
}

#[test]
fn mcts_prefers_higher_expected_reward_action() {
    let yaml = r#"
start: s0
states:
  - id: s0
    actions:
      - id: a0
        outcomes:
          - next: s1
            prob: 1.0
            reward: 1.0
      - id: a1
        outcomes:
          - next: s2
            prob: 1.0
            reward: 5.0
  - id: s1
    terminal: true
  - id: s2
    terminal: true
"#;

    let spec: MdpSpec = serde_yaml::from_str(yaml).expect("valid syntax");
    let compiled = spec.compile().expect("compile should succeed");
    let start = compiled.start();

    let simulator = RefCell::new(MdpSimulator::new(compiled, 7));

    let mut tree = Tree::new(CoreStateKey::from(start.index() as u64), false);
    let config = SearchConfig {
        iterations: 20,
        c: 0.0,
        gamma: 1.0,
        max_steps: 2,
        return_type: ReturnType::Discounted,
        fixed_horizon_steps: 2,
    };

    let run = tree
        .run(
            &config,
            |state| {
                simulator
                    .borrow()
                    .num_actions(StateKey::from(state.value() as usize))
            },
            |state, action| {
                let (next, reward, terminal) = simulator
                    .borrow_mut()
                    .step(StateKey::from(state.value() as usize), action.index());
                (CoreStateKey::from(next.index() as u64), reward, terminal)
            },
            |_state, _num_actions| ActionId::from(0),
        )
        .expect("run should succeed");

    assert_eq!(run.iterations_completed, config.iterations);

    let best = tree
        .best_root_action_by_value()
        .expect("lookup should succeed")
        .expect("action should exist");

    assert_eq!(best.index(), 1);
}
