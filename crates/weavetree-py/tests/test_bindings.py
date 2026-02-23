import pytest

from weavetree.mdp import MdpSimulator, TypedSimulator, compile_yaml_str
from weavetree.mcts import SearchConfig, tree

VALID_MDP_YAML = """
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
"""


def test_yaml_parse_and_compile_success():
    compiled = compile_yaml_str(VALID_MDP_YAML)
    start = compiled.start_state_key()

    assert compiled.state_count() == 3
    assert start == 0
    assert compiled.state_id(start) == "s0"


def test_validation_fails_for_probability_sum():
    yaml = """
start: s0
states:
  - id: s0
    actions:
      - id: a0
        outcomes:
          - next: s0
            prob: 0.9
            reward: 1.0
"""

    with pytest.raises(ValueError):
        compile_yaml_str(yaml)


def test_validation_fails_for_unknown_state_reference():
    yaml = """
start: s0
states:
  - id: s0
    actions:
      - id: a0
        outcomes:
          - next: missing
            prob: 1.0
            reward: 1.0
"""

    with pytest.raises(ValueError):
        compile_yaml_str(yaml)


def test_sampling_is_deterministic_for_fixed_seed():
    yaml = """
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
"""

    compiled = compile_yaml_str(yaml)

    sim_a = MdpSimulator(compiled, 42)
    sim_b = MdpSimulator(compiled, 42)

    trace_a = [sim_a.step(0, 0) for _ in range(20)]
    trace_b = [sim_b.step(0, 0) for _ in range(20)]

    assert trace_a == trace_b


def test_mcts_prefers_higher_expected_reward_action_for_compiled_mdp():
    yaml = """
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
"""

    compiled = compile_yaml_str(yaml)
    sim = MdpSimulator(compiled, 7)

    start_key = compiled.start_state_key()
    t = tree(start_key, compiled.is_terminal(start_key))
    config = SearchConfig(
        iterations=20,
        c=0.0,
        gamma=1.0,
        max_steps=2,
        return_type="discounted",
        fixed_horizon_steps=2,
    )

    run = t.run(sim, config, rollout_action=0)

    assert run.iterations_completed == 20
    assert t.best_root_action_by_value() == 1


class CounterDomain:
    def start_state(self):
        return {"count": 0, "phase": "running"}

    def state_token(self, state):
        return f"{state['count']}:{state['phase']}"

    def is_terminal(self, state):
        return state["phase"] == "finished"

    def num_actions(self, state):
        return 0 if self.is_terminal(state) else 2

    def step(self, state, action_id, _sample):
        if self.is_terminal(state):
            return state, 0.0, True

        reward = 1.0 if action_id == 0 else 3.0 if action_id == 1 else 0.0
        next_state = {"count": state["count"] + 1, "phase": "finished"}
        return next_state, reward, True


def test_mcts_runs_with_custom_typed_domain():
    sim = TypedSimulator(CounterDomain(), 11)

    t = tree(sim.start_state_key(), sim.is_terminal_by_key(sim.start_state_key()))
    config = SearchConfig(
        iterations=20,
        c=0.0,
        gamma=1.0,
        max_steps=2,
        return_type="discounted",
        fixed_horizon_steps=2,
    )

    run = t.run(sim, config, rollout_action=0)

    assert run.iterations_completed == 20
    assert t.best_root_action_by_value() == 1


def test_typed_simulator_state_token_must_be_str_or_bytes():
    class BadDomain:
        def start_state(self):
            return 0

        def state_token(self, state):
            return 1234

        def is_terminal(self, state):
            return True

        def num_actions(self, state):
            return 0

        def step(self, state, action_id, sample):
            return state, 0.0, True

    with pytest.raises(TypeError):
        TypedSimulator(BadDomain(), 1)


def test_mcts_custom_rollout_policy_callback_is_used():
    yaml = """
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
"""

    compiled = compile_yaml_str(yaml)
    sim = MdpSimulator(compiled, 7)
    t = tree(compiled.start_state_key(), compiled.is_terminal(compiled.start_state_key()))
    config = SearchConfig(iterations=10, c=0.0, gamma=1.0, max_steps=2)

    def rollout_policy(_state_key: int, num_actions: int) -> int:
        assert num_actions > 0
        return 1

    run = t.run(sim, config, rollout_policy=rollout_policy)
    assert run.iterations_completed == 10
    assert t.best_root_action_by_value() == 1


def test_typed_simulator_token_collision_debug_check_fails_fast():
    class CollisionDomain:
        def start_state(self):
            return {"value": 0}

        def state_token(self, state):
            return "same-token"

        def is_terminal(self, state):
            return False

        def num_actions(self, state):
            return 1

        def step(self, state, action_id, sample):
            return {"value": state["value"] + 1}, 0.0, False

    sim = TypedSimulator(CollisionDomain(), 3, check_token_collisions=True)
    with pytest.raises(ValueError):
        sim.step_by_key(sim.start_state_key(), 0)


def test_tree_run_propagates_rollout_policy_exception():
    yaml = """
start: s0
states:
      - id: s0
        actions:
          - id: a0
            outcomes:
              - next: s1
                prob: 1.0
                reward: 0.0
      - id: s1
        actions:
          - id: a0
            outcomes:
              - next: s2
                prob: 1.0
                reward: 0.0
      - id: s2
        terminal: true
"""

    compiled = compile_yaml_str(yaml)
    sim = MdpSimulator(compiled, 1)
    t = tree(compiled.start_state_key(), compiled.is_terminal(compiled.start_state_key()))
    config = SearchConfig(iterations=20, c=0.0, gamma=1.0, max_steps=2)

    def boom(state_key: int, num_actions: int) -> int:
        raise RuntimeError("policy failure")

    with pytest.raises(RuntimeError):
        t.run(sim, config, rollout_policy=boom)
