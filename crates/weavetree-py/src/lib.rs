#![allow(unsafe_op_in_unsafe_fn)]

use std::{cell::RefCell, collections::HashMap};

use ::weavetree_core::{
    ActionId, ReturnType, RunError, RunMetrics, SearchConfig, StateKey as CoreStateKey, Tree,
    TreeError,
};
use ::weavetree_mdp::{CompiledMdp, MdpError, MdpSimulator, MdpSpec, StateKey, compile_yaml};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn mdp_err_to_py(err: MdpError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn tree_err_to_py(err: TreeError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn parse_state_key(index: usize) -> StateKey {
    StateKey::from(index)
}

fn parse_return_type(value: &str) -> PyResult<ReturnType> {
    match value {
        "discounted" => Ok(ReturnType::Discounted),
        "episodic_undiscounted" => Ok(ReturnType::EpisodicUndiscounted),
        "fixed_horizon" => Ok(ReturnType::FixedHorizon),
        _ => Err(PyValueError::new_err(
            "invalid return_type; expected one of: discounted, episodic_undiscounted, fixed_horizon",
        )),
    }
}

#[pyclass(name = "CompiledMdp", module = "weavetree.mdp")]
#[derive(Clone)]
/// CompiledMdp()
/// --
///
/// Immutable compiled MDP model loaded from YAML.
///
/// Use this object to inspect state/action layout and to construct
/// a deterministic `MdpSimulator`.
pub struct PyCompiledMdp {
    inner: CompiledMdp,
}

#[pymethods]
impl PyCompiledMdp {
    /// start_state_key($self, /)
    /// --
    ///
    /// Return the integer key of the start state.
    #[pyo3(text_signature = "($self, /)")]
    fn start_state_key(&self) -> usize {
        self.inner.start().index()
    }

    /// state_count($self, /)
    /// --
    ///
    /// Return the total number of compiled states.
    #[pyo3(text_signature = "($self, /)")]
    fn state_count(&self) -> usize {
        self.inner.state_count()
    }

    /// is_terminal($self, state_key, /)
    /// --
    ///
    /// Return whether the given state key is terminal.
    ///
    /// Raises:
    ///     KeyError: If `state_key` does not exist.
    #[pyo3(text_signature = "($self, state_key, /)")]
    fn is_terminal(&self, state_key: usize) -> PyResult<bool> {
        self.inner
            .is_terminal(parse_state_key(state_key))
            .ok_or_else(|| PyKeyError::new_err(format!("unknown state key: {state_key}")))
    }

    /// num_actions($self, state_key, /)
    /// --
    ///
    /// Return the number of actions available in a state.
    ///
    /// Raises:
    ///     KeyError: If `state_key` does not exist.
    #[pyo3(text_signature = "($self, state_key, /)")]
    fn num_actions(&self, state_key: usize) -> PyResult<usize> {
        self.inner
            .num_actions(parse_state_key(state_key))
            .ok_or_else(|| PyKeyError::new_err(format!("unknown state key: {state_key}")))
    }

    /// state_id($self, state_key, /)
    /// --
    ///
    /// Return the original string state id for a compiled key.
    ///
    /// Raises:
    ///     KeyError: If `state_key` does not exist.
    #[pyo3(text_signature = "($self, state_key, /)")]
    fn state_id(&self, state_key: usize) -> PyResult<String> {
        self.inner
            .state_id(parse_state_key(state_key))
            .map(str::to_owned)
            .ok_or_else(|| PyKeyError::new_err(format!("unknown state key: {state_key}")))
    }

    /// state_key($self, state_id, /)
    /// --
    ///
    /// Return the compiled integer key for a string state id.
    ///
    /// Raises:
    ///     KeyError: If `state_id` does not exist.
    #[pyo3(text_signature = "($self, state_id, /)")]
    fn state_key(&self, state_id: &str) -> PyResult<usize> {
        self.inner
            .state_key(state_id)
            .map(|key| key.index())
            .ok_or_else(|| PyKeyError::new_err(format!("unknown state id: {state_id}")))
    }
}

#[pyclass(name = "MdpSimulator", module = "weavetree.mdp")]
/// MdpSimulator(compiled, seed, /)
/// --
///
/// Seeded simulator over a compiled MDP.
///
/// Transitions are sampled deterministically from `seed`.
pub struct PyMdpSimulator {
    inner: RefCell<MdpSimulator>,
}

#[pymethods]
impl PyMdpSimulator {
    #[new]
    #[pyo3(text_signature = "(compiled, seed, /)")]
    fn new(compiled: PyRef<'_, PyCompiledMdp>, seed: u64) -> Self {
        Self {
            inner: RefCell::new(MdpSimulator::new(compiled.inner.clone(), seed)),
        }
    }

    /// num_actions($self, state_key, /)
    /// --
    ///
    /// Return the number of actions available from `state_key`.
    ///
    /// Invalid keys return `0`.
    #[pyo3(text_signature = "($self, state_key, /)")]
    fn num_actions(&self, state_key: usize) -> usize {
        self.inner.borrow().num_actions(parse_state_key(state_key))
    }

    /// step($self, state_key, action_id, /)
    /// --
    ///
    /// Sample one transition from `(state_key, action_id)`.
    ///
    /// Returns:
    ///     tuple[int, float, bool]: `(next_state_key, reward, terminal)`.
    ///
    /// Invalid inputs are treated as a terminal no-op transition.
    #[pyo3(text_signature = "($self, state_key, action_id, /)")]
    fn step(&self, state_key: usize, action_id: usize) -> (usize, f64, bool) {
        let (next, reward, terminal) = self
            .inner
            .borrow_mut()
            .step(parse_state_key(state_key), action_id);
        (next.index(), reward, terminal)
    }
}

#[pyclass(name = "TypedSimulator", module = "weavetree.mdp")]
/// TypedSimulator(domain, seed, check_token_collisions=False, /)
/// --
///
/// Seeded simulator over a Python-defined typed domain.
///
/// `domain` must define:
/// - `start_state() -> Any`
/// - `state_token(state) -> str | bytes` (stable canonical token)
/// - `is_terminal(state) -> bool`
/// - `num_actions(state) -> int`
/// - `step(state, action_id, sample) -> (next_state, reward, terminal)`
///
/// States are deep-copied on ingestion and when returned by `state_for_key`
/// to preserve stable interning semantics.
/// Set `check_token_collisions=True` to detect token collisions at runtime.
pub struct PyTypedSimulator {
    domain: Py<PyAny>,
    states: RefCell<Vec<Py<PyAny>>>,
    token_to_key: RefCell<HashMap<Vec<u8>, u64>>,
    rng: RefCell<ChaCha8Rng>,
    check_token_collisions: bool,
}

impl PyTypedSimulator {
    fn deep_copy(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let copy = py.import_bound("copy")?;
        copy.call_method1("deepcopy", (value,))?.extract()
    }

    fn token_bytes(token_obj: Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
        if let Ok(bytes) = token_obj.extract::<Vec<u8>>() {
            return Ok([b'b'].into_iter().chain(bytes).collect());
        }
        if let Ok(text) = token_obj.extract::<String>() {
            return Ok([b's']
                .into_iter()
                .chain(text.into_bytes())
                .collect::<Vec<u8>>());
        }
        Err(PyTypeError::new_err(
            "state_token(state) must return str or bytes",
        ))
    }

    fn intern_state(&self, py: Python<'_>, state: Py<PyAny>) -> PyResult<u64> {
        let frozen_state = Self::deep_copy(py, &state.bind(py))?;
        let token_obj = self
            .domain
            .bind(py)
            .call_method1("state_token", (frozen_state.bind(py),))?;
        let token = Self::token_bytes(token_obj)?;

        if let Some(existing) = self.token_to_key.borrow().get(&token) {
            if self.check_token_collisions {
                let existing_state = self
                    .states
                    .borrow()
                    .get(*existing as usize)
                    .map(|s| s.clone_ref(py))
                    .ok_or_else(|| {
                        PyValueError::new_err("internal state key map is inconsistent")
                    })?;
                let equal = existing_state.bind(py).eq(frozen_state.bind(py))?;
                if !equal {
                    return Err(PyValueError::new_err(
                        "state_token collision detected: same token mapped to non-equal states",
                    ));
                }
            }
            return Ok(*existing);
        }

        let key = self.states.borrow().len() as u64;
        self.states.borrow_mut().push(frozen_state);
        self.token_to_key.borrow_mut().insert(token, key);
        Ok(key)
    }

    fn state_by_key(&self, py: Python<'_>, key: u64) -> Option<Py<PyAny>> {
        self.states
            .borrow()
            .get(key as usize)
            .map(|state| state.clone_ref(py))
    }

    fn num_actions_by_key_impl(&self, state_key: u64) -> PyResult<usize> {
        Python::with_gil(|py| {
            let Some(state) = self.state_by_key(py, state_key) else {
                return Ok(0);
            };
            self.domain
                .bind(py)
                .call_method1("num_actions", (state.bind(py),))?
                .extract()
        })
    }

    fn step_by_key_impl(&self, state_key: u64, action_id: usize) -> PyResult<(u64, f64, bool)> {
        Python::with_gil(|py| {
            let Some(state) = self.state_by_key(py, state_key) else {
                return Ok((state_key, 0.0, true));
            };

            let sample = {
                let mut rng = self.rng.borrow_mut();
                (rng.next_u64() as f64) / ((u64::MAX as f64) + 1.0)
            };

            let (next_state, reward, terminal): (Py<PyAny>, f64, bool) = self
                .domain
                .bind(py)
                .call_method1("step", (state.bind(py), action_id, sample))?
                .extract()?;
            let next_key = self.intern_state(py, next_state)?;
            Ok((next_key, reward, terminal))
        })
    }
}

#[pymethods]
impl PyTypedSimulator {
    #[new]
    #[pyo3(signature = (domain, seed, check_token_collisions=false))]
    #[pyo3(text_signature = "(domain, seed, check_token_collisions=False, /)")]
    fn new(domain: Py<PyAny>, seed: u64, check_token_collisions: bool) -> PyResult<Self> {
        let simulator = Self {
            domain,
            states: RefCell::new(Vec::new()),
            token_to_key: RefCell::new(HashMap::new()),
            rng: RefCell::new(ChaCha8Rng::seed_from_u64(seed)),
            check_token_collisions,
        };

        Python::with_gil(|py| {
            let start_state: Py<PyAny> = simulator
                .domain
                .bind(py)
                .call_method0("start_state")?
                .extract()?;
            let key = simulator.intern_state(py, start_state)?;
            if key != 0 {
                return Err(PyValueError::new_err("typed simulator start key must be 0"));
            }
            Ok(())
        })?;

        Ok(simulator)
    }

    /// start_state_key($self, /)
    /// --
    ///
    /// Return the interned key of the domain start state.
    #[pyo3(text_signature = "($self, /)")]
    fn start_state_key(&self) -> u64 {
        0
    }

    /// state_for_key($self, state_key, /)
    /// --
    ///
    /// Return the decoded Python state object for an interned key.
    ///
    /// Raises:
    ///     KeyError: If `state_key` does not exist.
    #[pyo3(text_signature = "($self, state_key, /)")]
    fn state_for_key(&self, state_key: u64) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let state = self
                .state_by_key(py, state_key)
                .ok_or_else(|| PyKeyError::new_err(format!("unknown state key: {state_key}")))?;
            Self::deep_copy(py, &state.bind(py))
        })
    }

    /// is_terminal_by_key($self, state_key, /)
    /// --
    ///
    /// Return whether an interned state key is terminal.
    ///
    /// Invalid keys are treated as terminal.
    #[pyo3(text_signature = "($self, state_key, /)")]
    fn is_terminal_by_key(&self, state_key: u64) -> PyResult<bool> {
        Python::with_gil(|py| {
            let Some(state) = self.state_by_key(py, state_key) else {
                return Ok(true);
            };
            self.domain
                .bind(py)
                .call_method1("is_terminal", (state.bind(py),))?
                .extract()
        })
    }

    /// num_actions_by_key($self, state_key, /)
    /// --
    ///
    /// Return the number of actions available for an interned state key.
    ///
    /// Invalid keys return `0`.
    #[pyo3(text_signature = "($self, state_key, /)")]
    fn num_actions_by_key(&self, state_key: u64) -> PyResult<usize> {
        self.num_actions_by_key_impl(state_key)
    }

    /// step_by_key($self, state_key, action_id, /)
    /// --
    ///
    /// Sample one transition `(next_state_key, reward, terminal)`.
    ///
    /// Invalid state/action inputs are treated as terminal no-op transitions.
    #[pyo3(text_signature = "($self, state_key, action_id, /)")]
    fn step_by_key(&self, state_key: u64, action_id: usize) -> PyResult<(u64, f64, bool)> {
        self.step_by_key_impl(state_key, action_id)
    }
}

#[pyfunction]
#[pyo3(text_signature = "(path, /)")]
/// compile_yaml_file(path, /)
/// --
///
/// Load and compile an MDP from a YAML file path.
///
/// Raises:
///     ValueError: If file loading, YAML parsing, or MDP validation fails.
fn compile_yaml_file(path: &str) -> PyResult<PyCompiledMdp> {
    let mdp = compile_yaml(path).map_err(mdp_err_to_py)?;
    Ok(PyCompiledMdp { inner: mdp })
}

#[pyfunction]
#[pyo3(text_signature = "(yaml, /)")]
/// compile_yaml_str(yaml, /)
/// --
///
/// Compile an MDP directly from a YAML string.
///
/// Raises:
///     ValueError: If YAML parsing or MDP validation fails.
fn compile_yaml_str(yaml: &str) -> PyResult<PyCompiledMdp> {
    let spec: MdpSpec =
        serde_yaml::from_str(yaml).map_err(|err| mdp_err_to_py(MdpError::Yaml(err)))?;
    let mdp = spec.compile().map_err(mdp_err_to_py)?;
    Ok(PyCompiledMdp { inner: mdp })
}

#[pyclass(name = "SearchConfig", module = "weavetree.mcts")]
#[derive(Clone)]
/// SearchConfig(iterations=256, c=1.4, gamma=1.0, max_steps=128, return_type='discounted', fixed_horizon_steps=32, /)
/// --
///
/// MCTS search configuration.
pub struct PySearchConfig {
    inner: SearchConfig,
}

#[pymethods]
impl PySearchConfig {
    #[new]
    #[pyo3(signature = (iterations=256, c=1.4, gamma=1.0, max_steps=128, return_type="discounted", fixed_horizon_steps=32))]
    #[pyo3(
        text_signature = "(iterations=256, c=1.4, gamma=1.0, max_steps=128, return_type='discounted', fixed_horizon_steps=32, /)"
    )]
    fn new(
        iterations: usize,
        c: f64,
        gamma: f64,
        max_steps: usize,
        return_type: &str,
        fixed_horizon_steps: usize,
    ) -> PyResult<Self> {
        let rt = parse_return_type(return_type)?;

        if iterations == 0 {
            return Err(PyValueError::new_err("iterations must be greater than 0"));
        }
        if !c.is_finite() || c < 0.0 {
            return Err(PyValueError::new_err("c must be finite and >= 0"));
        }
        if !gamma.is_finite() || gamma < 0.0 {
            return Err(PyValueError::new_err("gamma must be finite and >= 0"));
        }
        if max_steps == 0 {
            return Err(PyValueError::new_err("max_steps must be greater than 0"));
        }
        if fixed_horizon_steps == 0 {
            return Err(PyValueError::new_err(
                "fixed_horizon_steps must be greater than 0",
            ));
        }

        Ok(Self {
            inner: SearchConfig {
                iterations,
                c,
                gamma,
                max_steps,
                return_type: rt,
                fixed_horizon_steps,
            },
        })
    }

    /// iterations($self, /)
    /// --
    ///
    /// Number of MCTS iterations.
    #[pyo3(text_signature = "($self, /)")]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }
}

#[pyclass(name = "RunMetrics", module = "weavetree.mcts")]
/// RunMetrics()
/// --
///
/// Aggregate metrics for a full MCTS run.
pub struct PyRunMetrics {
    #[pyo3(get)]
    iterations_requested: usize,
    #[pyo3(get)]
    iterations_completed: usize,
    #[pyo3(get)]
    total_return_sum: f64,
    #[pyo3(get)]
    average_total_return: f64,
}

impl From<RunMetrics> for PyRunMetrics {
    fn from(value: RunMetrics) -> Self {
        Self {
            iterations_requested: value.iterations_requested,
            iterations_completed: value.iterations_completed,
            total_return_sum: value.total_return_sum,
            average_total_return: value.average_total_return,
        }
    }
}

#[pyclass(name = "Tree", module = "weavetree.mcts")]
/// Tree(root_state_key, root_is_terminal, /)
/// --
///
/// Search tree for MCTS.
pub struct PyTree {
    inner: Tree,
}

#[pymethods]
impl PyTree {
    #[new]
    #[pyo3(text_signature = "(root_state_key, root_is_terminal, /)")]
    fn new(root_state_key: u64, root_is_terminal: bool) -> Self {
        Self {
            inner: Tree::new(CoreStateKey::from(root_state_key), root_is_terminal),
        }
    }

    /// node_count($self, /)
    /// --
    ///
    /// Return the number of nodes currently in the tree.
    #[pyo3(text_signature = "($self, /)")]
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// best_root_action_by_value($self, /)
    /// --
    ///
    /// Return the best root action index by mean value, or `None`.
    #[pyo3(text_signature = "($self, /)")]
    fn best_root_action_by_value(&self) -> PyResult<Option<usize>> {
        self.inner
            .best_root_action_by_value()
            .map(|opt| opt.map(|a| a.index()))
            .map_err(tree_err_to_py)
    }

    /// best_root_action_by_visits($self, /)
    /// --
    ///
    /// Return the best root action index by visit count, or `None`.
    #[pyo3(text_signature = "($self, /)")]
    fn best_root_action_by_visits(&self) -> PyResult<Option<usize>> {
        self.inner
            .best_root_action_by_visits()
            .map(|opt| opt.map(|a| a.index()))
            .map_err(tree_err_to_py)
    }

    /// run($self, simulator, config, rollout_action=0, rollout_policy=None, /)
    /// --
    ///
    /// Run MCTS using either `MdpSimulator` or `TypedSimulator`.
    ///
    /// If `rollout_policy` is provided, it must be callable:
    /// `(state_key: int, num_actions: int) -> action_id: int`.
    /// Otherwise `rollout_action` is used and clamped to valid range.
    /// Callback failures are propagated immediately.
    #[pyo3(signature = (simulator, config, rollout_action=0, rollout_policy=None))]
    #[pyo3(text_signature = "($self, simulator, config, rollout_action=0, rollout_policy=None, /)")]
    fn run(
        &mut self,
        simulator: &Bound<'_, PyAny>,
        config: PyRef<'_, PySearchConfig>,
        rollout_action: usize,
        rollout_policy: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyRunMetrics> {
        let rollout_policy: Option<Py<PyAny>> =
            rollout_policy.map(|policy| policy.clone().unbind());

        if let Ok(simulator) = simulator.extract::<PyRef<'_, PyMdpSimulator>>() {
            let sim_cell = &simulator.inner;
            let metrics = self
                .inner
                .run_fallible(
                    &config.inner,
                    |state| {
                        Ok::<usize, PyErr>(
                            sim_cell
                                .borrow()
                                .num_actions(StateKey::from(state.value() as usize)),
                        )
                    },
                    |state, action| {
                        Ok::<(CoreStateKey, f64, bool), PyErr>({
                            let mut sim = sim_cell.borrow_mut();
                            let (next, reward, terminal) =
                                sim.step(StateKey::from(state.value() as usize), action.index());
                            (CoreStateKey::from(next.index() as u64), reward, terminal)
                        })
                    },
                    |state, num_actions| {
                        if let Some(policy) = &rollout_policy {
                            let action_id = Python::with_gil(|py| -> PyResult<usize> {
                                policy
                                    .bind(py)
                                    .call1((state.value(), num_actions))?
                                    .extract()
                            })?;
                            Ok(ActionId::from(action_id))
                        } else {
                            let clamped = if num_actions == 0 {
                                0
                            } else {
                                rollout_action.min(num_actions - 1)
                            };
                            Ok(ActionId::from(clamped))
                        }
                    },
                )
                .map_err(|err| match err {
                    RunError::Tree(tree_err) => tree_err_to_py(tree_err),
                    RunError::Callback(py_err) => py_err,
                })?;
            return Ok(metrics.into());
        }

        if let Ok(simulator) = simulator.extract::<PyRef<'_, PyTypedSimulator>>() {
            let sim = simulator;
            let metrics = self
                .inner
                .run_fallible(
                    &config.inner,
                    |state| sim.num_actions_by_key_impl(state.value()),
                    |state, action| {
                        sim.step_by_key_impl(state.value(), action.index()).map(
                            |(next, reward, terminal)| (CoreStateKey::from(next), reward, terminal),
                        )
                    },
                    |state, num_actions| {
                        if let Some(policy) = &rollout_policy {
                            let action_id = Python::with_gil(|py| -> PyResult<usize> {
                                policy
                                    .bind(py)
                                    .call1((state.value(), num_actions))?
                                    .extract()
                            })?;
                            Ok(ActionId::from(action_id))
                        } else {
                            let clamped = if num_actions == 0 {
                                0
                            } else {
                                rollout_action.min(num_actions - 1)
                            };
                            Ok(ActionId::from(clamped))
                        }
                    },
                )
                .map_err(|err| match err {
                    RunError::Tree(tree_err) => tree_err_to_py(tree_err),
                    RunError::Callback(py_err) => py_err,
                })?;
            return Ok(metrics.into());
        }

        Err(PyTypeError::new_err(
            "simulator must be weavetree.mdp.MdpSimulator or weavetree.mdp.TypedSimulator",
        ))
    }
}

#[pyfunction]
#[pyo3(text_signature = "(root_state_key, root_is_terminal, /)")]
/// tree(root_state_key, root_is_terminal, /)
/// --
///
/// Convenience constructor for `Tree`.
fn tree(root_state_key: u64, root_is_terminal: bool) -> PyTree {
    PyTree::new(root_state_key, root_is_terminal)
}

#[pymodule]
fn weavetree(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let mdp_mod = PyModule::new_bound(py, "mdp")?;
    mdp_mod.add_class::<PyCompiledMdp>()?;
    mdp_mod.add_class::<PyMdpSimulator>()?;
    mdp_mod.add_class::<PyTypedSimulator>()?;
    mdp_mod.add_function(wrap_pyfunction!(compile_yaml_file, &mdp_mod)?)?;
    mdp_mod.add_function(wrap_pyfunction!(compile_yaml_str, &mdp_mod)?)?;

    let mcts_mod = PyModule::new_bound(py, "mcts")?;
    mcts_mod.add_class::<PySearchConfig>()?;
    mcts_mod.add_class::<PyRunMetrics>()?;
    mcts_mod.add_class::<PyTree>()?;
    mcts_mod.add_function(wrap_pyfunction!(tree, &mcts_mod)?)?;

    module.add_submodule(&mdp_mod)?;
    module.add_submodule(&mcts_mod)?;

    let sys_modules = py.import_bound("sys")?.getattr("modules")?;
    sys_modules.set_item("weavetree.mdp", &mdp_mod)?;
    sys_modules.set_item("weavetree.mcts", &mcts_mod)?;

    Ok(())
}
