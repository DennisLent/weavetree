use crate::{ActionSpec, CompiledMdp, MdpError, MdpSpec, OutcomeSpec, StateSpec};

#[derive(Debug, Clone, Default)]
/// Struct to build MDPs
pub struct MdpBuilder {
    start: Option<String>,
    states: Vec<StateSpec>,
}

impl MdpBuilder {
    /// Create a new MDPBuilder
    pub fn new() -> Self {
        Self::default()
    }

    /// Define the start position of the MDP
    pub fn set_start(&mut self, state: impl Into<String>) -> &mut Self {
        self.start = Some(state.into());
        self
    }

    /// Add a new state
    /// Terminal flag if this state is the final one
    pub fn add_state(&mut self, id: impl Into<String>, terminal: bool) -> &mut Self {
        self.states.push(StateSpec {
            id: id.into(),
            terminal: Some(terminal),
            actions: Some(Vec::new()),
        });
        self
    }

    /// Add an action to a state
    pub fn add_action(
        &mut self,
        state_id: impl AsRef<str>,
        action_id: impl Into<String>,
    ) -> Result<&mut Self, MdpError> {
        let state_id = state_id.as_ref();
        let action_id = action_id.into();

        let state = self
            .states
            .iter_mut()
            .find(|s| s.id == state_id)
            .ok_or_else(|| MdpError::BuilderUnknownState {
                state: state_id.to_string(),
            })?;

        let actions = state.actions.get_or_insert_with(Vec::new);
        actions.push(ActionSpec {
            id: action_id,
            outcomes: Vec::new(),
        });

        Ok(self)
    }

    /// Add an outcome to an action
    /// Action can be stochastic so we need to push it to the OutcomeSet
    pub fn add_outcome(
        &mut self,
        state_id: impl AsRef<str>,
        action_id: impl AsRef<str>,
        next: impl Into<String>,
        prob: f64,
        reward: f64,
    ) -> Result<&mut Self, MdpError> {
        let state_id = state_id.as_ref();
        let action_id = action_id.as_ref();

        let state = self
            .states
            .iter_mut()
            .find(|s| s.id == state_id)
            .ok_or_else(|| MdpError::BuilderUnknownState {
                state: state_id.to_string(),
            })?;

        let actions = state.actions.get_or_insert_with(Vec::new);
        let action = actions
            .iter_mut()
            .find(|a| a.id == action_id)
            .ok_or_else(|| MdpError::BuilderUnknownAction {
                state: state_id.to_string(),
                action: action_id.to_string(),
            })?;

        action.outcomes.push(OutcomeSpec {
            next: next.into(),
            prob,
            reward,
        });

        Ok(self)
    }

    pub fn build_spec(self) -> Result<MdpSpec, MdpError> {
        let start = self.start.ok_or(MdpError::MissingStart)?;
        let spec = MdpSpec {
            version: Some(1),
            start,
            states: self.states,
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn compile(self) -> Result<CompiledMdp, MdpError> {
        let spec = self.build_spec()?;
        spec.compile()
    }
}
