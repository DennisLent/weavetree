/**
 * Validates that a value is a non empty string.
 * Throws when the value is empty or not a string.
 */
function assertNonEmptyString(value, fieldName) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${fieldName} must be a non-empty string`)
  }
}

/**
 * Validates that a value is a finite number.
 * Error when the value is missing, NaN, or infinite.
 */
function assertFiniteNumber(value, fieldName) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`${fieldName} must be a finite number`)
  }
}

/**
 * Validates that a probability is in the inclusive range [0, 1].
 */
function assertProbability(value) {
  assertFiniteNumber(value, 'probability')
  if (value < 0 || value > 1) {
    throw new Error('probability must be between 0 and 1')
  }
}

// Basic in-memory MDP graph model:
// - state nodes
// - action nodes bound to source states
// - stochastic outcomes from action -> target state
export class MdpGraph {
  /**
   * Builds a graph from optional state/action/outcome arrays.
   */
  constructor(input = {}) {
    const { states = [], actions = [], outcomes = [] } = input

    this.stateNodes = new Map()
    this.actionNodes = new Map()
    this.outcomeEdges = new Map()

    this.nextStateId = 0
    this.nextActionNodeId = 0
    this.nextOutcomeEdgeId = 0

    states.forEach((state) => this.addState(state))
    actions.forEach((action) => this.addAction(action))
    outcomes.forEach((outcome) => this.addOutcome(outcome))
  }

  /**
   * Add new state node with optional manual id and terminal flag.
   * If no id is provided, use sequential numeric string id is generated
   */
  addState(input = {}) {
    const id = input.id ?? String(this.nextStateId)
    const terminal = input.terminal ?? false

    assertNonEmptyString(id, 'state.id')
    if (typeof terminal !== 'boolean') {
      throw new Error('state.terminal must be a boolean')
    }
    if (this.stateNodes.has(id)) {
      throw new Error(`State already exists: ${id}`)
    }

    // Auto ids simple and sequential: "0", "1", "2", ...
    const numericId = Number.parseInt(id, 10)
    if (Number.isInteger(numericId) && numericId >= this.nextStateId) {
      this.nextStateId = numericId + 1
    } else if (id === String(this.nextStateId)) {
      this.nextStateId += 1
    }

    const state = { id, terminal }
    this.stateNodes.set(id, state)
    return state
  }

  /**
   * Adds an action node to a source state.
   */
  addAction(input) {
    if (!input || typeof input !== 'object') {
      throw new Error('action input is required')
    }

    const id = input.id ?? `a${this.nextActionNodeId++}`
    const { state_id, action_id } = input

    assertNonEmptyString(id, 'action.id')
    assertNonEmptyString(state_id, 'action.state_id')
    assertNonEmptyString(action_id, 'action.action_id')

    if (!this.stateNodes.has(state_id)) {
      throw new Error(`Unknown source state: ${state_id}`)
    }
    if (this.actionNodes.has(id)) {
      throw new Error(`Action already exists: ${id}`)
    }

    const action = { id, state_id, action_id }
    this.actionNodes.set(id, action)
    return action
  }

  /**
   * Add a stochastic outcome edge from an action node to a target state.
   * Reward defaults to 0
   */
  addOutcome(input) {
    if (!input || typeof input !== 'object') {
      throw new Error('outcome input is required')
    }

    const id = input.id ?? `o${this.nextOutcomeEdgeId++}`
    const { action_node_id, target_state_id, probability, reward = 0 } = input

    assertNonEmptyString(id, 'outcome.id')
    assertNonEmptyString(action_node_id, 'outcome.action_node_id')
    assertNonEmptyString(target_state_id, 'outcome.target_state_id')
    assertProbability(probability)
    assertFiniteNumber(reward, 'outcome.reward')

    if (!this.actionNodes.has(action_node_id)) {
      throw new Error(`Unknown action node: ${action_node_id}`)
    }
    if (!this.stateNodes.has(target_state_id)) {
      throw new Error(`Unknown target state: ${target_state_id}`)
    }
    if (this.outcomeEdges.has(id)) {
      throw new Error(`Outcome already exists: ${id}`)
    }

    const outcome = {
      id,
      action_node_id,
      target_state_id,
      probability,
      reward,
    }
    this.outcomeEdges.set(id, outcome)
    return outcome
  }

  /**
   * Checks each (source state, action id) outcome distribution.
   * Returns entries where probabilities do not sum to 1 within epsilon.
   */
  validateActionDistributions(epsilon = 1e-9) {
    const sums = new Map()

    for (const outcome of this.outcomeEdges.values()) {
      const action = this.actionNodes.get(outcome.action_node_id)
      if (!action) {
        continue
      }

      // Validate per (source state, action name) distribution.
      const key = `${action.state_id}::${action.action_id}`
      const current = sums.get(key) ?? 0
      sums.set(key, current + outcome.probability)
    }

    const invalid = []
    for (const [key, sum] of sums.entries()) {
      if (Math.abs(sum - 1) > epsilon) {
        const [state_id, action_id] = key.split('::')
        invalid.push({ state_id, action_id, probability_sum: sum })
      }
    }

    return invalid
  }

  /**
   * Serializes graph data into plain JSON-friendly arrays.
   */
  toJSON() {
    return {
      states: [...this.stateNodes.values()],
      actions: [...this.actionNodes.values()],
      outcomes: [...this.outcomeEdges.values()],
    }
  }

  /**
   * Convert the graph model into React Flow nodes and edges.
   * Optional state positions can be provided
   */
  toReactFlowData(positionByStateId = {}) {
    // Convert model objects into React Flow nodes/edges with simple defaults.
    const stateNodes = [...this.stateNodes.values()].map((state) => {
      const position = positionByStateId[state.id] ?? { x: 80, y: 80 }

      return {
        id: `s-${state.id}`,
        position,
        data: { label: `S${state.id}${state.terminal ? ' (T)' : ''}` },
        style: {
          width: 76,
          height: 76,
          borderRadius: '50%',
          border: '2px solid #1f2937',
          background: state.terminal ? '#fef3c7' : '#e5e7eb',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontWeight: 600,
        },
      }
    })

    const actionCountByState = new Map()
    const actionNodes = [...this.actionNodes.values()].map((action) => {
      const sourcePosition = positionByStateId[action.state_id] ?? { x: 80, y: 80 }
      const count = actionCountByState.get(action.state_id) ?? 0
      actionCountByState.set(action.state_id, count + 1)

      return {
        id: `a-${action.id}`,
        position: {
          x: sourcePosition.x + 130,
          y: sourcePosition.y + count * 52 + 10,
        },
        data: { label: action.action_id },
        style: {
          width: 62,
          height: 30,
          borderRadius: 6,
          border: '1.5px solid #1f2937',
          background: '#dbeafe',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 12,
          fontWeight: 600,
        },
      }
    })

    const linkEdges = [...this.actionNodes.values()].map((action) => ({
      id: `link-${action.id}`,
      source: `s-${action.state_id}`,
      target: `a-${action.id}`,
      animated: true,
      selectable: false,
      style: { stroke: '#6b7280', strokeDasharray: '4 2' },
    }))

    const outcomeEdges = [...this.outcomeEdges.values()].map((outcome) => ({
      id: `o-${outcome.id}`,
      source: `a-${outcome.action_node_id}`,
      target: `s-${outcome.target_state_id}`,
      label: `p=${outcome.probability}, r=${outcome.reward}`,
      style: { stroke: '#111827', strokeWidth: 1.8 },
      labelStyle: { fill: '#111827', fontWeight: 600, fontSize: 11 },
    }))

    return {
      nodes: [...stateNodes, ...actionNodes],
      edges: [...linkEdges, ...outcomeEdges],
    }
  }
}
