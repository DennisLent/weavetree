const PROBABILITY_TOLERANCE = 1e-9

function quoteYamlString(value) {
  return JSON.stringify(String(value))
}

function formatYamlNumber(value) {
  if (!Number.isFinite(value)) {
    throw new Error(`Invalid numeric value: ${value}`)
  }
  return String(Object.is(value, -0) ? 0 : value)
}

function toMdpSpecFromGraph(nodes, edges, startStateOverride = '') {
  const stateNodes = nodes.filter((node) => node.type === 'state')
  const actionNodes = nodes.filter((node) => node.type === 'action')

  if (stateNodes.length === 0) {
    throw new Error('Add at least one state before exporting YAML.')
  }

  const stateByNodeId = new Map()
  const stateIdSet = new Set()
  stateNodes.forEach((node) => {
    const stateId = node.data?.stateId?.trim()
    if (!stateId) {
      throw new Error('All states must have a non-empty id.')
    }
    if (stateIdSet.has(stateId)) {
      throw new Error(`Duplicate state id "${stateId}".`)
    }
    stateIdSet.add(stateId)
    stateByNodeId.set(node.id, { id: stateId, terminal: Boolean(node.data?.terminal) })
  })

  const actionByNodeId = new Map()
  actionNodes.forEach((node) => {
    const actionId = node.data?.actionId?.trim()
    if (!actionId) {
      throw new Error('All actions must have a non-empty action id.')
    }
    actionByNodeId.set(node.id, { id: actionId })
  })

  const incomingStateByActionNodeId = new Map()
  edges
    .filter((edge) => edge.data?.edgeType === 'stateAction')
    .forEach((edge) => {
      const sourceState = stateByNodeId.get(edge.source)
      const targetAction = actionByNodeId.get(edge.target)
      if (!sourceState || !targetAction) {
        throw new Error('State-action edges must connect state -> action.')
      }
      if (incomingStateByActionNodeId.has(edge.target)) {
        throw new Error(
          `Action "${targetAction.id}" is connected to multiple source states.`,
        )
      }
      incomingStateByActionNodeId.set(edge.target, edge.source)
    })

  const outcomesByActionNodeId = new Map()
  edges
    .filter((edge) => edge.data?.edgeType === 'outcome')
    .forEach((edge) => {
      const sourceAction = actionByNodeId.get(edge.source)
      const targetState = stateByNodeId.get(edge.target)
      if (!sourceAction || !targetState) {
        throw new Error('Outcome edges must connect action -> state.')
      }

      const probability = edge.data?.probability
      const reward = edge.data?.reward
      if (!Number.isFinite(probability) || probability < 0) {
        throw new Error(`Action "${sourceAction.id}" has an invalid probability.`)
      }
      if (!Number.isFinite(reward)) {
        throw new Error(`Action "${sourceAction.id}" has an invalid reward.`)
      }

      const outcomes = outcomesByActionNodeId.get(edge.source) ?? []
      outcomes.push({
        next: targetState.id,
        prob: probability,
        reward,
      })
      outcomesByActionNodeId.set(edge.source, outcomes)
    })

  const actionsByStateNodeId = new Map()
  actionNodes.forEach((actionNode) => {
    const sourceStateNodeId = incomingStateByActionNodeId.get(actionNode.id)
    if (!sourceStateNodeId) {
      throw new Error(`Action "${actionNode.data?.actionId}" is not connected to a state.`)
    }

    const existing = actionsByStateNodeId.get(sourceStateNodeId) ?? []
    existing.push(actionNode)
    actionsByStateNodeId.set(sourceStateNodeId, existing)
  })

  const states = stateNodes.map((stateNode) => {
    const state = stateByNodeId.get(stateNode.id)
    const actionNodesForState = actionsByStateNodeId.get(stateNode.id) ?? []
    const actionIdSet = new Set()

    const actions = actionNodesForState.map((actionNode) => {
      const action = actionByNodeId.get(actionNode.id)
      if (actionIdSet.has(action.id)) {
        throw new Error(`State "${state.id}" has duplicate action id "${action.id}".`)
      }
      actionIdSet.add(action.id)

      const outcomes = outcomesByActionNodeId.get(actionNode.id) ?? []
      if (outcomes.length === 0) {
        throw new Error(
          `Action "${action.id}" in state "${state.id}" must have at least one outcome.`,
        )
      }

      const probabilitySum = outcomes.reduce((sum, outcome) => sum + outcome.prob, 0)
      if (Math.abs(probabilitySum - 1) > PROBABILITY_TOLERANCE) {
        throw new Error(
          `Action "${action.id}" in state "${state.id}" has probability sum ${probabilitySum.toFixed(6)} (must be 1).`,
        )
      }

      return {
        id: action.id,
        outcomes,
      }
    })

    if (state.terminal && actions.length > 0) {
      throw new Error(`Terminal state "${state.id}" cannot have actions.`)
    }

    if (actions.length === 0) {
      return {
        id: state.id,
        terminal: state.terminal,
      }
    }

    return {
      id: state.id,
      terminal: state.terminal,
      actions,
    }
  })

  const start = startStateOverride.trim() === '' ? states[0]?.id : startStateOverride.trim()

  if (!stateIdSet.has(start)) {
    throw new Error(`Start state "${start}" does not exist.`)
  }

  return {
    version: 1,
    start,
    states,
  }
}

function mdpSpecToYaml(spec) {
  const lines = []
  lines.push(`version: ${spec.version}`)
  lines.push(`start: ${quoteYamlString(spec.start)}`)
  lines.push('states:')

  spec.states.forEach((state) => {
    lines.push(`  - id: ${quoteYamlString(state.id)}`)
    lines.push(`    terminal: ${state.terminal ? 'true' : 'false'}`)

    if (!state.actions || state.actions.length === 0) {
      return
    }

    lines.push('    actions:')
    state.actions.forEach((action) => {
      lines.push(`      - id: ${quoteYamlString(action.id)}`)
      lines.push('        outcomes:')
      action.outcomes.forEach((outcome) => {
        lines.push(`          - next: ${quoteYamlString(outcome.next)}`)
        lines.push(`            prob: ${formatYamlNumber(outcome.prob)}`)
        lines.push(`            reward: ${formatYamlNumber(outcome.reward)}`)
      })
    })
  })

  return `${lines.join('\n')}\n`
}

export { PROBABILITY_TOLERANCE, toMdpSpecFromGraph, mdpSpecToYaml }
