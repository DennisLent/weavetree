function assertObject(value, field) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${field} must be an object`)
  }
}

function assertArray(value, field) {
  if (!Array.isArray(value)) {
    throw new Error(`${field} must be an array`)
  }
}

function assertFiniteNumber(value, field) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`${field} must be a finite number`)
  }
}

function assertNonNegativeInteger(value, field) {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${field} must be a non-negative integer`)
  }
}

function formatValue(value) {
  return Number(value.toFixed(4))
}

function stateLabel(node) {
  const terminal = node.is_terminal ? ' (T)' : ''
  return `N${node.node_id} | S${node.state_key}${terminal}`
}

function actionLabel(action) {
  return `a${action.action_id} | n=${action.visits} | q=${formatValue(action.q)}`
}

export function parseTreeSnapshot(rawValue) {
  assertObject(rawValue, 'snapshot')
  const snapshot = rawValue

  assertNonNegativeInteger(snapshot.schema_version, 'schema_version')
  assertNonNegativeInteger(snapshot.root_node_id, 'root_node_id')
  assertNonNegativeInteger(snapshot.node_count, 'node_count')
  assertArray(snapshot.nodes, 'nodes')

  if (snapshot.nodes.length !== snapshot.node_count) {
    throw new Error('node_count does not match nodes length')
  }

  const byId = new Map()
  snapshot.nodes.forEach((node, index) => {
    assertObject(node, `nodes[${index}]`)
    assertNonNegativeInteger(node.node_id, `nodes[${index}].node_id`)
    assertFiniteNumber(node.state_key, `nodes[${index}].state_key`)
    assertNonNegativeInteger(node.depth, `nodes[${index}].depth`)
    if (typeof node.is_terminal !== 'boolean') {
      throw new Error(`nodes[${index}].is_terminal must be a boolean`)
    }
    assertArray(node.edges, `nodes[${index}].edges`)
    if (byId.has(node.node_id)) {
      throw new Error(`duplicate node_id: ${node.node_id}`)
    }
    byId.set(node.node_id, node)
  })

  if (!byId.has(snapshot.root_node_id)) {
    throw new Error(`root_node_id ${snapshot.root_node_id} does not exist in nodes`)
  }

  snapshot.nodes.forEach((node, nodeIndex) => {
    node.edges.forEach((edge, edgeIndex) => {
      assertObject(edge, `nodes[${nodeIndex}].edges[${edgeIndex}]`)
      assertNonNegativeInteger(
        edge.action_id,
        `nodes[${nodeIndex}].edges[${edgeIndex}].action_id`,
      )
      assertFiniteNumber(edge.visits, `nodes[${nodeIndex}].edges[${edgeIndex}].visits`)
      assertFiniteNumber(edge.value_sum, `nodes[${nodeIndex}].edges[${edgeIndex}].value_sum`)
      assertFiniteNumber(edge.q, `nodes[${nodeIndex}].edges[${edgeIndex}].q`)
      assertArray(edge.outcomes, `nodes[${nodeIndex}].edges[${edgeIndex}].outcomes`)

      edge.outcomes.forEach((outcome, outcomeIndex) => {
        assertObject(
          outcome,
          `nodes[${nodeIndex}].edges[${edgeIndex}].outcomes[${outcomeIndex}]`,
        )
        assertFiniteNumber(
          outcome.next_state_key,
          `nodes[${nodeIndex}].edges[${edgeIndex}].outcomes[${outcomeIndex}].next_state_key`,
        )
        assertNonNegativeInteger(
          outcome.child_node_id,
          `nodes[${nodeIndex}].edges[${edgeIndex}].outcomes[${outcomeIndex}].child_node_id`,
        )
        assertFiniteNumber(
          outcome.count,
          `nodes[${nodeIndex}].edges[${edgeIndex}].outcomes[${outcomeIndex}].count`,
        )
        if (!byId.has(outcome.child_node_id)) {
          throw new Error(`outcome references unknown child_node_id ${outcome.child_node_id}`)
        }
      })
    })
  })

  return snapshot
}

export function treeSnapshotToReactFlow(snapshot) {
  const parsed = parseTreeSnapshot(snapshot)
  const sortedNodes = [...parsed.nodes].sort((a, b) => a.node_id - b.node_id)

  const depthOrderByNodeId = new Map()
  const depthCounts = new Map()
  sortedNodes.forEach((node) => {
    const nextOrder = depthCounts.get(node.depth) ?? 0
    depthOrderByNodeId.set(node.node_id, nextOrder)
    depthCounts.set(node.depth, nextOrder + 1)
  })

  const flowNodes = sortedNodes.map((node) => {
    const row = depthOrderByNodeId.get(node.node_id) ?? 0
    return {
      id: `s-${node.node_id}`,
      type: 'state',
      position: { x: 130 + node.depth * 280, y: 80 + row * 160 },
      data: {
        stateId: String(node.state_key),
        terminal: node.is_terminal,
        label: stateLabel(node),
      },
    }
  })

  const flowEdges = []
  sortedNodes.forEach((node) => {
    const sourceRow = depthOrderByNodeId.get(node.node_id) ?? 0
    const sourceY = 80 + sourceRow * 160

    const sortedEdges = [...node.edges].sort((a, b) => a.action_id - b.action_id)
    sortedEdges.forEach((action, actionIndex) => {
      const actionNodeId = `a-${node.node_id}-${action.action_id}`
      const actionY = sourceY - 42 + actionIndex * 84
      flowNodes.push({
        id: actionNodeId,
        type: 'action',
        position: { x: 240 + node.depth * 280, y: actionY },
        data: {
          actionId: String(action.action_id),
          label: actionLabel(action),
        },
      })

      flowEdges.push({
        id: `sa-${node.node_id}-${action.action_id}`,
        source: `s-${node.node_id}`,
        target: actionNodeId,
        sourceHandle: 's-right',
        targetHandle: 't-left',
        data: { edgeType: 'stateAction' },
        animated: false,
        selectable: false,
        style: { stroke: '#6b7280', strokeDasharray: '4 2' },
      })

      action.outcomes.forEach((outcome, outcomeIndex) => {
        flowEdges.push({
          id: `o-${node.node_id}-${action.action_id}-${outcomeIndex}`,
          source: actionNodeId,
          target: `s-${outcome.child_node_id}`,
          sourceHandle: 's-right',
          targetHandle: 't-left',
          data: {
            edgeType: 'outcome',
            count: outcome.count,
            nextStateKey: outcome.next_state_key,
          },
          label: `count=${outcome.count}`,
          style: { stroke: '#111827', strokeWidth: 1.8 },
          labelStyle: { fill: '#111827', fontWeight: 600, fontSize: 11 },
        })
      })
    })
  })

  return {
    nodes: flowNodes,
    edges: flowEdges,
  }
}
