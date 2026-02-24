import { describe, expect, it } from 'vitest'

import { parseTreeSnapshot, treeSnapshotToReactFlow } from './treeSnapshotGraph'

const sampleSnapshot = {
  schema_version: 1,
  root_node_id: 0,
  node_count: 2,
  nodes: [
    {
      node_id: 0,
      state_key: 42,
      depth: 0,
      is_terminal: false,
      parent_node_id: null,
      parent_action_id: null,
      edges: [
        {
          action_id: 0,
          visits: 3,
          value_sum: 9,
          q: 3,
          outcomes: [{ next_state_key: 7, child_node_id: 1, count: 3 }],
        },
      ],
    },
    {
      node_id: 1,
      state_key: 7,
      depth: 1,
      is_terminal: true,
      parent_node_id: 0,
      parent_action_id: 0,
      edges: [],
    },
  ],
}

describe('parseTreeSnapshot', () => {
  it('accepts valid snapshots', () => {
    expect(parseTreeSnapshot(sampleSnapshot)).toEqual(sampleSnapshot)
  })

  it('rejects snapshots with invalid child links', () => {
    const broken = structuredClone(sampleSnapshot)
    broken.nodes[0].edges[0].outcomes[0].child_node_id = 99
    expect(() => parseTreeSnapshot(broken)).toThrow(/unknown child_node_id/i)
  })
})

describe('treeSnapshotToReactFlow', () => {
  it('creates state/action graph elements', () => {
    const result = treeSnapshotToReactFlow(sampleSnapshot)

    const stateNodes = result.nodes.filter((node) => node.type === 'state')
    const actionNodes = result.nodes.filter((node) => node.type === 'action')
    const outcomeEdges = result.edges.filter((edge) => edge.data?.edgeType === 'outcome')

    expect(stateNodes).toHaveLength(2)
    expect(actionNodes).toHaveLength(1)
    expect(outcomeEdges).toHaveLength(1)
    expect(outcomeEdges[0].target).toBe('s-1')
    expect(String(actionNodes[0].data.label)).toContain('q=3')
  })
})
