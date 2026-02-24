import { describe, expect, it } from 'vitest'

import { renderTreeSnapshotSvg } from './renderTreeSvg'

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

describe('renderTreeSnapshotSvg', () => {
  it('renders valid snapshots to svg', () => {
    const svg = renderTreeSnapshotSvg(sampleSnapshot)
    expect(svg.startsWith('<svg')).toBe(true)
    expect(svg).toContain('N0')
    expect(svg).toContain('count=3')
  })

  it('throws on invalid snapshot shape', () => {
    expect(() => renderTreeSnapshotSvg({ nodes: [] })).toThrow(/node_count/i)
  })
})
