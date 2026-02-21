import { describe, expect, it } from 'vitest'

import { mdpSpecToYaml, toMdpSpecFromGraph } from './exportMdp'

const validNodes = [
  { id: 's-0', type: 'state', data: { stateId: 's0', terminal: false } },
  { id: 's-1', type: 'state', data: { stateId: 's1', terminal: true } },
  { id: 'a-0', type: 'action', data: { actionId: 'a0' } },
]

const validEdges = [
  { id: 'sa-0', source: 's-0', target: 'a-0', data: { edgeType: 'stateAction' } },
  {
    id: 'o-0',
    source: 'a-0',
    target: 's-1',
    data: { edgeType: 'outcome', probability: 1, reward: 1.5 },
  },
]

describe('toMdpSpecFromGraph', () => {
  it('builds an MDP spec with inferred start state', () => {
    const spec = toMdpSpecFromGraph(validNodes, validEdges)

    expect(spec).toEqual({
      version: 1,
      start: 's0',
      states: [
        {
          id: 's0',
          terminal: false,
          actions: [
            {
              id: 'a0',
              outcomes: [{ next: 's1', prob: 1, reward: 1.5 }],
            },
          ],
        },
        {
          id: 's1',
          terminal: true,
        },
      ],
    })
  })

  it('throws when outcome probabilities do not sum to 1', () => {
    const edges = [
      ...validEdges,
      {
        id: 'o-1',
        source: 'a-0',
        target: 's-0',
        data: { edgeType: 'outcome', probability: 0.2, reward: 0 },
      },
    ]

    expect(() => toMdpSpecFromGraph(validNodes, edges)).toThrow(/probability sum/i)
  })

  it('throws when start state does not exist', () => {
    expect(() => toMdpSpecFromGraph(validNodes, validEdges, 'missing')).toThrow(
      /does not exist/i,
    )
  })

  it('throws when terminal state has actions', () => {
    const nodes = [
      { id: 's-0', type: 'state', data: { stateId: 's0', terminal: true } },
      { id: 's-1', type: 'state', data: { stateId: 's1', terminal: true } },
      { id: 'a-0', type: 'action', data: { actionId: 'a0' } },
    ]

    expect(() => toMdpSpecFromGraph(nodes, validEdges)).toThrow(/terminal state/i)
  })
})

describe('mdpSpecToYaml', () => {
  it('serializes to expected yaml layout', () => {
    const spec = toMdpSpecFromGraph(validNodes, validEdges)
    const yaml = mdpSpecToYaml(spec)

    expect(yaml).toContain('version: 1')
    expect(yaml).toContain('start: "s0"')
    expect(yaml).toContain('- id: "s0"')
    expect(yaml).toContain('- id: "a0"')
    expect(yaml).toContain('next: "s1"')
    expect(yaml).toContain('prob: 1')
    expect(yaml).toContain('reward: 1.5')
  })
})
