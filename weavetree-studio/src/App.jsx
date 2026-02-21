import { useState, useCallback, useMemo, useRef } from 'react'
import {
  ReactFlow,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
  Background,
  Controls,
  MiniMap,
  Handle,
  Position,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import './App.css'
import { mdpSpecToYaml, toMdpSpecFromGraph } from './model/exportMdp'

function formatOutcomeLabel(probability, reward) {
  return `p=${probability}, r=${reward}`
}

// State nodes expose 4 source and 4 target handles (top/right/bottom/left)
// to make dense MDP layouts easier to route.
function StateNode({ data }) {
  return (
    <div className={`state-node ${data.terminal ? 'terminal' : ''}`}>
      <Handle
        type="target"
        id="t-top"
        position={Position.Top}
        className="dock-handle dock-target"
        style={{ left: '42%' }}
      />
      <Handle
        type="source"
        id="s-top"
        position={Position.Top}
        className="dock-handle dock-source"
        style={{ left: '58%' }}
      />
      <Handle
        type="target"
        id="t-right"
        position={Position.Right}
        className="dock-handle dock-target"
        style={{ top: '42%' }}
      />
      <Handle
        type="source"
        id="s-right"
        position={Position.Right}
        className="dock-handle dock-source"
        style={{ top: '58%' }}
      />
      <Handle
        type="target"
        id="t-bottom"
        position={Position.Bottom}
        className="dock-handle dock-target"
        style={{ left: '42%' }}
      />
      <Handle
        type="source"
        id="s-bottom"
        position={Position.Bottom}
        className="dock-handle dock-source"
        style={{ left: '58%' }}
      />
      <Handle
        type="target"
        id="t-left"
        position={Position.Left}
        className="dock-handle dock-target"
        style={{ top: '42%' }}
      />
      <Handle
        type="source"
        id="s-left"
        position={Position.Left}
        className="dock-handle dock-source"
        style={{ top: '58%' }}
      />
      <span>{data.label}</span>
    </div>
  )
}

// Action nodes are compact decision points that receive one incoming
// state link and emit one or more outcome links to states.
function ActionNode({ data }) {
  return (
    <div className="action-node">
      <Handle
        type="target"
        id="t-left"
        position={Position.Left}
        className="dock-handle dock-target"
      />
      <Handle
        type="source"
        id="s-right"
        position={Position.Right}
        className="dock-handle dock-source"
      />
      <span>{data.label}</span>
    </div>
  )
}

const nodeTypes = {
  state: StateNode,
  action: ActionNode,
}

// Starter graph for immediate interaction in the studio UI.
const initialNodes = [
  {
    id: 's-0',
    type: 'state',
    position: { x: 160, y: 140 },
    data: { stateId: '0', terminal: false, label: 'S0' },
  },
  {
    id: 's-1',
    type: 'state',
    position: { x: 540, y: 120 },
    data: { stateId: '1', terminal: false, label: 'S1' },
  },
  {
    id: 'a-0',
    type: 'action',
    position: { x: 340, y: 160 },
    data: { actionId: 'move', label: 'move' },
  },
]

const initialEdges = [
  {
    id: 'sa-0',
    source: 's-0',
    target: 'a-0',
    sourceHandle: 's-right',
    targetHandle: 't-left',
    data: { edgeType: 'stateAction' },
    animated: true,
    style: { stroke: '#6b7280', strokeDasharray: '4 2' },
  },
  {
    id: 'o-0',
    source: 'a-0',
    target: 's-1',
    sourceHandle: 's-right',
    targetHandle: 't-left',
    data: { edgeType: 'outcome', probability: 1, reward: 0 },
    label: formatOutcomeLabel(1, 0),
    style: { stroke: '#111827', strokeWidth: 1.8 },
    labelStyle: { fill: '#111827', fontWeight: 600, fontSize: 11 },
  },
]

export default function App() {
  const [nodes, setNodes] = useState(initialNodes)
  const [edges, setEdges] = useState(initialEdges)
  const [stateIdInput, setStateIdInput] = useState('')
  const [stateTerminalInput, setStateTerminalInput] = useState(false)
  const [actionNameInput, setActionNameInput] = useState('')
  const [startStateInput, setStartStateInput] = useState('')
  const [exportFileNameInput, setExportFileNameInput] = useState('model.mdp.yaml')
  const [selectedNodeId, setSelectedNodeId] = useState(null)
  const [selectedEdgeId, setSelectedEdgeId] = useState(null)
  const [errorMessage, setErrorMessage] = useState('')

  const nodeCounterRef = useRef(2)
  const edgeCounterRef = useRef(1)
  const nextStateIdRef = useRef(2)

  const onNodesChange = useCallback(
    (changes) => setNodes((nodesSnapshot) => applyNodeChanges(changes, nodesSnapshot)),
    [],
  )
  const onEdgesChange = useCallback(
    (changes) => setEdges((edgesSnapshot) => applyEdgeChanges(changes, edgesSnapshot)),
    [],
  )

  // Connection rules:
  // 1) state -> action (an action can belong to only one state)
  // 2) action -> state (outcomes with probability/reward metadata)
  // Any other connection shape is rejected.
  const onConnect = useCallback(
    (params) => {
      if (!params.source || !params.target) {
        return
      }

      setErrorMessage('')
      setEdges((edgesSnapshot) => {
        const sourceNode = nodes.find((node) => node.id === params.source)
        const targetNode = nodes.find((node) => node.id === params.target)

        if (!sourceNode || !targetNode || params.source === params.target) {
          setErrorMessage('Invalid connection.')
          return edgesSnapshot
        }

        if (sourceNode.type === 'state' && targetNode.type === 'action') {
          const actionIncoming = edgesSnapshot.filter(
            (edge) =>
              edge.data?.edgeType === 'stateAction' && edge.target === targetNode.id,
          )
          if (actionIncoming.length > 0) {
            setErrorMessage('Action already belongs to a source state.')
            return edgesSnapshot
          }

          const duplicate = edgesSnapshot.some(
            (edge) =>
              edge.data?.edgeType === 'stateAction' &&
              edge.source === sourceNode.id &&
              edge.target === targetNode.id,
          )
          if (duplicate) {
            return edgesSnapshot
          }

          return addEdge(
            {
              id: `sa-${edgeCounterRef.current++}`,
              source: sourceNode.id,
              target: targetNode.id,
              sourceHandle: params.sourceHandle,
              targetHandle: params.targetHandle,
              data: { edgeType: 'stateAction' },
              animated: true,
              style: { stroke: '#6b7280', strokeDasharray: '4 2' },
            },
            edgesSnapshot,
          )
        }

        if (sourceNode.type === 'action' && targetNode.type === 'state') {
          const currentOutcomes = edgesSnapshot.filter(
            (edge) =>
              edge.data?.edgeType === 'outcome' && edge.source === sourceNode.id,
          )
          const currentSum = currentOutcomes.reduce(
            (sum, edge) => sum + (edge.data?.probability ?? 0),
            0,
          )

          if (currentSum >= 1) {
            setErrorMessage(
              'Cannot add more outcomes: this action already has probability sum 1.',
            )
            return edgesSnapshot
          }

          const remaining = Number((1 - currentSum).toFixed(4))
          const probability = currentOutcomes.length === 0 ? 1 : remaining
          const reward = 0

          // New outcomes default to reward=0 and consume remaining probability.
          return addEdge(
            {
              id: `o-${edgeCounterRef.current++}`,
              source: sourceNode.id,
              target: targetNode.id,
              sourceHandle: params.sourceHandle,
              targetHandle: params.targetHandle,
              data: { edgeType: 'outcome', probability, reward },
              label: formatOutcomeLabel(probability, reward),
              style: { stroke: '#111827', strokeWidth: 1.8 },
              labelStyle: { fill: '#111827', fontWeight: 600, fontSize: 11 },
            },
            edgesSnapshot,
          )
        }

        setErrorMessage('Only state -> action and action -> state connections are allowed.')
        return edgesSnapshot
      })
    },
    [nodes],
  )

  const actionDistributionIssues = useMemo(() => {
    const sumsByAction = new Map()
    edges
      .filter((edge) => edge.data?.edgeType === 'outcome')
      .forEach((edge) => {
        const current = sumsByAction.get(edge.source) ?? 0
        sumsByAction.set(edge.source, current + (edge.data?.probability ?? 0))
      })

    const issues = []
    for (const [actionNodeId, sum] of sumsByAction.entries()) {
      if (Math.abs(sum - 1) > 1e-9) {
        issues.push({
          actionNodeId,
          sum: Number(sum.toFixed(6)),
        })
      }
    }
    return issues
  }, [edges])

  const issueActionNodeIds = useMemo(
    () => new Set(actionDistributionIssues.map((issue) => issue.actionNodeId)),
    [actionDistributionIssues],
  )

  const renderedEdges = useMemo(
    () =>
      edges.map((edge) => {
        if (edge.data?.edgeType !== 'outcome') {
          return edge
        }

        if (!issueActionNodeIds.has(edge.source)) {
          return edge
        }

        return {
          ...edge,
          style: { ...(edge.style ?? {}), stroke: '#b91c1c', strokeWidth: 2 },
        }
      }),
    [edges, issueActionNodeIds],
  )

  const selectedNode = nodes.find((node) => node.id === selectedNodeId)
  const selectedEdge = edges.find((edge) => edge.id === selectedEdgeId)

  // Partial data patch helper used by side-panel editors.
  const updateNodeData = (nodeId, partialData) => {
    setNodes((snapshot) =>
      snapshot.map((node) => {
        if (node.id !== nodeId) {
          return node
        }

        const nextData = { ...node.data, ...partialData }
        return { ...node, data: nextData }
      }),
    )
  }

  const addState = () => {
    setErrorMessage('')
    const trimmedId = stateIdInput.trim()
    const stateId = trimmedId === '' ? String(nextStateIdRef.current++) : trimmedId

    if (
      nodes.some((node) => node.type === 'state' && node.data?.stateId === stateId)
    ) {
      setErrorMessage(`State id "${stateId}" already exists.`)
      return
    }

    const nextNode = {
      id: `s-${nodeCounterRef.current++}`,
      type: 'state',
      position: { x: 100 + nodes.length * 24, y: 100 + nodes.length * 24 },
      data: {
        stateId,
        terminal: stateTerminalInput,
        label: `S${stateId}${stateTerminalInput ? ' (T)' : ''}`,
      },
    }

    setNodes((snapshot) => [...snapshot, nextNode])
    setStateIdInput('')
    setStateTerminalInput(false)
  }

  const addAction = () => {
    setErrorMessage('')
    const actionName = actionNameInput.trim()
    if (actionName === '') {
      setErrorMessage('Action name is required.')
      return
    }

    const nextNode = {
      id: `a-${nodeCounterRef.current++}`,
      type: 'action',
      position: { x: 260 + nodes.length * 18, y: 180 + nodes.length * 18 },
      data: { actionId: actionName, label: actionName },
    }

    setNodes((snapshot) => [...snapshot, nextNode])
    setActionNameInput('')
  }

  const deleteSelected = () => {
    if (selectedNodeId) {
      setNodes((snapshot) => snapshot.filter((node) => node.id !== selectedNodeId))
      setEdges((snapshot) =>
        snapshot.filter(
          (edge) => edge.source !== selectedNodeId && edge.target !== selectedNodeId,
        ),
      )
      setSelectedNodeId(null)
      return
    }

    if (selectedEdgeId) {
      setEdges((snapshot) => snapshot.filter((edge) => edge.id !== selectedEdgeId))
      setSelectedEdgeId(null)
    }
  }

  const updateOutcomeEdge = (edgeId, patch) => {
    setEdges((snapshot) =>
      snapshot.map((edge) => {
        if (edge.id !== edgeId || edge.data?.edgeType !== 'outcome') {
          return edge
        }

        const probability =
          patch.probability !== undefined ? patch.probability : edge.data.probability
        const reward = patch.reward !== undefined ? patch.reward : edge.data.reward

        // Keep edge label in sync with editable outcome data.
        return {
          ...edge,
          data: { ...edge.data, ...patch },
          label: formatOutcomeLabel(probability, reward),
        }
      }),
    )
  }

  const exportYaml = () => {
    setErrorMessage('')
    try {
      const spec = toMdpSpecFromGraph(nodes, edges, startStateInput)
      const yaml = mdpSpecToYaml(spec)

      const requestedName = exportFileNameInput.trim() || 'model.mdp.yaml'
      const downloadName = /\.ya?ml$/i.test(requestedName)
        ? requestedName
        : `${requestedName}.mdp.yaml`

      const blob = new Blob([yaml], { type: 'application/x-yaml;charset=utf-8' })
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = downloadName
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      URL.revokeObjectURL(url)
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Failed to export YAML.')
    }
  }

  return (
    <div className="app-shell">
      <aside className="side-panel">
        <h2>MDP Studio</h2>

        <div className="panel-section">
          <h3>Add State</h3>
          <label>
            State id (optional)
            <input
              value={stateIdInput}
              onChange={(event) => setStateIdInput(event.target.value)}
              placeholder="auto: 0,1,2..."
            />
          </label>
          <label className="inline-checkbox">
            <input
              type="checkbox"
              checked={stateTerminalInput}
              onChange={(event) => setStateTerminalInput(event.target.checked)}
            />
            Terminal
          </label>
          <button onClick={addState}>Place state</button>
        </div>

        <div className="panel-section">
          <h3>Add Action</h3>
          <label>
            Action name
            <input
              value={actionNameInput}
              onChange={(event) => setActionNameInput(event.target.value)}
              placeholder="e.g. move"
            />
          </label>
          <button onClick={addAction}>Place action</button>
        </div>

        <div className="panel-section">
          <h3>Editing</h3>
          {!selectedNode && !selectedEdge ? (
            <p className="muted">Select a node or edge to edit.</p>
          ) : null}

          {selectedNode?.type === 'state' ? (
            <div className="editor-block">
              <p className="item-title">State {selectedNode.data.stateId}</p>
              <label>
                State id
                <input
                  value={selectedNode.data.stateId}
                  onChange={(event) => {
                    const nextId = event.target.value
                    const inUse = nodes.some(
                      (node) =>
                        node.id !== selectedNode.id &&
                        node.type === 'state' &&
                        node.data.stateId === nextId,
                    )
                    if (inUse) {
                      setErrorMessage(`State id "${nextId}" already exists.`)
                      return
                    }
                    setErrorMessage('')
                    updateNodeData(selectedNode.id, {
                      stateId: nextId,
                      label: `S${nextId}${selectedNode.data.terminal ? ' (T)' : ''}`,
                    })
                  }}
                />
              </label>
              <label className="inline-checkbox">
                <input
                  type="checkbox"
                  checked={selectedNode.data.terminal}
                  onChange={(event) => {
                    const terminal = event.target.checked
                    updateNodeData(selectedNode.id, {
                      terminal,
                      label: `S${selectedNode.data.stateId}${terminal ? ' (T)' : ''}`,
                    })
                  }}
                />
                Terminal
              </label>
            </div>
          ) : null}

          {selectedNode?.type === 'action' ? (
            <div className="editor-block">
              <p className="item-title">Action</p>
              <label>
                Action name
                <input
                  value={selectedNode.data.actionId}
                  onChange={(event) =>
                    updateNodeData(selectedNode.id, {
                      actionId: event.target.value,
                      label: event.target.value,
                    })
                  }
                />
              </label>
            </div>
          ) : null}

          {selectedEdge?.data?.edgeType === 'outcome' ? (
            <div className="editor-block">
              <p className="item-title">Outcome Edge</p>
              <label>
                Probability
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  value={selectedEdge.data.probability}
                  onChange={(event) => {
                    const value = Number(event.target.value)
                    if (Number.isNaN(value) || value < 0 || value > 1) {
                      return
                    }
                    updateOutcomeEdge(selectedEdge.id, { probability: value })
                  }}
                />
              </label>
              <label>
                Reward
                <input
                  type="number"
                  step="0.1"
                  value={selectedEdge.data.reward}
                  onChange={(event) => {
                    const value = Number(event.target.value)
                    if (Number.isNaN(value)) {
                      return
                    }
                    updateOutcomeEdge(selectedEdge.id, { reward: value })
                  }}
                />
              </label>
            </div>
          ) : null}

          {(selectedNode || selectedEdge) && (
            <button className="danger" onClick={deleteSelected}>
              Delete selected
            </button>
          )}
        </div>

        <div className="panel-section">
          <h3>Validation</h3>
          {actionDistributionIssues.length === 0 ? (
            <p className="ok">All action outcome distributions sum to 1.</p>
          ) : (
            actionDistributionIssues.map((issue) => {
              const actionNode = nodes.find((node) => node.id === issue.actionNodeId)
              const actionName = actionNode?.data?.actionId ?? issue.actionNodeId
              return (
                <p key={issue.actionNodeId} className="warn">
                  {actionName}: sum={issue.sum}
                </p>
              )
            })
          )}
        </div>

        <div className="panel-section">
          <h3>Export YAML</h3>
          <label>
            Start state id (optional)
            <input
              value={startStateInput}
              onChange={(event) => setStartStateInput(event.target.value)}
              placeholder="auto: first state in graph"
            />
          </label>
          <label>
            File name
            <input
              value={exportFileNameInput}
              onChange={(event) => setExportFileNameInput(event.target.value)}
              placeholder="model.mdp.yaml"
            />
          </label>
          <button onClick={exportYaml}>Download YAML</button>
        </div>

        {errorMessage ? <p className="error">{errorMessage}</p> : null}
      </aside>

      <div className="canvas-wrap">
        <ReactFlow
          nodes={nodes}
          edges={renderedEdges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={(_, node) => {
            setSelectedNodeId(node.id)
            setSelectedEdgeId(null)
          }}
          onEdgeClick={(_, edge) => {
            setSelectedEdgeId(edge.id)
            setSelectedNodeId(null)
          }}
          onPaneClick={() => {
            setSelectedNodeId(null)
            setSelectedEdgeId(null)
          }}
          fitView
        >
          <Background />
          <MiniMap />
          <Controls />
        </ReactFlow>
      </div>
    </div>
  )
}
