function escapeXml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;')
}

function assertSnapshotShape(snapshot) {
  if (!snapshot || typeof snapshot !== 'object' || Array.isArray(snapshot)) {
    throw new Error('snapshot must be an object')
  }
  if (!Array.isArray(snapshot.nodes)) {
    throw new Error('snapshot.nodes must be an array')
  }
  if (!Number.isInteger(snapshot.node_count) || snapshot.node_count < 0) {
    throw new Error('snapshot.node_count must be a non-negative integer')
  }
  if (!Number.isInteger(snapshot.root_node_id) || snapshot.root_node_id < 0) {
    throw new Error('snapshot.root_node_id must be a non-negative integer')
  }
  if (snapshot.nodes.length !== snapshot.node_count) {
    throw new Error('snapshot.node_count must match snapshot.nodes length')
  }
}

function layoutNodes(snapshot) {
  const sorted = [...snapshot.nodes].sort((a, b) => a.node_id - b.node_id)
  const depthCounts = new Map()
  const positions = new Map()

  for (const node of sorted) {
    const depth = Number(node.depth) || 0
    const row = depthCounts.get(depth) ?? 0
    depthCounts.set(depth, row + 1)
    positions.set(node.node_id, { x: 140 + depth * 300, y: 100 + row * 170 })
  }

  return { sorted, positions }
}

export function renderTreeSnapshotSvg(snapshot) {
  assertSnapshotShape(snapshot)
  const { sorted, positions } = layoutNodes(snapshot)
  const svgParts = []
  const width = Math.max(...[...positions.values()].map((p) => p.x), 300) + 320
  const height = Math.max(...[...positions.values()].map((p) => p.y), 180) + 200

  svgParts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
  )
  svgParts.push('<rect x="0" y="0" width="100%" height="100%" fill="#f8fafc" />')

  for (const node of sorted) {
    const source = positions.get(node.node_id)
    const edges = Array.isArray(node.edges) ? node.edges : []
    edges.forEach((edge, edgeIndex) => {
      const actionX = source.x + 120
      const actionY = source.y - 44 + edgeIndex * 86
      const outcomes = Array.isArray(edge.outcomes) ? edge.outcomes : []

      svgParts.push(
        `<line x1="${source.x + 36}" y1="${source.y}" x2="${actionX - 34}" y2="${actionY}" stroke="#6b7280" stroke-width="1.5" stroke-dasharray="4 3" />`,
      )
      svgParts.push(
        `<rect x="${actionX - 34}" y="${actionY - 16}" width="68" height="32" rx="6" fill="#dbeafe" stroke="#1f2937" stroke-width="1.5" />`,
      )
      svgParts.push(
        `<text x="${actionX}" y="${actionY + 4}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#0f172a">a${escapeXml(edge.action_id)}</text>`,
      )

      outcomes.forEach((outcome) => {
        const target = positions.get(outcome.child_node_id)
        if (!target) {
          return
        }
        svgParts.push(
          `<line x1="${actionX + 34}" y1="${actionY}" x2="${target.x - 38}" y2="${target.y}" stroke="#111827" stroke-width="1.7" />`,
        )
        svgParts.push(
          `<text x="${(actionX + target.x) / 2}" y="${(actionY + target.y) / 2 - 6}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#111827">count=${escapeXml(outcome.count)}</text>`,
        )
      })
    })
  }

  for (const node of sorted) {
    const position = positions.get(node.node_id)
    const fill = node.is_terminal ? '#fef3c7' : '#e5e7eb'
    svgParts.push(
      `<circle cx="${position.x}" cy="${position.y}" r="38" fill="${fill}" stroke="#1f2937" stroke-width="2" />`,
    )
    svgParts.push(
      `<text x="${position.x}" y="${position.y - 6}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#111827">N${escapeXml(node.node_id)}</text>`,
    )
    svgParts.push(
      `<text x="${position.x}" y="${position.y + 11}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#111827">S${escapeXml(node.state_key)}</text>`,
    )
  }

  svgParts.push('</svg>')
  return svgParts.join('')
}
