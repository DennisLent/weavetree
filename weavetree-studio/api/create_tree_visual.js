import { renderTreeSnapshotSvg } from '../src/model/renderTreeSvg.js'

export default function handler(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST')
    res.status(405).json({ error: 'Method not allowed. Use POST.' })
    return
  }

  try {
    const snapshot = typeof req.body === 'string' ? JSON.parse(req.body) : req.body
    const svg = renderTreeSnapshotSvg(snapshot)
    res.setHeader('Content-Type', 'image/svg+xml; charset=utf-8')
    res.status(200).send(svg)
  } catch (error) {
    res.status(400).json({
      error: error instanceof Error ? error.message : 'Invalid snapshot payload',
    })
  }
}
