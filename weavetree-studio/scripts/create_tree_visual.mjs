#!/usr/bin/env node

import fs from 'node:fs'
import path from 'node:path'

import { renderTreeSnapshotSvg } from '../src/model/renderTreeSvg.js'

function usage() {
  const tool = 'npm --prefix weavetree-studio run render-tree --'
  console.error(`Usage: ${tool} --input <tree_snapshot.json> [--output <tree.svg>]`)
}

function parseArgs(argv) {
  let inputPath = ''
  let outputPath = 'tree.svg'

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]
    if (arg === '--input' || arg === '-i') {
      inputPath = argv[i + 1] ?? ''
      i += 1
      continue
    }
    if (arg === '--output' || arg === '-o') {
      outputPath = argv[i + 1] ?? outputPath
      i += 1
      continue
    }
    if (arg === '--help' || arg === '-h') {
      usage()
      process.exit(0)
    }
    console.error(`Unknown argument: ${arg}`)
    usage()
    process.exit(2)
  }

  if (!inputPath) {
    usage()
    process.exit(2)
  }

  return { inputPath, outputPath }
}

function main() {
  const { inputPath, outputPath } = parseArgs(process.argv.slice(2))
  const inputText = fs.readFileSync(inputPath, 'utf8')
  const snapshot = JSON.parse(inputText)
  const svg = renderTreeSnapshotSvg(snapshot)
  fs.writeFileSync(outputPath, svg, 'utf8')
  const absoluteOutput = path.resolve(outputPath)
  console.log(`Wrote ${absoluteOutput}`)
}

try {
  main()
} catch (error) {
  console.error(error instanceof Error ? error.message : 'Failed to render tree SVG')
  process.exit(1)
}
