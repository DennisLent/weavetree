# Weavetree Studio

`weavetree-studio` is hosted alongside this book so you can work in two modes:

[Open Weavetree Studio](./studio/)

- `Create MDP`: interactive state/action editor with YAML export.
- `Visualize Tree`: upload `TreeSnapshot` JSON (from `weavetree-core`) and inspect the tree graph.

In `Visualize Tree`, uploaded snapshots can be re-downloaded as:

- original tree JSON snapshot
- rendered graph JSON payload

## Tree Rendering From Terminal (Works with GitHub Pages hosting)

GitHub Pages is static hosting, so runtime `/api/*` routes are not executed there.
For terminal-driven rendering, use the bundled CLI renderer:

```bash
npm --prefix weavetree-studio run render-tree -- \
  --input tree_snapshot.json \
  --output tree.svg
```

This command validates the snapshot and writes an SVG file directly.

## API Rendering (Only for non-Pages deployments with serverless routes)

The studio workspace includes a serverless endpoint template at:

- `weavetree-studio/api/create_tree_visual.js`

When deployed on a platform that supports this route shape (for example Vercel-style `/api/*`),
you can render a snapshot to SVG directly from terminal:

```bash
curl -X POST "https://<your-studio-host>/api/create_tree_visual" \
  -H "content-type: application/json" \
  --data-binary @tree_snapshot.json \
  -o tree.svg
```

If the link above is opened from a copied URL and does not resolve, go to the book root and then open `/studio/`.
