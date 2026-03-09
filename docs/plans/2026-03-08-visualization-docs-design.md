# Visualization Documentation System — Design

## Goal

Create a Docsify-powered documentation site covering the full shader-playground system — rendering pipeline, genetics, taste engine, audio, and config. Living docs updated alongside code changes. Serves future Chad, AI agents, and external contributors.

## Tooling

- **Docsify v4** with search, pagination, copy-code, and syntax highlighting (Rust, WGSL, JSON, Bash)
- `docs/index.html` — single-file Docsify setup, no build step
- `docs/_sidebar.md` — navigation tree
- Add rule to `CLAUDE.md`: "When modifying rendering, genetics, taste, audio, or config systems, update the corresponding doc in `docs/`"

## Structure

```
docs/
├── index.html
├── _sidebar.md
├── README.md                  ← Landing page
├── reference/
│   ├── README.md              ← "When to read" index
│   ├── vocabulary.md          ← Terminology glossary
│   ├── uniform-layout.md     ← Complete uniform buffer map
│   └── weights-config.md     ← Every RuntimeConfig field documented
├── rendering/
│   ├── README.md              ← Pipeline overview + ASCII flow
│   ├── chaos-game.md          ← REFERENCE: compute shader deep dive
│   ├── tonemapping.md         ← REFERENCE: log-density, ACES, histogram EQ
│   ├── feedback-trail.md      ← Trail decay, temporal reprojection
│   ├── post-effects.md        ← Bloom, DoF, velocity blur, edge glow
│   └── luminosity.md          ← Per-point luminosity factors
├── genetics/
│   ├── README.md              ← Overview
│   ├── genome-format.md       ← FlameGenome struct, serialization
│   ├── breeding.md            ← breed() flow, slot allocation
│   ├── mutation.md            ← Mutation operators, clamping
│   └── persistence.md         ← Directory layout, lineage, archiving
├── taste-engine/
│   ├── README.md              ← End-to-end taste learning
│   ├── palette-model.md       ← Palette features, scoring
│   └── transform-model.md     ← Transform features, biased generation
├── audio/
│   ├── README.md              ← Audio signal pipeline
│   └── signal-mapping.md      ← Bands, normalization, weight mapping
└── config/
    ├── README.md              ← weights.json structure, hot-reload
    └── signal-weights.md      ← Audio/time signal modulation
```

## Doc Conventions

- **Vocabulary centralization** — `reference/vocabulary.md` is the authority. Other docs link to it, don't redefine terms.
- **ASCII diagrams** for system flows (box-drawing characters, no images)
- **"Quick Reference" tables** at top of each category README — "Working on X? Read Y"
- **Config callouts** — whenever a doc mentions a tunable parameter, note the `weights.json` field name and default
- **Cross-references** — relative links between docs (`/rendering/chaos-game.md`)
- **Depth scaling** — reference-level (300-500 lines) for chaos-game.md, tonemapping.md, taste engine. Working-knowledge (100-200 lines) for everything else.

## Living Docs Rule

Docs are the source of truth, updated alongside code changes. CLAUDE.md will include:

```
### Documentation
- Docs live in `docs/` (Docsify site)
- When modifying rendering, genetics, taste, audio, or config systems, update the corresponding doc
- `reference/vocabulary.md` is the single source of truth for terminology
- Run `npx docsify-cli serve docs` to preview locally
```

## Audience

- **Future Chad** — "why did we do this?" and "what knobs exist?"
- **AI agents** — precise enough to reason about the pipeline without re-exploring source
- **External contributors** — approachable for someone new to the project

## Depth

- **Reference-level** (300-500 lines): chaos-game.md, tonemapping.md, taste engine docs
- **Working-knowledge** (100-200 lines): everything else
