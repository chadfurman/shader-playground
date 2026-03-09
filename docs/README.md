# Shader Playground

Real-time fractal flame renderer with genetic evolution, taste learning, and audio reactivity.

## Quick Links

- [Reference](reference/README.md) — vocabulary, uniform layout, config fields
- [Rendering](rendering/README.md) — chaos game, tonemapping, feedback, post effects
- [Genetics](genetics/README.md) — genome format, breeding, mutation, persistence
- [Taste Engine](taste-engine/README.md) — palette and transform preference models
- [Audio](audio/README.md) — signal mapping and audio-reactive modulation
- [Config](config/README.md) — weights.json structure and signal weights

## What is this?

Shader Playground is a GPU-accelerated fractal flame renderer built in Rust with wgpu. It combines several systems into a live, evolving visual experience:

- **Chaos game rendering** — iterative function systems rendered via the chaos game algorithm on the GPU, producing fractal flame images in real time.
- **Genetic breeding** — each flame is defined by a genome. Genomes can be bred together, mutated, and selected for, creating an evolutionary process that explores the space of possible flames.
- **Taste model** — a learned preference system that observes which flames you keep and which you skip, building a model of your aesthetic taste across palettes and transforms.
- **Audio reactivity** — audio input drives parameter modulation so the visuals respond to music in real time, with configurable signal weights controlling how different frequency bands map to renderer parameters.

## Getting Started

Run the renderer:

```bash
cargo run
```

Serve the docs locally:

```bash
cd docs
npx docsify-cli serve .
```
