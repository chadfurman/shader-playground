# Spanda: The Divine Creative Pulse

## Game Design Specification

**Date:** 2026-03-17
**Status:** Draft — open questions remain
**Working repo:** TBD (fork of shader-playground)

---

## The Soul

You are energy drifting through a world made of light and music.
The world evolves around you — blooming, mutating, becoming.
It learns what you find beautiful and gives you more of it.
Sometimes another light drifts alongside yours.
You don't know who they are. You don't need to.
Your flames touch, and something new is born between you.

There is no score. There is no failure.
There is only the pulse — the spanda —
and what you make of it together.

---

## Design Pillars

### 1. Beauty is the gameplay
The fractal flames ARE the game. Not a backdrop, not a reward — the thing itself. Every system serves beauty: evolution creates it, music shapes it, taste curates it, players share it.

### 2. The music is the physics
Audio drives everything: terrain, evolution pressure, intensity, biome shifts. Bass warps geometry. Beats trigger mutations. Song transitions reshape the world. Bring your own music — your library becomes your universe.

### 3. Flow over friction
No fail state. No enemies. No game over. The zen experience is the default. Challenge and scoring exist as optional layers — never forced, never in the way. Inspired by Flower, Journey, fl0w.

### 4. Presence, not performance
Multiplayer is silent, anonymous, beautiful. Like Journey: no voice, no text, no usernames. Just another light beside you. Proximity is communication. Cross-pollination is collaboration. You meet as energy, not as profiles.

### 5. The world learns you
The taste model isn't a feature — it's the soul of the game. Spanda watches what you linger on, what you chase, what you flee from. Over time, the universe becomes a reflection of your aesthetic unconscious. No two players' Spanda is the same.

### 6. Actions snap to the beat
Rez's secret: quantize everything. Mutations fire on downbeats. Path forks appear on phrase boundaries. Genome transitions land on drops. Even imprecise play feels musical. Every player is a musician and doesn't know it.

---

## Inspirations

| Game | What we take from it |
|------|---------------------|
| **Flower** (thatgamecompany) | Zen/flow pillar. No score, no fail state. Wind carries you. Beauty is the reward. |
| **Dyad** (][) | Synesthesia pillar. Music intensifies with play quality. Color-sound-touch fusion. Tunnel energy. |
| **Rez** (Mizuguchi) | Beat quantization. Every action snaps to rhythm. Imprecise play feels musical. |
| **Journey** (thatgamecompany) | Anonymous multiplayer. No voice, no text. Proximity is communication. Reveal at end. |
| **Audiosurf** (Fitterer) | Supplements music with gameplay. Player's own library. Layered modes (zen + competitive). |
| **Fantasia: Music Evolved** (Harmonix) | Conducting feeling. Player authorship over the music's direction. |

---

## Systems

### Flight & Navigation

The player is a point of light — a flame seed — drifting through fractal space. Third-person camera, close enough to feel immersed, far enough to see the world blooming.

**Three input modes to prototype (build all, keep what feels like magic):**

- **Drift** — Gentle analog steering. Tilt, float, carried by wind. Flower energy. Low intensity, high immersion.
- **Tunnel** — Fast forward motion. Weave and hook. Dyad energy. High intensity, synesthetic rush.
- **Gestural** — Shape and sculpt. Wave, draw, conduct. Fantasia energy. Creative, expressive.

These may become separate game modes, or one may emerge as the core with others as variants.

### Fractal Evolution

**Inherited from shader-playground:**
- 26 variation functions (linear through curl)
- Genome mutation & breeding (FlameGenome)
- Morph system (smooth genome transitions)
- MAP-Elites diversity archive
- Novelty search scoring
- Palette evolution
- 3D rendering pipeline (camera pitch, yaw, focal)
- Attractor estimation

**New for Spanda:**
- Player movement → breeding selection (where you go = what survives)
- Proximity to flames = implicit vote (lingering is a preference signal)
- Beat-quantized mutation triggers
- Biome system (genome families with distinct visual character)
- Cross-pollination (multiplayer genome breeding)
- Evolution timeline (rewind/revisit lineages)
- Genome seeds for sharing between players

**Key insight:** In shader-playground, you watch evolution happen TO you. In Spanda, your movement IS the selection pressure. The world is your garden.

### Audio Engine

**Inherited from shader-playground:**
- FFT analysis (16 bands)
- Beat detection + beat accumulation
- Bass / mids / highs separation
- Energy tracking
- Change signal (EMA divergence detector for song transitions)
- Signal → shader parameter mapping via weights.json

**New for Spanda:**
- Music file loading (not just system audio capture)
- Beat quantization grid (snap actions to musical time)
- Phrase/section detection (for biome transitions, path forks)
- Tempo-adaptive flight speed
- Cross-platform audio input (cpal already cross-platform for mic/file)
- Microphone input option (play to ambient sound)

### Taste Model

**Inherited:** IGMM clustering, 74-dimension feature vectors, favorite profiles, variation frequency analysis, palette feature extraction, vote ledger.

**New for Spanda:** In shader-playground, taste is fed by explicit votes (up/down keys). In Spanda, taste is fed by *behavior*. Lingering near a flame is a vote. Chasing a mutation is a vote. Turning away is a vote. The game reads movement patterns as preference data. Over a session, the universe subtly reshapes itself around the player's aesthetic unconscious.

### Multiplayer: Silent Communion

Journey model — anonymous, beautiful, optional.

- No voice chat, no text, no usernames
- Another player appears as a flame with a different genome / different taste
- **Proximity = cross-pollination** — get close and genomes blend; the world between you becomes a child of both tastes
- **Session end = reveal** — you see their name when you leave
- Completely optional — the single-player experience is complete on its own

### Game Modes

| Mode | Description |
|------|------------|
| **Zen** | Default. No score, no UI. Just you, the music, and the flames. Pure flow state. |
| **Challenge** | Optional scoring: genome diversity, aesthetic novelty, taste alignment. Leaderboards per song. |
| **Garden** | Sandbox. Breed and curate flame collections. Share genome seeds. Build worlds. |

---

## Architecture

### Approach: Fork First, Extract Later

**Phase 1 (weeks 1-3): Rapid prototyping**
- Copy shader-playground into new `spanda/` repo
- Strip HUD/dev tooling and `sck_audio.rs` (macOS ScreenCaptureKit — not needed, Spanda uses file/mic input via cpal)
- Add game systems (flight, input, modes)
- Prototype all three input modes against the fractal renderer
- Find the fun through play, not planning
- **Done when:** One input mode feels good with music-reactive fractal rendering — the "oh HELL yes" moment
- **Note:** Phase 1 uses single-genome rendering (morph between genomes, not multiple simultaneously). Multi-flame rendering (OQ3) is a Phase 2 R&D spike — don't let it block prototyping

**Phase 2 (week 4+): Engine extraction**
- Once the game design solidifies through prototyping, extract shared systems into `flame-engine` library crate
- Cargo workspace: `flame-engine` (lib) + `shader-playground` (bin) + `spanda` (bin)
- Shared WGSL shaders via build.rs
- Platform layer abstracted behind traits in the engine

**Phase 3: Platform targets**
- Desktop first (macOS, Windows, Linux) — already works via wgpu
- Web (WebGPU/wasm) — wgpu compiles to WebGPU
- Mobile (iOS, Android) — winit supports mobile, wgpu targets Metal/Vulkan

### What's Already Cross-Platform
- wgpu → Metal, Vulkan, DX12, WebGPU
- winit → macOS, Windows, Linux, iOS, Android, Web
- cpal → cross-platform audio (mic input, file loading)
- All WGSL shaders → platform-agnostic
- Genome system, taste model, weights system → pure Rust, no platform deps

### What Needs Platform Abstraction
- Audio capture: `sck_audio.rs` (macOS ScreenCaptureKit) is NOT carried forward into Spanda — strip it during fork. Spanda uses file/mic input via cpal (cross-platform)
- File I/O: genome saves, config — trivial to abstract
- Windowing/input: winit handles it, web needs thin adapter

---

## Monetization

- **Base game: $10-15** — single-player, all modes, full experience
- **Multiplayer subscription: $2-3/month** — optional, funds servers
- **No microtransactions, no ads, no loot boxes**
- Aligns with the zen philosophy — no friction, no extraction

---

## Open Questions & Concerns

### OQ1: What is "terrain" in fractal flame space?
Fractal flames are attractor systems, not terrain meshes. How does the player navigate "through" them? Options to explore:
- Flames exist as nodes/regions in 3D space; you fly between them
- The flame IS the space — you're inside the attractor, and different regions have different visual character
- Hybrid: a generated flight path through 3D space with flames rendered at various depths/positions
- **Needs prototyping to discover what feels right**

### OQ2: What do player lights look like?
How is the player visually represented? A simple glowing point? A small flame with its own genome? A particle trail? How does it read against the fractal background without getting lost?

### OQ3: How do multiple flames coexist visually?
In shader-playground, one genome fills the screen. In Spanda, multiple genomes need to be visible simultaneously (the world around you, nearby flames, other players). How?
- Multiple render targets composited?
- Spatial partitioning — different genomes in different regions?
- Blended genome fields with smooth transitions?
- **This is a core rendering architecture question that needs R&D**

### OQ4: What does "close to a flame" mean?
How does the player know they're near a flame for voting/cross-pollination? Visual proximity cues? Haptic feedback? Audio cues? A glow or resonance effect?

### OQ5: Fractal rendering performance vs game responsiveness
shader-playground already has performance concerns (render thread was added to fix macOS drag lag). Adding game logic, physics, multiple flames, multiplayer networking... how do we keep frame times low?
- LOD system for distant flames (fewer iterations, lower resolution)?
- Temporal reprojection to amortize compute cost?
- Adaptive quality based on frame budget?
- **The render thread architecture helps — game logic on main thread, GPU work on render thread**

### OQ6: Taste engine in a game context
The IGMM taste model works for a single user making deliberate votes. In Spanda, taste signals are implicit (movement patterns) and noisy. How do we:
- Filter meaningful preference signals from random movement?
- Prevent the taste model from converging too fast (boring) or too slow (unresponsive)?
- Handle multiplayer taste blending without one player dominating?

### OQ7: Server architecture & security
Multiplayer means servers. Genome sharing means data exchange. Subscriptions mean payment processing.
- Server architecture: peer-to-peer vs dedicated? (Journey used dedicated servers)
- Anti-cheat: does it matter in a zen game? Probably minimal
- Genome validation: malicious genome data could crash clients
- Payment: Stripe or platform-native (App Store, Google Play, Steam)?
- Privacy: anonymous multiplayer means the server knows identities but clients don't
- **This is a later concern — single-player first, multiplayer in phase 2+**

### OQ8: Music licensing & input
"Bring your own music" avoids licensing issues. But:
- File format support: MP3, FLAC, OGG, WAV?
- Streaming service integration (Spotify, Apple Music)? Probably not — licensing nightmare
- Generated/procedural music as a fallback for players without local files?
- Microphone input: ambient sound as music source?

---

## Name & Identity

**Name:** Spanda (स्पन्द)
**Meaning:** "The divine creative pulse" — the subtle vibration of consciousness as it manifests into living form. From 9th century Kashmir Śaivism.
**Tagline:** TBD
**Trademark status:** No conflicts found in games/apps/software space

---

## References

- [Spanda: The Pulse of Consciousness](https://scienceandnonduality.com/article/spanda-the-universal-activity-of-absolute-consciousness/)
- [Rez: Where Design Meets Music](https://www.feedme.design/rez-a-cultural-and-design-revolution-in-gaming/)
- [Reimagining Co-op: Journey's Multiplayer](https://www.gamedeveloper.com/programming/reimagining-co-op-a-case-study-of-journey-s-multiplayer-gameplay)
- [Jenova Chen's Flow in Games (MFA Thesis)](https://www.jenovachen.com/flowingames/Flow_in_games_final.pdf)
- [Cognitive Flow: Psychology of Great Game Design](https://www.gamedeveloper.com/design/cognitive-flow-the-psychology-of-great-game-design)
- [Flower — thatgamecompany](https://thatgamecompany.com/flower/)
- [Dyad — Wikipedia](https://en.wikipedia.org/wiki/Dyad_(video_game))
- [Rasa (aesthetics) — Wikipedia](https://en.wikipedia.org/wiki/Rasa_(aesthetics))
