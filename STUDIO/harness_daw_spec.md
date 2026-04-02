# STUDIO — Multi-Agent DAW Harness Spec

## Phenomenon
Single-agent generation produces incomplete DAW systems (UI without interaction, broken audio graph, missing state wiring).

## Architecture
Three-agent harness adapted to ScoreFriction STUDIO:

### 1. PLANNER
Input: "Build DAW inside STUDIO"
Output:
- Audio Engine (Web Audio API graph)
- Timeline (clips draggable, quantized grid)
- Mixer (gain, pan, sends)
- Instruments (synth, sampler)
- Effects (reverb, delay)
- State model (tracks, clips, automation)
- AI Composer node (ScoreFriction integration)

### 2. BUILDER
Stack:
- Frontend: React + Vite
- Audio: Web Audio API
- State: Zustand or Redux
- Backend: FastAPI (optional for persistence)

Rules:
- Implement per feature block
- Maintain audio graph integrity
- No stubbed UI allowed

### 3. EVALUATOR (QA)
Runs Playwright-style checks:

Criteria:
1. Interaction: All UI elements must change state
2. Audio Validity: Graph produces sound (oscillator → gain → destination)
3. Timeline Integrity: Clips draggable and time-synced
4. State Consistency: No desync between UI and audio engine

Fail if any criterion < threshold.

## Generator–Evaluator Loop
- Builder writes feature
- Evaluator tests real behavior
- Feedback returned as diff-level fixes
- Iterate until pass

## ScoreFriction Integration
- Treat DAW as "STUDIO node"
- Each track = node
- Each effect = transformation edge
- Timeline = temporal vector field

## Codex Execution Prompt

"You are Builder Agent inside a multi-agent harness. You must implement a fully functional browser DAW inside the ScoreFriction STUDIO module.

Constraints:
- No placeholder UI
- Every control must affect audio output
- Use Web Audio API graph explicitly
- Ensure clips are draggable and audible

Evaluator will reject any non-interactive component."

## Outcome
Transforms STUDIO from static interface → executable audio system.

Cost ↑
Reliability ↑
Completeness ↑
