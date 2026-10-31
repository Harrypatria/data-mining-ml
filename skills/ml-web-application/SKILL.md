---
name: ml-web-application
description: ML Web Application. Build a React front end for a machine learning model with the Tastemaker design workflow - predictor workbench with schema-driven sliders, result and explanation panels, sensitivity chart, scenario comparison, AI explainer chat with streaming, model information drawer, every loading, empty, error, not-ready, rate-limited and LLM-unavailable state, typed API client from OpenAPI, Vite, TypeScript, Tailwind, shadcn/ui, TanStack Query, Recharts, Vitest. Use for a UI, dashboard, web app, front end or what-if tool for a model.
---

# ML Web Application

Scaffold: `templates/ml-web/` (becomes `web/`). Contract: the FastAPI service from model-deployment.

## Design process (mandatory order)

1. Load the Tastemaker skill before writing UI code. If installed, run its workflow: taste-brief, taste-direction, taste-tokens, taste-components, taste-guard. If not installed, tell the user: `/plugin marketplace add Harrypatria/tastemaker`, then `/plugin install tastemaker@tastemaker`; read a local `tastemaker.md` or `skills/tastemaker/SKILL.md` if present; otherwise apply its rules from memory and say the fallback was used.
2. Write the 8-line Design Decision (style, palette, type, layout, motion, charts, slop to avoid, audience risk), each line with a reason drawn from ANALYSIS_MANIFEST.md.
3. Generate `design-tokens.json` and its CSS and Tailwind outputs (`tastemaker tokens design-tokens.json --out src/styles`); all text pairs pass WCAG 2.2 AA in light and dark.
4. Build with Vite, React, TypeScript, Tailwind, shadcn/ui (Radix), Lucide, TanStack Query, Recharts. No emoji icons, no default AI gradients, no placeholder copy.
5. Run `npx tastemaker scan web/src --project` after each step; fix errors at the cause, at most 3 cycles; then `npm run typecheck`, `npm test`, `npm run build`.

## Screens

- **Predictor workbench**: controls generated from `GET /v1/schema`; numeric sliders show value, unit, step and the training range under the track; categories as segmented controls (up to 4) or selects; booleans as switches; tooltip from the feature dictionary; keyboard operation and `aria-valuetext`. Reset to median profile and Save scenario at the top. A switch allows extrapolation.
- **Result panel**: probability or value, label at the stored threshold, baseline comparison, signed delta chip, extrapolation warning with icon and text.
- **Explanation panel**: diverging contribution bars with signed labels (color never carries meaning alone), sensitivity line for the last-moved predictor with the training range shaded, scenario comparison table.
- **AI explainer chat**: Explain this change, free-text follow-ups, suggested questions, auto-explain switch (off by default), streaming answer, note that explanations describe the model, not causes, and a visible notice when the template fallback was used.
- **Model information drawer**: version, training date, metrics with intervals, intended use, limitations.
- **States** for every region: loading skeleton, empty, error, API unavailable, model not ready, rate limited, LLM unavailable, success.

## Interaction rules

- Debounce slider changes (about 250 ms) before `/v1/whatif`; TanStack Query's AbortSignal cancels stale requests; keep the previous result visible with a pending indicator.
- The browser talks only to the FastAPI backend (`VITE_API_URL`); it never holds an LLM key. In production nginx proxies `/api` and adds the API key.
- Typed client from OpenAPI: `python api/scripts/export_openapi.py models api/openapi.json && npm run gen:api`.
- Respect `prefers-reduced-motion`; dark mode through tokens; theme follows the system until the user chooses.
- Tests: Vitest and Testing Library for slider-to-request debouncing, state rendering and the streaming chat; Playwright smoke test when available.

Architecture, file map and test recipes: `references/react-app.md`.
