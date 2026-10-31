---
description: Build the React what-if workbench with the Tastemaker design workflow
argument-hint: [api url]
---

Use the `ml-web-application` skill.

Load Tastemaker first (taste-direction, taste-tokens, taste-components, taste-guard; if it is not installed, give the install commands and state the fallback). Write the 8-line Design Decision, build tokens, scaffold `web/` from `${CLAUDE_PLUGIN_ROOT}/templates/ml-web`, regenerate the typed client from the API's OpenAPI document, then run `npm run typecheck`, `npm test`, `npm run build` and `npx tastemaker scan web/src --project` until clean (at most 3 cycles).
