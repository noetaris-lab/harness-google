# @noetaris/harness-google

Google Gemini adapter for [@noetaris/harness](../core).

> **Status:** not yet released. Implementation tracked in F22.

## Overview

`@noetaris/harness-google` provides a `Gemini` class that implements the `LLM` and `ObserverAware` interfaces from `@noetaris/harness`. It handles translation between the harness message format and the Google Generative AI SDK (`generateContent`) format, and emits telemetry events (token usage, model ID) through an attached `Observer`.

## Installation

```sh
pnpm add @noetaris/harness-google
```

Peer dependencies:

```sh
pnpm add @noetaris/harness @noetaris/harness-types
```

Requires Node.js ≥ 22.

## Usage

```ts
import { Gemini } from '@noetaris/harness-google'

const llm = new Gemini({ apiKey: process.env.GOOGLE_API_KEY })

// Wire into a harness provider slot
h.provide('model', runtime())

const agent = createAgent(h, { prompts: { system: '...' } })
const run = agent.run(initialState, { model: llm })
```

## API

### `Gemini`

Implements `LLM` and `ObserverAware`.

- **`invoke(messages, options?)`** — translates harness `Message[]` and `Tool[]` to Google Generative AI SDK format, calls `generateContent()`, and maps the response back to `LLMResponse` (including function call extraction from candidates).
- **`bindObserver(observer)`** — attaches an `Observer`; after each `invoke`, emits an `"llm.response"` event with `{ tokens: { input, output }, modelId }` from `usageMetadata` in the response.

### `MockGemini`

A deterministic test double for use in tests and demos without a real API key.

## Related Packages

- [`@noetaris/harness`](https://github.com/noetaris-lab/harness) — core execution engine
- [`@noetaris/harness-types`](https://github.com/noetaris-lab/harness-types) — shared LLM type contract
- [`@noetaris/harness-anthropic`](https://github.com/noetaris-lab/harness-anthropic) — Anthropic Claude adapter
- [`@noetaris/harness-openai`](https://github.com/noetaris-lab/harness-openai) — OpenAI adapter
- [`@noetaris/harness-otel`](https://github.com/noetaris-lab/harness-otel) — OpenTelemetry observer bridge

## License

MIT
