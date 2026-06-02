import { randomUUID } from 'node:crypto'
import type { LLM, Message, Tool, ToolCall, LLMResponse, LLMUsageEvent, LLMRequestEvent } from '@noetaris/harness-types'
import type { ObserverAware, Observer, StepContext } from '@noetaris/harness'
import { GoogleGenAI } from '@google/genai'
import type { Content, Part, FunctionDeclaration } from '@google/genai'

/** Options for {@link Gemini}. */
export interface GeminiOptions {
  /** Google AI API key. Defaults to the `GEMINI_API_KEY` environment variable. */
  apiKey?: string
  /**
   * Sampling temperature. Higher values produce more varied output.
   * When absent, the provider default applies.
   */
  temperature?: number
  /**
   * Maximum number of tokens to generate.
   * When absent, the provider default applies.
   */
  maxTokens?: number
  /**
   * Top-p nucleus sampling probability.
   * When absent, the provider default applies.
   */
  topP?: number
  /**
   * Top-k sampling — number of highest-probability tokens considered.
   * When absent, the provider default applies.
   */
  topK?: number
  /**
   * Thinking configuration for models that support extended reasoning.
   * When present, enables the thinking feature with the specified token budget.
   */
  thinkingConfig?: {
    thinkingBudget: number
  }
}

function translateMessages(messages: Message[], signatureCache: Map<string, string>): Content[] {
  const result: Content[] = []
  let lastAssistantToolCalls: ToolCall[] | undefined

  for (const msg of messages) {
    if (msg.role === 'user') {
      result.push({ role: 'user', parts: [{ text: msg.content }] })
    } else if (msg.role === 'assistant') {
      lastAssistantToolCalls = msg.toolCalls
      const parts: Part[] = []
      if (msg.content) {
        parts.push({ text: msg.content })
      }
      if (msg.toolCalls) {
        for (const tc of msg.toolCalls) {
          const sig = signatureCache.get(tc.id)
          parts.push({
            functionCall: {
              name: tc.name,
              // as: ToolCall.input is unknown; Google SDK args expects Record<string,unknown>
              args: tc.input as Record<string, unknown>,
            },
            ...(sig !== undefined ? { thoughtSignature: sig } : {}),
          })
        }
      }
      result.push({ role: 'model', parts })
    } else if (msg.role === 'tool') {
      const matchedTc = lastAssistantToolCalls?.find((tc) => tc.id === msg.toolCallId)
      const name = matchedTc?.name ?? msg.toolCallId
      const responsePart: Part = {
        functionResponse: {
          name,
          response: { result: msg.content },
        },
      }

      const last = result[result.length - 1]
      if (last !== undefined && last.role === 'user') {
        last.parts ??= []
        last.parts.push(responsePart)
      } else {
        result.push({ role: 'user', parts: [responsePart] })
      }
    }
  }

  return result
}

function translateTools(tools: Tool[]): FunctionDeclaration[] {
  return tools.map((t) => {
    const decl: FunctionDeclaration = {
      name: t.name,
      description: t.description,
      parametersJsonSchema: t.inputSchema,
    }
    return decl
  })
}

function normalizeResponse(parts: Part[], finishReason: string | undefined, signatureCache: Map<string, string>): LLMResponse {
  let text = ''
  const toolCalls: ToolCall[] = []

  // Collect any standalone thoughtSignature part (a part with a signature but no functionCall).
  // Gemini may emit the signature as a separate part preceding the functionCall part.
  let standaloneSignature: string | undefined
  for (const part of parts) {
    if (part.thoughtSignature !== undefined && part.functionCall === undefined) {
      standaloneSignature = part.thoughtSignature
    }
  }

  for (const part of parts) {
    if ('text' in part && typeof part.text === 'string' && part.thought !== true) {
      text += part.text
    } else if ('functionCall' in part && part.functionCall !== undefined) {
      const id = part.functionCall.id ?? randomUUID()
      // Signature may sit on the same part as the functionCall, or arrive as the standalone part above.
      const sig = part.thoughtSignature ?? standaloneSignature
      if (sig !== undefined) {
        signatureCache.set(id, sig)
      }
      toolCalls.push({
        id,
        name: part.functionCall.name ?? '',
        input: part.functionCall.args,
      })
    }
  }

  let stopReason: LLMResponse['stopReason']
  if (toolCalls.length > 0) {
    stopReason = 'tool_use'
  } else if (finishReason === 'MAX_TOKENS') {
    stopReason = 'max_tokens'
  } else {
    stopReason = 'end'
  }

  return { text, toolCalls, stopReason }
}

const ZEROED_STEP_CONTEXT: StepContext = { agentId: '', sessionId: '', stepName: '' }

/**
 * {@link LLM} adapter for the Google Gen AI (Gemini) API.
 *
 * Implements {@link ObserverAware} — emits an `'llm.response'` event with an
 * `LLMUsageEvent` payload after each successful invocation.
 *
 * @example
 * ```ts
 * const llm = new Gemini('gemini-2.0-flash')
 * const response = await llm.invoke(messages)
 * ```
 */
export class Gemini implements LLM, ObserverAware {
  private readonly ai: GoogleGenAI
  private readonly modelId: string
  private readonly options?: GeminiOptions
  private observer: Observer = {}
  private stepContext: StepContext = ZEROED_STEP_CONTEXT
  // Keyed by ToolCall.id; survives across turns within the same Gemini instance.
  private readonly thoughtSignatures = new Map<string, string>()

  /**
   * @param model - Gemini model ID, e.g. `'gemini-2.0-flash'`.
   * @param options - Optional configuration including API key and generation params.
   */
  constructor(model: string, options?: GeminiOptions) {
    this.modelId = model
    this.options = options
    this.ai = new GoogleGenAI({ apiKey: options?.apiKey ?? process.env['GEMINI_API_KEY'] ?? '' })
  }

  bindObserver(observer: Observer): void {
    this.observer = observer
  }

  setStepContext(ctx: StepContext): void {
    this.stepContext = ctx
  }

  async invoke(messages: Message[], options?: { tools?: Tool[] }): Promise<LLMResponse> {
    const contents = translateMessages(messages, this.thoughtSignatures)
    const tools = options?.tools

    const generationConfig: Record<string, unknown> = {}
    if (this.options?.temperature !== undefined) generationConfig['temperature'] = this.options.temperature
    if (this.options?.maxTokens !== undefined) generationConfig['maxOutputTokens'] = this.options.maxTokens
    if (this.options?.topP !== undefined) generationConfig['topP'] = this.options.topP
    if (this.options?.topK !== undefined) generationConfig['topK'] = this.options.topK
    const hasGenerationConfig = Object.keys(generationConfig).length > 0
    const hasThinkingConfig = this.options?.thinkingConfig !== undefined
    const hasTools = tools !== undefined

    const config = hasTools || hasGenerationConfig || hasThinkingConfig
      ? {
          ...(hasTools ? { tools: [{ functionDeclarations: translateTools(tools!) }] } : {}),
          ...(hasGenerationConfig ? { generationConfig } : {}),
          ...(hasThinkingConfig ? { thinkingConfig: this.options!.thinkingConfig } : {}),
        }
      : undefined

    const requestEvent: LLMRequestEvent = { modelId: this.modelId, providerName: 'google' }
    this.observer.onEvent?.(this.stepContext, 'llm.request', requestEvent)

    const result = await this.ai.models.generateContent({
      model: this.modelId,
      contents,
      ...(config !== undefined ? { config } : {}),
    })

    const candidates = result.candidates

    if (!candidates || candidates.length === 0) {
      throw new Error('Gemini response contained no candidates')
    }

    // noUncheckedIndexedAccess: guarded by the length check above
    const firstCandidate = candidates[0] as NonNullable<typeof candidates[0]>
    const parts = firstCandidate.content?.parts ?? []
    const finishReason = firstCandidate.finishReason

    const response = normalizeResponse(parts, finishReason, this.thoughtSignatures)

    const usageMetadata = result.usageMetadata
    const event: LLMUsageEvent = {
      tokens: {
        input: usageMetadata?.promptTokenCount ?? 0,
        output: usageMetadata?.candidatesTokenCount ?? 0,
      },
      modelId: this.modelId,
      stopReason: response.stopReason,
      providerName: 'google',
    }
    this.observer.onEvent?.(this.stepContext, 'llm.response', event)

    return response
  }
}
