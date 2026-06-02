import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { Tool } from '@noetaris/harness-types'
import type { StepContext } from '@noetaris/harness'

// mockGenerateContent is declared in outer scope so the mock factory can close over it
let mockGenerateContent = vi.fn()

vi.mock('@google/genai', () => {
  const MockGoogleGenAI = vi.fn(function (this: { models: unknown }, _opts: unknown) {
    this.models = { generateContent: (...args: unknown[]) => mockGenerateContent(...args) }
  })
  return { GoogleGenAI: MockGoogleGenAI }
})

import { Gemini } from './gemini.js'
import { GoogleGenAI } from '@google/genai'

const minimalTextResponse = {
  candidates: [{ content: { parts: [{ text: 'ok' }] }, finishReason: 'STOP' }],
  usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 1 },
}

beforeEach(() => {
  vi.clearAllMocks()
  mockGenerateContent.mockResolvedValue(minimalTextResponse)
})

describe('Gemini', () => {

  describe('construction', () => {

    it('constructs without throwing when given a model string', () => {
      expect(() => new Gemini('gemini-1.5-pro', { apiKey: 'test-key' })).not.toThrow()
    })

    it('initializes GoogleGenAI with GEMINI_API_KEY env var when no apiKey option is provided', () => {
      // arrange
      process.env['GEMINI_API_KEY'] = 'env-api-key'

      // act
      new Gemini('gemini-2.0-flash')

      // assert
      expect(GoogleGenAI).toHaveBeenCalledWith({ apiKey: 'env-api-key' })

      // cleanup
      delete process.env['GEMINI_API_KEY']
    })

  })

  describe('response normalization — text and stopReason', () => {

    it('returns text response and empty toolCalls when response has a single text part', async () => {
      // arrange
      mockGenerateContent.mockResolvedValue({
        candidates: [{ content: { parts: [{ text: 'hello back' }] }, finishReason: 'STOP' }],
        usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5 },
      })
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      gemini.bindObserver({})

      // act
      const result = await gemini.invoke([{ role: 'user', content: 'hello' }])

      // assert
      expect(mockGenerateContent).toHaveBeenCalledWith(expect.objectContaining({
        model: 'gemini-1.5-pro',
        contents: [{ role: 'user', parts: [{ text: 'hello' }] }],
      }))
      expect(result.text).toBe('hello back')
      expect(result.toolCalls).toEqual([])
      expect(result.stopReason).toBe('end')
    })

    it('returns single ToolCall with UUID id and empty text when response has one functionCall part', async () => {
      // arrange
      mockGenerateContent.mockResolvedValue({
        candidates: [{
          content: { parts: [{ functionCall: { name: 'get_weather', args: { city: 'Paris' } } }] },
          finishReason: 'STOP',
        }],
        usageMetadata: { promptTokenCount: 8, candidatesTokenCount: 3 },
      })
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      gemini.bindObserver({})

      // act
      const result = await gemini.invoke([{ role: 'user', content: 'weather?' }])

      // assert
      expect(result.text).toBe('')
      expect(result.toolCalls).toHaveLength(1)
      expect(result.toolCalls[0]?.name).toBe('get_weather')
      expect(result.toolCalls[0]?.input).toEqual({ city: 'Paris' })
      expect(result.toolCalls[0]?.id).toMatch(/^[0-9a-f-]{36}$/)
      expect(result.stopReason).toBe('tool_use')
    })

    it('returns text and one ToolCall when response has both a text part and a functionCall part', async () => {
      // arrange
      mockGenerateContent.mockResolvedValue({
        candidates: [{
          content: { parts: [
            { text: "I'll call a tool" },
            { functionCall: { name: 'lookup', args: { q: 'x' } } },
          ]},
          finishReason: 'STOP',
        }],
        usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 6 },
      })
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      gemini.bindObserver({})

      // act
      const result = await gemini.invoke([{ role: 'user', content: 'go' }])

      // assert
      expect(result.text).toBe("I'll call a tool")
      expect(result.toolCalls).toHaveLength(1)
      expect(result.toolCalls[0]?.name).toBe('lookup')
      expect(result.stopReason).toBe('tool_use')
    })

    it('concatenates multiple text parts with no separator', async () => {
      // arrange
      mockGenerateContent.mockResolvedValue({
        candidates: [{
          content: { parts: [{ text: 'Hello ' }, { text: 'world' }] },
          finishReason: 'STOP',
        }],
        usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 4 },
      })
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      gemini.bindObserver({})

      // act
      const result = await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(result.text).toBe('Hello world')
      expect(result.toolCalls).toEqual([])
    })

    it('returns one ToolCall per functionCall part with distinct UUIDs when response has multiple functionCall parts', async () => {
      // arrange
      mockGenerateContent.mockResolvedValue({
        candidates: [{
          content: { parts: [
            { functionCall: { name: 'tool_a', args: { x: 1 } } },
            { functionCall: { name: 'tool_b', args: { y: 2 } } },
          ]},
          finishReason: 'STOP',
        }],
        usageMetadata: { promptTokenCount: 12, candidatesTokenCount: 8 },
      })
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      gemini.bindObserver({})

      // act
      const result = await gemini.invoke([{ role: 'user', content: 'go' }])

      // assert
      expect(result.toolCalls).toHaveLength(2)
      expect(result.toolCalls[0]?.name).toBe('tool_a')
      expect(result.toolCalls[0]?.input).toEqual({ x: 1 })
      expect(result.toolCalls[1]?.name).toBe('tool_b')
      expect(result.toolCalls[1]?.input).toEqual({ y: 2 })
      expect(result.toolCalls[0]?.id).toMatch(/^[0-9a-f-]{36}$/)
      expect(result.toolCalls[1]?.id).toMatch(/^[0-9a-f-]{36}$/)
      expect(result.toolCalls[0]?.id).not.toBe(result.toolCalls[1]?.id)
      expect(result.stopReason).toBe('tool_use')
    })

    it('returns stopReason "max_tokens" when finishReason is "MAX_TOKENS" and no functionCall parts', async () => {
      // arrange
      mockGenerateContent.mockResolvedValue({
        candidates: [{
          content: { parts: [{ text: 'partial response' }] },
          finishReason: 'MAX_TOKENS',
        }],
        usageMetadata: { promptTokenCount: 20, candidatesTokenCount: 100 },
      })
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      gemini.bindObserver({})

      // act
      const result = await gemini.invoke([{ role: 'user', content: 'generate' }])

      // assert
      expect(result.stopReason).toBe('max_tokens')
      expect(result.text).toBe('partial response')
    })

    it('returns stopReason "end" when finishReason is an unrecognized value and no functionCall parts', async () => {
      // arrange
      mockGenerateContent.mockResolvedValue({
        candidates: [{
          content: { parts: [{ text: 'blocked' }] },
          finishReason: 'SAFETY',
        }],
        usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 1 },
      })
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      gemini.bindObserver({})

      // act
      const result = await gemini.invoke([{ role: 'user', content: 'risky?' }])

      // assert
      expect(result.stopReason).toBe('end')
    })

  })

  describe('message and tool translation', () => {

    it('translates tools array into config.tools with functionDeclarations using parametersJsonSchema', async () => {
      // arrange
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      gemini.bindObserver({})
      const tools: Tool[] = [{
        name: 'search',
        description: 'search the web',
        inputSchema: { type: 'object', properties: { q: { type: 'string' } } },
      }]

      // act
      await gemini.invoke([{ role: 'user', content: 'go' }], { tools })

      // assert
      expect(mockGenerateContent).toHaveBeenCalledWith(expect.objectContaining({
        config: { tools: [{ functionDeclarations: [{ name: 'search', description: 'search the web', parametersJsonSchema: { type: 'object', properties: { q: { type: 'string' } } } }] }] },
      }))
    })

    it('translates assistant message with toolCalls-only into model content with only functionCall parts', async () => {
      // arrange
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      gemini.bindObserver({})
      const messages = [
        { role: 'user' as const, content: 'do it' },
        { role: 'assistant' as const, toolCalls: [{ id: 'tc1', name: 'run', input: { x: 1 } }] },
      ]

      // act
      await gemini.invoke(messages)

      // assert
      const capturedContents = // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
      mockGenerateContent.mock.calls[0]![0].contents
      // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
      expect(capturedContents[1]).toEqual({ role: 'model', parts: [{ functionCall: { name: 'run', args: { x: 1 } } }] })
      // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
      expect(capturedContents[1].parts).not.toContainEqual(expect.objectContaining({ text: expect.any(String) }))
    })

    it('translates assistant message with both content and toolCalls into model content with text part first then functionCall parts', async () => {
      // arrange
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      gemini.bindObserver({})
      const messages = [
        { role: 'user' as const, content: 'go' },
        { role: 'assistant' as const, content: 'thinking...', toolCalls: [{ id: 'tc1', name: 'calc', input: { n: 5 } }] },
      ]

      // act
      await gemini.invoke(messages)

      // assert
      const capturedContents = // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
      mockGenerateContent.mock.calls[0]![0].contents
      // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
      expect(capturedContents[1]).toEqual({
        role: 'model',
        parts: [{ text: 'thinking...' }, { functionCall: { name: 'calc', args: { n: 5 } } }],
      })
    })

    it('groups tool messages into a single user content with functionResponse parts and resolves name from preceding assistant toolCalls', async () => {
      // arrange
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      gemini.bindObserver({})
      const messages = [
        { role: 'user' as const, content: 'call the tool' },
        { role: 'assistant' as const, toolCalls: [{ id: 'tc1', name: 'weather_lookup', input: {} }] },
        { role: 'tool' as const, toolCallId: 'tc1', content: 'sunny' },
      ]

      // act
      await gemini.invoke(messages)

      // assert
      const capturedContents = // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
      mockGenerateContent.mock.calls[0]![0].contents
      expect(capturedContents).toHaveLength(3)
      expect(capturedContents[2]).toEqual({
        role: 'user',
        parts: [{ functionResponse: { name: 'weather_lookup', response: { result: 'sunny' } } }],
      })
    })

    it('falls back to toolCallId as functionResponse name when toolCallId does not match any preceding ToolCall id', async () => {
      // arrange
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      gemini.bindObserver({})
      const messages = [
        { role: 'user' as const, content: 'go' },
        { role: 'assistant' as const, toolCalls: [{ id: 'tc1', name: 'tool_a', input: {} }] },
        { role: 'tool' as const, toolCallId: 'tc-unknown', content: 'result' },
      ]

      // act
      await gemini.invoke(messages)

      // assert
      const capturedContents = // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
      mockGenerateContent.mock.calls[0]![0].contents
      expect(capturedContents[2]).toEqual({
        role: 'user',
        parts: [{ functionResponse: { name: 'tc-unknown', response: { result: 'result' } } }],
      })
    })

    it('omits config field from SDK call when invoke is called without options', async () => {
      // arrange
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      gemini.bindObserver({})

      // act
      await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
      const callArg = mockGenerateContent.mock.calls[0]![0]
      expect('config' in callArg).toBe(false)
    })

    it('groups two consecutive tool messages into a single user content with two functionResponse parts', async () => {
      // arrange
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      gemini.bindObserver({})
      const messages = [
        { role: 'user' as const, content: 'use both tools' },
        {
          role: 'assistant' as const,
          toolCalls: [
            { id: 'tc1', name: 'tool_a', input: {} },
            { id: 'tc2', name: 'tool_b', input: {} },
          ],
        },
        { role: 'tool' as const, toolCallId: 'tc1', content: 'result_a' },
        { role: 'tool' as const, toolCallId: 'tc2', content: 'result_b' },
      ]

      // act
      await gemini.invoke(messages)

      // assert
      const capturedContents = // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
      mockGenerateContent.mock.calls[0]![0].contents
      expect(capturedContents).toHaveLength(3)
      expect(capturedContents[2]).toEqual({
        role: 'user',
        parts: [
          { functionResponse: { name: 'tool_a', response: { result: 'result_a' } } },
          { functionResponse: { name: 'tool_b', response: { result: 'result_b' } } },
        ],
      })
    })

    it('merges user message text and following tool messages into a single user content entry', async () => {
      // arrange
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      gemini.bindObserver({})
      const messages = [
        { role: 'user' as const, content: 'here is the result' },
        { role: 'tool' as const, toolCallId: 'tc1', content: 'tool_output' },
      ]

      // act
      await gemini.invoke(messages)

      // assert
      const capturedContents = // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
      mockGenerateContent.mock.calls[0]![0].contents
      expect(capturedContents).toHaveLength(1)
      expect(capturedContents[0]).toEqual({
        role: 'user',
        parts: [
          { text: 'here is the result' },
          { functionResponse: { name: 'tc1', response: { result: 'tool_output' } } },
        ],
      })
    })

  })

  describe('observer and StepContext', () => {

    it('calls observer.onEvent with correct StepContext, event type, and payload after successful invoke', async () => {
      // arrange
      mockGenerateContent.mockResolvedValue({
        candidates: [{ content: { parts: [{ text: 'ok' }] }, finishReason: 'STOP' }],
        usageMetadata: { promptTokenCount: 15, candidatesTokenCount: 7 },
      })
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      const observer = { onEvent: vi.fn() }
      gemini.bindObserver(observer)
      const ctx: StepContext = { agentId: 'ag1', sessionId: 's1', stepName: 'step1' }
      gemini.setStepContext(ctx)

      // act
      await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(observer.onEvent).toHaveBeenCalledWith(
        { agentId: 'ag1', sessionId: 's1', stepName: 'step1' },
        'llm.response',
        { tokens: { input: 15, output: 7 }, modelId: 'gemini-1.5-pro', stopReason: 'end', providerName: 'google' },
      )
    })

    it('does not throw when observer has no onEvent method (NOOP_OBSERVER)', async () => {
      // arrange
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      gemini.bindObserver({})

      // act
      const act = () => gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      await expect(act()).resolves.not.toThrow()
    })

    it('uses zeroed StepContext in observer event when setStepContext was never called', async () => {
      // arrange
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      const observer = { onEvent: vi.fn() }
      gemini.bindObserver(observer)

      // act
      await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(observer.onEvent).toHaveBeenCalledWith(
        { agentId: '', sessionId: '', stepName: '' },
        'llm.response',
        expect.any(Object),
      )
    })

    it('uses the provided StepContext in observer event when setStepContext was called', async () => {
      // arrange
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      const observer = { onEvent: vi.fn() }
      gemini.bindObserver(observer)
      gemini.setStepContext({ agentId: 'a1', sessionId: 's1', stepName: 'step_x' })

      // act
      await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(observer.onEvent).toHaveBeenCalledWith(
        { agentId: 'a1', sessionId: 's1', stepName: 'step_x' },
        'llm.response',
        expect.any(Object),
      )
    })

    it('emits tokens.input 0 and tokens.output 0 when usageMetadata is absent', async () => {
      // arrange
      mockGenerateContent.mockResolvedValue({
        candidates: [{ content: { parts: [{ text: 'ok' }] }, finishReason: 'STOP' }],
      })
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      const observer = { onEvent: vi.fn() }
      gemini.bindObserver(observer)

      // act
      await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(observer.onEvent).toHaveBeenCalledWith(
        expect.any(Object),
        'llm.response',
        expect.objectContaining({ tokens: { input: 0, output: 0 } }),
      )
    })

  })

  describe('error propagation', () => {

    it('propagates API error unchanged and does not emit llm.response', async () => {
      // arrange
      const apiError = new Error('API key not valid')
      mockGenerateContent.mockRejectedValue(apiError)
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      const observer = { onEvent: vi.fn() }
      gemini.bindObserver(observer)

      // act
      const rejected = gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      await expect(rejected).rejects.toThrow('API key not valid')
      await expect(rejected).rejects.toBe(apiError)
      const eventTypes = observer.onEvent.mock.calls.map((c: unknown[]) => c[1])
      expect(eventTypes).not.toContain('llm.response')
    })

    it('propagates network error unchanged when generateContent rejects', async () => {
      // arrange
      const netError = new Error('fetch failed')
      mockGenerateContent.mockRejectedValue(netError)
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      gemini.bindObserver({})

      // act / assert
      const act = () => gemini.invoke([{ role: 'user', content: 'hi' }])
      await expect(act()).rejects.toThrow('fetch failed')
      await expect(act()).rejects.toBe(netError)
    })

    it('throws with message "Gemini response contained no candidates" and does not emit llm.response when candidates is empty', async () => {
      // arrange
      mockGenerateContent.mockResolvedValue({
        candidates: [],
        usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 0 },
      })
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      const observer = { onEvent: vi.fn() }
      gemini.bindObserver(observer)

      // act
      const act = () => gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      await expect(act()).rejects.toThrow('Gemini response contained no candidates')
      const eventTypes = observer.onEvent.mock.calls.map((c: unknown[]) => c[1])
      expect(eventTypes).not.toContain('llm.response')
    })

    it('throws with message "Gemini response contained no candidates" and does not emit llm.response when candidates is absent', async () => {
      // arrange
      mockGenerateContent.mockResolvedValue({ usageMetadata: {} })
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      const observer = { onEvent: vi.fn() }
      gemini.bindObserver(observer)

      // act
      const act = () => gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      await expect(act()).rejects.toThrow('Gemini response contained no candidates')
      const eventTypes = observer.onEvent.mock.calls.map((c: unknown[]) => c[1])
      expect(eventTypes).not.toContain('llm.response')
    })

  })

  describe('thought signature handling', () => {

    it('caches thoughtSignature from a functionCall Part and re-attaches it on the next invoke', async () => {
      // arrange — turn 1 response: functionCall part with thoughtSignature at Part level
      const toolCallId = 'call_abc'
      mockGenerateContent.mockResolvedValueOnce({
        candidates: [{
          content: {
            parts: [{
              functionCall: { id: toolCallId, name: 'get_date', args: {} },
              thoughtSignature: 'SIG_BLOB',
            }],
          },
          finishReason: 'STOP',
        }],
        usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 3 },
      })
      mockGenerateContent.mockResolvedValueOnce(minimalTextResponse)

      const gemini = new Gemini('gemini-2.5-flash', { apiKey: 'key' })
      gemini.bindObserver({})

      // act — turn 1
      const r1 = await gemini.invoke([{ role: 'user', content: 'what day is it?' }], {
        tools: [{ name: 'get_date', description: 'get date', inputSchema: {} }],
      })

      // assert turn 1 result
      expect(r1.stopReason).toBe('tool_use')
      expect(r1.toolCalls[0]?.id).toBe(toolCallId)

      // act — turn 2: send tool result back including the assistant message
      await gemini.invoke([
        { role: 'user', content: 'what day is it?' },
        { role: 'assistant', toolCalls: r1.toolCalls },
        { role: 'tool', toolCallId: r1.toolCalls[0]!.id, content: '2026-06-01' },
      ])

      // assert — the functionCall Part sent in turn 2 must carry thoughtSignature
      const turn2Contents = mockGenerateContent.mock.calls[1]![0].contents as { role: string; parts: { functionCall?: unknown; thoughtSignature?: string }[] }[]
      const modelContent = turn2Contents.find(c => c.role === 'model')
      expect(modelContent).toBeDefined()
      const fcPart = modelContent!.parts[0]!
      expect(fcPart.thoughtSignature).toBe('SIG_BLOB')
    })

    it('falls back to standalone thoughtSignature part when the functionCall Part itself has no signature', async () => {
      // arrange — turn 1: separate standalone signature part, then functionCall part
      const toolCallId = 'call_xyz'
      mockGenerateContent.mockResolvedValueOnce({
        candidates: [{
          content: {
            parts: [
              { thoughtSignature: 'STANDALONE_SIG' },                              // standalone
              { functionCall: { id: toolCallId, name: 'search', args: { q: 'x' } } }, // no direct sig
            ],
          },
          finishReason: 'STOP',
        }],
        usageMetadata: { promptTokenCount: 4, candidatesTokenCount: 2 },
      })
      mockGenerateContent.mockResolvedValueOnce(minimalTextResponse)

      const gemini = new Gemini('gemini-2.5-flash', { apiKey: 'key' })
      gemini.bindObserver({})

      // act — turn 1
      const r1 = await gemini.invoke([{ role: 'user', content: 'search for x' }], {
        tools: [{ name: 'search', description: 'search', inputSchema: {} }],
      })
      expect(r1.toolCalls[0]?.id).toBe(toolCallId)

      // act — turn 2
      await gemini.invoke([
        { role: 'user', content: 'search for x' },
        { role: 'assistant', toolCalls: r1.toolCalls },
        { role: 'tool', toolCallId: r1.toolCalls[0]!.id, content: 'results' },
      ])

      // assert — standalone signature propagated to the functionCall Part in turn 2
      const turn2Contents = mockGenerateContent.mock.calls[1]![0].contents as { role: string; parts: { thoughtSignature?: string }[] }[]
      const modelContent = turn2Contents.find(c => c.role === 'model')!
      expect(modelContent.parts[0]!.thoughtSignature).toBe('STANDALONE_SIG')
    })

    it('does not attach thoughtSignature when none was present in the response', async () => {
      // arrange — turn 1: functionCall with no thoughtSignature
      mockGenerateContent.mockResolvedValueOnce({
        candidates: [{
          content: { parts: [{ functionCall: { id: 'call_1', name: 'tool', args: {} } }] },
          finishReason: 'STOP',
        }],
        usageMetadata: { promptTokenCount: 3, candidatesTokenCount: 2 },
      })
      mockGenerateContent.mockResolvedValueOnce(minimalTextResponse)

      const gemini = new Gemini('gemini-2.0-flash', { apiKey: 'key' })
      gemini.bindObserver({})

      const r1 = await gemini.invoke([{ role: 'user', content: 'go' }], {
        tools: [{ name: 'tool', description: 'tool', inputSchema: {} }],
      })

      // act — turn 2
      await gemini.invoke([
        { role: 'user', content: 'go' },
        { role: 'assistant', toolCalls: r1.toolCalls },
        { role: 'tool', toolCallId: r1.toolCalls[0]!.id, content: 'done' },
      ])

      // assert — no thoughtSignature field on the Part
      const turn2Contents = mockGenerateContent.mock.calls[1]![0].contents as { role: string; parts: { functionCall?: unknown; thoughtSignature?: string }[] }[]
      const modelContent = turn2Contents.find(c => c.role === 'model')!
      expect('thoughtSignature' in modelContent.parts[0]!).toBe(false)
    })

    it('caches signatures for each tool call independently when response has multiple functionCall parts', async () => {
      // arrange — response with two functionCall parts each with their own signature
      mockGenerateContent.mockResolvedValueOnce({
        candidates: [{
          content: {
            parts: [
              { functionCall: { id: 'id_a', name: 'tool_a', args: {} }, thoughtSignature: 'SIG_A' },
              { functionCall: { id: 'id_b', name: 'tool_b', args: {} }, thoughtSignature: 'SIG_B' },
            ],
          },
          finishReason: 'STOP',
        }],
        usageMetadata: { promptTokenCount: 6, candidatesTokenCount: 4 },
      })
      mockGenerateContent.mockResolvedValueOnce(minimalTextResponse)

      const gemini = new Gemini('gemini-2.5-flash', { apiKey: 'key' })
      gemini.bindObserver({})

      const r1 = await gemini.invoke([{ role: 'user', content: 'both' }], {
        tools: [
          { name: 'tool_a', description: 'a', inputSchema: {} },
          { name: 'tool_b', description: 'b', inputSchema: {} },
        ],
      })
      expect(r1.toolCalls).toHaveLength(2)

      // act — turn 2
      await gemini.invoke([
        { role: 'user', content: 'both' },
        { role: 'assistant', toolCalls: r1.toolCalls },
        { role: 'tool', toolCallId: 'id_a', content: 'result_a' },
        { role: 'tool', toolCallId: 'id_b', content: 'result_b' },
      ])

      // assert — each Part carries its own signature
      const turn2Contents = mockGenerateContent.mock.calls[1]![0].contents as { role: string; parts: { thoughtSignature?: string }[] }[]
      const modelContent = turn2Contents.find(c => c.role === 'model')!
      expect(modelContent.parts[0]!.thoughtSignature).toBe('SIG_A')
      expect(modelContent.parts[1]!.thoughtSignature).toBe('SIG_B')
    })

    it('excludes thought parts (thought: true) from the returned text', async () => {
      // arrange — response with a thought part followed by a regular text part
      mockGenerateContent.mockResolvedValue({
        candidates: [{
          content: {
            parts: [
              { text: 'internal reasoning', thought: true },
              { text: 'final answer' },
            ],
          },
          finishReason: 'STOP',
        }],
        usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 3 },
      })

      const gemini = new Gemini('gemini-2.5-flash', { apiKey: 'key' })
      gemini.bindObserver({})

      // act
      const result = await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert — only non-thought text included
      expect(result.text).toBe('final answer')
    })

    it('uses the API-provided functionCall id when present instead of generating a UUID', async () => {
      // arrange
      mockGenerateContent.mockResolvedValue({
        candidates: [{
          content: { parts: [{ functionCall: { id: 'api_provided_id', name: 'do_thing', args: { n: 1 } } }] },
          finishReason: 'STOP',
        }],
        usageMetadata: { promptTokenCount: 3, candidatesTokenCount: 2 },
      })

      const gemini = new Gemini('gemini-2.5-flash', { apiKey: 'key' })
      gemini.bindObserver({})

      // act
      const result = await gemini.invoke([{ role: 'user', content: 'go' }])

      // assert — ToolCall.id equals the API-provided id, not a random UUID
      expect(result.toolCalls[0]?.id).toBe('api_provided_id')
    })

  })

  describe('generation params — config omission', () => {

    it('omits config from generateContent when no params and no tools are provided', async () => {
      // arrange
      const gemini = new Gemini('gemini-2.0-flash', { apiKey: 'test-key' })

      // act
      await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(mockGenerateContent).toHaveBeenCalledOnce()
      expect(mockGenerateContent).toHaveBeenCalledWith(expect.not.objectContaining({ config: expect.anything() }))
    })

  })

  describe('generation params — individual param forwarding', () => {

    it('maps maxTokens to maxOutputTokens in generationConfig', async () => {
      // arrange
      const gemini = new Gemini('gemini-2.0-flash', { apiKey: 'test-key', maxTokens: 2048 })

      // act
      await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(mockGenerateContent).toHaveBeenCalledWith(expect.objectContaining({ config: { generationConfig: { maxOutputTokens: 2048 } } }))
    })

    it('forwards temperature in generationConfig', async () => {
      // arrange
      const gemini = new Gemini('gemini-2.0-flash', { apiKey: 'test-key', temperature: 0.5 })

      // act
      await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(mockGenerateContent).toHaveBeenCalledWith(expect.objectContaining({ config: { generationConfig: { temperature: 0.5 } } }))
    })

    it('forwards topP in generationConfig', async () => {
      // arrange
      const gemini = new Gemini('gemini-2.0-flash', { apiKey: 'test-key', topP: 0.95 })

      // act
      await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(mockGenerateContent).toHaveBeenCalledWith(expect.objectContaining({ config: { generationConfig: { topP: 0.95 } } }))
    })

    it('forwards topK in generationConfig', async () => {
      // arrange
      const gemini = new Gemini('gemini-2.0-flash', { apiKey: 'test-key', topK: 40 })

      // act
      await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(mockGenerateContent).toHaveBeenCalledWith(expect.objectContaining({ config: { generationConfig: { topK: 40 } } }))
    })

    it('places thinkingConfig at the top level of config (not inside generationConfig)', async () => {
      // arrange
      const gemini = new Gemini('gemini-2.0-flash', { apiKey: 'test-key', thinkingConfig: { thinkingBudget: 1024 } })

      // act
      await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(mockGenerateContent).toHaveBeenCalledWith(expect.objectContaining({ config: { thinkingConfig: { thinkingBudget: 1024 } } }))
      expect(mockGenerateContent.mock.calls[0]![0]).not.toHaveProperty('config.generationConfig')
    })

  })

  describe('generation params — composite config', () => {

    it('includes all scalar params and thinkingConfig in one config when all params are set', async () => {
      // arrange
      const gemini = new Gemini('gemini-2.0-flash', { apiKey: 'test-key', temperature: 0.7, maxTokens: 512, topP: 0.9, topK: 20, thinkingConfig: { thinkingBudget: 500 } })

      // act
      await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(mockGenerateContent).toHaveBeenCalledWith(expect.objectContaining({
        config: {
          generationConfig: { temperature: 0.7, maxOutputTokens: 512, topP: 0.9, topK: 20 },
          thinkingConfig: { thinkingBudget: 500 },
        },
      }))
    })

    it('merges tools, generationConfig, and thinkingConfig into a single config object', async () => {
      // arrange
      const gemini = new Gemini('gemini-2.0-flash', { apiKey: 'test-key', temperature: 0.5, thinkingConfig: { thinkingBudget: 256 } })
      const tool: Tool = { name: 'get_weather', description: 'Gets weather', inputSchema: { type: 'object', properties: { location: { type: 'string' } }, required: ['location'] } }

      // act
      await gemini.invoke([{ role: 'user', content: 'hi' }], { tools: [tool] })

      // assert
      expect(mockGenerateContent).toHaveBeenCalledOnce()
      const callArg = mockGenerateContent.mock.calls[0]![0] as { config: Record<string, unknown> }
      expect(callArg.config).toHaveProperty('tools')
      expect(callArg.config).toHaveProperty('generationConfig.temperature', 0.5)
      expect(callArg.config).toHaveProperty('thinkingConfig.thinkingBudget', 256)
    })

  })

  describe('generation params — explicit undefined suppression', () => {

    it('excludes temperature from generationConfig when explicitly set to undefined', async () => {
      // arrange
      const gemini = new Gemini('gemini-2.0-flash', { apiKey: 'test-key', temperature: undefined })

      // act
      await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(mockGenerateContent).toHaveBeenCalledWith(expect.not.objectContaining({ config: expect.anything() }))
    })

    it('excludes thinkingConfig from config when explicitly set to undefined', async () => {
      // arrange
      const gemini = new Gemini('gemini-2.0-flash', { apiKey: 'test-key', thinkingConfig: undefined })

      // act
      await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(mockGenerateContent).toHaveBeenCalledWith(expect.not.objectContaining({ config: expect.anything() }))
    })

    it('excludes maxOutputTokens from generationConfig when maxTokens is explicitly undefined', async () => {
      // arrange
      const gemini = new Gemini('gemini-2.0-flash', { apiKey: 'test-key', maxTokens: undefined })

      // act
      await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(mockGenerateContent).toHaveBeenCalledWith(expect.not.objectContaining({ config: expect.anything() }))
    })

    it('excludes topP from generationConfig when explicitly set to undefined', async () => {
      // arrange
      const gemini = new Gemini('gemini-2.0-flash', { apiKey: 'test-key', topP: undefined })

      // act
      await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(mockGenerateContent).toHaveBeenCalledWith(expect.not.objectContaining({ config: expect.anything() }))
    })

    it('excludes topK from generationConfig when explicitly set to undefined', async () => {
      // arrange
      const gemini = new Gemini('gemini-2.0-flash', { apiKey: 'test-key', topK: undefined })

      // act
      await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(mockGenerateContent).toHaveBeenCalledWith(expect.not.objectContaining({ config: expect.anything() }))
    })

    it('includes only tools in config when no generation params are set but tools are provided', async () => {
      // arrange
      const gemini = new Gemini('gemini-2.0-flash', { apiKey: 'test-key' })
      const tool: Tool = { name: 'ping', description: 'Pings a host', inputSchema: { type: 'object', properties: {}, required: [] } }

      // act
      await gemini.invoke([{ role: 'user', content: 'hi' }], { tools: [tool] })

      // assert
      expect(mockGenerateContent).toHaveBeenCalledOnce()
      const callArg = mockGenerateContent.mock.calls[0]![0] as { config: Record<string, unknown> }
      expect(callArg).toHaveProperty('config')
      expect(callArg.config).toHaveProperty('tools')
      expect(callArg.config).not.toHaveProperty('generationConfig')
    })

  })

  describe('generation params — observer event', () => {

    it('emits llm.response observer event with correct shape when generation params are set', async () => {
      // arrange
      mockGenerateContent.mockResolvedValue({
        candidates: [{ content: { parts: [{ text: 'result text' }] }, finishReason: 'STOP', usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 20 } }],
        usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 20 },
      })
      const observer = { onEvent: vi.fn() }
      const gemini = new Gemini('gemini-2.0-flash', { apiKey: 'test-key', temperature: 0.8, maxTokens: 256 })
      gemini.bindObserver(observer)

      // act
      const result = await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(observer.onEvent).toHaveBeenCalledWith(
        expect.any(Object),
        'llm.response',
        expect.objectContaining({ modelId: 'gemini-2.0-flash', providerName: 'google', stopReason: 'end', tokens: { input: 10, output: 20 } }),
      )
      expect(result.text).toBe('result text')
    })

  })

  describe('"llm.request" emission', () => {

    it('emits "llm.request" with modelId and providerName: "google" before ai.models.generateContent', async () => {
      // arrange
      const mockObserver = { onEvent: vi.fn() }
      const adapter = new Gemini('gemini-2.0-flash')
      adapter.bindObserver(mockObserver)

      // act
      await adapter.invoke([{ role: 'user', content: 'hello' }])

      // assert
      expect(mockObserver.onEvent.mock.calls[0]?.[1]).toBe('llm.request')
      expect(mockObserver.onEvent.mock.calls[0]?.[2]).toEqual({ modelId: 'gemini-2.0-flash', providerName: 'google' })
      expect(mockGenerateContent).toHaveBeenCalledOnce()
      expect(mockObserver.onEvent.mock.invocationCallOrder[0] ?? 0).toBeLessThan(mockGenerateContent.mock.invocationCallOrder[0] ?? 0)
    })

    it('emits "llm.request" before "llm.response" on success; no optional content fields', async () => {
      // arrange
      const mockObserver = { onEvent: vi.fn() }
      const adapter = new Gemini('gemini-2.0-flash')
      adapter.bindObserver(mockObserver)

      // act
      await adapter.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(mockObserver.onEvent).toHaveBeenCalledTimes(2)
      expect(mockObserver.onEvent.mock.calls[0]?.[1]).toBe('llm.request')
      expect(mockObserver.onEvent.mock.calls[1]?.[1]).toBe('llm.response')
      expect(mockObserver.onEvent.mock.calls[0]?.[2]).not.toHaveProperty('messages')
      expect(mockObserver.onEvent.mock.calls[0]?.[2]).not.toHaveProperty('tools')
      expect(mockObserver.onEvent.mock.calls[1]?.[2]).not.toHaveProperty('output')
    })

    it('emits "llm.request" before SDK throw and does not emit "llm.response" on error', async () => {
      // arrange
      mockGenerateContent.mockRejectedValue(new Error('QuotaExceeded'))
      const mockObserver = { onEvent: vi.fn() }
      const adapter = new Gemini('gemini-2.0-flash')
      adapter.bindObserver(mockObserver)

      // act
      await expect(adapter.invoke([{ role: 'user', content: 'hi' }])).rejects.toThrow('QuotaExceeded')

      // assert
      expect(mockObserver.onEvent).toHaveBeenCalledTimes(1)
      expect(mockObserver.onEvent.mock.calls[0]?.[1]).toBe('llm.request')
    })

  })

  describe('edge cases', () => {

    it('sends contents as empty array when invoke is called with an empty messages array', async () => {
      // arrange
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      gemini.bindObserver({})

      // act
      await gemini.invoke([])

      // assert
      expect(mockGenerateContent).toHaveBeenCalledWith(expect.objectContaining({ contents: [] }))
    })

    it('uses the second observer when bindObserver is called twice', async () => {
      // arrange
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      const observer1 = { onEvent: vi.fn() }
      const observer2 = { onEvent: vi.fn() }
      gemini.bindObserver(observer1)
      gemini.bindObserver(observer2)

      // act
      await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(observer2.onEvent).toHaveBeenCalled()
      expect(observer1.onEvent).not.toHaveBeenCalled()
    })

    it('uses the most recently set StepContext when setStepContext is called multiple times before invoke', async () => {
      // arrange
      const gemini = new Gemini('gemini-1.5-pro', { apiKey: 'key' })
      const observer = { onEvent: vi.fn() }
      gemini.bindObserver(observer)
      gemini.setStepContext({ agentId: 'old', sessionId: 'old', stepName: 'old' })
      gemini.setStepContext({ agentId: 'new', sessionId: 'new', stepName: 'new' })

      // act
      await gemini.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(observer.onEvent).toHaveBeenCalledWith(
        { agentId: 'new', sessionId: 'new', stepName: 'new' },
        'llm.response',
        expect.any(Object),
      )
    })

  })

})
