import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { Message } from '@noetaris/harness-types'

// mockGet and mockGenerateContent are declared in outer scope so the mock factory can close over them
const mockGet = vi.fn()
const mockGenerateContent = vi.fn()

vi.mock('@google/genai', () => {
  const MockGoogleGenAI = vi.fn(function (this: { models: unknown }, _opts: unknown) {
    this.models = {
      get: mockGet,
      generateContent: (...args: unknown[]) => mockGenerateContent(...args),
    }
  })
  return { GoogleGenAI: MockGoogleGenAI }
})

import { Gemini } from './gemini.js'
import { MockGemini } from './mock-gemini.js'

const messages: Message[] = [{ role: 'user', content: 'Hello' }]

const minimalGenerateResponse = {
  candidates: [{ content: { parts: [{ text: 'Hi' }] }, finishReason: 'STOP' }],
  usageMetadata: { promptTokenCount: 50, candidatesTokenCount: 20 },
}

beforeEach(() => {
  vi.clearAllMocks()
  mockGet.mockResolvedValue({ inputTokenLimit: 1048576 })
  mockGenerateContent.mockResolvedValue(minimalGenerateResponse)
})

describe('Gemini — AdapterUsageF52', () => {

  describe('Group 3: Context Window Fetch, Cache, and Error Suppression', () => {

    it('populates contextWindowSize from inputTokenLimit on first invoke call; token counts correct', async () => {
      // arrange
      mockGet.mockResolvedValue({ inputTokenLimit: 1048576 })
      mockGenerateContent.mockResolvedValue({
        candidates: [{ content: { parts: [{ text: 'Hi' }] } }],
        usageMetadata: { promptTokenCount: 50, candidatesTokenCount: 20 },
      })
      const observer = { onEvent: vi.fn() }
      const gemini = new Gemini('gemini-2.0-flash', { apiKey: 'test-key' })
      gemini.bindObserver(observer)

      // act
      const result = await gemini.invoke(messages)

      // assert
      expect(mockGet).toHaveBeenCalledOnce()
      expect(mockGet).toHaveBeenCalledWith({ model: 'gemini-2.0-flash' })
      expect(result.usage.contextWindowSize).toBe(1048576)
      expect(result.usage.inputTokens).toBe(50)
      expect(result.usage.outputTokens).toBe(20)
      const event = observer.onEvent.mock.calls.find((call) => call[1] === 'llm.response')?.[2]
      expect(event.contextWindowSize).toBe(1048576)
    })

    it('suppresses ai.models.get error; invoke succeeds with contextWindowSize undefined', async () => {
      // arrange
      mockGet.mockRejectedValue(new Error('503 Service Unavailable'))
      mockGenerateContent.mockResolvedValue(minimalGenerateResponse)
      const gemini = new Gemini('gemini-2.0-flash', { apiKey: 'test-key' })

      // act
      const result = await gemini.invoke(messages)

      // assert
      expect(result.usage.contextWindowSize).toBeUndefined()
      expect(result.text).toBeDefined()
    })

    it('does not call ai.models.get on subsequent invoke calls after successful first fetch', async () => {
      // arrange
      mockGet.mockResolvedValue({ inputTokenLimit: 1048576 })
      const gemini = new Gemini('gemini-2.0-flash', { apiKey: 'test-key' })
      await gemini.invoke(messages)
      vi.clearAllMocks()
      mockGenerateContent.mockResolvedValue(minimalGenerateResponse)

      // act
      const result = await gemini.invoke(messages)

      // assert
      expect(mockGet).not.toHaveBeenCalled()
      expect(result.usage.contextWindowSize).toBe(1048576)
    })

  })

})

describe('MockGemini — AdapterUsageF52', () => {

  describe('Group 6: Fixed Zero Usage', () => {

    it('invoke returns usage = { inputTokens: 0, outputTokens: 0 } with no contextWindowSize; emitted event has no contextWindowSize', async () => {
      // arrange
      const observer = { onEvent: vi.fn() }
      const mockGemini = new MockGemini({
        text: 'Ok',
        toolCalls: [],
        stopReason: 'end',
        usage: { inputTokens: 0, outputTokens: 0 },
      })
      mockGemini.bindObserver(observer)

      // act
      const result = await mockGemini.invoke(messages)

      // assert
      expect(result.usage).toEqual({ inputTokens: 0, outputTokens: 0 })
      expect(result.usage.contextWindowSize).toBeUndefined()
      const event = observer.onEvent.mock.calls.find((call) => call[1] === 'llm.response')?.[2]
      expect(event).not.toHaveProperty('contextWindowSize')
    })

  })

})
