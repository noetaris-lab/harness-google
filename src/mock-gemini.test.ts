import { describe, it, expect, vi } from 'vitest'
import { MockGemini, MockGeminiEmptyQueueError } from './mock-gemini.js'
import type { LLMResponse, Message } from '@noetaris/harness-types'
import type { StepContext } from '@noetaris/harness'

describe('MockGemini', () => {

  describe('Group 1: Construction', () => {

    it('initializes lastMessages to empty array', () => {
      // arrange
      const mock = new MockGemini()

      // act / assert
      expect(mock.lastMessages).toEqual([])
    })

    it('construction with single LLMResponse does not throw', () => {
      // arrange
      const r1: LLMResponse = { text: 'hello', toolCalls: [], stopReason: 'end' }

      // act
      const mock = new MockGemini(r1)

      // assert
      expect(mock).toBeInstanceOf(MockGemini)
    })

  })

  describe('Group 2: Queue Drain and Sticky-Last Semantics', () => {

    it('returns same response on every call when constructed with single response', async () => {
      // arrange
      const r1: LLMResponse = { text: 'only', toolCalls: [], stopReason: 'end' }
      const mock = new MockGemini(r1)
      const msgs: Message[] = [{ role: 'user', content: 'hi' }]

      // act
      const res1 = await mock.invoke(msgs)
      const res2 = await mock.invoke(msgs)
      const res3 = await mock.invoke(msgs)

      // assert
      expect(res1).toEqual(r1)
      expect(res2).toEqual(r1)
      expect(res3).toEqual(r1)
    })

    it('drains queue FIFO then sticks to last element', async () => {
      // arrange
      const r1: LLMResponse = { text: 'first', toolCalls: [], stopReason: 'end' }
      const r2: LLMResponse = { text: 'second', toolCalls: [], stopReason: 'end' }
      const r3: LLMResponse = { text: 'third', toolCalls: [], stopReason: 'end' }
      const mock = new MockGemini([r1, r2, r3])
      const msgs: Message[] = []

      // act
      const res1 = await mock.invoke(msgs)
      const res2 = await mock.invoke(msgs)
      const res3 = await mock.invoke(msgs)

      // assert
      expect(res1).toEqual(r1)
      expect(res2).toEqual(r2)
      expect(res3).toEqual(r3)
    })

    it('returns last element repeatedly after queue is exhausted', async () => {
      // arrange
      const r1: LLMResponse = { text: 'first', toolCalls: [], stopReason: 'end' }
      const r2: LLMResponse = { text: 'second', toolCalls: [], stopReason: 'end' }
      const mock = new MockGemini([r1, r2])
      const msgs: Message[] = []

      // act
      const res1 = await mock.invoke(msgs)
      const res2 = await mock.invoke(msgs)
      const res3 = await mock.invoke(msgs)
      const res4 = await mock.invoke(msgs)

      // assert
      expect(res1).toEqual(r1)
      expect(res2).toEqual(r2)
      expect(res3).toEqual(r2)
      expect(res4).toEqual(r2)
    })

  })

  describe('Group 3: Enqueue', () => {

    it('enqueue after empty construction makes invoke succeed', async () => {
      // arrange
      const mock = new MockGemini()
      const r1: LLMResponse = { text: 'late', toolCalls: [], stopReason: 'end' }
      mock.enqueue(r1)

      // act
      const result = await mock.invoke([])

      // assert
      expect(result).toEqual(r1)
    })

    it('enqueue array after construction preserves correct merged order', async () => {
      // arrange
      const r1: LLMResponse = { text: 'a', toolCalls: [], stopReason: 'end' }
      const r2: LLMResponse = { text: 'b', toolCalls: [], stopReason: 'end' }
      const r3: LLMResponse = { text: 'c', toolCalls: [], stopReason: 'end' }
      const mock = new MockGemini([r1])
      mock.enqueue([r2, r3])

      // act
      const res1 = await mock.invoke([])
      const res2 = await mock.invoke([])
      const res3 = await mock.invoke([])

      // assert
      expect(res1).toEqual(r1)
      expect(res2).toEqual(r2)
      expect(res3).toEqual(r3)
    })

    it('enqueue with single non-array LLMResponse is treated as one-element array', async () => {
      // arrange
      const r1: LLMResponse = { text: 'scalar', toolCalls: [], stopReason: 'end' }
      const mock = new MockGemini()
      mock.enqueue(r1)

      // act
      const result = await mock.invoke([])

      // assert
      expect(result).toEqual(r1)
    })

  })

  describe('Group 4: Observer Event', () => {

    it('emits llm.response event with correct payload after invoke', async () => {
      // arrange
      const r1: LLMResponse = { text: 'ok', toolCalls: [], stopReason: 'end' }
      const mock = new MockGemini(r1)
      const onEvent = vi.fn()
      mock.bindObserver({ onEvent })
      const ctx: StepContext = { agentId: 'a1', sessionId: 's1', stepName: 'step1' }
      mock.setStepContext(ctx)

      // act
      await mock.invoke([])

      // assert
      expect(onEvent).toHaveBeenCalledOnce()
      expect(onEvent).toHaveBeenCalledWith(ctx, 'llm.response', { tokens: { input: 0, output: 0 }, modelId: 'mock', stopReason: 'end', providerName: 'mock' })
    })

    it('uses StepContext from setStepContext as first arg to onEvent', async () => {
      // arrange
      const r1: LLMResponse = { text: 'ok', toolCalls: [], stopReason: 'end' }
      const mock = new MockGemini(r1)
      const onEvent = vi.fn()
      mock.bindObserver({ onEvent })
      const ctx: StepContext = { agentId: 'agent-99', sessionId: 'sess-42', stepName: 'my-step' }
      mock.setStepContext(ctx)

      // act
      await mock.invoke([])

      // assert
      // noUncheckedIndexedAccess: calls[0] and [0] are guaranteed by the single invoke() above
      expect(onEvent.mock.calls[0]![0]).toEqual({ agentId: 'agent-99', sessionId: 'sess-42', stepName: 'my-step' })
    })

    it('uses default StepContext when setStepContext is never called', async () => {
      // arrange
      const r1: LLMResponse = { text: 'ok', toolCalls: [], stopReason: 'end' }
      const mock = new MockGemini(r1)
      const onEvent = vi.fn()
      mock.bindObserver({ onEvent })

      // act
      await mock.invoke([])

      // assert
      // noUncheckedIndexedAccess: calls[0] and [0] are guaranteed by the single invoke() above
      expect(onEvent.mock.calls[0]![0]).toEqual({ agentId: '', sessionId: '', stepName: '' })
    })

    it('does not throw when observer has no onEvent method', async () => {
      // arrange
      const r1: LLMResponse = { text: 'ok', toolCalls: [], stopReason: 'end' }
      const mock = new MockGemini(r1)
      mock.bindObserver({})

      // assert
      await expect(mock.invoke([])).resolves.toEqual(r1)
    })

    it('stopReason in observer event reflects the response stopReason', async () => {
      // arrange
      const r1: LLMResponse = { text: '', toolCalls: [{ id: 'c1', name: 'fn', input: {} }], stopReason: 'tool_use' }
      const mock = new MockGemini(r1)
      const onEvent = vi.fn()
      mock.bindObserver({ onEvent })
      mock.setStepContext({ agentId: '', sessionId: '', stepName: '' })

      // act
      await mock.invoke([])

      // assert
      expect(onEvent).toHaveBeenCalledWith(expect.anything(), 'llm.response', expect.objectContaining({ stopReason: 'tool_use' }))
    })

  })

  describe('Group 5: lastMessages and Misc Public API', () => {

    it('lastMessages is updated to messages arg after each invoke', async () => {
      // arrange
      const r1: LLMResponse = { text: 'ok', toolCalls: [], stopReason: 'end' }
      const mock = new MockGemini(r1)
      const msgs1: Message[] = [{ role: 'user', content: 'hello' }]
      const msgs2: Message[] = [{ role: 'user', content: 'hello' }, { role: 'assistant', content: 'hi' }]

      // act
      await mock.invoke(msgs1)

      // assert
      expect(mock.lastMessages).toEqual(msgs1)

      // act
      await mock.invoke(msgs2)

      // assert
      expect(mock.lastMessages).toEqual(msgs2)
    })

    it('only the most recently bound observer fires after bindObserver called multiple times', async () => {
      // arrange
      const r1: LLMResponse = { text: 'ok', toolCalls: [], stopReason: 'end' }
      const mock = new MockGemini(r1)
      const onEvent1 = vi.fn()
      const onEvent2 = vi.fn()
      mock.bindObserver({ onEvent: onEvent1 })
      mock.bindObserver({ onEvent: onEvent2 })

      // act
      await mock.invoke([])

      // assert
      expect(onEvent1).not.toHaveBeenCalled()
      expect(onEvent2).toHaveBeenCalledOnce()
    })

    it('returns configured response and sets lastMessages to empty array when invoked with empty messages', async () => {
      // arrange
      const r1: LLMResponse = { text: 'ok', toolCalls: [], stopReason: 'end' }
      const mock = new MockGemini(r1)

      // act
      const result = await mock.invoke([])

      // assert
      expect(result).toEqual(r1)
      expect(mock.lastMessages).toEqual([])
    })

  })

  describe('Group 6: Error Cases', () => {

    it('throws MockGeminiEmptyQueueError when invoke is called on empty queue', async () => {
      // arrange
      const mock = new MockGemini()

      // act
      const p = mock.invoke([])

      // assert
      await expect(p).rejects.toThrow(MockGeminiEmptyQueueError)
    })

  })

})
