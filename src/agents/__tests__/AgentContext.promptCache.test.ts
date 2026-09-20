import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type {
  BaseMessage,
  MessageContentComplex,
} from '@langchain/core/messages';
import type { PromptCacheTtl } from '@/messages/cache';
import { _convertMessagesToAnthropicPayload } from '@/llm/anthropic/utils/message_inputs';
import { partitionAndMarkAnthropicToolCache } from '@/messages/anthropicToolCache';
import { _convertMessagesToOpenAIParams } from '@/llm/openai/utils';
import { AgentContext } from '../AgentContext';
import { Providers } from '@/common';

const providers = [Providers.ANTHROPIC, Providers.OPENROUTER] as const;
const ttls = ['5m', '1h'] as const;

function createContext(
  provider: (typeof providers)[number],
  ttl: PromptCacheTtl = '1h',
  dynamicInstructions = 'Current timestamp: Monday'
): AgentContext {
  return AgentContext.fromConfig({
    agentId: 'cache-test',
    provider,
    clientOptions: {
      model: 'claude-sonnet-4-5',
      promptCache: true,
      promptCacheTtl: ttl,
    },
    instructions: 'Stable instructions',
    additional_instructions: dynamicInstructions,
  });
}

function toolExchange(id: string): BaseMessage[] {
  return [
    new AIMessage({
      content: '',
      tool_calls: [{ id, name: 'search', args: {} }],
    }),
    new ToolMessage({ content: `Result ${id}`, tool_call_id: id }),
  ];
}

function markedBlocks(message: BaseMessage): MessageContentComplex[] {
  return Array.isArray(message.content)
    ? message.content.filter((block) => 'cache_control' in block)
    : [];
}

function markedMessages(messages: BaseMessage[]): BaseMessage[] {
  return messages.filter((message) => markedBlocks(message).length > 0);
}

describe.each(providers)('%s dynamic prompt caching', (provider) => {
  it.each(ttls)(
    'advances the subagent loop tail with %s TTL without mutating history',
    async (ttl) => {
      const ctx = createContext(provider, ttl);
      const history: BaseMessage[] = [new HumanMessage('Research the task')];
      const cacheControl =
        ttl === '1h' ? { type: 'ephemeral', ttl } : { type: 'ephemeral' };

      for (const id of ['call_1', 'call_2', 'call_3']) {
        history.push(...toolExchange(id));
        const before = JSON.stringify(history);
        const result = await ctx.systemRunnable!.invoke(history);

        expect(markedMessages(result)).toEqual([result[0], result.at(-1)]);
        expect(markedBlocks(result.at(-1)!)).toEqual([
          { type: 'text', text: `Result ${id}`, cache_control: cacheControl },
        ]);
        expect(result.at(-1)).toBeInstanceOf(ToolMessage);
        expect((result.at(-1) as ToolMessage).tool_call_id).toBe(id);
        expect(JSON.stringify(history)).toBe(before);
        expect(result.map((message) => message.text)).toContain(
          'Current timestamp: Monday'
        );
      }
    }
  );

  it('keeps one stable-history marker and one current-turn marker', async () => {
    const history = [
      new HumanMessage('First'),
      new AIMessage('First answer'),
      new HumanMessage('Second'),
      new AIMessage('Second answer'),
      new HumanMessage('Research the task'),
      ...toolExchange('call_1'),
    ];
    const result =
      await createContext(provider).systemRunnable!.invoke(history);

    expect(markedMessages(result).map((message) => message.text)).toEqual([
      'Stable instructions',
      'Second answer',
      'Result call_1',
    ]);
    const changed = await createContext(
      provider,
      '1h',
      'Current timestamp: Tuesday'
    ).systemRunnable!.invoke(history);
    expect(changed.slice(0, 5)).toEqual(result.slice(0, 5));
    expect(changed[5].content).toBe('Current timestamp: Tuesday');
    expect(markedMessages(changed).map((message) => message.text)).toEqual([
      'Stable instructions',
      'Second answer',
      'Result call_1',
    ]);
  });

  it('normalizes inherited markers including the opening message', async () => {
    const history = [
      new HumanMessage({
        content: [
          { type: 'text', text: 'First', cache_control: { type: 'ephemeral' } },
          { type: 'text', text: 'More', cache_control: { type: 'ephemeral' } },
        ],
      }),
      new AIMessage({
        content: [
          {
            type: 'text',
            text: 'Answer',
            cache_control: { type: 'ephemeral' },
          },
        ],
      }),
      new HumanMessage('Latest'),
      ...toolExchange('call_1'),
    ];
    const before = JSON.stringify(history);
    const result =
      await createContext(provider).systemRunnable!.invoke(history);

    expect(result.flatMap(markedBlocks)).toHaveLength(3);
    expect(markedBlocks(result[1])).toHaveLength(0);
    for (const block of result.flatMap(markedBlocks)) {
      expect(block).toHaveProperty('cache_control', {
        type: 'ephemeral',
        ttl: '1h',
      });
    }
    expect(JSON.stringify(history)).toBe(before);
  });

  it.each([false, true])(
    'marks the conversation after a checkpoint (precedesMessages=%s)',
    async (precedesMessages) => {
      const ctx = createContext(provider);
      ctx.setSummary('Compacted history', 10, { precedesMessages });
      const history = precedesMessages
        ? toolExchange('call_1')
        : [new HumanMessage('Task'), ...toolExchange('call_1')];
      const result = await ctx.systemRunnable!.invoke(history);

      expect(markedMessages(result)).toEqual([result[0], result.at(-1)]);
      expect(result.at(-1)!.text).toBe('Result call_1');
      expect(result.map((message) => message.text).join('\n')).toContain(
        'Compacted history'
      );
    }
  );

  it('replaces stale markers within the current tool loop', async () => {
    const history = [new HumanMessage('Task'), ...toolExchange('call_1')];
    const first = await createContext(provider, '5m').systemRunnable!.invoke(
      history
    );
    const markedResult = first.at(-1)!;
    const resumed = [
      ...history.slice(0, -1),
      markedResult,
      ...toolExchange('call_2'),
    ];
    const before = JSON.stringify(resumed);
    const result =
      await createContext(provider).systemRunnable!.invoke(resumed);

    expect(markedMessages(result).map((message) => message.text)).toEqual([
      'Stable instructions',
      'Result call_2',
    ]);
    expect(markedBlocks(result.at(-1)!)[0]).toHaveProperty('cache_control', {
      type: 'ephemeral',
      ttl: '1h',
    });
    expect(JSON.stringify(resumed)).toBe(before);
  });

  it('keeps the no-dynamic-instructions tail strategy', async () => {
    const ctx = createContext(provider, '1h', '');
    const result = await ctx.systemRunnable!.invoke([
      new HumanMessage('Task'),
      ...toolExchange('call_1'),
    ]);
    expect(markedMessages(result)).toEqual([result[0], result.at(-1)]);
    expect(result).toHaveLength(4);
  });

  it('does not add markers when prompt caching is disabled', async () => {
    const ctx = AgentContext.fromConfig({
      agentId: 'uncached',
      provider,
      clientOptions: { promptCache: false },
      instructions: 'Stable instructions',
      additional_instructions: 'Dynamic instructions',
    });
    const result = await ctx.systemRunnable!.invoke([
      new HumanMessage('Task'),
      ...toolExchange('call_1'),
    ]);
    expect(markedMessages(result)).toHaveLength(0);
    expect(result[0].text).toBe('Stable instructions\n\nDynamic instructions');
  });

  it('preserves single-message instruction ordering without anchoring dynamic text', async () => {
    const ctx = createContext(provider);
    const result = await ctx.systemRunnable!.invoke([new HumanMessage('Task')]);
    const bodyText = result.slice(1).map((message) => message.text);
    expect(bodyText).toEqual(
      provider === Providers.ANTHROPIC
        ? ['Current timestamp: Monday', 'Task']
        : ['Task', 'Current timestamp: Monday']
    );
    const expectedAnchors =
      provider === Providers.ANTHROPIC
        ? ['Stable instructions', 'Task']
        : ['Stable instructions'];
    expect(markedMessages(result).map((message) => message.text)).toEqual(
      expectedAnchors
    );
  });

  it('does not anchor instruction-only or summary-only bodies', async () => {
    const ctx = createContext(provider);
    ctx.setSummary('Compacted history', 10, { precedesMessages: true });
    const result = await ctx.systemRunnable!.invoke([]);
    expect(markedMessages(result)).toEqual([result[0]]);
  });

  it('skips synthetic and reasoning-only tail messages without losing the conversation marker', async () => {
    const result = await createContext(provider).systemRunnable!.invoke([
      new HumanMessage('Task'),
      ...toolExchange('call_1'),
      new AIMessage({
        content: [
          { type: 'thinking', thinking: 'Reasoning', signature: 'signature' },
        ],
      }),
      new AIMessage({
        content: 'Volatile skill',
        additional_kwargs: { isMeta: true },
      }),
    ]);
    expect(markedMessages(result).map((message) => message.text)).toContain(
      'Result call_1'
    );
    expect(markedBlocks(result.at(-1)!)).toHaveLength(0);
  });
});

describe('dynamic prompt cache payload', () => {
  it('preserves the tool tail through OpenRouter chat conversion', async () => {
    const result = await createContext(
      Providers.OPENROUTER
    ).systemRunnable!.invoke([
      new HumanMessage('Task'),
      ...toolExchange('call_1'),
    ]);
    const payload = _convertMessagesToOpenAIParams(
      result,
      'anthropic/claude-sonnet-4-5',
      {
        preserveToolCacheControl: true,
      }
    );
    expect(payload.at(-1)).toMatchObject({
      role: 'tool',
      tool_call_id: 'call_1',
      content: [
        {
          type: 'text',
          text: 'Result call_1',
          cache_control: { type: 'ephemeral', ttl: '1h' },
        },
      ],
    });
    expect(JSON.stringify(payload).match(/"cache_control":/g)).toHaveLength(2);
  });

  it('ships a tool-result tail and stays within four total breakpoints', async () => {
    const ctx = createContext(Providers.ANTHROPIC);
    const messages = await ctx.systemRunnable!.invoke([
      new HumanMessage('First'),
      new AIMessage('First answer'),
      new HumanMessage('Second'),
      new AIMessage('Second answer'),
      new HumanMessage('Task'),
      ...toolExchange('call_1'),
    ]);
    const payload = _convertMessagesToAnthropicPayload(messages);
    const tools = partitionAndMarkAnthropicToolCache(
      [
        {
          name: 'search',
          description: 'Search',
          input_schema: { type: 'object', properties: {} },
        },
      ],
      () => false,
      '1h'
    );
    const blocks = payload.messages.flatMap((message) =>
      typeof message.content === 'string' ? [] : message.content
    );
    const toolResult = blocks.find((block) => block.type === 'tool_result');

    expect(toolResult).toMatchObject({
      type: 'tool_result',
      tool_use_id: 'call_1',
      cache_control: { type: 'ephemeral', ttl: '1h' },
    });
    expect(blocks.filter((block) => 'cache_control' in block)).toHaveLength(2);
    const request = { ...payload, tools };
    expect(JSON.stringify(request).match(/"cache_control":/g)).toHaveLength(4);
    expect(JSON.stringify(toolResult?.content)).not.toContain('cache_control');
  });
});
