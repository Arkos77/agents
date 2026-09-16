import { ChatGenerationChunk } from '@langchain/core/outputs';
import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { hasContextWindowExceeded } from '@/llm/truncation';
import { MultiAgentGraph } from '@/graphs/MultiAgentGraph';
import { FakeChatModel } from '@/llm/fake';
import { Providers } from '@/common';
import { Run } from '@/run';

const contextStop = 'model_context_window_exceeded';
const config = {
  configurable: { thread_id: 'context-stop' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

class ContextStopModel extends FakeChatModel {
  readonly requests: BaseMessage[][] = [];
  constructor(private readonly repeated = false) {
    super({ responses: ['unused'] });
  }

  async *_streamResponseChunks(
    messages: BaseMessage[]
  ): AsyncGenerator<ChatGenerationChunk> {
    this.requests.push(messages);
    const partial = this.requests.length % 2 === 1 || this.repeated;
    if (partial) {
      yield new ChatGenerationChunk({
        text: '',
        message: new AIMessageChunk({
          content: [
            {
              type: 'server_tool_use',
              id: 'srvtoolu_search',
              name: 'web_search',
              input: { query: 'documentation' },
            },
          ],
          tool_call_chunks: [
            {
              index: 0,
              id: 'srvtoolu_search',
              name: 'web_search',
              args: '{"query":"documentation"}',
            },
          ],
        }),
      });
    }
    yield new ChatGenerationChunk({
      text: partial ? 'Found a source.' : 'The completed answer.',
      message: new AIMessageChunk({
        content: partial
          ? [
            {
              type: 'web_search_tool_result',
              tool_use_id: 'srvtoolu_search',
              content: [
                {
                  type: 'web_search_result',
                  url: 'https://example.com/docs',
                  title: 'Docs',
                  encrypted_content: 'opaque-search-history',
                },
              ],
            },
            { type: 'text', text: 'Found a source.' },
          ]
          : 'The completed answer.',
        additional_kwargs: { stop_reason: partial ? contextStop : 'end_turn' },
        usage_metadata: {
          input_tokens: 50,
          output_tokens: 10,
          total_tokens: 60,
        },
      }),
    });
  }
}

async function createRun(model: ContextStopModel): Promise<Run<t.IState>> {
  const run = await Run.create<t.IState>({
    runId: 'context-stop',
    graphConfig: {
      type: 'standard',
      llmConfig: {
        provider: Providers.ANTHROPIC,
        model: 'claude-sonnet-4-5',
        apiKey: 'test',
      },
    },
    returnContent: true,
    skipCleanup: true,
  });
  if (!run.Graph) throw new Error('Missing graph');
  run.Graph.overrideModel = model;
  return run;
}

describe('context-window stop continuation', () => {
  it('reads both Anthropic metadata shapes without treating normal stops as context stops', () => {
    expect(hasContextWindowExceeded(undefined)).toBe(false);
    expect(
      hasContextWindowExceeded(
        new AIMessageChunk({
          content: '',
          response_metadata: { stop_reason: contextStop },
        })
      )
    ).toBe(true);
    expect(
      hasContextWindowExceeded(
        new AIMessageChunk({
          content: '',
          additional_kwargs: { stop_reason: contextStop },
        })
      )
    ).toBe(true);
    expect(
      hasContextWindowExceeded(
        new AIMessageChunk({
          content: '',
          response_metadata: { stop_reason: 'max_tokens' },
        })
      )
    ).toBe(false);
  });

  it('continues once with partial content and opaque search history intact', async () => {
    const model = new ContextStopModel();
    const run = await createRun(model);
    await run.processStream(
      { messages: [new HumanMessage('Search and answer')] },
      config
    );
    expect(model.requests).toHaveLength(2);
    expect(JSON.stringify(model.requests[1])).toContain(
      'opaque-search-history'
    );
    expect(JSON.stringify(model.requests[1])).toContain('Found a source.');
    expect(model.requests[1].at(-1)?.content).toContain(
      'Continue from where you stopped'
    );
    expect(run.getHaltReason()).toBeUndefined();
    expect(run.getOutputTruncated()).toBe(false);
  });

  it('ends as unfinished on a repeated context stop instead of looping', async () => {
    const model = new ContextStopModel(true);
    const run = await createRun(model);
    await run.processStream({ messages: [new HumanMessage('Search')] }, config);
    expect(model.requests).toHaveLength(2);
    expect(run.getOutputTruncated()).toBe(true);
    expect(run.getHaltReason()).toBe('output_truncated');
  });

  it.each([false, true])(
    'reserves one continuation step (member=%s)',
    async (member) => {
      const model = new ContextStopModel();
      if (member) {
        const graph = new MultiAgentGraph({
          runId: 'context-stop-member',
          agents: [
            {
              agentId: 'member',
              provider: Providers.ANTHROPIC,
              clientOptions: { model: 'claude-sonnet-4-5', apiKey: 'test' },
            },
          ],
          edges: [],
          memberRecursionLimit: 2,
        });
        graph.overrideModel = model;
        await graph
          .createWorkflow()
          .invoke({ messages: [new HumanMessage('Search')] }, config);
        expect(model.requests).toHaveLength(2);
        return;
      }
      const run = await createRun(model);
      await run.processStream(
        { messages: [new HumanMessage('Search')] },
        {
          ...config,
          recursionLimit: 2,
        }
      );
      expect(model.requests).toHaveLength(2);
      expect(run.getHaltReason()).toBeUndefined();
    }
  );

  it('resets the continuation allowance for a new invocation on the same run', async () => {
    const model = new ContextStopModel();
    const run = await createRun(model);
    for (const prompt of ['First search', 'Second search']) {
      await run.processStream({ messages: [new HumanMessage(prompt)] }, config);
      expect(run.getHaltReason()).toBeUndefined();
    }
    expect(model.requests).toHaveLength(4);
  });
});
