import React, { useState, useRef, useEffect } from 'react';
import { StopCircle, ArrowUp } from 'lucide-react';
import type { ChatMessage, MessageMetadata } from '../types';
import { MessageRenderer } from './MessageRenderer';

interface AssistantChatProps {
  onMetadataChange: (meta?: MessageMetadata) => void;
  onTopicDetected: (topic?: string | null) => void;
}

const SAMPLE_PROMPTS = [
  {
    title: 'Master Theorem',
    query: 'Solve the recurrence T(n) = 2T(n/2) + n using the Master Theorem.',
  },
  {
    title: '0/1 Knapsack',
    query: 'Solve the 0/1 knapsack problem for weights [1, 3, 4, 5], values [1, 4, 5, 7], capacity 7.',
  },
  {
    title: 'LCS Table',
    query: 'Compute the Longest Common Subsequence of ABCBDAB and BDCABA with directional arrows.',
  },
  {
    title: 'AVL Tree LL Rotation',
    query: 'Show the rotation that repairs an LL imbalance in an AVL tree and state balance factors.',
  },
  {
    title: 'Dijkstra Trace',
    query: 'Trace Dijkstra\'s algorithm from source A on the graph {"A": {"B": 4, "C": 2}, "B": {"C": 1, "D": 5}, "C": {"D": 8, "E": 10}, "D": {"E": 2}, "E": {}}',
  },
  {
    title: '4-Queens Backtracking',
    query: 'Draw the state space tree explored by backtracking for the 4-Queens problem and explain the bounding function.',
  },
];

export const AssistantChat: React.FC<AssistantChatProps> = ({
  onMetadataChange,
  onTopicDetected,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (queryText?: string) => {
    const query = (queryText || input).trim();
    if (!query || isLoading) return;

    setInput('');
    const userMsgId = `usr_${Date.now()}`;
    const botMsgId = `ast_${Date.now()}`;

    const userMsg: ChatMessage = {
      id: userMsgId,
      role: 'user',
      content: query,
    };

    const initialBotMsg: ChatMessage = {
      id: botMsgId,
      role: 'assistant',
      content: '',
      isStreaming: true,
      metadata: {},
    };

    setMessages((prev) => [...prev, userMsg, initialBotMsg]);
    setIsLoading(true);

    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          messages: [...messages, userMsg].map((m) => ({
            role: m.role,
            content: m.content,
          })),
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok || !response.body) {
        throw new Error(`Server returned ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let accumulatedContent = '';
      let currentMetadata: MessageMetadata = {};

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const block of lines) {
          if (!block.trim()) continue;
          let eventType = 'message';
          let eventData = '';

          for (const line of block.split('\n')) {
            if (line.startsWith('event: ')) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith('data: ')) {
              eventData = line.slice(6);
            }
          }

          if (!eventData) continue;

          try {
            const parsed = JSON.parse(eventData);

            if (eventType === 'metadata') {
              currentMetadata = {
                ...currentMetadata,
                classification: parsed.classification,
                chunks: parsed.chunks,
                offered_diagrams: parsed.offered_diagrams,
                refused: parsed.refused,
              };
              onMetadataChange(currentMetadata);
              if (parsed.classification?.topic) {
                onTopicDetected(parsed.classification.topic);
              }
            } else if (eventType === 'token') {
              accumulatedContent += parsed.content;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === botMsgId
                    ? {
                        ...m,
                        content: accumulatedContent,
                        metadata: currentMetadata,
                      }
                    : m
                )
              );
            } else if (eventType === 'tool_calls') {
              currentMetadata = {
                ...currentMetadata,
                tool_calls: parsed.tool_calls,
              };
              onMetadataChange(currentMetadata);
            } else if (eventType === 'done') {
              currentMetadata = {
                ...currentMetadata,
                citations: parsed.citations,
                diagrams: parsed.diagrams,
                latency_s: parsed.latency_s,
                refused: parsed.refused,
              };
              onMetadataChange(currentMetadata);
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === botMsgId
                    ? {
                        ...m,
                        content: parsed.answer || accumulatedContent,
                        metadata: currentMetadata,
                        isStreaming: false,
                      }
                    : m
                )
              );
            } else if (eventType === 'error') {
              accumulatedContent += `\n\n**Error:** ${parsed.error}`;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === botMsgId
                    ? {
                        ...m,
                        content: accumulatedContent,
                        isStreaming: false,
                      }
                    : m
                )
              );
            }
          } catch (e) {
            console.error('Error parsing SSE block:', e);
          }
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === botMsgId
              ? {
                  ...m,
                  content: `Error: ${err.message}`,
                  isStreaming: false,
                }
              : m
          )
        );
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-zinc-950 text-zinc-100 overflow-hidden relative">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 lg:px-8 py-6 space-y-5">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center max-w-xl mx-auto text-center space-y-6">
            <div>
              <h2 className="text-base font-semibold font-mono tracking-tight text-zinc-100">
                Design and Analysis of Algorithms
              </h2>
              <p className="text-xs text-zinc-400 mt-1">
                Enter a question or select an example prompt below.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full text-left">
              {SAMPLE_PROMPTS.map((prompt, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSend(prompt.query)}
                  className="p-3 rounded border border-zinc-800 bg-zinc-900/60 hover:bg-zinc-900 hover:border-zinc-700 transition-colors text-left cursor-pointer"
                >
                  <div className="text-xs font-mono font-semibold text-zinc-200">
                    {prompt.title}
                  </div>
                  <div className="text-[11px] text-zinc-400 mt-0.5 line-clamp-2">
                    {prompt.query}
                  </div>
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              onClick={() => msg.metadata && onMetadataChange(msg.metadata)}
              className={`flex max-w-3xl mx-auto ${
                msg.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`rounded px-4 py-3 text-xs leading-relaxed max-w-[90%] ${
                  msg.role === 'user'
                    ? 'bg-zinc-800 text-zinc-100 border border-zinc-700'
                    : 'bg-zinc-900/90 text-zinc-200 border border-zinc-800'
                }`}
              >
                {msg.role === 'user' ? (
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                ) : (
                  <div>
                    {msg.isStreaming && !msg.content ? (
                      <span className="font-mono text-zinc-400 text-xs animate-pulse">
                        computing...
                      </span>
                    ) : (
                      <MessageRenderer
                        content={msg.content}
                        diagrams={msg.metadata?.diagrams}
                      />
                    )}
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Composer */}
      <div className="p-4 border-t border-zinc-800 bg-zinc-950">
        <div className="max-w-3xl mx-auto flex items-center bg-zinc-900 border border-zinc-800 rounded px-3 py-1.5 focus-within:border-zinc-600">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="Type your question..."
            rows={1}
            className="w-full bg-transparent text-zinc-100 text-xs placeholder-zinc-500 focus:outline-none resize-none max-h-32 py-1"
          />

          <div className="flex items-center pl-2">
            {isLoading ? (
              <button
                onClick={handleStop}
                className="p-1 rounded bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
                title="Stop"
              >
                <StopCircle className="h-4 w-4" />
              </button>
            ) : (
              <button
                onClick={() => handleSend()}
                disabled={!input.trim()}
                className="p-1 rounded bg-zinc-100 disabled:bg-zinc-800 text-zinc-900 disabled:text-zinc-600 cursor-pointer"
                title="Send"
              >
                <ArrowUp className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
