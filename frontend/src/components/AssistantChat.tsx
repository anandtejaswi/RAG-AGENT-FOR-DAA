import React, { useState, useRef, useEffect } from 'react';
import { Sparkles, StopCircle, RefreshCw, ArrowUp } from 'lucide-react';
import type { ChatMessage, MessageMetadata } from '../types';
import { MessageRenderer } from './MessageRenderer';

interface AssistantChatProps {
  onMetadataChange: (meta?: MessageMetadata) => void;
  onTopicDetected: (topic?: string | null) => void;
}

const SAMPLE_PROMPTS = [
  {
    title: 'Master Theorem',
    query: 'Solve the recurrence T(n) = 2T(n/2) + n using the Master Theorem. State which case applies.',
    tag: 'Unit 1: Recurrences',
  },
  {
    title: '0/1 Knapsack',
    query: 'Solve the 0/1 knapsack problem for weights [1, 3, 4, 5], values [1, 4, 5, 7], capacity 7.',
    tag: 'Unit 4: Dynamic Programming',
  },
  {
    title: 'LCS with Arrows',
    query: 'Compute the Longest Common Subsequence of ABCBDAB and BDCABA with directional arrows.',
    tag: 'Unit 4: LCS Table',
  },
  {
    title: 'AVL Tree LL Rotation',
    query: 'Show the rotation that repairs an LL imbalance in an AVL tree and state balance factors.',
    tag: 'Unit 2: Balanced Trees',
  },
  {
    title: 'Dijkstra Trace',
    query: 'Trace Dijkstra\'s algorithm from source A on the graph {"A": {"B": 4, "C": 2}, "B": {"C": 1, "D": 5}, "C": {"D": 8, "E": 10}, "D": {"E": 2}, "E": {}}',
    tag: 'Unit 3: Shortest Paths',
  },
  {
    title: '4-Queens Backtracking',
    query: 'Draw the state space tree explored by backtracking for the 4-Queens problem and explain the bounding function.',
    tag: 'Unit 4: Backtracking',
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
            console.error('Error parsing SSE block:', e, block);
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
                  content: `Failed to receive answer from agent server: ${err.message}`,
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
    <div className="flex flex-col h-full bg-slate-900 overflow-hidden relative">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-4 lg:px-8 py-6 space-y-6">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center max-w-2xl mx-auto text-center space-y-6 animate-in fade-in duration-300">
            <div className="h-16 w-16 rounded-2xl bg-gradient-to-tr from-indigo-500 to-cyan-400 p-0.5 shadow-2xl shadow-indigo-500/30">
              <div className="h-full w-full bg-slate-950 rounded-[14px] flex items-center justify-center">
                <Sparkles className="h-8 w-8 text-cyan-400" />
              </div>
            </div>

            <div>
              <h2 className="text-xl font-bold text-slate-100 tracking-tight">
                AKTU Design & Analysis of Algorithms Assistant
              </h2>
              <p className="text-sm text-slate-400 mt-1.5 max-w-lg">
                Ask any examination question from KCS-503. The agent retrieves official notes, computes exact mathematical tables, and links visual schematics.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 w-full text-left">
              {SAMPLE_PROMPTS.map((prompt, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSend(prompt.query)}
                  className="group flex flex-col justify-between p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 hover:border-indigo-500/60 hover:bg-slate-950 transition-all text-left shadow-sm"
                >
                  <span className="text-[11px] font-mono text-cyan-400 font-semibold mb-1">
                    {prompt.tag}
                  </span>
                  <span className="text-xs font-medium text-slate-200 group-hover:text-slate-100 line-clamp-2">
                    {prompt.title}
                  </span>
                  <span className="text-[11px] text-slate-500 line-clamp-2 mt-1">
                    {prompt.query}
                  </span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              onClick={() => msg.metadata && onMetadataChange(msg.metadata)}
              className={`flex gap-3.5 max-w-3xl mx-auto ${
                msg.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {msg.role === 'assistant' && (
                <div className="h-8 w-8 rounded-xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center shrink-0 mt-0.5 shadow-sm">
                  <Sparkles className="h-4 w-4 text-indigo-400" />
                </div>
              )}

              <div
                className={`rounded-2xl px-5 py-4 text-sm max-w-[85%] shadow-md ${
                  msg.role === 'user'
                    ? 'bg-gradient-to-tr from-indigo-600 to-indigo-700 text-white font-medium rounded-tr-sm'
                    : 'bg-slate-950/90 border border-slate-800 rounded-tl-sm text-slate-200'
                }`}
              >
                {msg.role === 'user' ? (
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                ) : (
                  <div>
                    {msg.isStreaming && !msg.content ? (
                      <div className="flex items-center gap-2 text-slate-400 text-xs py-1">
                        <RefreshCw className="h-3.5 w-3.5 animate-spin text-indigo-400" />
                        <span>Searching syllabus notes & computing solver tables...</span>
                      </div>
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

      {/* Input Composer */}
      <div className="p-4 lg:px-8 border-t border-slate-800 bg-slate-900/90 backdrop-blur-md">
        <div className="max-w-3xl mx-auto relative flex items-center bg-slate-950 border border-slate-700/80 rounded-2xl p-1.5 focus-within:border-indigo-500 shadow-xl">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="Ask a DAA question (e.g., recurrence proof, DP table, graph trace)..."
            rows={1}
            className="w-full px-4 py-2.5 bg-transparent text-slate-100 text-xs sm:text-sm placeholder-slate-500 focus:outline-none resize-none max-h-32"
          />

          <div className="flex items-center gap-1.5 pr-1">
            {isLoading ? (
              <button
                onClick={handleStop}
                className="p-2 rounded-xl bg-rose-500/20 text-rose-400 hover:bg-rose-500/30 transition-colors"
                title="Stop generation"
              >
                <StopCircle className="h-5 w-5" />
              </button>
            ) : (
              <button
                onClick={() => handleSend()}
                disabled={!input.trim()}
                className="p-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-600 text-white transition-all shadow-md"
                title="Send query"
              >
                <ArrowUp className="h-5 w-5" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
