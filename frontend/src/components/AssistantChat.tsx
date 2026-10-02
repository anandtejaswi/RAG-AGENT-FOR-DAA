import React, { useState, useRef, useEffect } from 'react';
import { StopCircle, ArrowUp } from 'lucide-react';
import type { ChatMessage, MessageMetadata } from '../types';
import { MessageRenderer } from './MessageRenderer';

interface AssistantChatProps {
  onMetadataChange: (meta?: MessageMetadata) => void;
  onTopicDetected: (topic?: string | null) => void;
}

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
    <div className="flex flex-col h-full bg-white text-zinc-900 overflow-hidden relative">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 lg:px-8 py-6 space-y-4">
        {messages.length === 0 ? (
          <div className="h-full flex items-center justify-center text-zinc-400 text-xs font-mono">
            Type your question below to begin.
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
                    ? 'bg-zinc-100 text-zinc-900 border border-zinc-200'
                    : 'bg-white text-zinc-900 border border-zinc-200 shadow-xs'
                }`}
              >
                {msg.role === 'user' ? (
                  <p className="whitespace-pre-wrap font-medium">{msg.content}</p>
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
      <div className="p-4 border-t border-zinc-200 bg-white">
        <div className="max-w-3xl mx-auto flex items-center bg-zinc-50 border border-zinc-300 rounded px-3 py-1.5 focus-within:border-zinc-900 focus-within:bg-white transition-colors">
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
            className="w-full bg-transparent text-zinc-900 text-xs placeholder-zinc-400 focus:outline-none resize-none max-h-32 py-1"
          />

          <div className="flex items-center pl-2">
            {isLoading ? (
              <button
                onClick={handleStop}
                className="p-1 rounded bg-zinc-200 text-zinc-800 hover:bg-zinc-300 cursor-pointer"
                title="Stop"
              >
                <StopCircle className="h-4 w-4" />
              </button>
            ) : (
              <button
                onClick={() => handleSend()}
                disabled={!input.trim()}
                className="p-1 rounded bg-zinc-900 disabled:bg-zinc-200 text-white disabled:text-zinc-400 cursor-pointer transition-colors"
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
