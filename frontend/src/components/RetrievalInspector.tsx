import React from 'react';
import type { MessageMetadata } from '../types';

interface RetrievalInspectorProps {
  metadata?: MessageMetadata;
}

export const RetrievalInspector: React.FC<RetrievalInspectorProps> = ({ metadata }) => {
  if (!metadata) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 text-center text-zinc-500 font-mono text-xs">
        <p>No active query telemetry</p>
      </div>
    );
  }

  const { classification, chunks = [], tool_calls = [], diagrams = [], latency_s, refused } = metadata;

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4 text-xs font-mono">
      {/* Classification */}
      <div className="rounded border border-zinc-800 bg-zinc-950 p-3 space-y-2">
        <div className="flex items-center justify-between text-zinc-300 font-semibold">
          <span>Routing</span>
          <span className="text-[10px] text-zinc-400 border border-zinc-800 px-1.5 py-0.5 rounded">
            {refused ? 'Refused' : 'Accepted'}
          </span>
        </div>

        <div className="space-y-1 text-zinc-400 text-[11px]">
          <div className="flex justify-between">
            <span className="text-zinc-500">Topic:</span>
            <span className="text-zinc-200">
              {classification?.topic ? `U${classification.unit}: ${classification.topic}` : 'None'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-zinc-500">Confidence:</span>
            <span className="text-zinc-200">
              {classification?.confidence ? `${(classification.confidence * 100).toFixed(1)}%` : '0%'}
            </span>
          </div>
          {latency_s && (
            <div className="flex justify-between">
              <span className="text-zinc-500">Latency:</span>
              <span className="text-zinc-200">{latency_s}s</span>
            </div>
          )}
        </div>
      </div>

      {/* Solver Tool Calls */}
      {tool_calls.length > 0 && (
        <div className="rounded border border-zinc-800 bg-zinc-950 p-3 space-y-2">
          <div className="font-semibold text-zinc-300">
            Solvers ({tool_calls.length})
          </div>
          {tool_calls.map((t, idx) => (
            <div key={idx} className="border border-zinc-800 rounded p-2 bg-black space-y-1">
              <div className="text-[11px] text-zinc-200 font-bold">
                {t.name}()
              </div>
              <pre className="text-[10px] text-zinc-400 overflow-x-auto">
                {typeof t.result === 'string' ? t.result : JSON.stringify(t.result, null, 2)}
              </pre>
            </div>
          ))}
        </div>
      )}

      {/* Diagrams */}
      {diagrams.length > 0 && (
        <div className="rounded border border-zinc-800 bg-zinc-950 p-3 space-y-2">
          <div className="font-semibold text-zinc-300">
            Diagrams ({diagrams.length})
          </div>
          <div className="space-y-1.5">
            {diagrams.map((d) => (
              <div key={d.id} className="border border-zinc-800 rounded p-2 bg-black flex gap-2">
                <img src={d.url} alt={d.concept} className="h-10 w-14 object-contain bg-zinc-950 p-0.5 border border-zinc-800" />
                <div className="flex-1 min-w-0">
                  <div className="text-[10px] text-zinc-200 font-bold">{d.id}</div>
                  <div className="text-[11px] text-zinc-400 truncate">{d.concept}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Retrieved Chunks */}
      {chunks.length > 0 && (
        <div className="rounded border border-zinc-800 bg-zinc-950 p-3 space-y-2">
          <div className="font-semibold text-zinc-300">
            Retrieved Passages ({chunks.length})
          </div>

          <div className="space-y-1.5">
            {chunks.map((c) => (
              <details
                key={c.id}
                className="border border-zinc-800 rounded p-2 bg-black text-zinc-300"
              >
                <summary className="flex items-center justify-between cursor-pointer list-none text-[11px]">
                  <span className="font-bold text-zinc-200">{c.id}</span>
                  <span className="text-zinc-500 text-[10px]">
                    RRF {c.score.toFixed(4)}
                  </span>
                </summary>
                <div className="mt-1.5 pt-1.5 border-t border-zinc-900 text-[11px] text-zinc-400 font-sans leading-relaxed">
                  {c.text}
                </div>
              </details>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
