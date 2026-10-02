import React from 'react';
import type { MessageMetadata } from '../types';

interface RetrievalInspectorProps {
  metadata?: MessageMetadata;
}

export const RetrievalInspector: React.FC<RetrievalInspectorProps> = ({ metadata }) => {
  if (!metadata) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 text-center text-zinc-400 font-mono text-xs">
        <p>No active query telemetry</p>
      </div>
    );
  }

  const { classification, chunks = [], tool_calls = [], diagrams = [], latency_s, refused } = metadata;

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4 text-xs font-mono text-zinc-900">
      {/* Classification */}
      <div className="rounded border border-zinc-200 bg-white p-3 space-y-2 shadow-xs">
        <div className="flex items-center justify-between text-zinc-900 font-semibold">
          <span>Routing</span>
          <span className="text-[10px] text-zinc-700 border border-zinc-200 bg-zinc-50 px-1.5 py-0.5 rounded">
            {refused ? 'Refused' : 'Accepted'}
          </span>
        </div>

        <div className="space-y-1 text-zinc-600 text-[11px]">
          <div className="flex justify-between">
            <span className="text-zinc-500">Topic:</span>
            <span className="text-zinc-900 font-medium">
              {classification?.topic ? `U${classification.unit}: ${classification.topic}` : 'None'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-zinc-500">Confidence:</span>
            <span className="text-zinc-900 font-medium">
              {classification?.confidence ? `${(classification.confidence * 100).toFixed(1)}%` : '0%'}
            </span>
          </div>
          {latency_s && (
            <div className="flex justify-between">
              <span className="text-zinc-500">Latency:</span>
              <span className="text-zinc-900 font-medium">{latency_s}s</span>
            </div>
          )}
        </div>
      </div>

      {/* Solver Tool Calls */}
      {tool_calls.length > 0 && (
        <div className="rounded border border-zinc-200 bg-white p-3 space-y-2 shadow-xs">
          <div className="font-semibold text-zinc-900">
            Solvers ({tool_calls.length})
          </div>
          {tool_calls.map((t, idx) => (
            <div key={idx} className="border border-zinc-200 rounded p-2 bg-zinc-50 space-y-1">
              <div className="text-[11px] text-zinc-900 font-bold">
                {t.name}()
              </div>
              <pre className="text-[10px] text-zinc-800 overflow-x-auto">
                {typeof t.result === 'string' ? t.result : JSON.stringify(t.result, null, 2)}
              </pre>
            </div>
          ))}
        </div>
      )}

      {/* Diagrams */}
      {diagrams.length > 0 && (
        <div className="rounded border border-zinc-200 bg-white p-3 space-y-2 shadow-xs">
          <div className="font-semibold text-zinc-900">
            Diagrams ({diagrams.length})
          </div>
          <div className="space-y-1.5">
            {diagrams.map((d) => (
              <div key={d.id} className="border border-zinc-200 rounded p-2 bg-zinc-50 flex gap-2">
                <img src={d.url} alt={d.concept} className="h-10 w-14 object-contain bg-white p-0.5 border border-zinc-200" />
                <div className="flex-1 min-w-0">
                  <div className="text-[10px] text-zinc-900 font-bold">{d.id}</div>
                  <div className="text-[11px] text-zinc-600 truncate">{d.concept}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Retrieved Chunks */}
      {chunks.length > 0 && (
        <div className="rounded border border-zinc-200 bg-white p-3 space-y-2 shadow-xs">
          <div className="font-semibold text-zinc-900">
            Retrieved Passages ({chunks.length})
          </div>

          <div className="space-y-1.5">
            {chunks.map((c) => (
              <details
                key={c.id}
                className="border border-zinc-200 rounded p-2 bg-zinc-50 text-zinc-800"
              >
                <summary className="flex items-center justify-between cursor-pointer list-none text-[11px]">
                  <span className="font-bold text-zinc-900">{c.id}</span>
                  <span className="text-zinc-500 text-[10px]">
                    RRF {c.score.toFixed(4)}
                  </span>
                </summary>
                <div className="mt-1.5 pt-1.5 border-t border-zinc-200 text-[11px] text-zinc-700 font-sans leading-relaxed">
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
