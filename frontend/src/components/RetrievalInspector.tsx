import React from 'react';
import type { MessageMetadata } from '../types';
import { Database, FileText, Cpu, CheckCircle2 } from 'lucide-react';

interface RetrievalInspectorProps {
  metadata?: MessageMetadata;
}

export const RetrievalInspector: React.FC<RetrievalInspectorProps> = ({ metadata }) => {
  if (!metadata) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 text-center text-slate-500">
        <Database className="h-10 w-10 text-slate-700 mb-3" />
        <p className="text-xs font-medium text-slate-400">Retrieval & Solver Telemetry</p>
        <p className="text-[11px] text-slate-600 mt-1 max-w-[200px]">
          Ask a question to see real-time topic classification, retrieved passage scores, and solver traces.
        </p>
      </div>
    );
  }

  const { classification, chunks = [], tool_calls = [], diagrams = [], latency_s, refused } = metadata;

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4 text-xs">
      {/* Classification Card */}
      <div className="rounded-xl bg-slate-950/60 border border-slate-800 p-3.5 space-y-2">
        <div className="flex items-center justify-between">
          <span className="font-semibold text-slate-300 flex items-center gap-1.5">
            <Cpu className="h-4 w-4 text-indigo-400" />
            Topic Routing
          </span>
          <span
            className={`px-2 py-0.5 rounded text-[10px] font-mono ${
              refused
                ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
            }`}
          >
            {refused ? 'Refused / OOS' : 'In Scope'}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-2 pt-1">
          <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800/80">
            <div className="text-[10px] text-slate-500 uppercase tracking-wider">Unit & Topic</div>
            <div className="font-mono text-cyan-300 font-semibold truncate mt-0.5">
              {classification?.topic ? `U${classification.unit}: ${classification.topic}` : 'None (Off-syllabus)'}
            </div>
          </div>
          <div className="bg-slate-900/80 p-2 rounded-lg border border-slate-800/80">
            <div className="text-[10px] text-slate-500 uppercase tracking-wider">Confidence</div>
            <div className="font-mono text-emerald-400 font-semibold mt-0.5">
              {classification?.confidence ? `${(classification.confidence * 100).toFixed(1)}%` : '0%'}
            </div>
          </div>
        </div>

        {latency_s && (
          <div className="text-[11px] text-slate-500 font-mono flex justify-between pt-1">
            <span>Latency</span>
            <span>{latency_s}s</span>
          </div>
        )}
      </div>

      {/* Verified Solvers Card */}
      {tool_calls.length > 0 && (
        <div className="rounded-xl bg-slate-950/60 border border-indigo-500/30 p-3.5 space-y-2">
          <div className="flex items-center gap-1.5 font-semibold text-indigo-300">
            <CheckCircle2 className="h-4 w-4 text-indigo-400" />
            Verified Numeric Solvers ({tool_calls.length})
          </div>
          {tool_calls.map((t, idx) => (
            <div key={idx} className="bg-slate-900/90 rounded-lg p-2.5 border border-indigo-500/20 space-y-1">
              <div className="flex items-center justify-between text-[11px] font-mono text-indigo-300 font-bold">
                <span>{t.name}()</span>
                <span className="text-emerald-400 text-[10px]">Deterministic</span>
              </div>
              <pre className="text-[10px] font-mono bg-slate-950 p-2 rounded text-slate-300 overflow-x-auto">
                {typeof t.result === 'string' ? t.result : JSON.stringify(t.result, null, 2)}
              </pre>
            </div>
          ))}
        </div>
      )}

      {/* Diagram Reference Card */}
      {diagrams.length > 0 && (
        <div className="rounded-xl bg-slate-950/60 border border-slate-800 p-3.5 space-y-2">
          <div className="flex items-center gap-1.5 font-semibold text-slate-300">
            <FileText className="h-4 w-4 text-cyan-400" />
            Linked Diagram Assets ({diagrams.length})
          </div>
          <div className="space-y-2">
            {diagrams.map((d) => (
              <div key={d.id} className="bg-slate-900/80 rounded-lg p-2 border border-slate-800 flex gap-2.5">
                <img src={d.url} alt={d.concept} className="h-12 w-16 object-contain rounded bg-slate-950 p-1" />
                <div className="flex-1 min-w-0">
                  <div className="text-[10px] font-mono text-indigo-400 font-bold">{d.id}</div>
                  <div className="text-[11px] text-slate-200 truncate">{d.concept}</div>
                  <div className="text-[10px] text-slate-500 truncate">{d.caption}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Retrieved Passages Card */}
      {chunks.length > 0 && (
        <div className="rounded-xl bg-slate-950/60 border border-slate-800 p-3.5 space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-slate-300 flex items-center gap-1.5">
              <Database className="h-4 w-4 text-emerald-400" />
              Retrieved Passages ({chunks.length})
            </span>
          </div>

          <div className="space-y-2">
            {chunks.map((c) => (
              <details
                key={c.id}
                className="group bg-slate-900/80 rounded-lg border border-slate-800/80 p-2 text-slate-300 open:border-indigo-500/30"
              >
                <summary className="flex items-center justify-between cursor-pointer list-none font-mono text-[11px]">
                  <div className="flex items-center gap-1.5">
                    <span className="px-1.5 py-0.2 rounded bg-indigo-500/10 text-indigo-400 font-bold text-[10px]">
                      {c.id}
                    </span>
                    <span className="truncate max-w-[140px] text-slate-300">{c.heading}</span>
                  </div>
                  <span className="text-[10px] text-slate-500 font-mono">
                    RRF {c.score.toFixed(4)}
                  </span>
                </summary>
                <div className="mt-2 pt-2 border-t border-slate-800 text-[11px] text-slate-400 leading-relaxed font-sans">
                  {c.text}
                </div>
                <div className="mt-1.5 flex gap-2 text-[10px] text-slate-500 font-mono">
                  <span>Dense: {c.dense}</span>
                  <span>BM25: {c.bm25}</span>
                  <span>Src: {c.source}</span>
                </div>
              </details>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
