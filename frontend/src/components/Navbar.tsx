import React from 'react';
import { Image, Sparkles, Layers } from 'lucide-react';

interface NavbarProps {
  serverOnline: boolean;
  modelLabel: string;
  onOpenDiagrams: () => void;
  activeTopic?: string | null;
}

export const Navbar: React.FC<NavbarProps> = ({
  serverOnline,
  modelLabel,
  onOpenDiagrams,
  activeTopic,
}) => {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-800 bg-slate-900/90 backdrop-blur-md px-4 lg:px-8 py-3.5 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="h-9 w-9 rounded-xl bg-gradient-to-tr from-indigo-500 to-cyan-400 p-0.5 flex items-center justify-center shadow-lg shadow-indigo-500/20">
          <div className="h-full w-full bg-slate-950 rounded-[10px] flex items-center justify-center">
            <Sparkles className="h-4.5 w-4.5 text-cyan-400" />
          </div>
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-base font-semibold text-slate-100 tracking-tight">AKTU DAA Agent</h1>
            <span className="text-[11px] font-mono px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              KCS-503
            </span>
          </div>
          <p className="text-xs text-slate-400">Design & Analysis of Algorithms • Examiner-Grade RAG</p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {activeTopic && (
          <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-800 border border-slate-700 text-xs text-cyan-300 font-mono">
            <Layers className="h-3.5 w-3.5 text-cyan-400" />
            <span>{activeTopic}</span>
          </div>
        )}

        <button
          onClick={onOpenDiagrams}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs font-medium text-slate-200 transition-colors shadow-sm"
        >
          <Image className="h-4 w-4 text-indigo-400" />
          <span>Diagram Assets (21)</span>
        </button>

        <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-950/80 border border-slate-800 text-xs font-mono text-slate-400">
          <span className="flex h-2 w-2 relative">
            <span
              className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                serverOnline ? 'bg-emerald-400' : 'bg-rose-400'
              }`}
            />
            <span
              className={`relative inline-flex rounded-full h-2 w-2 ${
                serverOnline ? 'bg-emerald-500' : 'bg-rose-500'
              }`}
            />
          </span>
          <span className="truncate max-w-[200px]" title={modelLabel}>
            {modelLabel || 'Connecting...'}
          </span>
        </div>
      </div>
    </header>
  );
};
