import React from 'react';
import { Image } from 'lucide-react';

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
    <header className="sticky top-0 z-40 w-full border-b border-zinc-200 bg-white px-4 lg:px-6 py-2.5 flex items-center justify-between text-xs text-zinc-900">
      <div className="flex items-center gap-3">
        <span className="font-mono font-bold text-zinc-900 tracking-tight text-sm">
          AKTU DAA
        </span>
        <span className="font-mono text-[11px] text-zinc-600 border border-zinc-200 bg-zinc-50 px-1.5 py-0.5 rounded">
          KCS-503
        </span>
        {activeTopic && (
          <span className="hidden sm:inline font-mono text-[11px] text-zinc-700 border border-zinc-200 px-2 py-0.5 rounded bg-zinc-100">
            {activeTopic}
          </span>
        )}
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={onOpenDiagrams}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-zinc-50 hover:bg-zinc-100 border border-zinc-200 text-zinc-800 transition-colors cursor-pointer"
        >
          <Image className="h-3.5 w-3.5 text-zinc-600" />
          <span>Diagrams (21)</span>
        </button>

        <div className="flex items-center gap-1.5 font-mono text-[11px] text-zinc-600 border border-zinc-200 px-2 py-1 rounded bg-zinc-50">
          <span
            className={`inline-block h-1.5 w-1.5 rounded-full ${
              serverOnline ? 'bg-zinc-900' : 'bg-zinc-400'
            }`}
          />
          <span className="truncate max-w-[180px]">
            {modelLabel ? modelLabel.split(' ')[0] : 'Connecting...'}
          </span>
        </div>
      </div>
    </header>
  );
};
