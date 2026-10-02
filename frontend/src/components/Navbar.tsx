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
    <header className="sticky top-0 z-40 w-full border-b border-zinc-800 bg-zinc-950 px-4 lg:px-6 py-2.5 flex items-center justify-between text-xs">
      <div className="flex items-center gap-3">
        <span className="font-mono font-bold text-zinc-100 tracking-tight text-sm">
          AKTU DAA
        </span>
        <span className="font-mono text-[11px] text-zinc-400 border border-zinc-800 px-1.5 py-0.5 rounded">
          KCS-503
        </span>
        {activeTopic && (
          <span className="hidden sm:inline font-mono text-[11px] text-zinc-400 border border-zinc-800 px-2 py-0.5 rounded bg-zinc-900">
            {activeTopic}
          </span>
        )}
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={onOpenDiagrams}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-200 transition-colors cursor-pointer"
        >
          <Image className="h-3.5 w-3.5 text-zinc-400" />
          <span>Diagrams (21)</span>
        </button>

        <div className="flex items-center gap-1.5 font-mono text-[11px] text-zinc-400 border border-zinc-800 px-2 py-1 rounded bg-zinc-900">
          <span
            className={`inline-block h-1.5 w-1.5 rounded-full ${
              serverOnline ? 'bg-zinc-100' : 'bg-zinc-600'
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
