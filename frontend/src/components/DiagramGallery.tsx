import React, { useState } from 'react';
import { X, Search, ExternalLink } from 'lucide-react';
import type { DiagramDetail } from '../types';

interface DiagramGalleryProps {
  isOpen: boolean;
  onClose: () => void;
  diagrams: DiagramDetail[];
}

export const DiagramGallery: React.FC<DiagramGalleryProps> = ({
  isOpen,
  onClose,
  diagrams,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedUnit, setSelectedUnit] = useState<number | 'all'>('all');
  const [activePreview, setActivePreview] = useState<DiagramDetail | null>(null);

  if (!isOpen) return null;

  const filteredDiagrams = diagrams.filter((d) => {
    const matchesSearch =
      d.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      d.concept.toLowerCase().includes(searchTerm.toLowerCase()) ||
      d.caption.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesUnit = selectedUnit === 'all' || d.unit === selectedUnit;
    return matchesSearch && matchesUnit;
  });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-xs p-4 font-mono text-xs text-zinc-900">
      <div className="relative w-full max-w-5xl h-[85vh] bg-white border border-zinc-300 rounded-lg flex flex-col overflow-hidden shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-zinc-200 bg-zinc-50">
          <div>
            <h2 className="text-xs font-bold text-zinc-900 uppercase tracking-wider">
              Diagram Registry (21)
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded text-zinc-500 hover:text-zinc-900 hover:bg-zinc-200 cursor-pointer"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap items-center gap-2 px-5 py-2.5 border-b border-zinc-200 bg-white">
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-2.5 top-2 h-3.5 w-3.5 text-zinc-400" />
            <input
              type="text"
              placeholder="Search..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-8 pr-3 py-1 text-xs bg-zinc-50 border border-zinc-200 rounded text-zinc-900 placeholder-zinc-400 focus:outline-none focus:border-zinc-500"
            />
          </div>
          <div className="flex items-center gap-1">
            {(['all', 1, 2, 3, 4, 5] as const).map((unit) => (
              <button
                key={unit}
                onClick={() => setSelectedUnit(unit)}
                className={`px-2 py-0.5 text-[11px] rounded border transition-colors cursor-pointer ${
                  selectedUnit === unit
                    ? 'bg-zinc-900 text-white border-zinc-900 font-bold'
                    : 'bg-zinc-50 text-zinc-700 border-zinc-200 hover:bg-zinc-100'
                }`}
              >
                {unit === 'all' ? 'All' : `U${unit}`}
              </button>
            ))}
          </div>
        </div>

        {/* Grid */}
        <div className="flex-1 overflow-y-auto p-5 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
          {filteredDiagrams.map((diag) => (
            <div
              key={diag.id}
              onClick={() => setActivePreview(diag)}
              className="cursor-pointer flex flex-col bg-white border border-zinc-200 hover:border-zinc-400 rounded overflow-hidden shadow-xs hover:shadow-sm"
            >
              <div className="relative aspect-video w-full bg-zinc-50 flex items-center justify-center p-2 border-b border-zinc-200">
                <img
                  src={diag.url}
                  alt={diag.concept}
                  className="max-h-full max-w-full object-contain"
                  loading="lazy"
                />
                <div className="absolute top-1.5 left-1.5 px-1 py-0.5 rounded bg-white/90 border border-zinc-200 text-[9px] text-zinc-600 font-semibold">
                  U{diag.unit}
                </div>
              </div>
              <div className="p-2.5 space-y-1">
                <div className="text-[11px] font-bold text-zinc-900 truncate">
                  {diag.id}
                </div>
                <div className="text-[11px] text-zinc-600 truncate">
                  {diag.concept}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Modal Preview */}
        {activePreview && (
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
            <div className="relative max-w-2xl w-full bg-white border border-zinc-300 rounded-lg overflow-hidden flex flex-col shadow-2xl">
              <div className="flex items-center justify-between px-4 py-2 border-b border-zinc-200 bg-zinc-50">
                <span className="text-xs font-bold text-zinc-900">
                  {activePreview.id} • {activePreview.concept}
                </span>
                <button
                  onClick={() => setActivePreview(null)}
                  className="p-1 rounded text-zinc-500 hover:text-zinc-900 hover:bg-zinc-200 cursor-pointer"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              <div className="p-4 flex items-center justify-center bg-zinc-100">
                <img
                  src={activePreview.url}
                  alt={activePreview.concept}
                  className="max-h-[50vh] max-w-full object-contain rounded"
                />
              </div>
              <div className="p-3 border-t border-zinc-200 bg-white flex items-center justify-between">
                <span className="text-[11px] text-zinc-600">{activePreview.caption}</span>
                <a
                  href={activePreview.url}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-1 text-[11px] text-zinc-900 font-semibold hover:underline"
                >
                  <span>Full Asset</span>
                  <ExternalLink className="h-3 w-3" />
                </a>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
