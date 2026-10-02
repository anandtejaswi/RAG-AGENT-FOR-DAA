import React, { useState } from 'react';
import { X, Search, Image as ImageIcon, ExternalLink } from 'lucide-react';
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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
      <div className="relative w-full max-w-5xl h-[85vh] bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/90">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              <ImageIcon className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-slate-100">
                DAA Visual Schematic & Diagram Registry
              </h2>
              <p className="text-xs text-slate-400">
                21 Persistent Schematic Assets linked by persistent ID for examiner-grade derivations
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Filter bar */}
        <div className="flex flex-wrap items-center gap-3 px-6 py-3 border-b border-slate-800 bg-slate-950/50">
          <div className="relative flex-1 min-w-[240px]">
            <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-500" />
            <input
              type="text"
              placeholder="Search by concept, ID, or caption..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-4 py-1.5 text-xs bg-slate-900 border border-slate-700 rounded-lg text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
            />
          </div>
          <div className="flex items-center gap-1">
            {(['all', 1, 2, 3, 4, 5] as const).map((unit) => (
              <button
                key={unit}
                onClick={() => setSelectedUnit(unit)}
                className={`px-2.5 py-1 text-xs font-mono rounded-lg transition-colors ${
                  selectedUnit === unit
                    ? 'bg-indigo-600 text-white font-semibold'
                    : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200'
                }`}
              >
                {unit === 'all' ? 'All Units' : `Unit ${unit}`}
              </button>
            ))}
          </div>
        </div>

        {/* Gallery Grid */}
        <div className="flex-1 overflow-y-auto p-6 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
          {filteredDiagrams.map((diag) => (
            <div
              key={diag.id}
              onClick={() => setActivePreview(diag)}
              className="group cursor-pointer flex flex-col bg-slate-950/60 border border-slate-800 hover:border-indigo-500/60 rounded-xl overflow-hidden transition-all duration-200 hover:shadow-lg hover:shadow-indigo-500/10"
            >
              <div className="relative aspect-video w-full bg-slate-900 overflow-hidden flex items-center justify-center p-2">
                <img
                  src={diag.url}
                  alt={diag.concept}
                  className="max-h-full max-w-full object-contain group-hover:scale-105 transition-transform duration-200"
                  loading="lazy"
                />
                <div className="absolute top-2 left-2 px-1.5 py-0.5 rounded bg-slate-950/80 backdrop-blur-md border border-slate-800 text-[10px] font-mono text-cyan-400">
                  Unit {diag.unit}
                </div>
              </div>
              <div className="p-3.5 flex flex-col flex-1 justify-between gap-2 border-t border-slate-800/80">
                <div>
                  <div className="text-[11px] font-mono text-indigo-400 font-semibold truncate">
                    {diag.id}
                  </div>
                  <div className="text-xs font-medium text-slate-200 line-clamp-1">
                    {diag.concept}
                  </div>
                  <div className="text-[11px] text-slate-400 line-clamp-2 mt-0.5">
                    {diag.caption}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Single Image Preview Modal */}
        {activePreview && (
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-slate-950/90 backdrop-blur-md p-6">
            <div className="relative max-w-3xl w-full max-h-[90%] bg-slate-900 border border-slate-700 rounded-2xl overflow-hidden flex flex-col">
              <div className="flex items-center justify-between px-5 py-3 border-b border-slate-800">
                <div>
                  <span className="text-xs font-mono text-indigo-400 font-bold mr-2">
                    {activePreview.id}
                  </span>
                  <span className="text-sm font-semibold text-slate-100">
                    {activePreview.concept}
                  </span>
                </div>
                <button
                  onClick={() => setActivePreview(null)}
                  className="p-1.5 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
              <div className="flex-1 overflow-auto p-4 flex items-center justify-center bg-slate-950">
                <img
                  src={activePreview.url}
                  alt={activePreview.concept}
                  className="max-h-[60vh] max-w-full object-contain rounded-lg"
                />
              </div>
              <div className="p-4 border-t border-slate-800 bg-slate-900/90 flex items-center justify-between">
                <p className="text-xs text-slate-300 italic">{activePreview.caption}</p>
                <a
                  href={activePreview.url}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-1.5 text-xs text-indigo-400 hover:text-indigo-300"
                >
                  <span>Open Full Asset</span>
                  <ExternalLink className="h-3.5 w-3.5" />
                </a>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
