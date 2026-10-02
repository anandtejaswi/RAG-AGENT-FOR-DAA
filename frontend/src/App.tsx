import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { AssistantChat } from './components/AssistantChat';
import { RetrievalInspector } from './components/RetrievalInspector';
import { DiagramGallery } from './components/DiagramGallery';
import type { DiagramDetail, MessageMetadata } from './types';
import { SidebarClose, SidebarOpen } from 'lucide-react';

export const App: React.FC = () => {
  const [serverOnline, setServerOnline] = useState(false);
  const [modelLabel, setModelLabel] = useState('');
  const [diagrams, setDiagrams] = useState<DiagramDetail[]>([]);
  const [isDiagramsOpen, setIsDiagramsOpen] = useState(false);
  const [isInspectorOpen, setIsInspectorOpen] = useState(true);
  const [activeMetadata, setActiveMetadata] = useState<MessageMetadata | undefined>();
  const [activeTopic, setActiveTopic] = useState<string | null>(null);

  // Poll server health and fetch diagram list
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const healthRes = await fetch('/api/health');
        if (healthRes.ok) {
          const data = await healthRes.json();
          setServerOnline(true);
          setModelLabel(data.model || 'Agent Server Online');
        } else {
          setServerOnline(false);
        }
      } catch (err) {
        setServerOnline(false);
      }
    };

    const fetchDiagrams = async () => {
      try {
        const diagRes = await fetch('/api/diagrams');
        if (diagRes.ok) {
          const data = await diagRes.json();
          setDiagrams(data.diagrams || []);
        }
      } catch (err) {
        console.error('Failed to load diagrams:', err);
      }
    };

    fetchStatus();
    fetchDiagrams();
    const interval = setInterval(fetchStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-slate-950 text-slate-100 antialiased font-sans">
      <Navbar
        serverOnline={serverOnline}
        modelLabel={modelLabel}
        onOpenDiagrams={() => setIsDiagramsOpen(true)}
        activeTopic={activeTopic}
      />

      <div className="flex-1 flex overflow-hidden relative">
        {/* Main Assistant Chat View */}
        <main className="flex-1 h-full overflow-hidden flex flex-col">
          <AssistantChat
            onMetadataChange={(meta) => setActiveMetadata(meta)}
            onTopicDetected={(topic) => setActiveTopic(topic || null)}
          />
        </main>

        {/* Toggle Inspector Button */}
        <button
          onClick={() => setIsInspectorOpen(!isInspectorOpen)}
          className="absolute right-3 top-3 z-30 p-1.5 rounded-lg bg-slate-800/90 border border-slate-700 text-slate-400 hover:text-slate-100 hover:bg-slate-700 shadow-md transition-colors"
          title={isInspectorOpen ? 'Hide Telemetry Panel' : 'Show Telemetry Panel'}
        >
          {isInspectorOpen ? <SidebarClose className="h-4 w-4" /> : <SidebarOpen className="h-4 w-4" />}
        </button>

        {/* Right Sidebar: Retrieval & Solver Inspector */}
        {isInspectorOpen && (
          <aside className="w-80 lg:w-96 border-l border-slate-800 bg-slate-900/95 flex flex-col h-full shrink-0 shadow-2xl animate-in slide-in-from-right duration-200">
            <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300">Retrieval & Solvers</span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                Live Audit
              </span>
            </div>
            <div className="flex-1 overflow-y-auto">
              <RetrievalInspector metadata={activeMetadata} />
            </div>
          </aside>
        )}
      </div>

      {/* Diagrams Gallery Modal */}
      <DiagramGallery
        isOpen={isDiagramsOpen}
        onClose={() => setIsDiagramsOpen(false)}
        diagrams={diagrams}
      />
    </div>
  );
};

export default App;
