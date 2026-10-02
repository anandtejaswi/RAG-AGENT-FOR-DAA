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
          setModelLabel(data.model || 'Online');
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
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-zinc-950 text-zinc-100 antialiased font-sans">
      <Navbar
        serverOnline={serverOnline}
        modelLabel={modelLabel}
        onOpenDiagrams={() => setIsDiagramsOpen(true)}
        activeTopic={activeTopic}
      />

      <div className="flex-1 flex overflow-hidden relative">
        {/* Chat Area */}
        <main className="flex-1 h-full overflow-hidden flex flex-col">
          <AssistantChat
            onMetadataChange={(meta) => setActiveMetadata(meta)}
            onTopicDetected={(topic) => setActiveTopic(topic || null)}
          />
        </main>

        {/* Toggle Sidebar */}
        <button
          onClick={() => setIsInspectorOpen(!isInspectorOpen)}
          className="absolute right-3 top-3 z-30 p-1.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-zinc-100 transition-colors cursor-pointer"
          title={isInspectorOpen ? 'Hide Telemetry' : 'Show Telemetry'}
        >
          {isInspectorOpen ? <SidebarClose className="h-4 w-4" /> : <SidebarOpen className="h-4 w-4" />}
        </button>

        {/* Telemetry Sidebar */}
        {isInspectorOpen && (
          <aside className="w-72 lg:w-84 border-l border-zinc-800 bg-zinc-900 flex flex-col h-full shrink-0">
            <div className="px-4 py-2.5 border-b border-zinc-800 flex items-center justify-between text-xs font-mono">
              <span className="font-semibold text-zinc-300">Telemetry</span>
              <span className="text-[10px] text-zinc-500">Live</span>
            </div>
            <div className="flex-1 overflow-y-auto">
              <RetrievalInspector metadata={activeMetadata} />
            </div>
          </aside>
        )}
      </div>

      {/* Diagrams Modal */}
      <DiagramGallery
        isOpen={isDiagramsOpen}
        onClose={() => setIsDiagramsOpen(false)}
        diagrams={diagrams}
      />
    </div>
  );
};

export default App;
