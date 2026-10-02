export interface Chunk {
  id: string;
  unit: number | null;
  topic: string | null;
  heading: string;
  source: string;
  score: number;
  dense: number;
  bm25: number;
  text: string;
}

export interface DiagramDetail {
  id: string;
  concept: string;
  caption: string;
  url: string;
  unit?: number;
  topic?: string;
  file?: string;
}

export interface ToolCall {
  name: string;
  args: Record<string, any>;
  result: any;
}

export interface MessageMetadata {
  classification?: {
    unit: number | null;
    topic: string | null;
    confidence: number;
    source?: string;
    needs_diagram?: boolean;
    numeric_task?: string | null;
  };
  chunks?: Chunk[];
  offered_diagrams?: DiagramDetail[];
  diagrams?: DiagramDetail[];
  tool_calls?: ToolCall[];
  citations?: string[];
  latency_s?: number;
  refused?: boolean;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  metadata?: MessageMetadata;
  isStreaming?: boolean;
}
