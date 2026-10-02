import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import type { DiagramDetail } from '../types';
import { ExternalLink } from 'lucide-react';

interface MessageRendererProps {
  content: string;
  diagrams?: DiagramDetail[];
}

export const MessageRenderer: React.FC<MessageRendererProps> = ({ content, diagrams = [] }) => {
  return (
    <div className="prose prose-invert prose-zinc max-w-none text-xs leading-relaxed text-zinc-200">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          h1: ({ children }) => (
            <h1 className="text-sm font-bold font-mono text-zinc-100 border-b border-zinc-800 pb-1 mt-3 mb-1.5">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-xs font-semibold font-mono text-zinc-100 border-b border-zinc-800 pb-1 mt-2.5 mb-1.5">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-xs font-semibold font-mono text-zinc-300 mt-2 mb-1">{children}</h3>
          ),
          p: ({ children }) => <p className="mb-2 last:mb-0 text-zinc-200">{children}</p>,
          ul: ({ children }) => <ul className="list-disc list-inside space-y-0.5 mb-2 text-zinc-300">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal list-inside space-y-0.5 mb-2 text-zinc-300">{children}</ol>,
          li: ({ children }) => <li className="text-zinc-300">{children}</li>,
          strong: ({ children }) => <strong className="font-semibold text-zinc-100">{children}</strong>,
          code: ({ className, children, ...props }) => {
            const isInline = !className;
            const codeString = String(children);

            // Chunk citation format
            if (isInline && /^C\d{4}$/.test(codeString.trim())) {
              return (
                <span className="font-mono text-[10px] px-1 py-0.5 rounded bg-zinc-800 text-zinc-300 border border-zinc-700">
                  {children}
                </span>
              );
            }

            if (isInline) {
              return (
                <code className="px-1 py-0.5 rounded font-mono text-[11px] bg-zinc-800 text-zinc-200 border border-zinc-700" {...props}>
                  {children}
                </code>
              );
            }

            return (
              <div className="my-2 rounded border border-zinc-800 bg-black overflow-hidden">
                <pre className="p-3 text-[11px] font-mono text-zinc-300 overflow-x-auto leading-normal">
                  <code>{children}</code>
                </pre>
              </div>
            );
          },
          table: ({ children }) => (
            <div className="my-2 overflow-x-auto rounded border border-zinc-800 bg-zinc-950">
              <table className="min-w-full divide-y divide-zinc-800 text-[11px] text-left">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-zinc-900 text-zinc-300 font-semibold">{children}</thead>,
          th: ({ children }) => <th className="px-2.5 py-1.5 text-zinc-200 font-mono font-semibold">{children}</th>,
          td: ({ children }) => <td className="px-2.5 py-1.5 border-t border-zinc-800 text-zinc-300 font-mono">{children}</td>,
        }}
      >
        {content}
      </ReactMarkdown>

      {/* Embedded Diagrams */}
      {diagrams.length > 0 && (
        <div className="mt-3 pt-2.5 border-t border-zinc-800 space-y-2">
          <div className="text-[11px] font-mono font-semibold text-zinc-300">
            Diagram References
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {diagrams.map((d) => (
              <div
                key={d.id}
                className="rounded border border-zinc-800 bg-black overflow-hidden"
              >
                <div className="aspect-video w-full bg-zinc-950 flex items-center justify-center p-2 border-b border-zinc-800">
                  <img src={d.url} alt={d.concept} className="max-h-full max-w-full object-contain" />
                </div>
                <div className="p-2.5">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono font-bold text-zinc-200">{d.id}</span>
                    <a
                      href={d.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-zinc-500 hover:text-zinc-200"
                    >
                      <ExternalLink className="h-3 w-3" />
                    </a>
                  </div>
                  <div className="text-[11px] text-zinc-300 mt-0.5">{d.concept}</div>
                  <div className="text-[10px] text-zinc-500 mt-0.5">{d.caption}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
