import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import type { DiagramDetail } from '../types';
import { Image as ImageIcon, ExternalLink, Calculator, BookOpen } from 'lucide-react';

interface MessageRendererProps {
  content: string;
  diagrams?: DiagramDetail[];
}

export const MessageRenderer: React.FC<MessageRendererProps> = ({ content, diagrams = [] }) => {

  return (
    <div className="prose prose-invert prose-slate max-w-none text-sm leading-relaxed">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          h1: ({ children }) => (
            <h1 className="text-lg font-bold text-slate-100 border-b border-slate-800 pb-1.5 mt-4 mb-2">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-base font-semibold text-cyan-300 border-b border-slate-800/80 pb-1 mt-3 mb-2">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-sm font-semibold text-indigo-300 mt-2.5 mb-1.5">{children}</h3>
          ),
          p: ({ children }) => <p className="mb-2.5 last:mb-0 text-slate-200">{children}</p>,
          ul: ({ children }) => <ul className="list-disc list-inside space-y-1 mb-2.5 text-slate-300">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal list-inside space-y-1 mb-2.5 text-slate-300">{children}</ol>,
          li: ({ children }) => <li className="text-slate-300">{children}</li>,
          strong: ({ children }) => <strong className="font-semibold text-slate-100">{children}</strong>,
          code: ({ className, children, ...props }) => {
            const isInline = !className;
            const codeString = String(children);

            // Check if code contains chunk citation format e.g. C0042
            if (isInline && /^C\d{4}$/.test(codeString.trim())) {
              return (
                <span className="inline-flex items-center gap-1 font-mono text-[11px] px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                  <BookOpen className="h-3 w-3" />
                  {children}
                </span>
              );
            }

            if (isInline) {
              return (
                <code className="px-1.5 py-0.5 rounded font-mono text-xs bg-slate-800 text-cyan-300 border border-slate-700/60" {...props}>
                  {children}
                </code>
              );
            }

            return (
              <div className="relative my-3 rounded-xl overflow-hidden bg-slate-950 border border-slate-800 shadow-inner">
                <div className="px-4 py-1.5 bg-slate-900 border-b border-slate-800 flex items-center justify-between text-[11px] font-mono text-slate-400">
                  <div className="flex items-center gap-1.5">
                    <Calculator className="h-3.5 w-3.5 text-indigo-400" />
                    <span>Computation & Trace</span>
                  </div>
                </div>
                <pre className="p-3.5 text-xs font-mono text-slate-200 overflow-x-auto leading-normal">
                  <code>{children}</code>
                </pre>
              </div>
            );
          },
          table: ({ children }) => (
            <div className="my-3 overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/60">
              <table className="min-w-full divide-y divide-slate-800 text-xs text-left">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-slate-900/90 text-slate-300 font-semibold">{children}</thead>,
          th: ({ children }) => <th className="px-3 py-2 text-slate-200 font-semibold">{children}</th>,
          td: ({ children }) => <td className="px-3 py-2 border-t border-slate-800/80 text-slate-300">{children}</td>,
        }}
      >
        {content}
      </ReactMarkdown>

      {/* Render embedded diagram figures if mentioned */}
      {diagrams.length > 0 && (
        <div className="mt-4 pt-3 border-t border-slate-800/80 space-y-3">
          <div className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
            <ImageIcon className="h-4 w-4 text-cyan-400" />
            <span>Referenced Visual Schematic</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {diagrams.map((d) => (
              <div
                key={d.id}
                className="group relative rounded-xl border border-slate-800 bg-slate-950/80 overflow-hidden shadow-md hover:border-indigo-500/50 transition-colors"
              >
                <div className="aspect-video w-full bg-slate-900 flex items-center justify-center p-2">
                  <img src={d.url} alt={d.concept} className="max-h-full max-w-full object-contain" />
                </div>
                <div className="p-3 border-t border-slate-800/80">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-mono font-bold text-indigo-400">{d.id}</span>
                    <a
                      href={d.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-slate-400 hover:text-indigo-400"
                      title="Open full image"
                    >
                      <ExternalLink className="h-3.5 w-3.5" />
                    </a>
                  </div>
                  <div className="text-xs font-medium text-slate-200 mt-0.5">{d.concept}</div>
                  <div className="text-[11px] text-slate-400 mt-0.5 italic">{d.caption}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
