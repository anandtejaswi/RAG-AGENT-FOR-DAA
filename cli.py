#!/usr/bin/env python
"""Command line entry point for the AKTU DAA RAG system.

  python cli.py ingest              rebuild the chunk index and embeddings
  python cli.py diagrams            regenerate diagram assets and the registry
  python cli.py ask "question"      answer one question
  python cli.py eval [ids|category] run the labelled test set
  python cli.py report              build the PDF evaluation report
  python cli.py serve [port]        start the FastAPI Agent Server (default: 8000)
  python cli.py ui [port]           start both Agent Server and assistant-ui Frontend
"""

from __future__ import annotations

import json
import sys


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]

    if cmd == "ingest":
        from rag.ingest import main as run
        run()
    elif cmd == "diagrams":
        from tools.make_diagrams import main as run
        run()
    elif cmd == "ask":
        if not rest:
            print("usage: python cli.py ask \"your question\"")
            return 2
        from rag.answer import ask
        rec = ask(" ".join(rest))
        c = rec["classification"]
        print(f"topic      : {c.get('topic')} (unit {c.get('unit')}, "
              f"confidence {c.get('confidence')}, via {c.get('source')})")
        print(f"retrieval  : {rec['retrieval']}")
        print("chunks     : " + ", ".join(
            f"{ch['id']}({ch['topic']},{ch['score']:.4f})" for ch in rec["chunks"]))
        if rec["tool_calls"]:
            print("tools      : " + ", ".join(t["name"] for t in rec["tool_calls"]))
        if rec["diagram_ids"]:
            print(f"diagrams   : {', '.join(rec['diagram_ids'])}")
        if rec["error"]:
            print(f"error      : {rec['error']}")
        print(f"latency    : {rec['latency_s']}s\n")
        print(rec["answer"])
    elif cmd == "eval":
        from rag.evaluate import main as run
        run(rest[0] if rest else None)
    elif cmd == "report":
        from rag.report import main as run
        run()
    elif cmd == "serve":
        import uvicorn
        port = int(rest[0]) if rest else 8000
        print(f"Starting AKTU DAA Agent Server on http://localhost:{port}")
        uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
    elif cmd == "ui":
        import subprocess
        import os
        port = int(rest[0]) if rest else 8000
        print(f"Starting Agent Server (port {port}) and assistant-ui Frontend (port 5173)...")
        server_proc = subprocess.Popen([sys.executable, "cli.py", "serve", str(port)])
        try:
            subprocess.run(["npm", "run", "dev"], cwd="frontend", check=True)
        finally:
            server_proc.terminate()
    else:
        print(f"unknown command: {cmd}")
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
