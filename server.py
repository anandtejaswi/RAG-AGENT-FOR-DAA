"""FastAPI Agent Server for AKTU DAA RAG.

Serves streaming AI chat for assistant-ui frontend, static diagram assets,
topic routing, and deterministic algorithmic solvers.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag.answer import REFUSAL, build_context, _collect_tool_calls, _final_text, RECURSION_LIMIT, SYSTEM_PROMPT, _tools
from rag.config import chat_model, model_label
from rag.ingest import DATA, ROOT
from rag.retrieve import retrieve

DIAG_RE = re.compile(r"DIAG_DAA_[A-Z0-9_]+")

app = FastAPI(
    title="AKTU DAA RAG Agent Server",
    description="Examiner-grade agent server for AKTU Design & Analysis of Algorithms (KCS-503)",
    version="1.0.0",
)

# Enable CORS for frontend development servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount diagram images
DIAGRAMS_DIR = DATA / "diagrams"
if DIAGRAMS_DIR.exists():
    app.mount("/diagrams", StaticFiles(directory=str(DIAGRAMS_DIR)), name="diagrams")


class QueryRequest(BaseModel):
    query: str
    use_llm: bool = True


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    query: str | None = None
    use_llm: bool = True


@app.get("/api/health")
async def health():
    return {
        "status": "online",
        "service": "AKTU DAA Agent Server",
        "model": model_label(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/api/syllabus")
async def get_syllabus():
    syl_path = DATA / "syllabus.json"
    if syl_path.exists():
        return json.loads(syl_path.read_text())
    return {"units": []}


@app.get("/api/diagrams")
async def get_diagrams(request: Request):
    reg_path = DATA / "diagrams" / "diagrams.json"
    if not reg_path.exists():
        return {"diagrams": []}
    data = json.loads(reg_path.read_text())
    base_url = str(request.base_url).rstrip("/")
    for d in data.get("diagrams", []):
        d["url"] = f"{base_url}/diagrams/{d['file']}"
    return data


@app.post("/api/ask")
async def ask_endpoint(req: QueryRequest, request: Request):
    """Direct JSON endpoint returning the full AnswerRecord."""
    from rag.answer import ask
    record = ask(req.query, use_llm_classifier=req.use_llm)
    base_url = str(request.base_url).rstrip("/")
    
    # Attach absolute URLs to offered and emitted diagrams
    diag_reg = json.loads((DATA / "diagrams" / "diagrams.json").read_text())["diagrams"]
    diag_map = {d["id"]: d for d in diag_reg}
    
    record["diagram_details"] = [
        {
            "id": did,
            "concept": diag_map[did]["concept"],
            "caption": diag_map[did]["caption"],
            "url": f"{base_url}/diagrams/{diag_map[did]['file']}",
        }
        for did in record.get("diagram_ids", [])
        if did in diag_map
    ]
    return record


async def stream_generator(query: str, use_llm: bool, base_url: str) -> AsyncGenerator[str, None]:
    """Stream SSE events to assistant-ui."""
    started = time.time()
    
    # 1. Retrieval & Routing
    retrieval_res = retrieve(query, use_llm=use_llm)
    classification = retrieval_res["classification"]
    chunks = retrieval_res["chunks"]
    diagrams = retrieval_res["diagrams"]
    refuse = retrieval_res["refuse"]
    refusal_reason = retrieval_res.get("refusal_reason")

    diag_reg = json.loads((DATA / "diagrams" / "diagrams.json").read_text())["diagrams"]
    diag_map = {d["id"]: d for d in diag_reg}

    # Format offered diagram details with URLs
    offered_diagram_details = [
        {
            "id": d["id"],
            "concept": d["concept"],
            "caption": d["caption"],
            "url": f"{base_url}/diagrams/{d['file']}",
        }
        for d in diagrams
    ]

    meta_payload = {
        "type": "metadata",
        "query": query,
        "classification": classification,
        "chunks": [
            {
                "id": c.id,
                "unit": c.unit,
                "topic": c.topic,
                "heading": c.heading,
                "source": c.source,
                "score": c.score,
                "dense": c.dense,
                "bm25": c.bm25,
                "text": c.text,
            }
            for c in chunks
        ],
        "offered_diagrams": offered_diagram_details,
        "refused": refuse,
    }
    yield f"event: metadata\ndata: {json.dumps(meta_payload)}\n\n"

    # 2. Out of Scope Check
    if refuse:
        refusal_text = REFUSAL.format(reason=str(refusal_reason).capitalize())
        yield f"event: token\ndata: {json.dumps({'content': refusal_text})}\n\n"
        done_payload = {
            "type": "done",
            "answer": refusal_text,
            "refused": True,
            "citations": [],
            "diagrams": [],
            "latency_s": round(time.time() - started, 2),
        }
        yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
        return

    # 3. Agent Execution with Tool Support
    from langchain_core.messages import HumanMessage
    from deepagents import FilesystemMiddleware, create_deep_agent

    context = build_context(chunks, diagrams, classification.get("numeric_task"))
    message = f"{context}\n\nQUESTION\n{query}"

    agent_runner = create_deep_agent(
        model=chat_model(max_tokens=6000),
        tools=_tools(),
        system_prompt=SYSTEM_PROMPT,
        middleware=[FilesystemMiddleware(tools=["read_file"])],
    )

    try:
        # Run agent
        result = agent_runner.invoke(
            {"messages": [HumanMessage(content=message)]},
            config={"recursion_limit": RECURSION_LIMIT},
        )
        messages = result["messages"]
        tool_calls = _collect_tool_calls(messages)
        raw_answer = _final_text(messages)
    except Exception as exc:
        err_msg = f"Error during generation: {type(exc).__name__}: {exc}"
        yield f"event: error\ndata: {json.dumps({'error': err_msg})}\n\n"
        return

    # Emit tool calls if any took place
    if tool_calls:
        yield f"event: tool_calls\ndata: {json.dumps({'tool_calls': tool_calls})}\n\n"

    # Sanitize diagram references
    allowed = {d["id"] for d in diagrams}
    emitted = DIAG_RE.findall(raw_answer)
    invalid = sorted({d for d in emitted if d not in allowed})
    clean_answer = raw_answer
    for bad in invalid:
        clean_answer = clean_answer.replace(f"[Diagram: {bad}]", "").replace(bad, "")
    clean_answer = clean_answer.strip()

    valid_diagram_ids = sorted({d for d in emitted if d in allowed})
    valid_diagram_details = [
        {
            "id": did,
            "concept": diag_map[did]["concept"],
            "caption": diag_map[did]["caption"],
            "url": f"{base_url}/diagrams/{diag_map[did]['file']}",
        }
        for did in valid_diagram_ids
        if did in diag_map
    ]

    # Stream the final answer tokens
    chunk_size = 30
    for i in range(0, len(clean_answer), chunk_size):
        token_piece = clean_answer[i : i + chunk_size]
        yield f"event: token\ndata: {json.dumps({'content': token_piece})}\n\n"

    citations = sorted(set(re.findall(r"\[(C\d{4})\]", clean_answer)))

    done_payload = {
        "type": "done",
        "answer": clean_answer,
        "refused": False,
        "citations": citations,
        "diagrams": valid_diagram_details,
        "tool_calls": tool_calls,
        "latency_s": round(time.time() - started, 2),
    }
    yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest, request: Request):
    """Streaming chat endpoint for assistant-ui and SSE clients."""
    # Extract query from messages or explicit field
    query = req.query
    if not query and req.messages:
        query = req.messages[-1].content
    if not query:
        return JSONResponse(status_code=400, content={"error": "query or messages required"})

    base_url = str(request.base_url).rstrip("/")
    return StreamingResponse(
        stream_generator(query, req.use_llm, base_url),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting AKTU DAA Agent Server on http://localhost:{port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
