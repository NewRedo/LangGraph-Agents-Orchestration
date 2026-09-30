"""Run with uv run uvicorn main:app --reload. Try the API at /docs."""

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import LengthFinishReasonError
from pydantic import BaseModel, Field

from graph import build_graph

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
log = logging.getLogger("uvicorn.error")


class StructuredModel:
    """Each provider accepts a different structured-output method."""

    def __init__(self, model, method: str):
        self._model = model
        self._method = method

    def with_structured_output(self, schema, **kwargs):
        return self._model.with_structured_output(schema, method=self._method)


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Set DEEPSEEK_API_KEY in .env. See .env.example.")
    main_name = os.getenv("MAIN_MODEL", "deepseek-flash")
    critic_name = os.getenv("CRITIC_MODEL", "medgemma1.5")
    if main_name == critic_name:
        raise RuntimeError("Choose different MAIN_MODEL and CRITIC_MODEL values.")
    # DeepSeek rejects json_schema response_format. Tool calls work with thinking off.
    main_model = StructuredModel(ChatOpenAI(
        model=main_name,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key=api_key,
        temperature=0,
        timeout=60,
        max_retries=1,
        extra_body={"thinking": {"type": "disabled"}},
    ), "function_calling")
    # Ollama applies json_schema even for models without tool support.
    # LangChain renames max_tokens to max_completion_tokens, which Ollama ignores.
    # Send max_tokens in the raw body so a repeating review cannot fill the context.
    critic_model = StructuredModel(ChatOpenAI(
        model=critic_name,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
        api_key="ollama",
        temperature=0,
        timeout=60,
        max_retries=0,
        stop=["<end_of_turn>"],
        extra_body={
            "max_tokens": 640,
            "think": False,
            "options": {"num_ctx": 4096, "num_predict": 640, "repeat_penalty": 1.1},
        },
    ), "json_schema")
    
    server = StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp_server.py")])

    # One local MCP subprocess for the app's lifetime, no second terminal needed.
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=10)) as session:
            await session.initialize()

            async def call_tool(name, arguments):
                result = await session.call_tool(name, arguments)
                if result.isError or result.structuredContent is None:
                    raise RuntimeError(f"MCP tool failed: {name}")
                return result.structuredContent

            app.state.graph = build_graph(main_model, critic_model, call_tool)
            log.info("ready: main=%s critic=%s", main_name, critic_name)
            yield


app = FastAPI(title="Medical LangGraph demo", lifespan=lifespan)


class Question(BaseModel):
    user_id: str = Field(min_length=1, max_length=80, pattern=r"\S")
    question: str = Field(min_length=1, max_length=2000, pattern=r"\S")


class Answer(BaseModel):
    status: Literal["approved", "needs_review"]
    answer: str | None
    feedback: str
    drafts: int
    trace: list[str]
    evidence: dict
    model_context: dict
    notice: str = "Educational demo using fictional data. Not medical advice."


@app.post("/ask", response_model=Answer)
async def ask(body: Question, request: Request):
    preview = " ".join(body.question.split())
    if len(preview) > 80:
        preview = preview[:77] + "..."
    log.info("ask %s: %s", body.user_id, preview)
    started = time.perf_counter()
    try:
        state = await asyncio.wait_for(request.app.state.graph.ainvoke(
            {**body.model_dump(), "drafts": 0, "trace": []},
            config={"recursion_limit": 12},
        ), timeout=900)
    except TimeoutError as exc:
        log.error("ask %s: timed out after %.1fs", body.user_id, time.perf_counter() - started)
        raise HTTPException(504, "The demo timed out. Please retry.") from exc
    except LengthFinishReasonError as exc:
        log.exception("ask: model output limit reached")
        raise HTTPException(502, "A model response exceeded its output limit. Please retry.") from exc
    except Exception as exc:
        log.exception("ask %s: failed after %.1fs", body.user_id, time.perf_counter() - started)
        raise HTTPException(502, "A model or MCP call failed. Check your configuration.") from exc
    answer = answer_from_state(state)
    log.info("ask %s: %s, %s draft(s), %.1fs",
             body.user_id, answer.status, state["drafts"], time.perf_counter() - started)
    return answer


def answer_from_state(state):
    approved = state["review"]["approved"]
    status = "approved" if approved else "needs_review"
    return Answer(
        status=status,
        answer=state["draft"] if approved else None,
        feedback=state["review"]["feedback"].replace(chr(0x2014), "."),
        drafts=state["drafts"], trace=state["trace"],
        evidence={"patient": state["patient"], "terminology": state["terminology"]},
        model_context={"source": "model_generated", "text": state["medical_context"]},
    )


@app.get("/live", include_in_schema=False)
async def live():
    return FileResponse(ROOT / "docs" / "live.html")


@app.get("/diagram", include_in_schema=False)
async def diagram():
    return FileResponse(ROOT / "docs" / "orchestration.html")


@app.post("/ask/stream")
async def ask_stream(body: Question, request: Request):
    """Request-scoped NDJSON. After headers, failures are terminal error events."""
    async def events():
        started = time.perf_counter()

        def encode(event):
            return json.dumps({**event, "elapsed_ms": round(
                (time.perf_counter() - started) * 1000)}, ensure_ascii=False) + "\n"

        yield encode({"type": "started"})
        try:
            async with asyncio.timeout(900):
                async for mode, value in request.app.state.graph.astream(
                    {**body.model_dump(), "drafts": 0, "trace": []},
                    config={"recursion_limit": 12}, stream_mode=["custom", "values"],
                ):
                    if mode == "custom":
                        yield encode({"type": "progress", **value})
                    else:
                        state = value
                yield encode({"type": "result", **answer_from_state(state).model_dump()})
        except TimeoutError:
            yield encode({"type": "error", "message": "The demo timed out. Please retry."})
        except LengthFinishReasonError:
            log.exception("stream: model output limit reached")
            yield encode({"type": "error", "message": "A model response exceeded its output limit. Please retry."})
        except Exception:
            log.exception("stream: model or MCP failure")
            yield encode({"type": "error", "message": "A model or MCP call failed. Check your configuration."})

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
