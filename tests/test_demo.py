"""Exercise the real graph, FastAPI lifespan and MCP subprocess without paid LLM calls."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from openai import LengthFinishReasonError

import main
from graph import Review, Route, Text, ask_model, build_graph
from mcp_server import get_patient, search_umls


class ScriptedModel:
    def __init__(self, first="patient_worker", verdicts=(True,)):
        self.first = first
        self.verdicts = iter(verdicts)
        self.calls = []

    def with_structured_output(self, schema, **kwargs):
        async def invoke(messages):
            payload = json.loads(messages[1][1])
            self.calls.append((schema, payload))
            if schema is Route:
                return Route(next_worker=self.first)
            if schema is Review:
                return Review(approved=next(self.verdicts), feedback="Use only recorded facts.")
            return Text(text="Educational fixture answer.")
        return SimpleNamespace(ainvoke=invoke)


@pytest.fixture
def client_factory(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-no-network")
    monkeypatch.setenv("MAIN_MODEL", "test-main")
    monkeypatch.setenv("CRITIC_MODEL", "test-critic")

    def create(first="patient_worker", verdicts=(True,)):
        writer, reviewer = ScriptedModel(first), ScriptedModel(verdicts=verdicts)
        monkeypatch.setattr(main, "ChatOpenAI", lambda model, **kwargs:
                            writer if model == "test-main" else reviewer)
        return TestClient(main.app), writer, reviewer
    return create


@pytest.mark.parametrize("first", ["patient_worker", "terminology_worker"])
def test_both_routes_use_real_mcp_and_separate_models(client_factory, first):
    client, writer, reviewer = client_factory(first)
    with client:
        result = client.post("/ask", json={"user_id": "demo-001", "question": "Explain asthma."})
    assert result.status_code == 200
    body = result.json()
    assert body["status"] == "approved" and body["drafts"] == 1
    assert body["trace"][0] == f"supervisor: dispatch {first}"
    assert len(body["trace"]) == 6
    assert body["evidence"]["patient"]["patient"]["known_conditions"] == ["asthma"]
    assert body["evidence"]["terminology"]["matches"][0]["id"] == "DEMO-001"
    assert [schema for schema, _ in writer.calls] == [Route, Text, Text]
    assert [schema for schema, _ in reviewer.calls] == [Text, Review]
    # Review sees source evidence, not just another model's summary.
    assert reviewer.calls[1][1]["patient"] == body["evidence"]["patient"]
    assert reviewer.calls[0][1]["retrieved_terminology"] == body["evidence"]["terminology"]


@pytest.mark.parametrize("verdicts, status", [((False, True), "approved"), ((False, False), "needs_review")])
def test_bounded_revision_and_feedback(client_factory, verdicts, status):
    client, writer, reviewer = client_factory(verdicts=verdicts)
    with client:
        body = client.post("/ask", json={"user_id": "demo-001", "question": "Explain asthma."}).json()
    assert body["status"] == status and body["drafts"] == 2
    assert (body["answer"] is None) == (status == "needs_review")
    assert [schema for schema, _ in reviewer.calls] == [Text, Review, Review]
    assert writer.calls[-1][1]["review"]["feedback"] == "Use only recorded facts."
    assert writer.calls[-1][1]["previous_draft"] == "Educational fixture answer."
    assert body["trace"].count("patient_worker: get_patient via MCP") == 1
    assert body["trace"].count("terminology_worker: search_umls via MCP + medical model context") == 1
    assert writer.calls[-1][1]["model_generated_medical_context"] == body["model_context"]["text"]


def test_missing_evidence_is_explicit(client_factory):
    client, _, _ = client_factory()
    with client:
        body = client.post("/ask", json={"user_id": "unknown", "question": "Explain migraine."}).json()
    assert body["evidence"]["patient"]["found"] is False
    assert body["evidence"]["patient"]["patient"] is None
    assert body["evidence"]["terminology"]["matches"] == []


@pytest.mark.parametrize("question", ["", "   ", "x" * 2001])
def test_invalid_input(client_factory, question):
    client, writer, reviewer = client_factory()
    with client:
        result = client.post("/ask", json={"user_id": "demo-001", "question": question})
    assert result.status_code == 422
    assert not writer.calls and not reviewer.calls


@pytest.mark.parametrize("error, status", [(RuntimeError("private upstream detail"), 502), (TimeoutError(), 504)])
def test_upstream_errors_are_not_answers(client_factory, error, status):
    client, _, _ = client_factory()

    async def fail(*args, **kwargs):
        raise error

    with client:
        main.app.state.graph = SimpleNamespace(ainvoke=fail)
        result = client.post("/ask", json={"user_id": "demo-001", "question": "Explain asthma."})
    assert result.status_code == status
    assert "private upstream detail" not in result.text


def test_tool_lookup_is_case_insensitive_and_matches_whole_terms():
    assert search_umls("What is HIGH BLOOD PRESSURE?")["matches"][0]["id"] == "DEMO-002"
    assert search_umls("asthmatic")["matches"] == []
    assert get_patient("../../.env")["found"] is False


def test_requests_do_not_share_graph_state():
    async def run():
        async def call_tool(name, arguments):
            await asyncio.sleep(0)
            return {"get_patient": get_patient, "search_umls": search_umls}[name](**arguments)

        graph = build_graph(ScriptedModel(), ScriptedModel(verdicts=(True, True)), call_tool)
        return await asyncio.gather(*[
            graph.ainvoke({"user_id": user_id, "question": question, "trace": [], "drafts": 0})
            for user_id, question in [("demo-001", "asthma"), ("demo-002", "hypertension")]
        ])

    first, second = asyncio.run(run())
    assert first["patient"]["patient"]["name"] == "Alex Demo"
    assert second["patient"]["patient"]["name"] == "Sam Example"
    assert first["terminology"]["matches"][0]["id"] == "DEMO-001"
    assert second["terminology"]["matches"][0]["id"] == "DEMO-002"
    assert len(first["trace"]) == len(second["trace"]) == 6


@pytest.mark.parametrize("verdicts, status", [((True,), "approved"), ((False, True), "approved"),
                                            ((False, False), "needs_review")])
def test_stream_runs_once_and_reports_reviews(client_factory, verdicts, status):
    client, writer, reviewer = client_factory(verdicts=verdicts)
    with client:
        result = client.post("/ask/stream", json={"user_id": "demo-001", "question": "Explain asthma."})
        assert client.get("/live").status_code == 200
        assert client.get("/diagram").status_code == 200
    events = [json.loads(line) for line in result.text.splitlines()]
    assert result.headers["content-type"].startswith("application/x-ndjson")
    assert events[0]["type"] == "started"
    assert events[-1]["type"] == "result" and events[-1]["status"] == status
    assert (events[-1]["answer"] is None) == (status == "needs_review")
    progress = [event for event in events if event["type"] == "progress"]
    for start, end in zip(progress[::2], progress[1::2], strict=True):
        assert start["status"] == "running" and start["node"] == end["node"]
        assert end["duration_ms"] >= 0
        assert "patient" not in end
        if end["node"] != "critic":
            assert "draft" not in end
    reviews = [event for event in progress if "feedback" in event]
    assert [event["status"] for event in reviews] == ["approved" if v else "rejected" for v in verdicts]
    assert [event["draft"] for event in reviews] == list(range(1, len(verdicts) + 1))
    assert len(reviewer.calls) == len(verdicts) + 1
    assert len(writer.calls) == 3 + len(verdicts) - 1
    assert events[-1]["evidence"]["patient"]["patient"]["known_conditions"] == ["asthma"]


@pytest.mark.parametrize("error", [RuntimeError("private upstream detail"), TimeoutError()])
def test_stream_errors_are_terminal_and_sanitized(client_factory, error):
    client, _, _ = client_factory()

    async def fail(*args, **kwargs):
        yield "custom", {"node": "supervisor", "status": "running"}
        raise error

    with client:
        main.app.state.graph = SimpleNamespace(astream=fail)
        result = client.post("/ask/stream", json={"user_id": "demo-001", "question": "Explain asthma."})
    events = [json.loads(line) for line in result.text.splitlines()]
    assert events[-1]["type"] == "error"
    assert not any(event["type"] == "result" for event in events)
    assert "private upstream detail" not in result.text


def test_stream_rejects_invalid_input_before_headers(client_factory):
    client, writer, reviewer = client_factory()
    with client:
        result = client.post("/ask/stream", json={"user_id": "demo-001", "question": " "})
    assert result.status_code == 422 and not writer.calls and not reviewer.calls


def test_running_event_precedes_model_completion_and_cancellation_stops_model():
    async def run():
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def blocked(messages):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        model = SimpleNamespace(with_structured_output=lambda schema: SimpleNamespace(ainvoke=blocked))
        graph = build_graph(model, model, None)
        stream = graph.astream({"user_id": "demo-001", "question": "asthma", "trace": [], "drafts": 0},
                               stream_mode="custom")
        async with asyncio.timeout(3):
            event = await anext(stream)
            assert event == {"node": "supervisor", "status": "running"}
            await entered.wait()
            await stream.aclose()
            await cancelled.wait()

    asyncio.run(run())


@pytest.mark.parametrize("first", ["patient_worker", "terminology_worker"])
@pytest.mark.parametrize("endpoint", ["/ask", "/ask/stream"])
def test_symptom_question_combines_model_context_with_json(client_factory, monkeypatch, first, endpoint):
    client, writer, reviewer = client_factory(first)
    context = "Model-generated general information: possible causes need assessment. How long have symptoms lasted?"
    medical_invoke = reviewer.with_structured_output
    writer_invoke = writer.with_structured_output

    def medical_response(schema, **kwargs):
        scripted = medical_invoke(schema)
        async def invoke(messages):
            result = await scripted.ainvoke(messages)
            return Text(text=context) if schema is Text else result
        return SimpleNamespace(ainvoke=invoke)

    def combined_response(schema, **kwargs):
        scripted = writer_invoke(schema)
        async def invoke(messages):
            result = await scripted.ainvoke(messages)
            payload = json.loads(messages[1][1])
            if "model_generated_medical_context" in payload:
                conditions = payload["patient"]["patient"]["known_conditions"]
                return Text(text=f"Recorded conditions: {', '.join(conditions)}. " +
                            payload["model_generated_medical_context"])
            return result
        return SimpleNamespace(ainvoke=invoke)

    monkeypatch.setattr(reviewer, "with_structured_output", medical_response)
    monkeypatch.setattr(writer, "with_structured_output", combined_response)
    question = "I have fever, headache, nausea what could be wrong"
    with client:
        response = client.post(endpoint, json={"user_id": "demo-001", "question": question})
    assert response.status_code == 200
    if endpoint.endswith("stream"):
        body = json.loads(response.text.splitlines()[-1])
    else:
        body = response.json()
    assert body["status"] == "approved"
    assert body["evidence"]["terminology"]["matches"] == []
    assert body["model_context"] == {"source": "model_generated", "text": context}
    assert "Recorded conditions: asthma." in body["answer"] and context in body["answer"]
    assert reviewer.calls[0][1]["question"] == question
    assert reviewer.calls[0][1]["retrieved_terminology"]["matches"] == []
    assert reviewer.calls[1][1]["model_generated_medical_context"] == context
    assert reviewer.calls[1][1]["patient"] == body["evidence"]["patient"]


@pytest.mark.parametrize("approved", [True, False])
def test_truncated_review_retries_once_without_changing_verdict(approved):
    messages_seen = []

    async def invoke(messages):
        messages_seen.append(messages)
        if len(messages_seen) == 1:
            raise LengthFinishReasonError(completion=SimpleNamespace(usage=None))
        return Review(approved=approved, feedback="Check the stated evidence.")

    model = SimpleNamespace(with_structured_output=lambda schema: SimpleNamespace(ainvoke=invoke))
    review = asyncio.run(ask_model(model, Review, "Review this draft.",
                                  {"draft": "Original draft", "patient": {"found": False}}, "critic"))
    assert review.approved is approved
    assert len(messages_seen) == 2
    assert messages_seen[0][1] == messages_seen[1][1]  # Exact same draft and evidence.


@pytest.mark.parametrize("endpoint", ["/ask", "/ask/stream"])
def test_repeated_review_truncation_is_an_error_not_approval(client_factory, monkeypatch, endpoint):
    client, _, reviewer = client_factory()
    original = reviewer.with_structured_output
    attempts = []

    def response(schema, **kwargs):
        if schema is not Review:
            return original(schema)
        async def invoke(messages):
            attempts.append(messages)
            raise LengthFinishReasonError(completion=SimpleNamespace(usage=None))
        return SimpleNamespace(ainvoke=invoke)

    monkeypatch.setattr(reviewer, "with_structured_output", response)
    with client:
        result = client.post(endpoint, json={"user_id": "demo-001", "question": "Explain asthma."})
    assert len(attempts) == 2
    if endpoint == "/ask":
        assert result.status_code == 502
        assert "output limit" in result.json()["detail"]
    else:
        events = [json.loads(line) for line in result.text.splitlines()]
        assert events[-1]["type"] == "error" and "output limit" in events[-1]["message"]
        assert not any(event["type"] == "result" for event in events)


def test_truncated_medical_context_does_not_use_review_retry():
    calls = []
    async def invoke(messages):
        calls.append(messages)
        raise LengthFinishReasonError(completion=SimpleNamespace(usage=None))
    model = SimpleNamespace(with_structured_output=lambda schema: SimpleNamespace(ainvoke=invoke))
    with pytest.raises(LengthFinishReasonError):
        asyncio.run(ask_model(model, Text, "Generate context.", {"question": "asthma"}, "medical context"))
    assert len(calls) == 1
