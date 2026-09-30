# Learning LangGraph. How to coordinate AI agents, tools, and decisions

Calling a language model is often the easiest part of building an AI application. The harder part begins when answering one question requires several actions. You may need to retrieve information, ask a specialist to interpret it, check the answer, and try again if something is missing.

At that point, you are designing a process. What happens first. What information survives between steps. Who decides what happens next. What makes the process stop.

LangGraph helps you express and run that process. This article introduces the ideas behind it, explains when they are useful, and develops a medical information example to make them concrete. Basic familiarity with Python functions and dictionaries is enough to follow along.

LangGraph is a library for building stateful workflows and agents. You describe work as a graph, with connected steps and information that moves through the execution. It supplies orchestration infrastructure, while your code supplies the actual work, including model calls, retrieval, and validation. It can be used independently of LangChain, although the two are often used together. See the [official LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview).

Here, orchestration means coordinating those steps. Suppose a reviewer rejects an answer. Your application needs to preserve the feedback, send the answer back for revision, and decide how many revisions to allow. Those are orchestration decisions.

A graph gives that behavior an explicit structure. A step is called a node. A connection between steps is an edge. The information available during execution is the state. Together, these let you describe both a straightforward sequence and a process that branches or returns to earlier work. These are the core building blocks in the [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api).

You can build the same control flow with ordinary Python functions, loops, and conditionals. For a short sequence, that is often sufficient. LangGraph becomes useful when the relationships between steps deserve their own structure, especially when different outcomes lead to different actions.

The practical benefits are easier to understand as engineering decisions.

- **Explicit control flow.** You can identify which transitions are allowed and which conditions trigger them.
- **Defined intermediate data.** You can decide what later steps need, instead of passing an increasingly large block of text between every call.
- **Review and revision.** You can make rejection lead to another attempt, with an explicit stopping rule.
- **Inspectable execution.** You can examine intermediate results to locate where a process went wrong.

LangGraph also supports capabilities such as checkpointing, resumable execution, and human intervention. Those require deliberate configuration and workflow design. Merely creating a graph does not give an application durable memory or a human approval system. The [overview describes these capabilities](https://docs.langchain.com/oss/python/langgraph/overview) separately from the work your application performs.

There is a tradeoff. A graph introduces concepts and structure that a single model call may not need. Start with the process you actually require. If a question can be answered reliably by one retrieval step and one model call, several agents may add cost and latency without solving a useful problem.

Some situations make orchestration more valuable.

- A support assistant retrieves account details, checks a refund policy, and requests approval before issuing a refund.
- A research assistant gathers sources, drafts an explanation, and checks whether its claims have supporting evidence.
- A coding assistant proposes a change, runs tests, and uses a failure report to guide a revision.

In each case, an intermediate result affects what should happen next. That is a useful reason to consider LangGraph.

![Agents Diagram](../agents-diagram.png)

Before building an example, it helps to distinguish a workflow from an agent. A workflow follows routes defined by the developer. An agent uses a model to make some decisions about its actions. A system can combine both. You might let a model choose a specialist, while requiring an ordinary Python condition to enforce a retry limit. LangGraph supports both styles, as explained in its [workflows and agents guide](https://docs.langchain.com/oss/python/langgraph/workflows-agents).

Now consider an educational medical assistant receiving this question.

> What does asthma mean, and is it in my known conditions?

The question has two parts. Explaining a term requires terminology information. Answering whether it appears in the person's record requires patient information. These are different sources of evidence. Finding the word asthma in a terminology database does not establish that the person has asthma.

For learning purposes, imagine two small JSON files. One contains fictional patient profiles and known conditions. The other contains a few synthetic terminology definitions that mimic a tiny part of a UMLS lookup. These are teaching records, not real patient data or an official UMLS dataset.

We can divide the work into four roles.

| Role | Responsibility | Result |
| --- | --- | --- |
| Supervisor | Coordinate retrieval, combine findings, and revise the answer | A routing decision or a draft |
| Patient worker | Retrieve and summarize the fictional patient record | Recorded patient evidence |
| Medical knowledge worker (`terminology_worker`) | Retrieve matching terminology and use MedGemma for general medical information | Retrieved definitions plus separately labeled model-generated context |
| Critic | Compare the draft with the question, retrieved evidence, and model context | Approval or actionable feedback |

These roles are an architectural choice. LangGraph does not require a supervisor or a critic. This arrangement is useful for learning because it separates coordination, evidence gathering, and review.

Start with the information these roles need to share. The supervisor needs to know which workers have finished. The critic needs the draft and its supporting evidence. A revision needs the previous draft and the latest feedback.

In Python, we can describe that shared state with a `TypedDict`. This excerpt focuses on the fields needed to understand the example.

```python
from typing import TypedDict


class State(TypedDict, total=False):
    user_id: str
    question: str
    patient: dict
    terminology: dict
    patient_summary: str
    medical_context: str
    next_node: str
    draft: str
    drafts: int
    review: dict
```

At the beginning, only the user ID, question, and initial draft count need values. After the patient worker finishes, `patient` and `patient_summary` become available. After drafting, `draft` contains the candidate answer. The shape of the state tells us what information exists at each stage.

Notice the distinction between raw evidence and a summary. A summary is another model's interpretation. Preserving the original tool result gives the supervisor and critic something to check that interpretation against. If a summary accidentally adds a condition, the raw record makes the discrepancy visible.

State also differs from a model's prompt. Storing a field does not automatically send it to every model. Each node chooses which state fields to include in its call. In this example, the terminology worker needs the question and matching concepts. It does not need the patient's full profile to explain a retrieved term.

A node performs one step and returns an update. For example, a patient worker can return this shape after retrieving and summarizing a record.

```python
return {
    "patient": evidence,
    "patient_summary": summary,
}
```

LangGraph applies the returned fields to the state. By default, a new value replaces the previous value for that field. You can configure a reducer when values should be combined instead, such as appending execution events to a trace. See the [state and reducer documentation](https://docs.langchain.com/oss/python/langgraph/graph-api).

A node does not have to call a language model. It could read a file, validate an identifier, or perform a calculation. Registering a function as a node simply makes it a step in the graph. In our medical example, the workers use tools for retrieval and a model for summarization.

This is where tools fit into the picture. The patient worker has `get_patient(user_id)`. The terminology worker has `search_umls(query)`. A tool performs an operation and returns information. The worker uses that information to carry out its assigned responsibility.

The example exposes these operations through Model Context Protocol, or MCP. MCP supplies a standard interface between a client and a tool server. LangGraph coordinates when a worker runs. MCP carries the worker's tool request and result. LangGraph can also coordinate ordinary Python functions, so MCP is an integration choice rather than a requirement. The [official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) provides server and client implementations.

For a first version, fixed tool calls are easy to reason about. Python makes the patient worker call `get_patient` with the request's user ID. The model summarizes the result. It does not need to choose from a large catalog of tools. You can introduce model-directed tool selection later if the task actually needs it.

With responsibilities defined, the next question is who runs next. This is the job of edges.

A normal edge describes an unconditional handoff. Once the patient worker finishes, control returns to the supervisor.

```python
graph.add_edge("patient_worker", "supervisor")
```

A conditional edge selects a destination using a routing function. In this example, the supervisor writes its decision into `next_node`, and the graph uses that value to choose the next step.

```python
graph.add_conditional_edges(
    "supervisor",
    lambda state: state["next_node"],
    ["patient_worker", "terminology_worker", "critic"],
)
```

The model's routing response should have a small, validated set of possible values. For the first decision, it can select either worker. Once one worker has finished, code dispatches the remaining one. After both have finished, the supervisor writes a draft and routes to the critic.

That division makes the model's freedom precise. It can choose which retrieval happens first. It cannot skip required evidence, invent a destination, or repeatedly dispatch a worker that has already completed. These rules belong in code because they are requirements of the process.

For the question about asthma, a possible execution is straightforward.

1. The supervisor chooses the patient worker.
2. The patient worker retrieves the fictional profile and reports whether asthma is recorded.
3. The supervisor dispatches the terminology worker.
4. The terminology worker retrieves a matching definition and uses MedGemma to explain it and add general medical context.
5. The supervisor combines both results into an answer.
6. The critic checks that answer against the evidence.

The supervisor could choose the terminology worker first. Both orders still satisfy the same requirement, both evidence sources must be consulted before drafting.

These workers run sequentially in the teaching example. They could run concurrently because neither needs the other's result, but the graph would then need to wait for both before drafting. Concurrency is a separate design choice. It is not what makes a system an agent workflow.

Review becomes more useful when its result changes execution. Suppose a draft says the fictional patient has severe asthma, but the profile only records asthma and gives no severity. The critic can reject that unsupported detail and ask for a revision grounded in the record.

A structured review makes the next action clear.

```json
{
  "approved": false,
  "feedback": "The record does not specify severity. Remove that claim."
}
```

This JSON is an illustrative review, not a recorded model response. The useful part is its contract. The supervisor receives both the rejected draft and a specific correction. It can revise without retrieving the same unchanged records again.

A review loop also needs a limit. Two models can disagree indefinitely, and another attempt does not guarantee a better answer. For this example, allow one initial draft and one revision.

```python
from langgraph.graph import END

MAX_DRAFTS = 2


def after_review(state):
    if state["review"]["approved"]:
        return END
    if state["drafts"] >= MAX_DRAFTS:
        return END
    return "supervisor"
```

Both approval and an exhausted attempt limit end execution. They should produce different outcomes. An approved draft can be returned as the answer. Two rejected drafts should produce `needs_review` and no approved answer. A completed process does not necessarily mean a successful result.

Four roles do not require four distinct models. In this version, the supervisor and patient worker use the main model. The medical knowledge worker and critic use MedGemma in separate calls. Medical context is stored separately from tool evidence and is included in both drafting and review. Model choice and graph structure are separate decisions.

For symptom-only questions, a condition-name lookup can return no matches. MedGemma may still supply general medical information, cautious possible causes, follow-up questions, and relevant warning signs. Empty retrieval stays empty. The supervisor must distinguish recorded history from model suggestions and must not present possibilities as a confirmed diagnosis. Review allows this general information while checking for invented patient facts and unsupported certainty. These are prompt instructions, not a deterministic medical validation system.

Using a second model does not prove correctness or guarantee independent judgment. In this medical example, critic approval is an automated review result, not clinical validation. Its engineering value comes from making checks explicit and connecting a failed check to a defined response.

We can now connect the pieces. The following excerpt assumes the state definition and four node functions have already been defined. It shows the orchestration structure, not a complete standalone application.

```python
from langgraph.graph import END, START, StateGraph

builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
builder.add_node("patient_worker", patient_worker)
builder.add_node("terminology_worker", terminology_worker)
builder.add_node("critic", critic)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next_node"],
    ["patient_worker", "terminology_worker", "critic"],
)
builder.add_edge("patient_worker", "supervisor")
builder.add_edge("terminology_worker", "supervisor")
builder.add_conditional_edges(
    "critic", after_review, ["supervisor", END]
)

graph = builder.compile()
```

`START` marks the entry into execution. `END` marks a terminal path. Calling `compile()` prepares the defined graph to run. An async application can then start a request with initial state.

```python
result = await graph.ainvoke({
    "user_id": "demo-001",
    "question": "What does asthma mean, and is it in my known conditions?",
    "drafts": 0,
})
```

A web framework such as FastAPI can accept the HTTP request and invoke this graph. Its responsibility is the API boundary. The graph's responsibility is coordinating the work inside that request. Keeping those responsibilities distinct makes it possible to test the orchestration without starting a web server.

Testing should examine the paths through the process as well as the final text. For this example, useful checks include whether both workers run before drafting, whether a rejection reaches the supervisor with feedback, and whether a second rejection stops the loop. Scripted model responses make those paths repeatable. Separate evaluations with real models assess whether summaries and reviews are useful.

Missing evidence deserves its own test. An unknown patient record should remain unknown. An empty terminology result should mean the small lookup found nothing. Neither result should become a confident claim about the person's health. Explicit state makes these cases easier to represent, but your prompts, validation, and tests still have to handle them.

When designing your own graph, write down the responsibilities before choosing agent names. Identify what each step receives, what it produces, who can act on that result, and what ends execution. Then decide which decisions need a model and which can be expressed directly in Python.

In the medical example, those questions lead to two sources of evidence, a coordinator that combines them, and a reviewer that can request one revision. In another domain, the same ideas might coordinate a policy check, a test runner, or a human approval step. LangGraph gives those decisions an executable structure. The quality of the process still depends on the responsibilities, evidence, and limits you design.

For a concrete companion to these concepts, explore the [medical workflow diagram](orchestration.html) and the [small Python example](../README.md). Use them to connect each concept to code after the responsibilities and handoffs make sense.
