# LangGraph Notes

A summary of LangGraph concepts, patterns, and architectures covered while building agent projects — ReAct agents, Reflexion agents, RAG agents, human-in-the-loop workflows, and multi-agent systems.

---

## Table of Contents

1. [What is LangGraph](#what-is-langgraph)
2. [State in LangGraph](#state-in-langgraph)
3. [MessageGraph vs StateGraph](#messagegraph-vs-stategraph)
4. [The ReAct Pattern](#the-react-pattern)
5. [Reflexion Agents](#reflexion-agents)
6. [Tool Calling & Tool Execution](#tool-calling--tool-execution)
7. [Human-in-the-Loop](#human-in-the-loop)
8. [Multi-Agent Architecture](#multi-agent-architecture)
9. [Streaming](#streaming)
10. [RAG Agents (Retrieval-Augmented Generation)](#rag-agents-retrieval-augmented-generation)
11. [Checkpointing & Memory](#checkpointing--memory)
12. [Interview Questions](#interview-questions)

---

## What is LangGraph

LangGraph is a Python framework for building agents as a **graph** instead of a straight-line pipeline. Execution can loop, branch conditionally, and revisit earlier steps — unlike traditional linear chains that only move forward and terminate.

Core building blocks:
- **Node** — a step in the workflow (e.g. "call the LLM", "run a tool")
- **Edge** — the path connecting one node to the next
- **Conditional edge** — a path chosen dynamically based on the current state

LangGraph is the underlying execution engine; higher-level LangChain agent helpers (like `create_agent`) run on top of it.

## State in LangGraph

**State** is the data threaded through the whole graph. Every node reads the current state, does its work, and returns an update. LangGraph merges that update into the state before passing it to the next node — like a shared notebook passed along an assembly line.

- Defined with a `TypedDict` (or Pydantic model)
- Updates are controlled by **reducers** — e.g. `Annotated[list, add_messages]` tells LangGraph to *append* new messages instead of overwriting the list
- Without a reducer, returning a new value for a key **overwrites** the old one by default

## MessageGraph vs StateGraph

- **MessageGraph** — a simplified graph where state is just a list of messages. Deprecated as of LangGraph v1.0, to be removed in v2.0.
- **StateGraph** — the current, general-purpose graph. State can hold any fields you define, with `messages` as just one key among others (using `MessagesState` or a custom `TypedDict`).

## The ReAct Pattern

**ReAct** (Reasoning + Acting) is a loop where the LLM alternates between thinking and acting until it can answer:

```
Thought → Action → Action Input → Observation → (repeat) → Final Answer
```

- **Thought** — the LLM decides whether it needs a tool or can answer directly
- **Action / Action Input** — which tool to call, and what to pass it
- **Observation** — the tool's result, fed back into the next Thought
- The loop ends when the LLM produces a final answer with no further tool calls, or an iteration/recursion limit is hit

## Reflexion Agents

**Reflexion** is a self-improvement pattern: generate an answer, critique it, then revise — repeated a few times before returning a final result.

```
Generate draft → Reflect (critique) → Generate revised draft → Reflect → ... → Final answer
```

- The critique step often needs role-swapping (e.g. treating an AI's own draft as a "human" message) since some providers (like Gemini) reject a request that ends on an assistant-authored turn.
- A loop limit (e.g. counting messages or reflection rounds) prevents infinite refinement.
- Empty or truncated model output during reflection is often caused by too small a `max_tokens`/`max_output_tokens` budget.

## Tool Calling & Tool Execution

- LLMs propose tool calls (`tool_calls`) as structured data (name, arguments, id) rather than free text.
- `ToolNode` (from `langgraph.prebuilt`) reads `tool_calls` off the last AI message and executes the matching tools automatically — replacing the older, now-removed `ToolExecutor`.
- Every tool call **must** get a matching tool response (`ToolMessage` with the same `tool_call_id`), or providers like OpenAI will reject the next request with an error about an unanswered tool call.
- A common bug source: tool-name mismatches between what the LLM's schema is bound as (e.g. `RevisedAnswer`) and what the executor checks for (e.g. `ReviseAnswer`) — these must match exactly.

## Human-in-the-Loop

LangGraph supports pausing a graph mid-execution to get human input, then resuming from exactly that point:

- `interrupt(...)` — pauses execution and surfaces data to the caller (e.g. a draft for review)
- `Command(update=..., goto=...)` — resumes execution, updates state, and routes to the next node
- Requires a **checkpointer** (e.g. `MemorySaver`) to persist state across the pause/resume boundary

## Multi-Agent Architecture

Splitting a workflow into specialized agents (e.g. researcher → validator → writer) rather than one monolithic agent:

- Each agent has a narrow, well-defined responsibility and its own system prompt
- Agents pass results to each other via shared state (typically `messages`)
- A "validator" node commonly checks or grades another agent's output before the workflow proceeds

## Streaming

- `.invoke()` — blocks until the entire graph finishes, returns one final result
- `.stream()` — yields updates incrementally as each node completes
- `.astream_events()` — fine-grained event stream, including `on_chat_model_stream` for token-by-token output, `on_tool_end` for tool completions, etc.
- Stream modes: `"values"` (full state), `"updates"` (just the diff), `"messages"` (token streaming), `"debug"` (verbose internals)
- Powers real-time chat UIs via Server-Sent Events (SSE) — the browser's `EventSource` API consumes these events directly

## RAG Agents (Retrieval-Augmented Generation)

An advanced RAG pipeline does more than a single retrieval step:

```
Rewrite question → Classify topic (in scope?) → Retrieve documents →
Grade relevance → (weak → refine & retry) / (strong → generate answer)
```

- **Question rewriting** turns a vague follow-up ("What about weekends?") into a standalone query for retrieval
- **Topic classification** filters out questions outside the knowledge base's scope
- **Retrieval grading** checks whether retrieved documents actually address the question before generating an answer — preventing confidently wrong answers built on irrelevant matches
- If grading fails, the question is refined and search is retried, up to a limit

## Checkpointing & Memory

- A **checkpointer** (e.g. `MemorySaver`, or a SQLite/Postgres-backed one) saves graph state at each step
- A **thread_id** identifies a specific conversation, so a graph can resume exactly where it left off
- Persistent checkpointers (SQLite/Postgres) survive crashes and restarts; in-memory ones do not

---

## Interview Questions

### ReAct Pattern & Agents
1. What is the ReAct pattern, and what do the Thought, Action, and Observation steps each represent?
2. How does an agent decide whether it needs to call a tool or can answer directly?
3. What stops a ReAct loop from running forever?
4. What is the difference between the classic `AgentAction`/`AgentFinish` style agent and the modern message-based agent?
5. What is `create_agent`, and how does it differ from the older `create_react_agent`?

### LangGraph Core Concepts
6. What is "state" in LangGraph, and how does it get passed between nodes?
7. What is the difference between `MessageGraph` and `StateGraph`?
8. What is a reducer function (e.g. `add_messages`), and why is it needed?
9. What happens if a state key has no reducer and a node returns a new value for it?
10. What is the difference between a node and an edge in a LangGraph graph?
11. What is a conditional edge, and when would you use one over a regular edge?

### Reflexion Agents
12. What is a Reflexion agent, and how does it differ from a standard ReAct agent?
13. Why does adding a "reflect" step tend to improve output quality?
14. What is the purpose of the "generate → reflect → revise" loop, and what usually stops it?
15. Why might a reflection/critique step return empty content, and how would you guard against it?

### Tool Calling & Execution
16. How does an LLM's tool call get matched to its result in a conversation history?
17. What is `ToolNode`, and how does it know which tools to run?
18. What happens if an assistant message with tool calls isn't followed by a matching tool response?
19. Why might a model return `finish_reason: "tool_calls"` but an empty `tool_calls` list?

### Human-in-the-Loop
20. What does the `interrupt()` function do in a LangGraph workflow?
21. What is a `Command` object used for, and how does it differ from a normal node return value?
22. Why is a checkpointer (e.g. `MemorySaver`) required for human-in-the-loop workflows?
23. How would you resume a graph after it pauses for human feedback?

### Multi-Agent Architecture
24. What are the benefits of splitting a workflow into multiple specialized agents (e.g. researcher, validator, writer)?
25. How do agents pass information to each other in a multi-agent graph?
26. What role does a "validator" node typically play in a multi-agent pipeline?

### Streaming
27. What is the difference between `.invoke()` and `.stream()` in LangGraph?
28. What are the different `stream_mode` options, and when would you use each?
29. How does token-by-token streaming work with `astream_events`?
30. Why is streaming particularly useful for chat applications?

### RAG (Retrieval-Augmented Generation) Agents
31. What problem does query rewriting solve in a multi-turn RAG system?
32. Why would a RAG agent grade its retrieved documents before generating an answer?
33. What should happen if retrieved documents are graded as not relevant?
34. How does a RAG agent decide a question is "off-topic" versus something it should attempt to answer?
35. What are the risks of skipping a relevance-grading step in a RAG pipeline?

### Checkpointing & Memory
36. What does a checkpointer do in LangGraph?
37. What is a `thread_id`, and why does it matter for multi-turn conversations?
38. What's the difference between in-memory and persistent checkpointing?

### Practical/Debugging
39. Why might an assistant message ending a conversation cause an error with certain LLM providers (e.g. Gemini)?
40. What's a common cause of an LLM tool call returning truncated or empty output?
41. Why is it important to separate embedding models from chat models when switching LLM providers?
