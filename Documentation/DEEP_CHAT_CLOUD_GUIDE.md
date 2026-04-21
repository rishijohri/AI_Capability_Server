# Deep Chat (Cloud) Build Guide

Build a Cloud AI client that performs Deep Chat by connecting to the local server's MCP tool endpoint. The server handles filtering, semantic ranking, and prompt construction; your cloud AI handles reasoning and tool selection.

## Architecture

```
┌──────────────────────────────┐         ┌──────────────────────────┐
│       Cloud AI Client        │   WS    │     Local AI Server      │
│  (GPT-4, Claude, Gemini …)   │◄───────►│       /api/mcp           │
│                              │         │                          │
│  Tool-calling loop           │         │  • System prompt builder │
│  (standard function calling) │         │  • Library context       │
│                              │         │  • Metadata store        │
│  Uses server-provided        │         │  • FAISS embeddings      │
│  system prompt directly      │         │  • Compacted convos      │
│                              │         │  • Date/tag filtering    │
└──────────────────────────────┘         └──────────────────────────┘
```

The cloud AI replaces the local LLM for all reasoning and tool-selection steps. The server provides:
- A ready-to-use **system prompt** with pre-loaded library context already injected
- **Tool execution** for all 4 MCP tools (filtering, date exploration, semantic search, conversation search)
- **Budget annotations** appended to every tool result so the AI tracks remaining calls

Unlike the old approach (custom extraction + synthesis prompts + SHORT_EXTRACTION loop), this design uses **standard LLM function calling** — exactly the same protocol the local deep-chat loop uses. The Cloud AI simply forwards tool calls to the server and feeds results back into its own conversation context.

---

## Prerequisites

Before using `/api/mcp`, ensure:

1. **Storage metadata is set** — `POST /api/set-storage-metadata`
2. **Files are tagged/described** — `WS /api/tag` and `WS /api/describe`
3. **RAG is generated** — `WS /api/generate-rag`
4. **Conversations are compacted** (optional but recommended) — `WS /api/compact-conversations` or `WS /api/cloud-compact`

---

## The Four MCP Tools

> **Note:** The complete OpenAI-format tool schemas are returned dynamically in `data.tool_definitions` from `get_library_context`. Your client should use those directly rather than hardcoding schemas. The descriptions below are for human reference.

The server exposes exactly four tools. Their parameter names and types are described below.

### 1. `get_scoped_tags`

> List the most common tags from a date/tag-scoped subset of the library.

Use this to understand what topics exist in a time range before running a semantic search.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `start_date` | string (YYYY-MM-DD) | No | Inclusive start date filter |
| `end_date` | string (YYYY-MM-DD) | No | Inclusive end date filter |
| `min_tags` | array of strings | No | Pre-filter: only files matching these tags are counted |
| `strict` | boolean | No | `true` = all tags must match (AND). `false` (default) = any tag matches (OR) |
| `top_m` | integer | No | How many top tags to return (default 50) |

Returns: Plain text listing the top tags and their frequencies in the scoped subset.

---

### 2. `get_scoped_dates`

> List contiguous date ranges that contain matching files, sorted by file count.

Use this to discover when events occurred or to find the temporal distribution of a topic.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `start_date` | string (YYYY-MM-DD) | No | Restrict to files on or after this date |
| `end_date` | string (YYYY-MM-DD) | No | Restrict to files on or before this date |
| `min_tags` | array of strings | No | Tag filter applied before date grouping |
| `strict` | boolean | No | AND vs OR tag matching (default OR) |
| `top_k` | integer | No | Maximum number of date clusters to return (default 10) |

Returns: Plain text listing date ranges and file counts.

---

### 3. `scoped_rag_search`

> Semantic search for media files within a mandatory date and tag scope.

**This tool requires all three scoping fields.** The AI must have a concrete date range and at least one tag before calling it — obtain these from `get_scoped_tags` / `get_scoped_dates` first, or from the pre-loaded library context if that is sufficient.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | **Yes** | Natural language search phrase |
| `start_date` | string (YYYY-MM-DD) | **Yes** | Inclusive start date |
| `end_date` | string (YYYY-MM-DD) | **Yes** | Inclusive end date |
| `min_tags` | array of strings | **Yes** | At least one tag required |
| `strict` | boolean | No | AND vs OR tag matching (default OR) |
| `top_k` | integer | No | Number of results to return |

Returns: Plain text listing the top-K matching files and conversation summaries with name, date, tags, and description for each.

> **Note:** Conversation summaries (from compacted conversations) participate in semantic ranking regardless of the date range. Only file candidates are filtered by date. Conversations always participate in the tag filter and semantic ranking.

---

### 4. `get_conversation_rag`

> Semantic search across compacted past conversation summaries.

Use this when the question involves facts, preferences, or events from previous conversations that may not appear as media files.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | **Yes** | Natural language search phrase |
| `top_n` | integer | No | Number of conversation snippets to return (default 5) |

Returns: Plain text listing matching conversation summaries with their topics and key facts.

---

## Call Order Summary

```
1.  Open persistent WebSocket to /api/mcp
2.  Wait for: {"type": "status", "message": "Cloud Deep Chat MCP tools ready."}
3.  Send: {"action": "get_library_context", "query": "<user question>"}
4.  Receive: system_prompt (ready to use), library_context, tool_budget, top_tags, date_range, counts
5.  Use system_prompt directly as the system message for the Cloud AI
6.  Define the 4 MCP tools as function-calling schemas for the Cloud AI LLM
7.  Send initial user message to Cloud AI (system + user turn)
8.  Tool-calling loop:
      a. Cloud AI responds with one or more tool calls
      b. For each tool call, send to server:
            {"action": "<tool_name>", "budget_remaining": <N>, ...tool arguments}
      c. Receive result — the "message" field contains the tool output with budget annotation appended
      d. Add result "message" as the tool result in the LLM conversation
      e. Decrement your local budget counter by 1
      f. If budget reaches 0, the server annotates: "[Tool calls remaining: 0 — generate your final answer now]"
      g. On the next LLM call, send no tools (force a plain text response)
9.  Cloud AI produces a final text answer (no tool calls)
10. Extract the answer from <conclusion>...</conclusion> tags
11. Extract referenced file names from <files>...</files> tags
12. Close WebSocket connection
```

---

## Step-by-Step Flow

### Step 1 — Open the Connection

Open a WebSocket to `ws://localhost:8000/api/mcp`.

Wait for the first message from the server:

```json
{"type": "status", "message": "Cloud Deep Chat MCP tools ready."}
```

The connection stays open for the entire session. All tool calls for a single conversation use the same connection.

---

### Step 2 — Get Library Context and System Prompt

Send the user's question so the server can pre-load relevant conversation memory:

```json
{"action": "get_library_context", "query": "<user question>"}
```

Receive:

```json
{
  "type": "result",
  "message": "Library context retrieved",
  "data": {
    "system_prompt": "You are Persona, a thorough AI assistant...",
    "library_context": "PRE-LOADED LIBRARY CONTEXT\n\nGlobal tags (top 50 from 1250 files):\nbeach, sunset, family, ...",
    "tool_definitions": [{"type": "function", "function": {"name": "get_scoped_tags", ...}}, ...],
    "tool_budget": 10,
    "total_files": 1250,
    "date_range": {"min": "2023-01-15", "max": "2026-04-10"},
    "top_tags": ["beach", "sunset", "family", "vacation", "dogs"],
    "conversation_count": 8
  }
}
```

| Field | Description |
|-------|-------------|
| `system_prompt` | Full system prompt with library context already injected — use as-is as the system message for the Cloud AI LLM call |
| `library_context` | The context block alone — useful if you want to customize the system prompt around it |
| `tool_definitions` | OpenAI function-calling schemas for all registered MCP tools — pass directly as the `tools` parameter in every Cloud AI LLM call, no hardcoding needed |
| `tool_budget` | Number of tool calls allowed before the AI must produce a final answer |
| `total_files` | Total files in the library |
| `date_range` | `min` and `max` file creation dates |
| `top_tags` | Top 200 most common tags, sorted by frequency |
| `conversation_count` | Number of compacted conversations with embeddings |

**Use `data.system_prompt` directly.** It already contains the tool descriptions, budget rules, strategy, answer format, and the pre-loaded library context (global tags, date range, and relevant past conversations matched to the query). You do not need to build a system prompt yourself.

---

### Step 3 — Configure the LLM for Tool Calling

Use `data.tool_definitions` from the `get_library_context` response directly as the `tools` parameter in every Cloud AI LLM call — no schema hardcoding needed. The server always returns complete, up-to-date OpenAI-format schemas for all registered MCP tools.

```python
tools = library_context_response["data"]["tool_definitions"]
# Pass to your Cloud AI LLM call:
# openai_client.chat.completions.create(..., tools=tools, tool_choice="auto")
```

The schemas describe the parameter contracts already enforced by the server (refer to [The Four MCP Tools](#the-four-mcp-tools) for human-readable descriptions). Key points:
- `scoped_rag_search` requires `query`, `start_date`, `end_date`, and `min_tags`
- All other tools have no required fields (all parameters optional)
- All string dates must be in `YYYY-MM-DD` format

---

### Step 4 — First LLM Call

Send to your Cloud AI:

- **System message:** `data.system_prompt` from Step 2
- **User message:** the user's question
- **Tools:** `data.tool_definitions` from Step 2
- **Tool choice:** `auto`

The AI will either respond with tool calls or — if the pre-loaded library context is already sufficient — produce a final answer immediately.

---

### Step 5 — Tool-Calling Loop

Repeat for each iteration until the AI produces a plain text response:

#### 5a — Receive Tool Calls

The Cloud AI LLM responds with one or more tool calls. Each call has:
- A tool name (`get_scoped_tags`, `get_scoped_dates`, `scoped_rag_search`, or `get_conversation_rag`)
- A set of arguments (matching the tool's parameter schema)

#### 5b — Forward Each Tool Call to the Server

For each tool call, send a message to the server:

```json
{
  "action": "<tool_name>",
  "budget_remaining": <current_budget_remaining>,
  "<arg1>": <value1>,
  "<arg2>": <value2>
}
```

Include `budget_remaining` (your current count before this call). The server uses it to append the correct budget annotation to the result.

Optionally include `embedding_model` to override which embedding model the server uses for `scoped_rag_search`.

The server will first send a `status` message (`"Executing MCP tool: <name>..."`), then the `result` message.

#### 5c — Process the Result

The `result` message contains:

```json
{
  "type": "result",
  "message": "<tool output text>\n\n[Tool calls remaining: N]",
  "data": {
    "tool": "<tool_name>",
    "raw": "<tool output text without annotation>",
    "budget_after": <N>
  }
}
```

Use the `message` field (top-level in the WebSocket JSON) as the tool result string inserted into the LLM conversation for that tool call ID. It already includes the `[Tool calls remaining: N]` annotation that the system prompt tells the AI to watch.

Update your local budget counter: `budget_remaining = data.budget_after`.

#### 5d — Next LLM Call

Add all tool results from this iteration to the conversation, then **apply history truncation** (see [Tool Call History Truncation](#tool-call-history-truncation) below) before calling the Cloud AI again.

**Budget enforcement on the client side:**
- If `budget_after == 1`: On the next LLM call, send **only** `scoped_rag_search` as the available tool (omit the other three). This forces a final targeted search.
- If `budget_after == 0`: On the next LLM call, send **no tools** at all (force a plain text response). The AI has already received the `[Tool calls remaining: 0 — generate your final answer now]` annotation in the last result.

Repeat Steps 5a–5d until the LLM responds with no tool calls.

---

### Step 6 — Extract the Final Answer

The final Cloud AI response will be formatted as:

```
<conclusion>
Based on the search results, ...
</conclusion>

<files>
photo_2024_01_15.jpg
video_birthday.mp4
</files>
```

- Extract the content inside `<conclusion>...</conclusion>` as the answer text
- Extract filenames listed inside `<files>...</files>` (one per line) as the list of referenced files
- If no structured tags are present, use the full response text as-is

---

### Step 7 — Close the Connection

Close the WebSocket. Optionally re-use the same connection for follow-up questions in the same session without calling `get_library_context` again (the pre-loaded context is already set).

---

## Tool Call History Truncation

Each MCP tool result can be hundreds of lines long. Keeping the full text of every result in the conversation history will exhaust the context window after a few tool calls. You must **compact older tool results** before every LLM call.

### When to Apply

After ALL tool calls for the current iteration have been appended to the conversation, and **before** sending the messages list to the Cloud AI for the next iteration — truncate every `role:tool` message **except the most recently added one**. The newest result stays in full so the model can reason about it; older results only need to show a summary and their budget annotation.

```
[system]          ← never touch
[user]            ← never touch
[assistant]       ← never touch (tool call request)
[tool: result 1]  ← truncate  (older)
[tool: result 2]  ← truncate  (older)
[assistant]       ← never touch
[tool: result 3]  ← truncate  (older)
[tool: result 4]  ← KEEP FULL  ← most recently added

  ↓ Cloud AI call happens here

[assistant]       ← append new response
[tool: result 5]  ← KEEP FULL  ← newest after next iteration
[tool: result 4]  ← NOW truncate (was previously newest)
```

### The Algorithm

Implement two functions in your client — a **single-result shrinker** and a **message-list walker** that applies it. The server reference implementation is `_truncate_tool_result` and `_apply_history_truncation` in `app/services/deep_chat_handler.py`.

**Single-result shrinker** (`truncate_tool_result(content, max_results)`):

1. Scan the content for the budget annotation line matching `[Tool calls remaining: …]`. Extract it and set it aside — it must be re-appended at the end regardless of truncation.
2. Split the content by newline.
3. Treat the **first line** as the header summary (e.g., `"Top 5 results for 'beach'…"`). Always keep it.
4. Collect the remaining non-empty lines that are **not** the budget annotation line into a body list.
5. Keep only the first `max_results` entries from the body list. Count how many were omitted.
6. Rebuild the result: header + kept body lines + (if any omitted) a note such as `"... (N more results omitted)"` + the budget annotation line.

**Message-list walker** (`apply_history_truncation(messages, max_results)`):

1. Walk the full messages list and find the **index of the last** `role:tool` message.
2. Walk the list again. For every `role:tool` message whose index is **not** the last one, replace its `content` with the result of `truncate_tool_result(content, max_results)`.
3. Leave all other roles (`system`, `user`, `assistant`) completely untouched.

### Rules

| Rule | Detail |
|------|--------|
| **Never truncate `role:assistant`** | The LLM's own reasoning and `<think>…</think>` blocks must stay intact — truncating them corrupts the reasoning chain |
| **Never truncate `role:user` or `role:system`** | Only `role:tool` messages are touched |
| **Always preserve the budget annotation** | The `[Tool calls remaining: N]` line at the end of every tool message is always kept, even after truncation — the LLM reads it to track budget |
| **Newest tool result is always kept full** | Only results from previous iterations are truncated; the current iteration's results stay intact |
| **Apply before every LLM call** | Not after: apply the truncation immediately before forwarding the message list to the Cloud AI |

### Configuration

The server config setting that controls the body line limit for the local deep-chat loop is `tool_history_max_results` (default: 5). Use the same value for your Cloud AI client, or adjust it to suit the context window of your Cloud AI model. Read or update it via `GET /api/config` and `POST /api/config` with the `tool_history_max_results` field.

**Context-window guidance:**

| Cloud AI model context | Recommended `max_results` |
|------------------------|---------------------------|
| 8K tokens | 3 |
| 32K tokens | 5–7 (server default) |
| 128K+ tokens | 10–20 |

---

## Budget and Tool Availability

The budget is managed jointly by the client and the server:

| Remaining calls | Server behaviour | Client behaviour |
|-----------------|-----------------|-----------------|
| > 1 | Appends `[Tool calls remaining: N]` to result | Send all 4 tools in next LLM call |
| = 1 | Appends `[Tool calls remaining: 1]` to result | Send **only** `scoped_rag_search` in next LLM call |
| = 0 | Appends `[Tool calls remaining: 0 — generate your final answer now]` to result | Send **no tools** in next LLM call |

The system prompt already instructs the AI about these rules — the client only needs to enforce the tool list restriction on the LLM side.

---

## Strategy the AI Follows

The system prompt instructs the AI to use this search strategy:

1. **Consult pre-loaded context first.** The `get_library_context` response already includes global tags, date range, and the most relevant past conversation summaries. If this is sufficient to scope a `scoped_rag_search`, the AI should skip `get_scoped_tags` and `get_scoped_dates` and call `scoped_rag_search` directly.

2. **Explore the scope only when needed.** Use `get_scoped_tags` or `get_scoped_dates` only when the question requires a narrower or more specific scope than what was pre-loaded.

3. **`scoped_rag_search` always requires a concrete scope.** The AI must have a specific date range AND at least one tag before calling it. The pre-loaded context or an exploration tool call provides these.

4. **Use `get_conversation_rag` for conversation-based facts** not already covered by the pre-loaded conversation context.

5. **Quality evaluation at each step.** The system prompt requires the AI to assess in its reasoning: "Is this result sufficient? What is still missing? What should I search next given remaining budget?"

6. **Answer format is enforced.** The AI must wrap its final answer in `<conclusion>...</conclusion>` and list referenced files in `<files>...</files>`.

---

## Message Flow Diagram

```
CLIENT                                          SERVER
  │                                                │
  │── CONNECT ───────────────────────────────────►│
  │◄─ {"type":"status","message":"...ready."}      │
  │                                                │
  │── {"action":"get_library_context",             │
  │    "query":"<user question>"} ───────────────►│
  │◄─ {"type":"result","data":{                    │
  │       "system_prompt":"...",                   │
  │       "tool_budget":10, ...}} ─────────────── │
  │                                                │
  │  [Cloud AI call: system_prompt + user msg]     │
  │  → AI responds with tool calls                 │
  │                                                │
  │── {"action":"get_scoped_dates",                │
  │    "min_tags":["beach"],                       │
  │    "budget_remaining":10} ───────────────────►│
  │◄─ {"type":"status",...}                        │
  │◄─ {"type":"result","message":"...\n\n          │
  │    [Tool calls remaining: 9]",                 │
  │    "data":{"budget_after":9,...}} ─────────── │
  │                                                │
  │  [Feed result into LLM conversation]           │
  │  [Cloud AI call: AI responds with tool call]   │
  │                                                │
  │── {"action":"scoped_rag_search",               │
  │    "query":"beach vacation with family",       │
  │    "start_date":"2025-06-01",                  │
  │    "end_date":"2025-06-30",                    │
  │    "min_tags":["beach","family"],              │
  │    "budget_remaining":9} ────────────────────►│
  │◄─ {"type":"status",...}                        │
  │◄─ {"type":"result","message":"Top 5 results    │
  │    ...\n\n[Tool calls remaining: 8]",          │
  │    "data":{"budget_after":8,...}} ─────────── │
  │                                                │
  │  [Feed result into LLM conversation]           │
  │  [Cloud AI call: AI produces final answer]     │
  │  → Response contains no tool calls             │
  │                                                │
  │── CLOSE ─────────────────────────────────────►│
```

---

## Important Notes

### `scoped_rag_search` Requires Mandatory Scoping

Unlike the exploration tools, `scoped_rag_search` **always** requires `start_date`, `end_date`, and at least one entry in `min_tags`. The AI cannot call it with open-ended parameters. This is by design: the pre-loaded library context or one exploration call (at most) should always provide sufficient scoping data before the semantic search.

### Conversations Are Never Date-Filtered

Compacted conversation summaries always participate in `scoped_rag_search` regardless of the `start_date`/`end_date` sent. Only file candidates are date-filtered. This ensures past conversational memory is always available.

### Pre-Loaded Context Reduces Tool Calls

The server pre-fetches global tags, the overall date range, and relevant past conversations matched to the user's question, and injects them all into the system prompt. For many questions this eliminates the need for `get_scoped_tags` and `get_scoped_dates` entirely — the AI can go directly to `scoped_rag_search` on its first tool call.

### Tool Results Are Ready for LLM Injection

Use the `message` field of each tool result directly as the tool result in the Cloud AI LLM conversation. It already contains the formatted output plus the `[Tool calls remaining: N]` annotation. Use `data.raw` only if you need the unannotated text for other purposes.

### `budget_remaining` Is Optional but Strongly Recommended

If you omit `budget_remaining` from tool calls, the server will not append a budget annotation and the AI will not see how many calls remain. The system prompt's budget rules will still be stated at session start (from the static `tool_call_budget` value), but the AI won't receive per-call updates. Always pass `budget_remaining` for correct budget-aware behaviour.

---

## Related Endpoints

| Endpoint | Use |
|----------|-----|
| `POST /api/set-storage-metadata` | Set the library path (must be first) |
| `WS /api/generate-rag` | Build the FAISS index |
| `WS /api/compact-conversations` | Summarize + embed conversations locally |
| `WS /api/cloud-compact` | Embed client-provided summaries (cloud workflow) |
| `WS /api/deep-chat` | Local Deep Chat (server runs all LLM calls) |
| `WS /api/mcp` | Cloud Deep Chat — MCP tool endpoint (server does filtering + ranking; cloud AI drives the loop) |

See [API_REFERENCE.md](../API_REFERENCE.md) for full endpoint and message format documentation.

