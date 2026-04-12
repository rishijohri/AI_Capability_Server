# Deep Chat (Cloud) Build Guide

Build a Cloud AI client that performs Deep Chat by using the local server's scoped RAG search endpoint. The server handles filtering and semantic ranking; your cloud AI handles reasoning and synthesis.

## Architecture

```
┌─────────────────────────┐         ┌──────────────────────────┐
│     Cloud AI Client     │   WS    │     Local AI Server      │
│  (GPT-4, Claude, etc.)  │◄───────►│  /api/scoped-rag-search  │
│                         │         │                          │
│  1. Extraction prompt   │         │  • Metadata store        │
│  2. Synthesis prompt    │         │  • FAISS embeddings      │
│  3. Short Extraction    │         │  • Compacted convos      │
│                         │         │  • Date/tag filtering    │
└─────────────────────────┘         └──────────────────────────┘
```

The cloud AI replaces the local LLM for all reasoning steps. The server provides:
- Library metadata (tags, dates, conversation keywords)
- Date & tag filtering (files only — conversations always participate)
- FAISS semantic ranking with stored embeddings

---

## Prerequisites

Before using `/api/scoped-rag-search`, ensure:

1. **Storage metadata is set** — `POST /api/set-storage-metadata`
2. **Files are tagged/described** — `WS /api/tag` and `WS /api/describe`
3. **RAG is generated** — `WS /api/generate-rag`
4. **Conversations are compacted** (optional but recommended) — `WS /api/compact-conversations` or `WS /api/cloud-compact`

---

## Call Order Summary

```
1.  Open persistent WebSocket to /api/scoped-rag-search
2.  Wait for: {"type": "status", "message": "Scoped RAG Search ready."}
3.  Send: {"action": "get_library_context"}
4.  Receive: library tags, conversation keywords, date range, file/conversation counts
5.  Cloud AI — Extraction call:
      INPUT:  user question + library context
      OUTPUT: START_DATE, END_DATE, TAGS, RAG_QUERY
6.  Send: {"action": "scoped_search", ...extracted params}
7.  Receive: ranked list of candidate files and conversation summaries
8.  Cloud AI — Synthesis call:
      INPUT:  user question + formatted candidates
      OUTPUT: answer, OR SHORT_EXTRACTION directive if more data needed
9.  If SHORT_EXTRACTION received:
      a. Parse: START_DATE, END_DATE, TAGS, RAG_QUERY, INSIGHT
      b. Append INSIGHT to accumulated insights list
      c. Send: {"action": "scoped_search", ...new params}
      d. Receive: new candidates
      e. Merge new candidates into existing list (deduplicate by fileName / id, cap at 8)
      f. Cloud AI — Re-synthesis call:
           INPUT: user question + INSIGHTS block + merged candidates
           OUTPUT: answer, OR next SHORT_EXTRACTION (repeat from 9a, up to N times)
10. Strip any remaining SHORT_EXTRACTION directive lines from final response
11. Return final answer to user
12. Close WebSocket connection
```

---

## Step-by-Step Flow

### Step 1 — Open the Connection

Open a WebSocket to `ws://localhost:8000/api/scoped-rag-search`.

Wait for the first message from the server:

```json
{"type": "status", "message": "Scoped RAG Search ready."}
```

The connection stays open for the entire conversation. Multiple actions (library context requests, scoped searches) can be sent over the same connection without reconnecting.

---

### Step 2 — Get Library Context

Send:
```json
{"action": "get_library_context"}
```

Receive:
```json
{
  "type": "result",
  "data": {
    "top_tags": ["beach", "sunset", "family", ...],
    "total_tags": 342,
    "conv_tags": ["trip planning", "photo editing", ...],
    "date_range": {"min": "2023-01-15", "max": "2026-04-10"},
    "file_count": 1250,
    "conversation_count": 8
  }
}
```

| Field | Description |
|-------|-------------|
| `top_tags` | Up to 200 most common file tags, sorted by frequency |
| `total_tags` | Total unique tags across all files |
| `conv_tags` | Keywords extracted from compacted conversation summaries |
| `date_range` | `min`/`max` file creation dates in the library |
| `file_count` | Total files in the library |
| `conversation_count` | Compacted conversations available for search |

---

### Step 3 — Extraction: Cloud AI Determines Search Parameters

Call your cloud AI using the system prompt below. Substitute in the library context from Step 2.

**System Prompt:**
```
Extract search parameters from the user's question about their media library.

LIBRARY TAGS ({top_tags count} shown, {total_tags} total): {top_tags comma-separated}
CONVERSATION KEYWORDS ({conv_tags count} keywords from chat history): {conv_tags comma-separated}
LIBRARY DATE RANGE: {date_range.min} to {date_range.max}

RESPOND EXACTLY:
FILTER_ORDER:date_first or tags_first
START_DATE:YYYY-MM-DD or none
END_DATE:YYYY-MM-DD or none
TAGS:tag1,tag2,tag3 or none
RAG_QUERY:semantic search phrase
PLAN:one_step or two_step

RULES:
- FILTER_ORDER: date_first when the question mentions a specific date/time period.
  tags_first when the question is about a topic with no specific date.
- START_DATE / END_DATE: Use the same date for both if asking about a single day.
  Use none if no date is mentioned.
- TAGS: Pick from LIBRARY TAGS or CONVERSATION KEYWORDS. Choose 3-10 most relevant.
- RAG_QUERY: A descriptive search phrase for semantic ranking.
- PLAN: one_step for direct questions. two_step if the answer requires chaining
  (e.g. first find when an event happened, then search around that date).
- Use none for any field that does not apply.
- Start your response with FILTER_ORDER: immediately.
```

**User message:** the user's question.

**Parse the AI response** line by line to extract:

```
FILTER_ORDER → filter_order  (date_first | tags_first)
START_DATE   → start_date    (YYYY-MM-DD or null)
END_DATE     → end_date      (YYYY-MM-DD or null)
TAGS         → tags          (list of strings, or empty)
RAG_QUERY    → rag_query     (string)
PLAN         → plan          (one_step | two_step)
```

Parsing rules:
- Skip any line that does not contain `:`
- Split on the first `:` only — values may themselves contain colons
- If a value is the literal word `none`, treat it as null / empty

---

### Step 4 — Scoped Search: Server Filters and Ranks

Send the extracted parameters to the server:

```json
{
  "action": "scoped_search",
  "rag_query": "{rag_query}",
  "start_date": "{start_date or null}",
  "end_date": "{end_date or null}",
  "tags": ["{tag1}", "{tag2}"],
  "k": 8
}
```

The server may send one or more `status` messages before the result. Consume them (optionally display to the user) then wait for the `result` message:

```json
{
  "type": "result",
  "data": {
    "candidates": [ ... ],
    "file_count": 6,
    "conv_count": 2
  }
}
```

Each candidate in the `candidates` array is either a **file** or a **conversation**:

**File candidate:**
```json
{
  "type": "file",
  "fileName": "IMG_2847.jpg",
  "creationTime": "2025-06-15T10:30:00Z",
  "tags": ["beach", "family"],
  "description": "Family at the beach",
  "fileType": "image",
  "formatted": "• IMG_2847.jpg\n  Date: 2025-06-15\n  Tags: beach, family\n  Desc: Family at the beach"
}
```

**Conversation candidate:**
```json
{
  "type": "conversation",
  "id": "conv_20250610",
  "summary": "Discussed planning a beach trip...",
  "tags": ["beach", "vacation"],
  "compacted_at": "2025-06-20T14:00:00Z",
  "formatted": "• [Conversation conv_20250610]\n  Compacted: 2025-06-20\n  Keywords: beach, vacation\n  Facts: Discussed planning a beach trip..."
}
```

The `formatted` field is a pre-built text block ready to be inserted directly into your synthesis prompt. Use it as-is.

> **Note:** Conversation candidates are **never filtered by date**. `start_date`/`end_date` apply to files only. Conversations always participate in tag filtering and semantic ranking regardless of date range.

---

### Step 5 — Synthesis: Cloud AI Generates the Answer

Concatenate all `formatted` fields from the candidates to build the FILE DATA block, then call your cloud AI.

**FILE DATA block format:**
```
=== FILE DATA ===
{candidate_1.formatted}

{candidate_2.formatted}

...
=== END ===
```

**System Prompt (when N short extractions remain):**
```
You are a helpful assistant. Answer the user's question using ONLY the data below.
Context may include media files and past conversation summaries (labeled 'Conversation').

RULES:
- Start your answer immediately.
- Reference specific file names, dates, tags, descriptions, or conversation facts.
- If the data IS SUFFICIENT, answer directly and completely.
- If the data is INSUFFICIENT to answer confidently, you may request a Short Extraction —
  a targeted follow-up search with different parameters. You have {N} Short Extraction(s) remaining.
  To request one, output EXACTLY this block first (before any explanation):

SHORT_EXTRACTION
START_DATE:YYYY-MM-DD or none
END_DATE:YYYY-MM-DD or none
TAGS:tag1,tag2,tag3 or none
RAG_QUERY:focused phrase for the missing information
INSIGHT:one sentence summarizing what you learned so far from this extraction

  Then explain what specific information is still missing.
  INSIGHT is important — it carries your findings forward to the next search round.
```

**User message:**
```
{user_question}

=== FILE DATA ===
{formatted candidates joined by blank lines}
=== END ===
```

---

### Step 6 — Short Extraction Loop (if needed)

If the synthesis response contains `SHORT_EXTRACTION`, the AI needs more data. Repeat the following up to N times (recommended default: 2).

#### 6a — Parse the Directive

Read the SHORT_EXTRACTION block from the AI response:

```
SHORT_EXTRACTION  → signals a follow-up search is requested
START_DATE:       → se_start_date  (YYYY-MM-DD or null)
END_DATE:         → se_end_date    (YYYY-MM-DD or null)
TAGS:             → se_tags        (list of strings or null)
RAG_QUERY:        → se_rag_query   (string)
INSIGHT:          → insight        (single sentence string)
```

Use the same line-by-line parsing rules from Step 3.

#### 6b — Collect the Insight

Append the insight to a running list of all insights so far:

```
insights_list ← append "Round {round_number}: {insight}"
```

All insights from all prior rounds accumulate and are passed forward together in Step 6e.

#### 6c — Run Another Scoped Search

Send the new parameters parsed from the directive:

```json
{
  "action": "scoped_search",
  "rag_query": "{se_rag_query}",
  "start_date": "{se_start_date or null}",
  "end_date": "{se_end_date or null}",
  "tags": ["{se_tag1}", ...],
  "k": 8
}
```

Consume status messages and wait for the `result` message, same as Step 4.

#### 6d — Merge Candidates

Merge new candidates into the existing candidate list:
- Deduplicate: skip any candidate whose `fileName` (file) or `id` (conversation) is already in the list
- Cap the total list at 8 candidates

#### 6e — Re-synthesize with Insights

Compute remaining count: `remaining = N - round_number - 1`

Build the INSIGHTS block from all collected insights across all rounds:

```
=== INSIGHTS FROM PREVIOUS ROUNDS ===
Round 1: {insight from round 1}
Round 2: {insight from round 2}
=== END INSIGHTS ===
```

**System Prompt:** Same synthesis prompt as Step 5, with `{N}` replaced by `remaining`. If `remaining = 0`, use the final prompt below instead.

**User message:**
```
{user_question}

=== INSIGHTS FROM PREVIOUS ROUNDS ===
Round 1: {insight}
Round 2: {insight}
=== END INSIGHTS ===

=== FILE DATA ===
{formatted merged candidates joined by blank lines}
=== END ===
```

If the response again contains `SHORT_EXTRACTION` and `remaining > 0`, repeat from Step 6a.

**System Prompt (when 0 short extractions remain — final synthesis):**
```
You are a helpful assistant. Answer the user's question using ONLY the data below.
Context may include media files and past conversation summaries (labeled 'Conversation').

RULES:
- Start your answer immediately.
- Reference specific file names, dates, tags, descriptions, or conversation facts.
- This is the final answer. Do NOT request further searches. If the data is still
  insufficient, state what you found and what is missing.
```

---

### Step 7 — Clean Up and Return

Strip any remaining directive lines from the final response before returning it to the user.

Remove any line whose content (trimmed, case-insensitive) starts with one of:

```
SHORT_EXTRACTION
START_DATE:
END_DATE:
TAGS:
RAG_QUERY:
INSIGHT:
```

Return the cleaned text as the final answer.

---

## Important Notes

### Conversations Are Never Date-Filtered

Compacted conversation summaries always participate in search regardless of the `start_date`/`end_date` sent. Only file candidates are date-filtered. This ensures past conversational memory is always available.

### Insight Accumulation Is Lightweight Context

Do **not** pass the full previous AI response into re-synthesis rounds — it pollutes the context window. Pass only the `INSIGHT` sentence extracted from each round's directive. The AI uses it to remember what it already found without re-reading all prior output.

### Deduplication Cap

Always deduplicate and cap the candidate list at 8 when merging Short Extraction results. Sending more than ~8 candidates to the synthesis prompt starts to degrade reasoning quality.

### Two-Step Plan Handling

If the extraction AI outputs `PLAN:two_step`, the question requires chaining — for example: "Find photos from the same trip as my birthday party." This means:
1. **First search** finds the birthday party date
2. **Short Extraction** uses that discovered date to find related trip photos

No special handling is required. The AI will naturally use Short Extraction to chain searches when it recognises a two-step need.

---

## Message Flow Diagram

```
CLIENT                                        SERVER
  │                                              │
  │── CONNECT ───────────────────────────────────►│
  │◄─ {"type":"status","message":"...ready."}    │
  │                                              │
  │── {"action":"get_library_context"} ──────────►│
  │◄─ {"type":"result","data":{tags, dates, ...}}│
  │                                              │
  │  [Cloud AI: Extraction call]                  │
  │                                              │
  │── {"action":"scoped_search", ...params} ─────►│
  │◄─ {"type":"status",...}  (one or more)       │
  │◄─ {"type":"result","data":{"candidates":[…]}}│
  │                                              │
  │  [Cloud AI: Synthesis call]                   │
  │                                              │
  │  ── if SHORT_EXTRACTION ──────────────────    │
  │── {"action":"scoped_search", ...new params} ─►│  (repeat up to N times)
  │◄─ {"type":"status",...}                      │
  │◄─ {"type":"result","data":{"candidates":[…]}}│
  │  [Cloud AI: Re-synthesis with insights]       │
  │  ─────────────────────────────────────────── │
  │                                              │
  │── CLOSE ─────────────────────────────────────►│
```

---

## Related Endpoints

| Endpoint | Use |
|----------|-----|
| `POST /api/set-storage-metadata` | Set the library path (must be first) |
| `WS /api/generate-rag` | Build the FAISS index |
| `WS /api/compact-conversations` | Summarize + embed conversations locally |
| `WS /api/cloud-compact` | Embed client-provided summaries (cloud workflow) |
| `WS /api/deep-chat` | Local Deep Chat (server runs all LLM calls) |
| `WS /api/scoped-rag-search` | Cloud Deep Chat (server does filtering + ranking only) |

See [API_REFERENCE.md](../API_REFERENCE.md) for full endpoint documentation.
