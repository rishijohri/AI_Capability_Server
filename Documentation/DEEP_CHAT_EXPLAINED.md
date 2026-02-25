# Deep Chat Mode - Complete Explanation

## Overview

This document provides a detailed explanation of how Deep Chat mode works, including system prompts, RAG access patterns, and behavioral differences from regular chat.

## System Prompts Comparison

### Regular Chat (`/api/chat`) System Prompt

Regular chat uses the **configurable `chat_system_prompt`** from the settings:

```
You are a helpful AI assistant. Answer questions based on the provided context...
```

This is a general-purpose prompt focused on:
- Being helpful and accurate
- Using provided context
- Direct answering

### Deep Chat (`/api/deep-chat`) System Prompt

Deep Chat uses a **specialized Deep Thinking mode prompt** that combines:

1. **Function-focused instructions** ✨ NEW
2. **The configurable `chat_system_prompt`** (still included)
3. **Round-specific context**

**Full Deep Thinking System Prompt:**

```
You are an AI assistant in DEEP THINKING mode with access to information retrieval functions.

**Your Approach:**
1. ALWAYS use functions to gather information before answering
2. Think step-by-step through multiple rounds
3. Cross-reference multiple sources when possible
4. Only provide your final answer when you have comprehensive information

**Available Functions:**

1. query_media_rag(query: str, k: int = 5) -> str
   - Searches media files: images, videos, documents
   - Returns file names, types, tags, descriptions, dates
   - Use this to find specific files or learn about media content
   
2. query_fact_rag(query: str, k: int = 5) -> str
   - Searches conversation history and stored facts
   - Returns relevant information from past interactions
   - Use this for historical context or previously discussed topics

**Function Call Format:**
<function_call>
<name>function_name</name>
<arguments>
{
  "query": "specific search query",
  "k": 5
}
</arguments>
</function_call>

**Guidelines:**
- Make multiple function calls in early rounds to gather comprehensive information
- Use specific, targeted queries for better results
- Analyze function results before making your final answer
- For complex questions, query both media and fact RAGs
- Only use <final_answer> tags when you're ready with a complete response

**Final Answer Format:**
<final_answer>
Your comprehensive answer here...
</final_answer>

**Important:** Prefer gathering information over guessing. Use the functions!

[Plus the standard chat_system_prompt from config]

[Plus round-specific context including initial RAG results and previous thinking]
```

## RAG Access Patterns

### When RAG Can Be Called

#### 1. Initial Context Gathering (Automatic)

**Timing:** Before Round 1 begins
**Who Controls:** Server (automatic)
**Purpose:** Provide baseline context

**What Happens:**
```python
# Media RAG Search
relevant_files = await rag_service.search(search_query, k=3)  # Top 3 only

# Fact RAG Search  
relevant_facts = knowledge_service.select_knowledge(
    user_message_embedding,
    token_budget=500,  # Limited to 500 tokens
    min_relevance=config.min_knowledge_relevance,
)
```

**Results Provided to LLM:**
```
Initial Media Context (consider using query_media_rag for more):
Here are relevant files from the knowledge base:
- vacation_beach.jpg
  Type: image
  Tags: beach, summer, vacation
  ...

Initial Fact Context (consider using query_fact_rag for more):
- User mentioned planning a beach trip in July
- Previous conversation about vacation photography
- ...
```

**Key Points:**
- ✅ Automatic - no LLM decision needed
- ✅ Limited results - intentionally incomplete
- ✅ Hints at using functions for more details
- ✅ Gives LLM a starting point

#### 2. Function-Based Access (LLM-Controlled)

**Timing:** During any thinking round (Round 1, 2, 3, etc.)
**Who Controls:** LLM (via function calls)
**Purpose:** Gather comprehensive, targeted information

**What Happens:**
```python
# LLM makes a function call
<function_call>
<name>query_media_rag</name>
<arguments>
{
  "query": "beach photos taken in summer 2024",
  "k": 10
}
</arguments>
</function_call>

# Server executes:
relevant_files = await rag_service.search(
    "beach photos taken in summer 2024", 
    k=10  # LLM-specified
)
```

**Key Points:**
- ✅ LLM controls: what to search, when to search, how many results
- ✅ Can be called multiple times per round
- ✅ More comprehensive than initial context
- ✅ Targeted queries based on LLM's analysis

### Regular Chat (`/api/chat`) RAG Pattern

**For Comparison:**

```python
# Single automatic RAG search (comprehensive)
relevant_files = await rag_service.search(
    search_query,  # Built from full conversation history
    k=config.top_k  # Default: 5, but configurable
)

# Results included directly in system prompt
# No additional RAG access possible
```

**Key Differences:**

| Aspect | Regular Chat | Deep Chat |
|--------|--------------|-----------|
| **Number of searches** | 1 (one-time) | 1 initial + N function calls |
| **Search control** | Server-controlled | Initial: server, Then: LLM-controlled |
| **Search queries** | Automatic (conversation history) | Initial: automatic, Then: LLM-crafted |
| **Result limit** | Full (top_k configured) | Initial: limited (3 files, 500 tokens), Then: LLM-specified |
| **Timing** | Before generation | Initial: before Round 1, Functions: during rounds |

## Complete Flow Comparison

### Regular Chat Flow

```
1. User sends message
   ↓
2. Server builds search query from conversation history
   ↓
3. Server performs RAG search (top_k results)
   ↓
4. Server adds results to system prompt
   ↓
5. LLM generates response (single pass)
   ↓
6. Response sent to client
```

**Total RAG Access:** 1 time (automatic)

### Deep Chat Flow

```
1. User sends message
   ↓
2. Server performs limited RAG search (3 files, 500 tokens)
   ↓
3. Server provides initial context to LLM
   ↓
4. Round 1: LLM receives initial context
   ↓
   ├─ LLM analyzes question + initial results
   ├─ LLM calls query_media_rag("specific query", k=10)
   ├─ Server executes → switches to embedding model → searches → switches back
   ├─ LLM calls query_fact_rag("another query", k=5)
   └─ Server executes and provides results
   ↓
5. Round 2: LLM receives function results
   ↓
   ├─ LLM analyzes all gathered information
   ├─ LLM may make additional function calls if needed
   └─ LLM may provide final answer if sufficient
   ↓
6. Round 3 (if needed): Final synthesis
   ↓
7. Response sent to client
```

**Total RAG Access:** 1 initial + N function calls (LLM decides N)

## Model Switching Behavior

### Regular Chat

```
Embedding Model → Chat Model
      (RAG)       (Generation)
```

**2 model switches total**

### Deep Chat

```
Embedding Model → Chat Model → (Function: Embedding) → Chat → (Function: Embedding) → Chat → ...
   (Initial)      (Round 1)         (RAG query)      (Round 2)    (RAG query)      (Round 3)
```

**3+ model switches** (depends on function call count)

**Why This Matters:**
- Each model switch takes time (~2-5 seconds)
- More function calls = more switches = longer response time
- Trade-off: Depth vs. Speed

## System Prompt Strategy

### Why Deep Chat Has a Different Prompt

**Problem:** Generic prompts don't encourage function usage

```
❌ Generic: "You have access to functions..."
   Result: LLM often ignores functions, answers directly

✅ Deep Thinking: "ALWAYS use functions to gather information..."
   Result: LLM actively uses functions before answering
```

**Key Prompt Elements:**

1. **Imperative Tone**: "ALWAYS use functions" (not "you can use")
2. **Process Description**: "Think step-by-step through multiple rounds"
3. **Quality Gate**: "Only provide final answer when comprehensive"
4. **Anti-Pattern Warning**: "Prefer gathering information over guessing"

### How It Affects Behavior

**Without Deep Thinking Prompt:**
```
User: "What beach photos do I have?"

LLM: "Based on the initial context mentioning 3 files, 
      you have vacation_beach.jpg, summer_2024.jpg..."
      
[Never calls functions, answers from limited initial context]
```

**With Deep Thinking Prompt:**
```
User: "What beach photos do I have?"

LLM Round 1:
<function_call>
<name>query_media_rag</name>
<arguments>{"query": "beach photos", "k": 15}</arguments>
</function_call>

LLM Round 2:
<function_call>
<name>query_media_rag</name>
<arguments>{"query": "ocean coastal seaside", "k": 10}</arguments>
</function_call>

LLM Round 3:
<final_answer>
You have 23 beach-related photos in your collection...
[Comprehensive answer based on 25 retrieved files]
</final_answer>
```

## Use Cases and Recommendations

### When to Use Regular Chat

✅ **Use `/api/chat` for:**
- Simple, direct questions
- Questions where a single RAG search is sufficient
- Time-sensitive requests
- General conversation

**Example Questions:**
- "What files do I have?"
- "Show me my photos"
- "What's in this image?"

### When to Use Deep Chat

✅ **Use `/api/deep-chat` for:**
- Complex, multi-faceted questions
- Questions requiring cross-referencing
- Historical context + current data
- Analytical queries

**Example Questions:**
- "Compare my vacation photos from 2023 and 2024"
- "What activities did I do last summer based on my photos and notes?"
- "Find progression photos of my garden project over time"
- "What themes appear most frequently in my photo collection?"

## Configuration Impact

### `chat_rounds` Parameter

Controls maximum thinking rounds:

```bash
# Fast but potentially less thorough (1-2 rounds)
curl -X POST http://localhost:8000/api/config \
  -d '{"chat_rounds": 2}'

# Balanced (3-4 rounds) - DEFAULT
curl -X POST http://localhost:8000/api/config \
  -d '{"chat_rounds": 3}'

# Deep analysis (5-7 rounds)
curl -X POST http://localhost:8000/api/config \
  -d '{"chat_rounds": 6}'

# Maximum depth (8-10 rounds)
curl -X POST http://localhost:8000/api/config \
  -d '{"chat_rounds": 10}'
```

**Note:** LLM can terminate early if it provides `<final_answer>` before using all rounds.

### Other Relevant Settings

```python
enable_knowledge_storage: bool  # Enable fact RAG
objectivity_threshold: float     # Fact storage threshold
max_knowledge_tokens: int        # Fact RAG token budget
min_knowledge_relevance: float   # Fact similarity threshold
top_k: int                       # Default RAG result count
llm_timeout: int                 # Per-round timeout
```

## Performance Characteristics

### Response Time Breakdown

**Regular Chat (5-15 seconds):**
```
Embedding model load: ~2s
RAG search: ~1s
Chat model load: ~2s
Generation: ~5-10s
Total: ~10-15s
```

**Deep Chat (15-45 seconds):**
```
Embedding model load: ~2s
Initial RAG search: ~1s
Chat model load: ~2s

Round 1:
  Generation: ~5s
  Function call (switch to embed): ~2s
  RAG search: ~1s
  Switch back to chat: ~2s
  
Round 2:
  Generation: ~5s
  Function call: ~3s
  
Round 3:
  Final answer: ~5s

Total: ~28s (varies with function calls)
```

## Summary

### Key Takeaways

1. **Deep Chat has a specialized system prompt** that strongly encourages function usage and multi-round thinking

2. **Initial RAG search happens automatically** in Deep Chat, providing baseline context (unlike regular chat's comprehensive search)

3. **RAG can be called on-demand** via LLM-controlled function calls during thinking rounds

4. **LLM has full control** over when to search, what to search for, and how many results to retrieve

5. **Trade-off: Depth vs Speed** - Deep Chat is slower but more thorough

6. **Different use cases** - Choose the right tool for the job:
   - Simple questions → Regular Chat
   - Complex questions → Deep Chat

### Architecture Benefits

This design provides:
- **Flexibility**: LLM can adapt search strategy to question complexity
- **Efficiency**: Initial context prevents blind first search
- **Depth**: Multi-round thinking enables comprehensive analysis
- **Transparency**: Client interface unchanged from regular chat
- **Control**: Both automatic (initial) and manual (functions) RAG access
