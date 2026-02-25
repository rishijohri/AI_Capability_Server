# Deep Chat Guide

## Overview

The Deep Chat endpoint (`/api/deep-chat`) provides an advanced chat interface that allows the local LLM to perform **multi-round thinking** with access to **function calls** for querying RAG databases. This enables more comprehensive and accurate responses for complex queries.

## Key Features

### 1. Multi-Round Thinking

Instead of generating a response in a single pass, the LLM can:
- Analyze the question in the first round
- Make function calls to gather information
- Synthesize results in subsequent rounds
- Provide a well-informed final answer

The number of rounds is controlled by the `chat_rounds` configuration parameter (configurable via `/api/config`).

### 2. RAG Function Access

During thinking rounds, the LLM has access to two functions:

#### `query_media_rag(query: str, k: int = 5) -> str`
- Searches the media RAG database (images, videos, documents)
- Returns relevant file information including:
  - File names
  - File types
  - Tags
  - Descriptions
  - Creation times
  - Custom metadata fields

#### `query_fact_rag(query: str, k: int = 5) -> str`
- Searches the conversation fact RAG database
- Returns relevant facts from previous conversations
- Helps maintain context across sessions

**When RAG Can Be Called:**

1. **Initial Context** (Automatic): Limited search before Round 1 starts
   - Top 3 media files
   - Up to 500 tokens of facts
   - Purpose: Baseline context

2. **During Thinking Rounds** (LLM-controlled via function calls):
   - `query_media_rag()`: LLM decides what to search, how many results (k)
   - `query_fact_rag()`: LLM decides search queries independently
   - Can be called multiple times per round
   - More comprehensive than initial context

**Key Difference from Regular Chat:**
- Regular `/api/chat`: Single automatic RAG search with full results
- Deep Chat `/api/deep-chat`: Limited initial + LLM-controlled detailed queries

### 3. Transparent to Client

From the client's perspective, the Deep Chat endpoint works exactly like the regular `/api/chat` endpoint:
- Same request format
- Same response format
- Same WebSocket message types
- Single request-response cycle per connection

The multi-round thinking happens internally and is hidden from the client (except for status messages).

## How It Works

### Initial Context Gathering (New!)

Before multi-round thinking begins, Deep Chat now performs an **optional initial RAG search** to provide baseline context:

1. **Media RAG Search**: Searches for relevant files (top 3 results)
2. **Fact RAG Search**: Searches conversation history (limited to 500 tokens)
3. **Results**: Provided to the LLM in Round 1 as "Initial Context"

**Purpose**: Gives the LLM a starting point without requiring it to blindly make the first function call.

**Note**: This initial context is intentionally limited. The LLM is encouraged to use function calls for more comprehensive information gathering.

### Internal Process Flow

```
User Question
    ↓
Initial Context Gathering (automatic)
    ├─ Quick media RAG search (top 3)
    └─ Quick fact RAG search (500 tokens)
    ↓
Round 1: LLM receives initial context + analyzes question
    ↓
    ├─ Call query_media_rag("specific query", k=10) for more details
    ├─ Call query_fact_rag("related topic", k=5)
    └─ Receive comprehensive function results
    ↓
Round 2: LLM synthesizes all information
    ↓
    ├─ Analyze retrieved files and facts
    ├─ Make additional targeted queries if needed
    └─ Decide if ready to answer
    ↓
Round 3: Provide final answer (if needed)
    ↓
Final Response
```

### Function Calling Mechanism

The LLM is given a specialized **Deep Thinking mode system prompt** that:

1. **Strongly encourages function usage**: "ALWAYS use functions to gather information before answering"
2. **Promotes step-by-step thinking**: "Think step-by-step through multiple rounds"
3. **Emphasizes comprehensiveness**: "Only provide your final answer when you have comprehensive information"
4. **Discourages guessing**: "Prefer gathering information over guessing"

This is different from regular chat, where the system uses the standard `chat_system_prompt`.

**Function Call Format:**

```xml
<function_call>
<name>query_media_rag</name>
<arguments>
{
  "query": "photos of the beach",
  "k": 5
}
</arguments>
</function_call>
```

The server automatically:
1. Parses the function call
2. Switches to embedding model if needed
3. Executes the function
4. Switches back to chat model
5. Provides results to the LLM
6. Continues to next round

### Final Answer Detection

The LLM signals completion by wrapping its response in `<final_answer>` tags:

```xml
<final_answer>
You have 12 beach photos from your summer vacation...
</final_answer>
```

When the server detects this, it stops iterating and returns the final answer to the client.

## Configuration

### Setting Chat Rounds

The `chat_rounds` parameter controls the maximum number of thinking rounds:

```bash
# Set to 5 rounds for complex queries
curl -X POST http://localhost:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"chat_rounds": 5}'

# Set to 1 round (equivalent to regular chat)
curl -X POST http://localhost:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"chat_rounds": 1}'
```

**Recommended values:**
- `1-2 rounds`: Simple queries, fast responses
- `3-4 rounds`: Moderate complexity (default is 3)
- `5-7 rounds`: Complex multi-source queries
- `8-10 rounds`: Maximum depth (may be slower)

## Usage Examples

### Python Client

```python
import asyncio
import websockets
import json

async def deep_chat(message, history=None):
    """Send a message to deep chat and get response."""
    uri = "ws://localhost:8000/api/deep-chat"
    
    async with websockets.connect(uri) as ws:
        # Wait for ready message
        ready = False
        while not ready:
            msg = await ws.recv()
            data = json.loads(msg)
            if "ready" in data['message'].lower():
                ready = True
                break
        
        # Send message
        request = {
            "message": message,
            "history": history or []
        }
        await ws.send(json.dumps(request))
        
        # Receive response
        response_text = ""
        thinking_rounds = 0
        
        async for msg in ws:
            data = json.loads(msg)
            
            if data['type'] == 'status':
                print(f"[Status] {data['message']}")
                if 'thinking round' in data['message'].lower():
                    thinking_rounds += 1
            
            elif data['type'] == 'progress':
                chunk = data['message']
                print(chunk, end='', flush=True)
                response_text += chunk
            
            elif data['type'] == 'result':
                print(f"\n\nThinking rounds used: {data['data']['thinking_rounds']}")
                return response_text
            
            elif data['type'] == 'error':
                raise Exception(data['message'])

# Example usage
async def main():
    response = await deep_chat(
        "What photos do I have of animals and when were they taken?"
    )
    print(f"Response: {response}")

asyncio.run(main())
```

### JavaScript Client

```javascript
async function deepChat(message, history = []) {
  const ws = new WebSocket('ws://localhost:8000/api/deep-chat');
  
  return new Promise((resolve, reject) => {
    let ready = false;
    let response = '';
    let thinkingRounds = 0;
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'status') {
        console.log(`[Status] ${data.message}`);
        
        if (data.message.toLowerCase().includes('ready')) {
          ready = true;
          // Send message
          ws.send(JSON.stringify({ message, history }));
        }
        
        if (data.message.toLowerCase().includes('thinking round')) {
          thinkingRounds++;
        }
      }
      
      else if (data.type === 'progress') {
        process.stdout.write(data.message);
        response += data.message;
      }
      
      else if (data.type === 'result') {
        console.log(`\n\nThinking rounds: ${data.data.thinking_rounds}`);
        resolve(response);
      }
      
      else if (data.type === 'error') {
        reject(new Error(data.message));
      }
    };
    
    ws.onerror = (error) => reject(error);
  });
}

// Example usage
(async () => {
  const response = await deepChat(
    "What photos do I have of animals and when were they taken?"
  );
  console.log(`Response: ${response}`);
})();
```

## Use Cases

### 1. Complex Multi-Source Queries

**Query:** "Compare my beach vacation photos from 2023 and 2024"

The LLM can:
1. Round 1: Call `query_media_rag("beach vacation 2023")`
2. Round 2: Call `query_media_rag("beach vacation 2024")`
3. Round 3: Synthesize comparison

### 2. Historical Context Queries

**Query:** "What was I doing last summer?"

The LLM can:
1. Round 1: Call `query_fact_rag("summer activities last year")`
2. Round 2: Call `query_media_rag("summer photos")`
3. Round 3: Combine conversation facts and media files

### 3. Sequential Information Gathering

**Query:** "Show me progression photos of my garden project"

The LLM can:
1. Round 1: Find all garden-related photos
2. Round 2: Analyze timestamps and tags
3. Round 3: Order chronologically and describe progression

## Performance Considerations

### Response Time

Deep Chat takes longer than regular chat because:
- Multiple model calls (chat model for each round)
- Function execution (embedding model switches for RAG queries)
- More complex reasoning

**Typical timings (3 rounds):**
- Regular chat: 5-15 seconds
- Deep chat: 15-45 seconds (depending on function calls)

### Optimization Tips

1. **Adjust `chat_rounds`**: Use fewer rounds for faster responses
2. **Pre-load models**: Keep embedding model loaded if using knowledge storage
3. **Limit RAG results**: Use smaller `k` values in function calls
4. **Cache results**: Store common queries in conversation facts

## Comparison with Regular Chat

| Feature | `/api/chat` | `/api/deep-chat` |
|---------|-------------|------------------|
| **Initial RAG search** | ✅ Yes (automatic, comprehensive) | ✅ Yes (automatic, limited) |
| **Additional RAG access** | ❌ No | ✅ Yes (via function calls) |
| **Thinking rounds** | 1 (single pass) | Configurable (1-10) |
| **Function calling** | ❌ No | ✅ Yes |
| **System prompt** | Standard chat prompt | Deep thinking prompt (encourages function use) |
| **Context gathering** | Single upfront RAG query | Initial + on-demand function calls |
| **Response time** | Fast (5-15s) | Slower (15-45s) |
| **Best for** | Simple questions | Complex multi-source questions |
| **Model switches** | 2 (embedding + chat) | 3+ (initial + multiple for functions) |
| **Accuracy** | Good | Better (multi-round reasoning) |
| **Control over RAG** | Automatic only | Initial + LLM-controlled queries |

## Troubleshooting

### Issue: Deep chat hangs or times out

**Cause:** Model taking too long to generate in each round

**Solution:**
- Reduce `chat_rounds`
- Increase `llm_timeout` in config
- Use faster models

### Issue: Final answer never arrives

**Cause:** LLM not using `<final_answer>` tags

**Solution:**
- The system automatically uses last round's response if no final answer tags are found
- Improve the system prompt to train the model to use tags

### Issue: Function calls fail

**Cause:** RAG databases not loaded or embedding model issues

**Solution:**
- Ensure RAG is built and loaded before using deep chat
- Check that embedding model is configured correctly
- Verify knowledge storage is enabled if using fact RAG

## Future Enhancements

Potential improvements for future versions:

1. **More functions**: Add functions for filtering, sorting, metadata updates
2. **Streaming thinking**: Optionally expose thinking process to client
3. **Parallel function calls**: Execute multiple RAG queries simultaneously
4. **Dynamic round adjustment**: Automatically determine optimal number of rounds
5. **Function result caching**: Cache RAG results within a session
6. **Custom functions**: Allow users to define custom functions

## Related Documentation

- [API Reference - Deep Chat](../API_REFERENCE.md#ws-apideep-chat)
- [Chat Configuration](../API_REFERENCE.md#post-apiconfig)
- [RAG Service Guide](../Documentation/RAG_GUIDE.md)
- [Knowledge Storage](../Documentation/KNOWLEDGE_STORAGE.md)
