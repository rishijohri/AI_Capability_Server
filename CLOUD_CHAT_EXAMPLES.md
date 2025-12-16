# Cloud Chat Integration Examples

This document provides practical examples of using the `/api/cloud-chat` WebSocket endpoint with various cloud LLM providers.

## Overview

The `/api/cloud-chat` endpoint allows you to leverage the AI Server's RAG capabilities while using external cloud LLMs for response generation. This provides the best of both worlds:
- **Local RAG**: Keep your file indexing and search local for privacy
- **Cloud LLM**: Use powerful models like GPT-4, Claude, or Gemini for high-quality responses

## Basic Flow

1. Connect to `/api/cloud-chat` WebSocket endpoint
2. Send your user message (optionally with conversation history)
3. Receive RAG context, system prompt, and relevant files
4. Call your cloud LLM provider with the context
5. Return the response to your user

---

## Example 1: OpenAI GPT-4

```python
import asyncio
import websockets
import json
from openai import OpenAI

async def chat_with_gpt4(user_message: str, history: list = None):
    """
    Use AI Server's RAG with OpenAI GPT-4.
    
    Args:
        user_message: The user's question
        history: Previous conversation history (optional)
    
    Returns:
        str: GPT-4's response
    """
    client = OpenAI()  # Requires OPENAI_API_KEY environment variable
    
    # Step 1: Get RAG context from AI Server
    async with websockets.connect('ws://localhost:8000/api/cloud-chat') as ws:
        # Wait for ready
        while True:
            message = await ws.recv()
            data = json.loads(message)
            if "ready" in data['message'].lower():
                break
        
        # Send request
        await ws.send(json.dumps({
            "message": user_message,
            "history": history or []
        }))
        
        # Get context
        context_data = None
        async for message in ws:
            data = json.loads(message)
            if data['type'] == 'result':
                context_data = data['data']
                break
            elif data['type'] == 'error':
                raise Exception(f"RAG error: {data['message']}")
    
    # Step 2: Build messages for GPT-4
    messages = [
        {
            "role": "system",
            "content": context_data['system_prompt']
        },
        {
            "role": "system",
            "content": f"Here are relevant files from the user's collection:\n\n{context_data['rag_context']}"
        }
    ]
    
    # Add conversation history if provided
    if history:
        messages.extend(history)
    
    # Add current user message
    messages.append({
        "role": "user",
        "content": user_message
    })
    
    # Step 3: Call GPT-4
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )
    
    return response.choices[0].message.content


# Usage example
async def main():
    # First question
    response1 = await chat_with_gpt4("What beach photos do I have?")
    print(f"GPT-4: {response1}\n")
    
    # Follow-up question with history
    history = [
        {"role": "user", "content": "What beach photos do I have?"},
        {"role": "assistant", "content": response1}
    ]
    
    response2 = await chat_with_gpt4("Show me the sunset ones", history)
    print(f"GPT-4: {response2}")

asyncio.run(main())
```

---

## Example 2: Anthropic Claude

```python
import asyncio
import websockets
import json
from anthropic import Anthropic

async def chat_with_claude(user_message: str, history: list = None):
    """
    Use AI Server's RAG with Anthropic Claude.
    
    Args:
        user_message: The user's question
        history: Previous conversation history (optional)
    
    Returns:
        str: Claude's response
    """
    client = Anthropic()  # Requires ANTHROPIC_API_KEY environment variable
    
    # Step 1: Get RAG context from AI Server
    async with websockets.connect('ws://localhost:8000/api/cloud-chat') as ws:
        while True:
            message = await ws.recv()
            data = json.loads(message)
            if "ready" in data['message'].lower():
                break
        
        await ws.send(json.dumps({
            "message": user_message,
            "history": history or []
        }))
        
        context_data = None
        async for message in ws:
            data = json.loads(message)
            if data['type'] == 'result':
                context_data = data['data']
                break
            elif data['type'] == 'error':
                raise Exception(f"RAG error: {data['message']}")
    
    # Step 2: Build system prompt with RAG context
    system_prompt = f"""{context_data['system_prompt']}

Here are relevant files from the user's collection:

{context_data['rag_context']}"""
    
    # Step 3: Build messages
    messages = []
    if history:
        messages.extend(history)
    
    messages.append({
        "role": "user",
        "content": user_message
    })
    
    # Step 4: Call Claude
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=messages
    )
    
    return response.content[0].text


# Usage example
async def main():
    response = await chat_with_claude("Describe my vacation photos from 2024")
    print(f"Claude: {response}")

asyncio.run(main())
```

---

## Example 3: Google Gemini

```python
import asyncio
import websockets
import json
import google.generativeai as genai

async def chat_with_gemini(user_message: str, history: list = None):
    """
    Use AI Server's RAG with Google Gemini.
    
    Args:
        user_message: The user's question
        history: Previous conversation history (optional)
    
    Returns:
        str: Gemini's response
    """
    genai.configure(api_key="YOUR_GOOGLE_API_KEY")
    model = genai.GenerativeModel('gemini-pro')
    
    # Step 1: Get RAG context from AI Server
    async with websockets.connect('ws://localhost:8000/api/cloud-chat') as ws:
        while True:
            message = await ws.recv()
            data = json.loads(message)
            if "ready" in data['message'].lower():
                break
        
        await ws.send(json.dumps({
            "message": user_message,
            "history": history or []
        }))
        
        context_data = None
        async for message in ws:
            data = json.loads(message)
            if data['type'] == 'result':
                context_data = data['data']
                break
    
    # Step 2: Build prompt with RAG context
    prompt = f"""{context_data['system_prompt']}

Relevant files from user's collection:
{context_data['rag_context']}

User question: {user_message}"""
    
    # Step 3: Call Gemini
    response = model.generate_content(prompt)
    return response.text


# Usage example
async def main():
    response = await chat_with_gemini("What photos do I have from the mountains?")
    print(f"Gemini: {response}")

asyncio.run(main())
```

---

## Example 4: Multi-turn Conversation

```python
import asyncio
import websockets
import json
from openai import OpenAI

class CloudChatBot:
    """A chatbot that uses AI Server's RAG with cloud LLMs."""
    
    def __init__(self, provider="openai"):
        self.provider = provider
        self.history = []
        
        if provider == "openai":
            self.client = OpenAI()
            self.model = "gpt-4-turbo-preview"
        elif provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic()
            self.model = "claude-3-5-sonnet-20241022"
    
    async def get_rag_context(self, user_message: str):
        """Get RAG context from AI Server."""
        async with websockets.connect('ws://localhost:8000/api/cloud-chat') as ws:
            # Wait for ready
            while True:
                message = await ws.recv()
                data = json.loads(message)
                if "ready" in data['message'].lower():
                    break
            
            # Send request with conversation history
            await ws.send(json.dumps({
                "message": user_message,
                "history": self.history
            }))
            
            # Get context
            async for message in ws:
                data = json.loads(message)
                if data['type'] == 'result':
                    return data['data']
                elif data['type'] == 'error':
                    raise Exception(f"RAG error: {data['message']}")
    
    async def chat(self, user_message: str) -> str:
        """Send a message and get a response."""
        # Get RAG context
        context_data = await self.get_rag_context(user_message)
        
        # Build messages
        if self.provider == "openai":
            messages = [
                {"role": "system", "content": context_data['system_prompt']},
                {"role": "system", "content": f"Relevant files:\n{context_data['rag_context']}"}
            ]
            messages.extend(self.history)
            messages.append({"role": "user", "content": user_message})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            assistant_message = response.choices[0].message.content
        
        elif self.provider == "anthropic":
            system_prompt = f"{context_data['system_prompt']}\n\nRelevant files:\n{context_data['rag_context']}"
            messages = self.history.copy()
            messages.append({"role": "user", "content": user_message})
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=messages
            )
            assistant_message = response.content[0].text
        
        # Update history
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message


# Usage example
async def main():
    bot = CloudChatBot(provider="openai")
    
    # Multi-turn conversation
    print("User: What photos do I have from 2024?")
    response1 = await bot.chat("What photos do I have from 2024?")
    print(f"Bot: {response1}\n")
    
    print("User: Show me the beach ones")
    response2 = await bot.chat("Show me the beach ones")
    print(f"Bot: {response2}\n")
    
    print("User: Which ones have sunsets?")
    response3 = await bot.chat("Which ones have sunsets?")
    print(f"Bot: {response3}")

asyncio.run(main())
```

---

## Example 5: With Image Context

```python
import asyncio
import websockets
import json
from openai import OpenAI

async def chat_with_image_context(user_message: str, image_name: str):
    """
    Chat about a specific image using cloud LLM.
    
    Args:
        user_message: Question about the image
        image_name: Filename of the image
    
    Returns:
        str: LLM response
    """
    client = OpenAI()
    
    # Get RAG context with image
    async with websockets.connect('ws://localhost:8000/api/cloud-chat') as ws:
        while True:
            message = await ws.recv()
            data = json.loads(message)
            if "ready" in data['message'].lower():
                break
        
        await ws.send(json.dumps({
            "message": user_message,
            "image_name": image_name,
            "history": []
        }))
        
        context_data = None
        async for message in ws:
            data = json.loads(message)
            if data['type'] == 'result':
                context_data = data['data']
                break
    
    # Build messages with image context
    image_ctx = context_data.get('image_context')
    context_text = f"""Image: {image_ctx['image_name']}
Tags: {', '.join(image_ctx['tags'])}
Description: {image_ctx['description']}

Related files:
{context_data['rag_context']}"""
    
    messages = [
        {"role": "system", "content": context_data['system_prompt']},
        {"role": "system", "content": context_text},
        {"role": "user", "content": user_message}
    ]
    
    # For vision models, you can also include the base64 image
    # messages.append({
    #     "role": "user",
    #     "content": [
    #         {"type": "text", "text": user_message},
    #         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_ctx['image_base64']}"}}
    #     ]
    # })
    
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=messages
    )
    
    return response.choices[0].message.content


# Usage
async def main():
    response = await chat_with_image_context(
        "Tell me about this photo",
        "beach_sunset.jpg"
    )
    print(response)

asyncio.run(main())
```

---

## Example 6: Error Handling

```python
import asyncio
import websockets
import json
from openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def robust_cloud_chat(user_message: str, max_retries: int = 3):
    """
    Chat with robust error handling and retries.
    """
    client = OpenAI()
    
    for attempt in range(max_retries):
        try:
            # Get RAG context
            async with websockets.connect(
                'ws://localhost:8000/api/cloud-chat',
                ping_interval=20,
                ping_timeout=10
            ) as ws:
                # Wait for ready
                timeout = 30  # seconds
                ready = False
                start_time = asyncio.get_event_loop().time()
                
                while not ready:
                    if asyncio.get_event_loop().time() - start_time > timeout:
                        raise TimeoutError("Server did not become ready in time")
                    
                    message = await asyncio.wait_for(ws.recv(), timeout=5)
                    data = json.loads(message)
                    
                    if data['type'] == 'error':
                        logger.error(f"Server error: {data['message']}")
                        raise Exception(data['message'])
                    
                    if "ready" in data['message'].lower():
                        ready = True
                
                # Send request
                await ws.send(json.dumps({
                    "message": user_message,
                    "history": []
                }))
                
                # Get response
                async for message in ws:
                    data = json.loads(message)
                    
                    if data['type'] == 'result':
                        context_data = data['data']
                        break
                    elif data['type'] == 'error':
                        logger.error(f"RAG error: {data['message']}")
                        raise Exception(data['message'])
            
            # Call OpenAI
            messages = [
                {"role": "system", "content": context_data['system_prompt']},
                {"role": "system", "content": f"Context:\n{context_data['rag_context']}"},
                {"role": "user", "content": user_message}
            ]
            
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                timeout=30
            )
            
            return response.choices[0].message.content
        
        except (ConnectionRefusedError, TimeoutError) as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
        
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise


# Usage
async def main():
    try:
        response = await robust_cloud_chat("What photos do I have?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Failed after retries: {e}")

asyncio.run(main())
```

---

## Tips and Best Practices

### 1. Cost Optimization
```python
# Cache RAG context for multiple related questions
cached_context = await get_rag_context(initial_question)

# Use the same context for follow-up questions if they're related
# This saves on embedding API calls
```

### 2. Streaming Responses
```python
# For streaming responses from cloud LLMs
async def stream_chat_with_openai(user_message: str):
    context_data = await get_rag_context(user_message)
    
    # ... build messages ...
    
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='')
```

### 3. Batch Processing
```python
# Process multiple questions efficiently
async def batch_questions(questions: list[str]):
    results = []
    
    for question in questions:
        context_data = await get_rag_context(question)
        # Process with cloud LLM
        result = await call_llm(context_data, question)
        results.append(result)
    
    return results
```

### 4. Fallback Strategy
```python
async def chat_with_fallback(user_message: str):
    """Try cloud LLM first, fall back to local if it fails."""
    try:
        # Try cloud LLM with RAG
        return await chat_with_gpt4(user_message)
    except Exception as e:
        logger.warning(f"Cloud LLM failed: {e}, falling back to local")
        # Fall back to local /api/chat endpoint
        return await local_chat(user_message)
```

---

## Performance Considerations

1. **Latency**: Cloud-chat adds local RAG search time (~100-500ms) + cloud LLM API time (~1-3s)
2. **Network**: Requires stable internet connection for cloud LLM calls
3. **Costs**: Each cloud LLM call has associated API costs (check provider pricing)
4. **Rate Limits**: Respect cloud provider rate limits (implement backoff/retry)

## Security Considerations

1. **API Keys**: Store cloud LLM API keys securely (environment variables, secrets manager)
2. **Data Privacy**: RAG context is sent to cloud providers - ensure compliance
3. **Sensitive Files**: Consider filtering sensitive files from RAG context before sending to cloud
4. **Audit Logging**: Log all cloud LLM calls for audit trails

---

## Conclusion

The `/api/cloud-chat` endpoint provides a flexible way to combine local file indexing with powerful cloud LLMs. Choose this approach when:
- You need the best response quality
- You want features specific to cloud models
- Your use case justifies the API costs
- You're comfortable sending RAG context to cloud providers

For fully local, private operation, use the standard `/api/chat` endpoint instead.
