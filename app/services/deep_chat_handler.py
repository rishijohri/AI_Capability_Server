"""Deep Chat Handler — OpenAI Tool-Calling Loop with budget enforcement.

Architecture:

    System prompt: Describes tools, budget, and quality-evaluation instructions.
                   Zero library data pre-loaded — agent discovers everything via tools.

    Tool-calling loop:
        1. Build messages (system + history + user).
        2. Call generate_with_tools() with remaining tool definitions.
        3. If tool_calls present  → execute each tool → append result → decrement budget.
        4. If content only        → final answer, break.
        5. Budget == 1            → only scoped_rag_search offered.
        6. Budget == 0            → tools omitted, model forced to answer.

    History truncation:
        After each tool result is consumed, the stored role:tool message content
        is truncated to a compact summary.  Budget annotations are always preserved.
        Assistant <think>…</think> blocks are NEVER truncated.
"""

import asyncio
import json
import logging
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from app.models.responses import WebSocketMessage
from app.services.conversation_compaction_service import (
    get_conversation_compaction_service,
)
from app.services.deep_chat_prompts import build_deep_chat_system_prompt
from app.services.mcp_tools import MCPToolRegistry

logger = logging.getLogger(__name__)

# Safety cap — never run more than this many iterations regardless of budget
MAX_ITERATIONS = 20


async def _build_library_context(
    user_message: str,
    tool_registry: "MCPToolRegistry",
    metadata_store,
    config,
) -> str:
    """Gather global library tags, overall date range, and relevant conversation history.

    Returns a formatted string injected into the system prompt so the agent
    can skip initial exploration calls when the global context is sufficient.
    """
    parts: List[str] = []

    # ----- Global tags + date range from all files -----
    try:
        all_meta = metadata_store.get_all_metadata()
        if all_meta:
            dates: List[datetime] = []
            tag_counter: Counter = Counter()
            for m in all_meta:
                try:
                    if m.creationTime:
                        dt = datetime.fromisoformat(
                            m.creationTime.replace("Z", "+00:00")
                        ).replace(tzinfo=None)
                        dates.append(dt)
                except (ValueError, AttributeError):
                    pass
                for tag in (m.tags or []):
                    tag_counter[tag.lower()] += 1

            top_m = getattr(config, "max_tags_per_scope", 50) if config else 50
            top_tags = tag_counter.most_common(top_m)

            overview_lines = [f"Total files in library: {len(all_meta)}"]
            if dates:
                min_d = min(dates).date()
                max_d = max(dates).date()
                overview_lines.append(f"Overall date range: {min_d} to {max_d}")
            if top_tags:
                tag_str = ", ".join(f"{t}({c})" for t, c in top_tags)
                overview_lines.append(f"Top tags (tag: file count): {tag_str}")

            parts.append("LIBRARY OVERVIEW:\n" + "\n".join(f"  {line}" for line in overview_lines))
    except Exception as exc:
        logger.warning(f"_build_library_context: failed to compute library overview: {exc}")

    # ----- Relevant past conversations -----
    try:
        conv_result = await tool_registry._get_conversation_rag(query=user_message)
        if conv_result:
            parts.append("RELEVANT PAST CONVERSATIONS (pre-searched for your query):\n" + conv_result)
    except Exception as exc:
        logger.warning(f"_build_library_context: conversation RAG failed: {exc}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_conclusion(text: str) -> str:
    """Extract text from <conclusion>…</conclusion> tags, fallback to cleaned full text."""
    m = re.search(r"<conclusion>(.*?)</conclusion>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: strip XML/think tags
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    return cleaned.strip()


def _extract_files(text: str) -> List[str]:
    """Extract file names listed in <files>…</files> tags."""
    m = re.search(r"<files>(.*?)</files>", text, re.DOTALL)
    if not m:
        return []
    raw = m.group(1).strip()
    files = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-• ")
        if line:
            files.append(line)
    return files


def _truncate_tool_result(content: str, config) -> str:
    """Shorten a stored tool result, preserving the budget annotation line."""
    # Always keep the budget line if present
    budget_line = ""
    budget_match = re.search(r"\[Tool calls remaining:.*?\]", content)
    if budget_match:
        budget_line = "\n" + budget_match.group(0)

    limit = getattr(config, "tool_history_max_results", 5) if config else 5
    lines = content.split("\n")

    # Keep first line (summary header) + up to limit result lines
    header = lines[0] if lines else ""
    body_lines = [l for l in lines[1:] if l.strip() and "[Tool calls remaining" not in l]
    kept = body_lines[:limit]
    omitted = len(body_lines) - len(kept)

    truncated = header
    if kept:
        truncated += "\n" + "\n".join(kept)
    if omitted > 0:
        truncated += f"\n... ({omitted} more results omitted)"
    truncated += budget_line
    return truncated


def _apply_history_truncation(messages: List[Dict[str, Any]], config) -> None:
    """Truncate role:tool messages in-place. Never touch role:assistant messages."""
    for msg in messages:
        if msg.get("role") == "tool":
            msg["content"] = _truncate_tool_result(msg.get("content", ""), config)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

async def run_deep_chat(
    websocket,
    user_message: str,
    active_history: List[Dict[str, Any]],
    config,
    metadata_store,
    rag_service,
    llm_service,
    face_service,
    rag_available: bool,
    embedding_loaded: bool,
    use_vision: bool,
    image_base64: Optional[str],
    image_name: Optional[str],
    image_tags: List[str],
    image_description: Optional[str],
    chat_model: str,
    vision_model: Optional[str],
    embedding_model: str,
    mmproj_file: Optional[str],
):
    """Deep Chat — OpenAI tool-calling loop with budget enforcement."""
    await websocket.send_json(
        WebSocketMessage(type="status", message="Deep Chat: Starting tool-calling loop...").to_json()
    )

    # ------------------------------------------------------------------
    # Inject attached image context into user message if present
    # ------------------------------------------------------------------
    if image_name and (image_tags or image_description):
        image_context = "[Attached image context: "
        if image_description:
            image_context += f"Description '{image_description}'. "
        if image_tags:
            image_context += f"Tags '{', '.join(image_tags)}'"
        image_context += "]\n\n"
        user_message = image_context + user_message

    # ------------------------------------------------------------------
    # Initialise compaction service and tool registry
    # ------------------------------------------------------------------
    compaction_service = get_conversation_compaction_service()
    if not compaction_service.is_loaded():
        compaction_service.load()

    tool_registry = MCPToolRegistry(
        metadata_store=metadata_store,
        llm_service=llm_service,
        embedding_model=embedding_model,
        chat_model=chat_model,
        compaction_service=compaction_service,
        config=config,
    )

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------
    budget: int = max(1, getattr(config, "chat_rounds", 3))
    tools_called: int = 0

    # ------------------------------------------------------------------
    # Pre-load library context (global tags, date range, conversation RAG)
    # ------------------------------------------------------------------
    await websocket.send_json(
        WebSocketMessage(type="status", message="Gathering library context...").to_json()
    )
    library_context = await _build_library_context(
        user_message=user_message,
        tool_registry=tool_registry,
        metadata_store=metadata_store,
        config=config,
    )
    logger.info(f"Library context built ({len(library_context)} chars)")

    # ------------------------------------------------------------------
    # Build initial message list
    # ------------------------------------------------------------------
    system_prompt = build_deep_chat_system_prompt(
        tool_call_budget=budget,
        library_context=library_context or None,
    )

    # Build user content (with optional image)
    if image_name and image_base64:
        user_content: Any = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            {"type": "text", "text": user_message},
        ]
    else:
        user_content = user_message

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        *active_history,
        {"role": "user", "content": user_content},
    ]

    logger.info(
        f"Deep chat start: budget={budget}, "
        f"history={len(active_history)} messages"
    )

    # ------------------------------------------------------------------
    # Tool-calling loop
    # ------------------------------------------------------------------
    remaining_budget = budget
    final_text = ""

    for iteration in range(MAX_ITERATIONS):
        # Choose tools based on remaining budget
        if remaining_budget <= 0:
            tools: List[Dict[str, Any]] = []
        elif remaining_budget == 1:
            tools = tool_registry.get_tool_definitions(only_rag=True)
        else:
            tools = tool_registry.get_tool_definitions(only_rag=False)

        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message=f"[Iteration {iteration + 1}] Thinking... ({remaining_budget} tool call(s) remaining)",
                data={"iteration": iteration + 1, "budget_remaining": remaining_budget, "tools_called": tools_called},
            ).to_json()
        )

        try:
            if tools:
                response_msg = await llm_service.generate_with_tools(
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
            else:
                # No tools — force plain text response
                raw = ""
                async for chunk in llm_service.generate(messages, stream=False):
                    raw += chunk
                response_msg = {"role": "assistant", "content": raw}
        except Exception as exc:
            logger.error(f"LLM call failed (iteration {iteration}): {exc}", exc_info=True)
            final_text = "An error occurred while generating the response. Please try again."
            break

        # Append assistant message to history
        messages.append({"role": "assistant", "content": response_msg.get("content", ""), **{
            k: v for k, v in response_msg.items() if k not in ("role", "content")
        }})

        tool_calls = response_msg.get("tool_calls")

        # -------------------------------------------------------------------
        # No tool calls → final answer
        # -------------------------------------------------------------------
        if not tool_calls:
            final_text = response_msg.get("content", "")
            logger.info(f"Final answer after {tools_called} tool call(s), {iteration + 1} iteration(s)")
            break

        # Intermediate iteration — full LLM output is thinking; send with rich metadata
        thinking_content = response_msg.get("content", "")
        if thinking_content:
            await websocket.send_json(
                WebSocketMessage(
                    type="thinking",
                    message=thinking_content,
                    data={
                        "iteration": iteration + 1,
                        "budget_remaining": remaining_budget,
                        "tools_called": tools_called,
                        "pending_tool_calls": len(tool_calls),
                    },
                ).to_json()
            )

        # -------------------------------------------------------------------
        # Execute tool calls
        # -------------------------------------------------------------------
        for tc_index, tc in enumerate(tool_calls):
            if remaining_budget <= 0:
                logger.warning("Budget exhausted mid-batch — skipping remaining tool calls")
                break

            tool_name = tc.get("function", {}).get("name", "")
            tool_call_id = tc.get("id", f"call_{tools_called}")

            try:
                arguments = json.loads(tc.get("function", {}).get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}

            logger.info(
                f"Tool call {tools_called + 1}: {tool_name}({json.dumps(arguments)[:120]})"
            )

            # Send tool invocation with pretty-printed args
            args_pretty = json.dumps(arguments, indent=2, ensure_ascii=False)
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"[Tool call {tools_called + 1}] {tool_name}\n{args_pretty}",
                    data={
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "tool_call_index": tc_index + 1,
                        "tool_call_number": tools_called + 1,
                        "budget_remaining": remaining_budget - 1,
                    },
                ).to_json()
            )

            result_text = await tool_registry.execute_tool(tool_name, arguments)

            # Send full tool result with metadata
            await websocket.send_json(
                WebSocketMessage(
                    type="progress",
                    message=result_text,
                    data={
                        "tool_name": tool_name,
                        "tool_call_number": tools_called + 1,
                        "result_length": len(result_text),
                        "budget_after": remaining_budget - 1,
                    },
                ).to_json()
            )

            remaining_budget -= 1
            tools_called += 1

            # Append budget annotation
            if remaining_budget == 0:
                budget_note = "\n\n[Tool calls remaining: 0 — generate your final answer now]"
            else:
                budget_note = f"\n\n[Tool calls remaining: {remaining_budget}]"
            result_with_budget = result_text + budget_note

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result_with_budget,
            })

            logger.info(
                f"Tool {tool_name} result: {len(result_text)} chars. "
                f"Budget now: {remaining_budget}"
            )

        # Truncate historic tool results AFTER they've been consumed (keep budget lines)
        _apply_history_truncation(messages[:-1], config)  # spare the just-added message

        # If budget exhausted after tool execution, force one more LLM call for the answer
        if remaining_budget <= 0 and tool_calls:
            # Will loop once more with tools=[] to get final text
            continue

    else:
        # Safety: hit MAX_ITERATIONS
        logger.warning(f"Hit MAX_ITERATIONS ({MAX_ITERATIONS}) — using last content as answer")
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                final_text = msg["content"]
                break

    # ------------------------------------------------------------------
    # Parse conclusion and file list
    # ------------------------------------------------------------------
    conclusion = _extract_conclusion(final_text) if final_text else ""
    if not conclusion:
        logger.warning("Empty conclusion generated")
        conclusion = (
            "I couldn't find enough data to answer confidently. "
            "Try specifying a date range or topic more precisely."
        )

    files_list = _extract_files(final_text)

    # Always include the attached image if it was used
    if image_name and image_name not in files_list:
        files_list.append(image_name)

    # ------------------------------------------------------------------
    # Stream conclusion to client
    # ------------------------------------------------------------------
    chunk_size = 80
    for i in range(0, len(conclusion), chunk_size):
        await websocket.send_json(
            WebSocketMessage(type="progress", message=conclusion[i : i + chunk_size]).to_json()
        )
        await asyncio.sleep(0.02)

    await websocket.send_json(
        WebSocketMessage(type="conclusion", message=conclusion).to_json()
    )

    if files_list:
        await websocket.send_json(
            WebSocketMessage(
                type="files",
                message=", ".join(files_list),
                data={"files": files_list},
            ).to_json()
        )

    await websocket.send_json(
        WebSocketMessage(
            type="full_response",
            message=conclusion,
            data={"tools_called": tools_called, "files": files_list},
        ).to_json()
    )
