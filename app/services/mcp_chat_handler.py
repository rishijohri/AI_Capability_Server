"""MCP Chat Handler — multi-round tool-calling chat loop for MCP mode."""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.config import get_config
from app.models.responses import WebSocketMessage
from app.services.mcp_tools import MCPToolRegistry

logger = logging.getLogger(__name__)


def _select_system_prompt(round_idx: int, total_rounds: int, config) -> str:
    """Select the appropriate system prompt based on round position.
    
    Round selection logic:
    - total_rounds == 1 → final only (no tools)
    - total_rounds == 2 → first + final
    - total_rounds == 3 → first + penultimate + final
    - total_rounds >= 4 → first + intermediate(s) + penultimate + final
    """
    last = total_rounds - 1
    if round_idx == last:
        return config.mcp_final_round_prompt
    if round_idx == 0:
        return config.mcp_first_round_prompt
    if round_idx == last - 1:
        return config.mcp_penultimate_round_prompt
    return config.mcp_intermediate_round_prompt


def _parse_structured_output(text: str):
    """Parse <conclusion> and <files> from model output. Returns (conclusion, files_list)."""
    sanitized = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    conclusion_match = re.search(r'<conclusion>(.*?)</conclusion>', sanitized, re.DOTALL)
    files_match = re.search(r'<files>(.*?)</files>', sanitized, re.DOTALL)

    conclusion = conclusion_match.group(1).strip() if conclusion_match else None
    files_list = []
    if files_match:
        files_content = files_match.group(1).strip()
        if files_content:
            parts = re.split(r'\r?\n|,|\-', files_content)
            for p in parts:
                p = p.strip()
                if p:
                    p = re.sub(r'^[\-*]\s*', '', p)
                    files_list.append(p)
    return conclusion, files_list


async def run_mcp_chat(
    websocket,
    user_message: str,
    active_history: List[Dict[str, Any]],
    config,
    metadata_store,
    rag_service,
    knowledge_service,
    llm_service,
    face_service,
    rag_available: bool,
    embedding_loaded: bool,
    use_vision: bool = False,
    image_base64: Optional[str] = None,
    image_tags: Optional[List[str]] = None,
    image_description: Optional[str] = None,
    chat_model: str = "",
    mmproj_file: Optional[str] = None,
):
    """Run multi-round MCP tool-calling chat.

    Sends WebSocket messages identical to RAG mode (status, progress, conclusion, files, result).
    """
    total_rounds = config.chat_rounds
    current_date = datetime.now().strftime("%B %d, %Y")

    # Build the tool registry
    registry = MCPToolRegistry(
        metadata_store=metadata_store,
        rag_service=rag_service,
        knowledge_service=knowledge_service,
        llm_service=llm_service,
        face_service=face_service,
        embedding_loaded=embedding_loaded,
        rag_available=rag_available,
    )
    tools = registry.get_tool_definitions()

    # Working message history for the tool-calling loop.
    # Start with the user's conversation history (which already has the user message appended).
    working_history: List[Dict[str, Any]] = list(active_history)

    response_text = ""
    all_referenced_files: List[str] = []
    skip_to_final = False

    for round_idx in range(total_rounds):
        is_last_round = round_idx == total_rounds - 1
        prompt_template = _select_system_prompt(round_idx, total_rounds, config)

        # Inject dynamic context into system prompt
        system_content = f"{prompt_template}\n\nCurrent Date: {current_date}"
        system_content += f"\nRound {round_idx + 1} of {total_rounds}."
        if not is_last_round and not skip_to_final:
            system_content += f" {total_rounds - round_idx - 1} round(s) remaining after this one."

        # Add vision context if present
        if use_vision and image_base64 and image_tags:
            image_ctx_parts = []
            if image_tags:
                image_ctx_parts.append(f"Image Tags: {', '.join(image_tags)}")
            if image_description:
                image_ctx_parts.append(f"Image Description: {image_description}")
            if image_ctx_parts:
                system_content += "\n\nImage Context:\n" + "\n".join(image_ctx_parts)

        # Build messages for this round
        messages = [{"role": "system", "content": system_content}] + working_history

        if is_last_round or skip_to_final:
            # Final round — stream without tools
            if skip_to_final:
                # Use final round prompt when skipping ahead
                final_system = f"{config.mcp_final_round_prompt}\n\nCurrent Date: {current_date}"
                final_system += f"\nProvide your final answer based on the information gathered so far."
                if use_vision and image_base64 and image_tags:
                    image_ctx_parts = []
                    if image_tags:
                        image_ctx_parts.append(f"Image Tags: {', '.join(image_tags)}")
                    if image_description:
                        image_ctx_parts.append(f"Image Description: {image_description}")
                    if image_ctx_parts:
                        final_system += "\n\nImage Context:\n" + "\n".join(image_ctx_parts)
                messages = [{"role": "system", "content": final_system}] + working_history

            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"Round {round_idx + 1}/{total_rounds}: Generating final answer..."
                ).to_json()
            )

            # If vision model is active and image provided, use vision generation
            if use_vision and image_base64:
                import base64
                image_bytes = base64.b64decode(image_base64)
                prompt = f"{system_content}\n\nUser: {user_message}\n\nAssistant:"
                response_text = await llm_service.generate_vision(image_bytes, prompt, mmproj_file)
                await websocket.send_json(
                    WebSocketMessage(
                        type="progress",
                        message=response_text,
                        data={"partial_response": response_text}
                    ).to_json()
                )
            else:
                async for chunk in llm_service.generate(messages, stream=True):
                    response_text += chunk
                    await websocket.send_json(
                        WebSocketMessage(
                            type="progress",
                            message=chunk,
                            data={"partial_response": response_text}
                        ).to_json()
                    )
            break  # Done after final streaming round
        else:
            # Non-final round — call with tools
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"Round {round_idx + 1}/{total_rounds}: Analyzing with tools..."
                ).to_json()
            )

            try:
                result_msg = await llm_service.generate_with_tools(
                    messages, tools, tool_choice="auto"
                )
            except Exception as e:
                logger.error(f"MCP round {round_idx + 1} generate_with_tools failed: {e}", exc_info=True)
                await websocket.send_json(
                    WebSocketMessage(
                        type="status",
                        message=f"Tool calling error in round {round_idx + 1}, falling back to direct answer: {str(e)}"
                    ).to_json()
                )
                # Skip to final streaming round instead of aborting
                skip_to_final = True
                continue

            tool_calls = result_msg.get("tool_calls")
            content = result_msg.get("content")

            if tool_calls:
                # Append assistant message with tool_calls + execute each tool
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tool_calls,
                }
                working_history.append(assistant_msg)

                for tc in tool_calls:
                    fn = tc.get("function", {})
                    tool_name = fn.get("name", "unknown")
                    tool_args_raw = fn.get("arguments", "{}")
                    tc_id = tc.get("id", "")

                    # Parse arguments
                    if isinstance(tool_args_raw, str):
                        try:
                            tool_args = json.loads(tool_args_raw)
                        except json.JSONDecodeError:
                            tool_args = {}
                    else:
                        tool_args = tool_args_raw

                    await websocket.send_json(
                        WebSocketMessage(
                            type="status",
                            message=f"Calling tool: {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:200]})"
                        ).to_json()
                    )

                    tool_result = await registry.execute_tool(tool_name, tool_args)

                    # Append tool result message
                    working_history.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": tool_result,
                    })

                    await websocket.send_json(
                        WebSocketMessage(
                            type="status",
                            message=f"Tool {tool_name} returned {len(tool_result)} chars"
                        ).to_json()
                    )
            else:
                # No tool calls — check if model produced a direct answer
                if content:
                    conclusion, files = _parse_structured_output(content)
                    if conclusion:
                        response_text = content
                        all_referenced_files = files
                        await websocket.send_json(
                            WebSocketMessage(
                                type="status",
                                message=f"Model provided early answer in round {round_idx + 1}."
                            ).to_json()
                        )
                        break

                # No conclusion found — skip to final streaming round.
                # Do NOT append the bare assistant message to avoid consecutive
                # assistant messages which llama-server rejects.
                await websocket.send_json(
                    WebSocketMessage(
                        type="status",
                        message=f"No tool calls in round {round_idx + 1}, moving to final answer..."
                    ).to_json()
                )
                skip_to_final = True

    # Parse and send structured output
    if response_text:
        conclusion, files_list = _parse_structured_output(response_text)
        if files_list:
            all_referenced_files = files_list
    else:
        conclusion = None

    sent_any = False

    if conclusion:
        await websocket.send_json(
            WebSocketMessage(
                type="conclusion",
                message=conclusion
            ).to_json()
        )
        sent_any = True

    # Send files
    await websocket.send_json(
        WebSocketMessage(
            type="files",
            message=", ".join(all_referenced_files) if all_referenced_files else "",
            data={"relevant_files": all_referenced_files}
        ).to_json()
    )
    if all_referenced_files:
        sent_any = True

    # Fallback if no structured tags found
    if not sent_any:
        clean = re.sub(r'<.*?>', '', response_text, flags=re.DOTALL).strip() if response_text else "No response generated."
        await websocket.send_json(
            WebSocketMessage(
                type="conclusion",
                message=clean
            ).to_json()
        )
        await websocket.send_json(
            WebSocketMessage(
                type="files",
                message="",
                data={"relevant_files": []}
            ).to_json()
        )

    # Send completion
    await websocket.send_json(
        WebSocketMessage(
            type="result",
            message="Response complete"
        ).to_json()
    )
