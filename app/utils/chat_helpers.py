"""Helper functions for chat endpoints."""

from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime

from app.models import WebSocketMessage, MetadataStore, FileMetadata
from app.config import get_config
from app.services import get_llm_service, get_rag_service, get_knowledge_service


async def prepare_chat_session(websocket, metadata_store: MetadataStore):
    """
    Prepare chat session by loading RAG and embedding models.
    
    Returns:
        Tuple[bool, bool]: (rag_available, embedding_loaded)
    """
    config = get_config()
    llm_service = get_llm_service()
    rag_service = get_rag_service()
    
    # Check if metadata file has been updated
    if await metadata_store.reload_if_modified():
        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message="Storage metadata file was updated. Reloaded metadata."
            ).to_json()
        )
    
    # Try to load RAG
    rag_available = rag_service.is_loaded()
    if not rag_available:
        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message="Loading RAG database..."
            ).to_json()
        )
        load_result = rag_service.load_rag(metadata_store)
        rag_available = isinstance(load_result, dict) and load_result.get("success", False)
        if not rag_available:
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message="RAG not available — chatting without document context."
                ).to_json()
            )
    
    # Load embedding model if needed
    embedding_loaded = False
    if rag_available or config.enable_knowledge_storage:
        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message=f"Loading embedding model {config.embedding_model}..."
            ).to_json()
        )
        
        await llm_service.load_model(config.embedding_model)
        embedding_loaded = True
        
        # Send embedding model startup command
        embedding_startup_cmd = llm_service.get_startup_command()
        if embedding_startup_cmd:
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"Embedding Model Command: {embedding_startup_cmd}",
                    data={"embedding_startup_command": embedding_startup_cmd}
                ).to_json()
            )
            import asyncio
            await asyncio.sleep(0.1)
    
    return rag_available, embedding_loaded


async def validate_and_setup_history(websocket, provided_history: Optional[List[Dict]]) -> Optional[List[Dict]]:
    """
    Validate conversation history format.
    
    Returns:
        List of validated history or None if validation failed
    """
    if provided_history is not None:
        if not isinstance(provided_history, list):
            await websocket.send_json(
                WebSocketMessage(
                    type="error",
                    message="history parameter must be a list of message objects"
                ).to_json()
            )
            return None
        
        for msg in provided_history:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                await websocket.send_json(
                    WebSocketMessage(
                        type="error",
                        message="Each history item must be a dict with 'role' and 'content' keys"
                    ).to_json()
                )
                return None
            if msg["role"] not in ["user", "assistant"]:
                await websocket.send_json(
                    WebSocketMessage(
                        type="error",
                        message="History role must be 'user' or 'assistant'"
                    ).to_json()
                )
                return None
        
        return provided_history.copy()
    
    return []


async def load_image_for_chat(
    websocket, 
    image_name: str, 
    metadata_store: MetadataStore
) -> Tuple[Optional[str], List[str], Optional[str], Optional[str]]:
    """
    Load and process image for visual chat.
    
    Returns:
        Tuple[image_base64, image_tags, image_description, error_message]
    """
    config = get_config()
    
    # Get file metadata to determine type
    image_metadata = metadata_store.get_metadata_by_filename(image_name)
    file_type = image_metadata.type if image_metadata and hasattr(image_metadata, 'type') else "image"
    
    type_label = "PDF document" if file_type == "pdf" else "video" if file_type == "video" else "image"
    await websocket.send_json(
        WebSocketMessage(
            type="status",
            message=f"Processing {type_label}: {image_name}..."
        ).to_json()
    )
    
    image_tags = []
    image_description = None
    if image_metadata:
        image_tags = image_metadata.tags
        image_description = image_metadata.description
    
    # Load image
    from app.utils import ImageProcessor
    image_base64, error_msg = await ImageProcessor.load_image_as_base64(
        image_name, 
        metadata_store, 
        config.image_quality
    )
    
    if error_msg:
        await websocket.send_json(
            WebSocketMessage(
                type="error",
                message=error_msg
            ).to_json()
        )
        return None, image_tags, image_description, error_msg
    
    # Send success message
    import base64
    image_bytes_len = len(base64.b64decode(image_base64))
    tag_info = f" (Tags: {', '.join(image_tags)})" if image_tags else ""
    await websocket.send_json(
        WebSocketMessage(
            type="status",
            message=f"Image loaded: {image_name} ({image_bytes_len / 1024:.1f} KB){tag_info}"
        ).to_json()
    )
    
    return image_base64, image_tags, image_description, None


def build_rag_context_from_results(relevant_files: List[FileMetadata]) -> Tuple[str, List[str]]:
    """
    Build formatted RAG context and file list from search results.
    
    Args:
        relevant_files: List of FileMetadata objects from RAG search
        
    Returns:
        Tuple of (formatted_context_string, list_of_filenames)
    """
    context_parts = ["Here are relevant files from the knowledge base:\n"]
    file_list = []
    
    for file_meta in relevant_files:
        context_parts.append(f"- {file_meta.fileName}")
        context_parts.append(f"  Type: {file_meta.type}")
        if file_meta.creationTime:
            context_parts.append(f"  Created: {file_meta.creationTime}")
        context_parts.append(f"  Tags: {', '.join(file_meta.tags)}")
        if file_meta.description:
            context_parts.append(f"  Description: {file_meta.description}")
        
        # Include any extra/unknown metadata fields (Pydantic v2 stores in __pydantic_extra__)
        if hasattr(file_meta, '__pydantic_extra__') and file_meta.__pydantic_extra__:
            for field_name, field_value in file_meta.__pydantic_extra__.items():
                if field_value is not None:
                    if isinstance(field_value, list):
                        context_parts.append(f"  {field_name}: {', '.join(str(v) for v in field_value)}")
                    elif isinstance(field_value, dict):
                        import json
                        context_parts.append(f"  {field_name}: {json.dumps(field_value)}")
                    else:
                        context_parts.append(f"  {field_name}: {field_value}")
        
        context_parts.append("")
        file_list.append(file_meta.fileName)
    
    context = "\n".join(context_parts)
    return context, file_list


async def gather_initial_context(
    websocket,
    user_message: str,
    active_history: List[Dict],
    rag_available: bool,
    embedding_loaded: bool
) -> str:
    """
    Gather initial context from RAG databases (limited for Deep Chat).
    
    Returns:
        String containing initial context or empty string
    """
    config = get_config()
    llm_service = get_llm_service()
    rag_service = get_rag_service()
    knowledge_service = get_knowledge_service()
    
    if not (rag_available or (config.enable_knowledge_storage and embedding_loaded)):
        return ""
    
    await websocket.send_json(
        WebSocketMessage(
            type="status",
            message="Gathering initial context..."
        ).to_json()
    )
    
    context_parts = []
    
    try:
        # Search media RAG if available
        if rag_available:
            search_query_parts = []
            for msg in active_history:
                role = msg["role"]
                content = msg["content"]
                search_query_parts.append(f'{role}: "{content}"')
            search_query = ", ".join(search_query_parts)
            
            relevant_files = await rag_service.search(search_query, k=3)
            media_context, _ = build_rag_context_from_results(relevant_files)
            if media_context and "Here are relevant files" in media_context:
                context_parts.append(
                    f"Initial Media Context (consider using query_media_rag for more):\n{media_context}"
                )
        
        # Search fact RAG if available
        if config.enable_knowledge_storage and knowledge_service.is_loaded():
            user_message_embedding = await llm_service.embed(user_message)
            relevant_facts = knowledge_service.select_knowledge(
                user_message_embedding,
                token_budget=500,
                min_relevance=config.min_knowledge_relevance,
            )
            if relevant_facts:
                fact_lines = [f"- {f['message']}" for f in relevant_facts]
                context_parts.append(
                    f"Initial Fact Context (consider using query_fact_rag for more):\n" + 
                    "\n".join(fact_lines)
                )
        
        if context_parts:
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"Initial context gathered from {len(context_parts)} source(s)"
                ).to_json()
            )
            return "\n\n".join(context_parts)
    
    except Exception as e:
        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message=f"Initial context gathering skipped: {e}"
            ).to_json()
        )
    
    return ""


async def search_rag_for_context(
    websocket,
    user_message: str,
    active_history: List[Dict],
    rag_available: bool,
    embedding_loaded: bool
) -> Tuple[str, Optional[List[float]]]:
    """
    Perform comprehensive RAG search for regular chat context.
    
    Returns:
        Tuple of (context_string, user_message_embedding)
    """
    config = get_config()
    llm_service = get_llm_service()
    rag_service = get_rag_service()
    knowledge_service = get_knowledge_service()
    
    context = ""
    user_message_embedding = None
    
    # Search media RAG
    if rag_available:
        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message="Searching knowledge base..."
            ).to_json()
        )
        
        # Build search query from history
        search_query_parts = []
        for msg in active_history:
            role = msg["role"]
            content = msg["content"]
            search_query_parts.append(f'{role}: "{content}"')
        search_query = ", ".join(search_query_parts)
        
        try:
            relevant_files = await rag_service.search(search_query)
            context, _ = build_rag_context_from_results(relevant_files)
        except Exception as e:
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"RAG search failed: {e}"
                ).to_json()
            )
    
    # Search conversation knowledge base
    knowledge_context = ""
    if config.enable_knowledge_storage and embedding_loaded:
        try:
            user_message_embedding = await llm_service.embed(user_message)
            if knowledge_service.is_loaded():
                relevant_facts = knowledge_service.select_knowledge(
                    user_message_embedding,
                    token_budget=config.max_knowledge_tokens,
                    min_relevance=config.min_knowledge_relevance,
                )
                if relevant_facts:
                    fact_lines = [f"- {f['message']}" for f in relevant_facts]
                    knowledge_context = (
                        "\n\nRelevant information from previous conversations:\n" + 
                        "\n".join(fact_lines)
                    )
                    await websocket.send_json(
                        WebSocketMessage(
                            type="status",
                            message=f"Found {len(relevant_facts)} relevant facts from conversation history"
                        ).to_json()
                    )
        except Exception as e:
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"Knowledge retrieval skipped: {e}"
                ).to_json()
            )
    
    return context + knowledge_context, user_message_embedding


async def store_objective_facts(
    user_message: str,
    user_message_embedding: Optional[List[float]],
    embedding_loaded: bool
):
    """Store objective facts from conversation into knowledge base."""
    config = get_config()
    
    if not config.enable_knowledge_storage:
        return
    
    try:
        llm_service = get_llm_service()
        knowledge_service = get_knowledge_service()
        knowledge_service._objectivity_threshold = config.objectivity_threshold
        
        user_obj, user_score = knowledge_service.should_store(user_message)
        if user_obj:
            # If we don't have embedding, generate it
            if user_message_embedding is None and embedding_loaded:
                user_message_embedding = await llm_service.embed(user_message)
            
            knowledge_service.add_fact(
                user_message, 
                "user", 
                user_score,
                embedding=user_message_embedding,
            )
    except Exception as e:
        print(f"Knowledge storage error (non-fatal): {e}")
