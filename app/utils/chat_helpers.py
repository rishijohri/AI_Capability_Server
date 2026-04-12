"""Helper functions for chat endpoints."""

from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime

from app.models import WebSocketMessage, MetadataStore, FileMetadata
from app.models.rag_result import RAGResult
from app.config import get_config
from app.services import get_llm_service, get_rag_service


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
    if rag_available:
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


def build_rag_context_from_results(results: List[RAGResult]) -> Tuple[str, List[str]]:
    """
    Build formatted RAG context and file list from search results.
    
    Args:
        results: List of RAGResult objects from RAG search (files and/or conversations)
        
    Returns:
        Tuple of (formatted_context_string, list_of_identifiers)
    """
    context_parts = ["Here are relevant items from the knowledge base:\n"]
    id_list = []
    
    for result in results:
        if result.source == "file" and result.file_metadata:
            file_meta = result.file_metadata
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
            id_list.append(file_meta.fileName)
        
        elif result.source == "conversation":
            context_parts.append(f"- Memory from conversation ({result.identifier})")
            context_parts.append(f"  Type: conversation_memory")
            if result.summary:
                context_parts.append(f"  Summary: {result.summary}")
            if result.compacted_at:
                context_parts.append(f"  Compacted: {result.compacted_at}")
            context_parts.append("")
            id_list.append(result.identifier)
    
    context = "\n".join(context_parts)
    return context, id_list



async def search_rag_for_context(
    websocket,
    user_message: str,
    active_history: List[Dict],
    rag_available: bool,
    embedding_loaded: bool
) -> Tuple[str, Optional[List[float]]]:
    """
    Perform comprehensive RAG search for regular chat context.
    Conversation memories are now included in the main RAG index, so
    a single search covers both files and compacted conversations.
    
    Returns:
        Tuple of (context_string, user_message_embedding)
    """
    config = get_config()
    llm_service = get_llm_service()
    rag_service = get_rag_service()
    
    context = ""
    user_message_embedding = None
    
    # Search media RAG (includes file + conversation results)
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
            results = await rag_service.search(search_query)
            context, _ = build_rag_context_from_results(results)
        except Exception as e:
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"RAG search failed: {e}"
                ).to_json()
            )
    
    return context, user_message_embedding
