"""API routes for AI Server."""

import asyncio
import re
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pathlib import Path
import json
import traceback
import sys
import os
import signal
from typing import Optional
from datetime import datetime
import sys

from app.config import get_config, update_config, set_storage_metadata_path, get_available_models
from app.utils.bookmark_resolver import resolve_security_scoped_bookmark
from app.utils import chat_helpers
from app.models import (
    ConfigUpdateRequest,
    ConfigResponse,
    StorageMetadataRequest,
    ModelDirectoryRequest,
    StatusResponse,
    TagRequest,
    DescribeRequest,
    ChatRequest,
    WebSocketMessage,
    MetadataStore,
    FileMetadata,
    AvailableModelsRequest,
    ModelInfo,
    AvailableModelsResponse,
    ModelOption,
    ModelOptionsResponse,
    SetModelOptionsRequest
)
from app.utils.file_picker import is_permission_error
from app.services import (
    get_llm_service,
    get_embedding_service,
    get_rag_service,
    get_vision_service,
    get_face_service,
    get_knowledge_service
)

router = APIRouter()

# Global metadata store
_metadata_store: Optional[MetadataStore] = None


def get_metadata_store() -> MetadataStore:
    """Get the global metadata store."""
    if _metadata_store is None:
        raise HTTPException(
            status_code=400,
            detail="Storage metadata not set. Call /set-storage-metadata first."
        )
    return _metadata_store


def build_rag_context_from_results(relevant_files: list[FileMetadata]) -> tuple[str, list[str]]:
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


@router.get("/config", response_model=ConfigResponse)
async def get_configuration():
    """Get current configuration."""
    config = get_config()
    return ConfigResponse(
        reduced_embedding_size=config.reduced_embedding_size,
        chat_rounds=config.chat_rounds,
        image_quality=config.image_quality,
        llm_mode=config.llm_mode,
        top_k=config.top_k,
        recency_bias=config.recency_bias,
        enable_visual_chat=config.enable_visual_chat,
        chat_model=config.chat_model,
        embedding_model=config.embedding_model,
        vision_model=config.vision_model,
        mmproj_model=config.mmproj_model,
        chat_system_prompt=config.chat_system_prompt,
        tag_prompt=config.tag_prompt,
        describe_prompt=config.describe_prompt,
        vision_binary=config.vision_binary,
        backend=config.llm_mode,
        model_timeout=300,
        llm_timeout=config.llm_timeout,
        llm_params=config.llm_params.model_dump(),
        rag_directory_name=config.rag_directory_name,
        storage_metadata_path=config.storage_metadata_path,
        model_directory=config.model_directory,
        binary_config=config.binary_config,
        system_info=config.system_info,
        available_binary_configs=config.get_available_binary_configs()
    )


@router.post("/config", response_model=ConfigResponse)
async def update_configuration(request: ConfigUpdateRequest):
    """Update configuration."""
    update_data = request.model_dump(exclude_none=True)
    config = update_config(**update_data)
    
    return ConfigResponse(
        reduced_embedding_size=config.reduced_embedding_size,
        chat_rounds=config.chat_rounds,
        image_quality=config.image_quality,
        llm_mode=config.llm_mode,
        top_k=config.top_k,
        recency_bias=config.recency_bias,
        enable_visual_chat=config.enable_visual_chat,
        chat_model=config.chat_model,
        embedding_model=config.embedding_model,
        vision_model=config.vision_model,
        mmproj_model=config.mmproj_model,
        chat_system_prompt=config.chat_system_prompt,
        tag_prompt=config.tag_prompt,
        describe_prompt=config.describe_prompt,
        vision_binary=config.vision_binary,
        backend=config.llm_mode,
        model_timeout=300,
        llm_timeout=config.llm_timeout,
        llm_params=config.llm_params.model_dump(),
        rag_directory_name=config.rag_directory_name,
        storage_metadata_path=config.storage_metadata_path,
        model_directory=config.model_directory,
        binary_config=config.binary_config,
        system_info=config.system_info,
        available_binary_configs=config.get_available_binary_configs()
    )


@router.get("/available-models", response_model=AvailableModelsResponse)
async def get_models(task_type: Optional[str] = None):
    """
    Get available models filtered by task type.
    
    Args:
        task_type: Optional query parameter to filter by 'vision', 'chat', or 'embedding'.
                   If not provided, returns all models.
                   Note: task_type='chat' returns both chat and vision models since
                   vision models can also be used for chat conversations.
    
    Returns:
        List of available models with existence status for model files.
    
    Example:
        GET /api/available-models
        GET /api/available-models?task_type=vision
        GET /api/available-models?task_type=chat  (includes vision models)
        GET /api/available-models?task_type=embedding
    """
    # Validate task_type if provided
    if task_type and task_type not in ["vision", "chat", "embedding"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_type '{task_type}'. Must be one of: vision, chat, embedding"
        )
    
    # Get available models
    models_data = get_available_models(task_type)
    
    # Convert to ModelInfo objects
    models = [ModelInfo(**model_data) for model_data in models_data]
    
    return AvailableModelsResponse(
        models=models,
        total_count=len(models),
        task_type=task_type
    )


@router.get("/model-options", response_model=ModelOptionsResponse)
async def get_model_options(task_type: Optional[str] = None):
    """
    Get all models that can be downloaded, regardless of whether they exist locally.
    
    This endpoint returns all models defined in model_options, showing which ones
    have repo_id configured and are ready for download via /api/download-models.
    
    Args:
        task_type: Optional query parameter to filter by 'vision', 'chat', or 'embedding'.
                   If not provided, returns all models.
                   Note: task_type='chat' returns both chat and vision models since
                   vision models can also be used for chat conversations.
    
    Returns:
        List of all models with download configuration status.
    
    Example:
        GET /api/model-options
        GET /api/model-options?task_type=vision
        GET /api/model-options?task_type=chat  (includes vision models)
    """
    # Validate task_type if provided
    if task_type and task_type not in ["vision", "chat", "embedding"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_type '{task_type}'. Must be one of: vision, chat, embedding"
        )
    
    # Get model_options from settings
    from app.config.settings import model_options
    
    # Build list of ModelOption objects
    models = []
    configured_count = 0
    
    for model_id, model_info in model_options.items():
        model_type = model_info.get("type", "chat")
        
        # Filter by task_type if specified
        # Vision models can also be used as chat models
        if task_type:
            if task_type == "chat":
                # Include both chat and vision models for chat requests
                if model_type not in ["chat", "vision"]:
                    continue
            elif model_type != task_type:
                continue
        
        repo_id = model_info.get("repo_id", "")
        repo_id_configured = bool(repo_id and repo_id.strip())
        
        if repo_id_configured:
            configured_count += 1
        
        model_option = ModelOption(
            model_id=model_id,
            name=model_info.get("name", model_id),
            type=model_info.get("type", "chat"),
            model_file=model_info.get("model_file", ""),
            mmproj_file=model_info.get("mmproj_file"),
            repo_id=repo_id,
            repo_id_configured=repo_id_configured,
            llm_params=model_info.get("llm_params")
        )
        models.append(model_option)
    
    return ModelOptionsResponse(
        models=models,
        total_count=len(models),
        configured_count=configured_count,
        task_type=task_type
    )


@router.post("/set-model-options", response_model=StatusResponse)
async def set_model_options(request: SetModelOptionsRequest):
    """Set model options from the Flutter app.
    
    This endpoint receives model configurations from the Flutter client,
    which now manages persistent model options locally and syncs them
    to the server on startup. This replaces the static model_options
    dictionary with runtime configurations.
    
    Args:
        request: SetModelOptionsRequest with list of model options
        
    Returns:
        StatusResponse with success/failure status
    """
    from app.config.settings import model_options
    
    try:
        # Clear existing model_options and replace with client-provided ones
        model_options.clear()
        
        received_count = 0
        for model in request.models:
            model_dict = {
                "name": model.name,
                "type": model.type,
                "model_file": model.model_file,
                "repo_id": model.repo_id,
            }
            
            # Add optional fields if present
            if model.mmproj_file:
                model_dict["mmproj_file"] = model.mmproj_file
            if model.llm_params:
                model_dict["llm_params"] = model.llm_params
                
            model_options[model.model_id] = model_dict
            received_count += 1
        
        return StatusResponse(
            status="success",
            message=f"Successfully set {received_count} model options"
        )
        
    except Exception as e:
        return StatusResponse(
            status="error",
            message=f"Failed to set model options: {str(e)}"
        )


@router.websocket("/download-models")
async def download_models_ws(websocket: WebSocket):
    """Download models from Hugging Face repositories (WebSocket)."""
    await websocket.accept()
    
    try:
        # Receive download request
        data = await websocket.receive_json()
        
        # Import DownloadModelsRequest and response models
        from app.models import DownloadModelsRequest, DownloadModelsResponse, DownloadStatus
        
        # Validate request
        try:
            request = DownloadModelsRequest(**data)
        except Exception as e:
            await websocket.send_json(
                WebSocketMessage(
                    type="error",
                    message=f"Invalid request: {str(e)}"
                ).to_json()
            )
            return
        
        # Check if huggingface_hub is installed
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            await websocket.send_json(
                WebSocketMessage(
                    type="error",
                    message="huggingface_hub library not installed. Please install it with: pip install huggingface_hub"
                ).to_json()
            )
            return
        
        # Get configuration
        from app.config.settings import model_options
        config = get_config()
        
        # Determine download directory
        if request.download_location:
            download_dir = Path(request.download_location)
            download_dir.mkdir(parents=True, exist_ok=True)
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"Using custom download location: {download_dir}"
                ).to_json()
            )
        else:
            download_dir = config.get_model_path("")
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"Using default model directory: {download_dir}"
                ).to_json()
            )
        
        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message=f"Starting download for {len(request.model_ids)} model(s)..."
            ).to_json()
        )
        
        # Process each model
        for model_id in request.model_ids:
            # Validate model_id
            if model_id not in model_options:
                await websocket.send_json(
                    WebSocketMessage(
                        type="error",
                        message=f"Invalid model_id: {model_id}. Not found in model_options."
                    ).to_json()
                )
                continue
            
            model_info = model_options[model_id]
            repo_id = model_info.get("repo_id")
            
            # Check if repo_id is configured
            if not repo_id or not repo_id.strip():
                await websocket.send_json(
                    WebSocketMessage(
                        type="error",
                        message=f"Model {model_id} does not have a repo_id configured. Please add repo_id to model_options in settings.py"
                    ).to_json()
                )
                continue
            
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"Processing model: {model_id}"
                ).to_json()
            )
            
            # Get files to download
            files_to_download = []
            model_file = model_info.get("model_file")
            mmproj_file = model_info.get("mmproj_file")
            
            if model_file:
                files_to_download.append(("model", model_file))
            if mmproj_file:
                files_to_download.append(("mmproj", mmproj_file))
            
            if not files_to_download:
                await websocket.send_json(
                    WebSocketMessage(
                        type="error",
                        message=f"Model {model_id} has no files configured (model_file or mmproj_file)"
                    ).to_json()
                )
                continue
            
            # Download each file
            file_statuses = []
            
            for file_type, filename in files_to_download:
                # Check if file already exists
                model_path = download_dir / filename
                
                if model_path.exists() and not request.force_redownload:
                    await websocket.send_json(
                        WebSocketMessage(
                            type="progress",
                            message=f"Skipping {filename} (already exists)",
                            data={
                                "filename": filename,
                                "bytes_downloaded": model_path.stat().st_size,
                                "total_bytes": model_path.stat().st_size
                            }
                        ).to_json()
                    )
                    file_statuses.append(DownloadStatus(
                        filename=filename,
                        status="skipped",
                        error=None,
                        bytes_downloaded=model_path.stat().st_size,
                        total_bytes=model_path.stat().st_size
                    ))
                    continue
                
                # Download the file
                try:
                    await websocket.send_json(
                        WebSocketMessage(
                            type="status",
                            message=f"Downloading {filename} from {repo_id}..."
                        ).to_json()
                    )
                    
                    # Ensure download directory exists
                    download_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Download file
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=str(download_dir),
                        local_dir_use_symlinks=False,
                        resume_download=True
                    )
                    
                    # Get file size
                    file_size = Path(downloaded_path).stat().st_size
                    size_mb = file_size / (1024 * 1024)
                    
                    await websocket.send_json(
                        WebSocketMessage(
                            type="progress",
                            message=f"Downloaded {filename} ({size_mb:.1f} MB)",
                            data={
                                "filename": filename,
                                "bytes_downloaded": file_size,
                                "total_bytes": file_size
                            }
                        ).to_json()
                    )
                    
                    file_statuses.append(DownloadStatus(
                        filename=filename,
                        status="completed",
                        error=None,
                        bytes_downloaded=file_size,
                        total_bytes=file_size
                    ))
                    
                except Exception as e:
                    error_msg = str(e)
                    await websocket.send_json(
                        WebSocketMessage(
                            type="error",
                            message=f"Failed to download {filename}: {error_msg}",
                            data={
                                "filename": filename,
                                "error": error_msg,
                                "repo_id": repo_id
                            }
                        ).to_json()
                    )
                    file_statuses.append(DownloadStatus(
                        filename=filename,
                        status="failed",
                        error=error_msg,
                        bytes_downloaded=None,
                        total_bytes=None
                    ))
            
            # Determine overall status
            if not file_statuses:
                overall_status = "failed"
            elif all(f.status in ["completed", "skipped"] for f in file_statuses):
                overall_status = "completed"
            elif any(f.status in ["completed", "skipped"] for f in file_statuses):
                overall_status = "partial"
            else:
                overall_status = "failed"
            
            # Send result for this model
            result = DownloadModelsResponse(
                model_id=model_id,
                files=file_statuses,
                overall_status=overall_status
            )
            
            await websocket.send_json(
                WebSocketMessage(
                    type="result",
                    message=f"Completed processing {model_id}",
                    data=result.model_dump()
                ).to_json()
            )
        
        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message="All downloads completed"
            ).to_json()
        )
        
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json(
                WebSocketMessage(
                    type="error",
                    message=f"Download error: {str(e)}"
                ).to_json()
            )
        except:
            pass


@router.post("/set-storage-metadata", response_model=StatusResponse)
async def set_storage_metadata(request: StorageMetadataRequest):
    """Set storage metadata path.
    
    On macOS, if a security-scoped bookmark is provided, it will be resolved
    first to gain sandbox access to the user-selected folder.
    """
    global _metadata_store
    
    # On macOS, resolve security-scoped bookmark if provided
    bookmark_resolved = False
    if request.bookmark and sys.platform == "darwin":
        resolved_path, error = resolve_security_scoped_bookmark(request.bookmark)
        if error:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to resolve security-scoped bookmark: {error}"
            )
        # The bookmark resolves to the directory, so we append the filename
        resolved_dir = Path(resolved_path)
        # Check if the resolved path is the metadata file itself or the directory
        if resolved_dir.is_file() and resolved_dir.name == "storage_metadata.json":
            path = resolved_dir
        else:
            path = resolved_dir / "storage_metadata.json"
        bookmark_resolved = True
    else:
        path = Path(request.path)
    
    # Verify path exists
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {path}")
    
    # Try to load metadata
    try:
        _metadata_store = MetadataStore(str(path))
        set_storage_metadata_path(str(path))
        
        # Create RAG directory if it doesn't exist
        config = get_config()
        rag_dir = config.get_rag_directory()
        if rag_dir:
            rag_dir.mkdir(exist_ok=True)
        
        # Create saved_llm directory if it doesn't exist
        saved_llm_dir = config.get_saved_llm_directory()
        if saved_llm_dir:
            saved_llm_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing embeddings if available
        embedding_service = get_embedding_service()
        embeddings_loaded = embedding_service.load_embeddings()
        
        # Get identified metadata properties
        identified_properties = _metadata_store.get_identified_properties()
        
        response_data = {
            "metadata_count": len(_metadata_store.get_all_metadata()),
            "rag_directory": str(rag_dir) if rag_dir else None,
            "saved_llm_directory": str(saved_llm_dir) if saved_llm_dir else None,
            "embeddings_loaded": embeddings_loaded,
            "embeddings_count": len(embedding_service.get_all_embeddings()) if embeddings_loaded else 0,
            "identified_properties": identified_properties
        }
        
        if bookmark_resolved:
            response_data["bookmark_resolved"] = True
            response_data["resolved_path"] = str(path)
        
        return StatusResponse(
            status="success",
            message=f"Storage metadata set to {path}",
            data=response_data
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=403,
            detail=f"Permission denied: {str(e)}. The main app should provide a security-scoped bookmark."
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load metadata: {str(e)}")


# NOTE: select_storage_metadata endpoint has been removed.
# This server runs as a background helper process that cannot make GUI calls.
# The main app should use /set-storage-metadata with a security-scoped bookmark instead.


@router.post("/set-model-directory", response_model=StatusResponse)
async def set_model_directory(request: ModelDirectoryRequest):
    """Set model directory path with optional bookmark for sandbox access.
    
    On macOS, if a security-scoped bookmark is provided, it will be resolved
    first to gain sandbox access to the model directory. This access is then
    inherited by child processes like llama-server.
    
    Args:
        request: ModelDirectoryRequest with path and optional bookmark
        
    Returns:
        StatusResponse with success/failure and resolved path info
    """
    # On macOS, resolve security-scoped bookmark if provided
    bookmark_resolved = False
    resolved_path = request.path
    
    if request.bookmark and sys.platform == "darwin":
        path_from_bookmark, error = resolve_security_scoped_bookmark(request.bookmark)
        if error:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to resolve security-scoped bookmark: {error}"
            )
        resolved_path = path_from_bookmark
        bookmark_resolved = True
    
    # Verify path is a valid directory (or can be created)
    path = Path(resolved_path)
    
    try:
        if not path.exists():
            # Create the directory if it doesn't exist
            path.mkdir(parents=True, exist_ok=True)
        
        if not path.is_dir():
            raise HTTPException(
                status_code=400, 
                detail=f"Path is not a directory: {resolved_path}"
            )
        
        # Update the config with the new model directory
        config = get_config()
        update_config(model_directory=str(path))
        
        response_data = {
            "model_directory": str(path),
            "exists": path.exists(),
            "is_writable": os.access(str(path), os.W_OK),
        }
        
        if bookmark_resolved:
            response_data["bookmark_resolved"] = True
            response_data["original_path"] = request.path
        
        return StatusResponse(
            status="success",
            message=f"Model directory set to {path}",
            data=response_data
        )
        
    except PermissionError as e:
        raise HTTPException(
            status_code=403,
            detail=f"Permission denied: {str(e)}. Please provide a security-scoped bookmark."
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to set model directory: {str(e)}"
        )

@router.post("/load-rag", response_model=StatusResponse)
async def load_rag():
    """Load existing RAG database."""
    metadata_store = get_metadata_store()
    rag_service = get_rag_service()
    
    # Check if metadata file has been updated and reload if necessary
    await metadata_store.reload_if_modified()
    
    result = rag_service.load_rag(metadata_store)
    
    if result["success"]:
        # Also initialize/load conversation knowledge base
        knowledge_service = get_knowledge_service()
        knowledge_service.initialize()
        knowledge_service.load()
        
        return StatusResponse(
            status="success",
            message="RAG database loaded successfully",
            data={
                "dimension": result.get("dimension"),
                "indexed_files": result.get("indexed_files")
            }
        )
    else:
        raise HTTPException(
            status_code=404,
            detail=result.get("error", "RAG database not found. Generate RAG first using /generate-rag endpoint.")
        )


@router.websocket("/vector-embeddings")
async def vector_embeddings_ws(websocket: WebSocket):
    """Generate or regenerate embeddings for files (WebSocket)."""
    await websocket.accept()
    
    # Track tasks for cancellation
    generation_task = None
    llm_service = None
    
    try:
        # Receive configuration
        data = await websocket.receive_json()
        config = get_config()
        embedding_model = data.get("embedding_model") or config.embedding_model
        file_names = data.get("file_names")  # Optional list of specific files
        regenerate_all = data.get("regenerate_all", False)  # Regenerate all files
        
        metadata_store = get_metadata_store()
        embedding_service = get_embedding_service()
        llm_service = get_llm_service()
        
        # Track failed files
        failed_files = []
        
        # Check if metadata file has been updated and reload if necessary
        if await metadata_store.reload_if_modified():
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message="Storage metadata file was updated. Reloaded metadata."
                ).to_json()
            )
        
        # Determine which files to process
        if file_names:
            # Process specific files
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"Processing {len(file_names)} specific file(s)..."
                ).to_json()
            )
            files_to_process = []
            for file_name in file_names:
                metadata = metadata_store.get_metadata_by_filename(file_name)
                if not metadata:
                    # Report error for this file but continue with others
                    await websocket.send_json(
                        WebSocketMessage(
                            type="error",
                            message=f"File not found in metadata: {file_name}",
                            data={"filename": file_name, "error": "File not found in metadata", "continue": True}
                        ).to_json()
                    )
                    # Track as failed
                    failed_files.append({"filename": file_name, "error": "File not found in metadata"})
                    continue
                files_to_process.append(metadata)
        elif regenerate_all:
            # Process all files
            files_to_process = metadata_store.get_all_metadata()
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"Regenerating embeddings for all {len(files_to_process)} file(s)..."
                ).to_json()
            )
        else:
            # Default: process only files without embeddings
            all_files = metadata_store.get_all_metadata()
            # Load embeddings if they exist, otherwise start with empty dict
            embedding_service.load_embeddings()
            existing_embeddings = embedding_service.get_all_embeddings()
            # Ensure existing_embeddings is a dict (it should be, but safeguard)
            if not isinstance(existing_embeddings, dict):
                existing_embeddings = {}
            files_to_process = [f for f in all_files if f.fileName not in existing_embeddings]
            
            if not files_to_process:
                await websocket.send_json(
                    WebSocketMessage(
                        type="status",
                        message="All files already have embeddings"
                    ).to_json()
                )
                await websocket.send_json(
                    WebSocketMessage(
                        type="result",
                        message="Embeddings generated successfully",
                        data={"count": len(existing_embeddings), "processed": 0}
                    ).to_json()
                )
                return
            
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"Found {len(existing_embeddings)} existing embeddings, processing {len(files_to_process)} new file(s)..."
                ).to_json()
            )
        
        # Progress callback
        async def progress_callback(current: int, total: int, filename: str):
            # Handle startup command message (current=0, total=0)
            if current == 0 and total == 0 and filename.startswith("LLM Command:"):
                await websocket.send_json(
                    WebSocketMessage(
                        type="status",
                        message=filename,
                        data={"startup_command": filename.replace("LLM Command: ", "")}
                    ).to_json()
                )
            elif current == 0 and total == 0:
                # Status messages
                await websocket.send_json(
                    WebSocketMessage(
                        type="status",
                        message=filename
                    ).to_json()
                )
            else:
                await websocket.send_json(
                    WebSocketMessage(
                        type="progress",
                        message=f"Processing {filename}",
                        data={"current": current, "total": total, "filename": filename}
                    ).to_json()
                )
        
        # Error callback for individual file failures
        async def error_callback(filename: str, error_message: str):
            failed_files.append({"filename": filename, "error": error_message})
            await websocket.send_json(
                WebSocketMessage(
                    type="error",
                    message=f"Failed to generate embedding for {filename}: {error_message}",
                    data={"filename": filename, "error": error_message, "continue": True}
                ).to_json()
            )
        
        # Generate embeddings for selected files
        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message="Starting embedding generation..."
            ).to_json()
        )
        
        # Wrap in task for cancellation support
        async def generate_embeddings_task():
            return await embedding_service.generate_embeddings_for_files(
                files_to_process,
                embedding_model,
                progress_callback,
                error_callback
            )
        
        generation_task = asyncio.create_task(generate_embeddings_task())
        
        # Execute the task - no global timeout, let individual files handle their own timeouts
        try:
            embeddings = await generation_task
        except Exception as e:
            # Any exception from embedding generation should be logged but not stop the process
            # since individual file errors are already handled via error_callback
            if error_callback:
                await error_callback("batch_process", f"Unexpected error during batch processing: {str(e)}")
            # Use empty embeddings dict if generation failed completely
            embeddings = {}
        
        # Send final result with success/failure summary
        success_count = len(files_to_process) - len(failed_files)
        if failed_files:
            await websocket.send_json(
                WebSocketMessage(
                    type="result",
                    message=f"Embeddings generated with {len(failed_files)} failure(s)",
                    data={
                        "count": len(embeddings),
                        "processed": len(files_to_process),
                        "successful": success_count,
                        "failed": len(failed_files),
                        "failed_files": failed_files
                    }
                ).to_json()
            )
        else:
            await websocket.send_json(
                WebSocketMessage(
                    type="result",
                    message="Embeddings generated successfully",
                    data={
                        "count": len(embeddings),
                        "processed": len(files_to_process),
                        "successful": success_count,
                        "failed": 0
                    }
                ).to_json()
            )
        
    except WebSocketDisconnect:
        # Client disconnected - cancel any running generation
        if generation_task and not generation_task.done():
            generation_task.cancel()
            try:
                await generation_task
            except asyncio.CancelledError:
                pass
        
        # If using CLI mode, explicitly cancel the subprocess
        if llm_service and config.llm_mode == "cli":
            if hasattr(llm_service.current_backend, 'cancel_generation'):
                await llm_service.current_backend.cancel_generation()
    except asyncio.CancelledError:
        # Task was cancelled (likely due to disconnect)
        if llm_service and config.llm_mode == "cli":
            if hasattr(llm_service.current_backend, 'cancel_generation'):
                await llm_service.current_backend.cancel_generation()
    except Exception as e:
        # Log unexpected errors but don't close the socket
        # Individual file errors are already handled via error_callback
        print(f"Unexpected error in vector-embeddings WebSocket: {type(e).__name__}: {str(e)}")
        print(traceback.format_exc())
    finally:
        # Final cleanup - only cancel task if still running
        if generation_task and not generation_task.done():
            generation_task.cancel()
        await websocket.close()


@router.websocket("/generate-rag")
async def generate_rag_ws(websocket: WebSocket):
    """Generate RAG database (WebSocket)."""
    await websocket.accept()
    
    try:
        metadata_store = get_metadata_store()
        rag_service = get_rag_service()
        
        # Check if metadata file has been updated and reload if necessary
        if await metadata_store.reload_if_modified():
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message="Storage metadata file was updated. Reloaded metadata."
                ).to_json()
            )
        
        # Progress callback
        async def progress_callback(message: str):
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=message
                ).to_json()
            )
        
        await progress_callback("Building RAG database...")
        
        result = await rag_service.build_rag(metadata_store, progress_callback)
        
        # Also initialize conversation knowledge base
        knowledge_service = get_knowledge_service()
        knowledge_service.initialize()
        knowledge_service.load()
        
        # Check for mismatched embeddings
        if result.get("mismatched_files"):
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"Warning: {result['removed_count']} file(s) had mismatched embedding dimensions and were excluded from RAG",
                    data={
                        "mismatched_files": result["mismatched_files"],
                        "majority_dimension": result["majority_dimension"]
                    }
                ).to_json()
            )
        
        await websocket.send_json(
            WebSocketMessage(
                type="result",
                message="RAG database created and loaded successfully",
                data={
                    "total_indexed": result["total_indexed"],
                    "removed_count": result["removed_count"],
                    "mismatched_files": result.get("mismatched_files", []),
                    "majority_dimension": result["majority_dimension"]
                }
            ).to_json()
        )
        
    except Exception as e:
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        await websocket.send_json(
            WebSocketMessage(
                type="error",
                message=f"RAG generation failed: {type(e).__name__}: {str(e)}",
                data=error_details
            ).to_json()
        )
    finally:
        await websocket.close()


@router.websocket("/tag")
async def tag_files_ws(websocket: WebSocket):
    """Generate tags for files (WebSocket)."""
    await websocket.accept()
    
    # Track tasks for cancellation
    current_task = None
    vision_service = None
    llm_service = None
    
    try:
        # Receive request
        data = await websocket.receive_json()
        config = get_config()
        file_paths = data.get("file_paths", [])
        vision_model = data.get("vision_model") or config.vision_model
        mmproj_file = data.get("mmproj_file") or config.mmproj_model
        
        if not file_paths:
            await websocket.send_json(
                WebSocketMessage(
                    type="error",
                    message="file_paths is required"
                ).to_json()
            )
            await websocket.close()
            return
        
        metadata_store = get_metadata_store()
        vision_service = get_vision_service()
        llm_service = get_llm_service()
        
        # Check if metadata file has been updated and reload if necessary
        if await metadata_store.reload_if_modified():
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message="Storage metadata file was updated. Reloaded metadata."
                ).to_json()
            )
        
        try:
            for idx, file_path_str in enumerate(file_paths):
                # Get metadata
                metadata = metadata_store.get_metadata_by_filename(file_path_str)
                if not metadata:
                    await websocket.send_json(
                        WebSocketMessage(
                            type="error",
                            message=f"Metadata not found for {file_path_str}"
                        ).to_json()
                    )
                    continue
                
                # Get full path
                file_path = metadata_store.get_file_path(file_path_str)
                
                if not file_path.exists():
                    await websocket.send_json(
                        WebSocketMessage(
                            type="error",
                            message=f"File not found: {file_path}"
                        ).to_json()
                    )
                    continue
                
                # Ask for confirmation before processing next
                if idx > 0:
                    await websocket.send_json(
                        WebSocketMessage(
                            type="confirmation_needed",
                            message=f"Ready to tag {file_path_str}. Send 'continue' to proceed.",
                            data={"current": idx + 1, "total": len(file_paths)}
                        ).to_json()
                    )
                    
                    response = await websocket.receive_json()
                    if response.get("action") != "continue":
                        break
                
                # Generate tags
                await websocket.send_json(
                    WebSocketMessage(
                        type="status",
                        message=f"Generating tags for {file_path_str}..."
                    ).to_json()
                )
                
                try:
                    # Define callback to send dimension info before LLM call
                    async def dimension_callback(original_dims: tuple, processed_dims: tuple, size_bytes: int):
                        await websocket.send_json(
                            WebSocketMessage(
                                type="status",
                                message=f"Image Info - Original: {original_dims[0]}x{original_dims[1]}, Processed: {processed_dims[0]}x{processed_dims[1]}, Size: {size_bytes / 1024:.1f} KB",
                                data={
                                    "original_width": original_dims[0],
                                    "original_height": original_dims[1],
                                    "processed_width": processed_dims[0],
                                    "processed_height": processed_dims[1],
                                    "size_bytes": size_bytes
                                }
                            ).to_json()
                        )
                        await asyncio.sleep(0.1)
                    
                    # Define callback to send startup command immediately (first file only)
                    async def startup_callback(cmd: str):
                        if file_path_str == file_paths[0]:
                            await websocket.send_json(
                                WebSocketMessage(
                                    type="status",
                                    message=f"LLM Command: {cmd}",
                                    data={"startup_command": cmd}
                                ).to_json()
                            )
                            await asyncio.sleep(0.1)  # Yield control to ensure message is sent
                    
                    # Wrap in task for cancellation support
                    async def generate_tags_task():
                        return await vision_service.generate_tags(
                            file_path,
                            metadata.type,
                            vision_model,
                            mmproj_file,
                            startup_callback,
                            dimension_callback,
                            keep_loaded=True  # Keep model loaded for batch processing
                        )
                    
                    current_task = asyncio.create_task(generate_tags_task())
                    try:
                        thinking, tags = await asyncio.wait_for(current_task, timeout=config.llm_timeout)
                    except asyncio.TimeoutError:
                        current_task.cancel()
                        raise RuntimeError(f"Tag generation timed out after {config.llm_timeout} seconds")
                    current_task = None
                    
                    # Send thinking part if available
                    if thinking:
                        await websocket.send_json(
                            WebSocketMessage(
                                type="thinking",
                                message=f"Analysis for {file_path_str}",
                                data={"filename": file_path_str, "thinking": thinking}
                            ).to_json()
                        )
                    
                    # Send final tags
                    await websocket.send_json(
                        WebSocketMessage(
                            type="result",
                            message=f"Tags generated for {file_path_str}",
                            data={"filename": file_path_str, "tags": tags}
                        ).to_json()
                    )
                except Exception as tag_error:
                    error_details = {
                        "filename": file_path_str,
                        "error_type": type(tag_error).__name__,
                        "error_message": str(tag_error),
                        "traceback": traceback.format_exc(),
                        "vision_model": vision_model,
                        "mmproj_file": mmproj_file,
                        "file_type": metadata.type,
                        "file_exists": file_path.exists(),
                        "file_size": file_path.stat().st_size if file_path.exists() else None
                    }
                    await websocket.send_json(
                        WebSocketMessage(
                            type="error",
                            message=f"Failed to generate tags for {file_path_str}: {type(tag_error).__name__}: {str(tag_error)}",
                            data=error_details
                        ).to_json()
                    )
                    continue
                
                await websocket.send_json(
                    WebSocketMessage(
                        type="status",
                        message="Tagging complete"
                    ).to_json()
                )
        finally:
            # Unload model after all files are processed or on error
            if vision_service:
                await vision_service.unload_model()
        
    except WebSocketDisconnect:
        # Client disconnected - cancel any running task
        if current_task and not current_task.done():
            current_task.cancel()
            try:
                await current_task
            except asyncio.CancelledError:
                pass
        
        # If using CLI mode, explicitly cancel the subprocess
        if llm_service and config.llm_mode == "cli":
            if hasattr(llm_service.current_backend, 'cancel_generation'):
                await llm_service.current_backend.cancel_generation()
        
        # Ensure model is unloaded on unexpected disconnect
        if vision_service:
            await vision_service.unload_model()
    except asyncio.CancelledError:
        # Task was cancelled (likely due to disconnect)
        if llm_service and config.llm_mode == "cli":
            if hasattr(llm_service.current_backend, 'cancel_generation'):
                await llm_service.current_backend.cancel_generation()
        
        if vision_service:
            await vision_service.unload_model()
    except asyncio.TimeoutError:
        # Task timed out
        if current_task and not current_task.done():
            current_task.cancel()
        
        if llm_service and config.llm_mode == "cli":
            if hasattr(llm_service.current_backend, 'cancel_generation'):
                await llm_service.current_backend.cancel_generation()
        
        await websocket.send_json(
            WebSocketMessage(
                type="error",
                message=f"Tag generation timed out after {config.llm_timeout} seconds",
                data={"timeout": config.llm_timeout}
            ).to_json()
        )
        
        if vision_service:
            await vision_service.unload_model()
    except Exception as e:
        # Cancel task if running
        if current_task and not current_task.done():
            current_task.cancel()
            try:
                await current_task
            except asyncio.CancelledError:
                pass
        
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        await websocket.send_json(
            WebSocketMessage(
                type="error",
                message=f"WebSocket error in /tag: {type(e).__name__}: {str(e)}",
                data=error_details
            ).to_json()
        )
        
        if vision_service:
            await vision_service.unload_model()
    finally:
        # Final cleanup
        if current_task and not current_task.done():
            current_task.cancel()
        await websocket.close()


@router.websocket("/describe")
async def describe_files_ws(websocket: WebSocket):
    """Generate descriptions for files (WebSocket)."""
    await websocket.accept()
    
    # Track tasks for cancellation
    current_task = None
    vision_service = None
    llm_service = None
    
    try:
        # Receive request
        data = await websocket.receive_json()
        config = get_config()
        file_paths = data.get("file_paths", [])
        vision_model = data.get("vision_model") or config.vision_model
        mmproj_file = data.get("mmproj_file") or config.mmproj_model
        
        if not file_paths:
            await websocket.send_json(
                WebSocketMessage(
                    type="error",
                    message="file_paths is required"
                ).to_json()
            )
            await websocket.close()
            return
        
        metadata_store = get_metadata_store()
        vision_service = get_vision_service()
        llm_service = get_llm_service()
        
        # Check if metadata file has been updated and reload if necessary
        if await metadata_store.reload_if_modified():
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message="Storage metadata file was updated. Reloaded metadata."
                ).to_json()
            )
        
        try:
            for idx, file_path_str in enumerate(file_paths):
                # Get metadata
                metadata = metadata_store.get_metadata_by_filename(file_path_str)
                if not metadata:
                    await websocket.send_json(
                        WebSocketMessage(
                            type="error",
                            message=f"Metadata not found for {file_path_str}"
                        ).to_json()
                    )
                    continue
                
                # Get full path
                file_path = metadata_store.get_file_path(file_path_str)
                
                if not file_path.exists():
                    await websocket.send_json(
                        WebSocketMessage(
                            type="error",
                            message=f"File not found: {file_path}"
                        ).to_json()
                    )
                    continue
                
                # Ask for confirmation before processing next
                if idx > 0:
                    await websocket.send_json(
                        WebSocketMessage(
                            type="confirmation_needed",
                            message=f"Ready to describe {file_path_str}. Send 'continue' to proceed.",
                            data={"current": idx + 1, "total": len(file_paths)}
                        ).to_json()
                    )
                    
                    response = await websocket.receive_json()
                    if response.get("action") != "continue":
                        break
                
                # Generate description
                await websocket.send_json(
                    WebSocketMessage(
                        type="status",
                        message=f"Generating description for {file_path_str}..."
                    ).to_json()
                )
                
                try:
                    # Define callback to send dimension info before LLM call
                    async def dimension_callback(original_dims: tuple, processed_dims: tuple, size_bytes: int):
                        await websocket.send_json(
                            WebSocketMessage(
                                type="status",
                                message=f"Image Info - Original: {original_dims[0]}x{original_dims[1]}, Processed: {processed_dims[0]}x{processed_dims[1]}, Size: {size_bytes / 1024:.1f} KB",
                                data={
                                    "original_width": original_dims[0],
                                    "original_height": original_dims[1],
                                    "processed_width": processed_dims[0],
                                    "processed_height": processed_dims[1],
                                    "size_bytes": size_bytes
                                }
                            ).to_json()
                        )
                        await asyncio.sleep(0.1)
                    
                    # Define callback to send startup command immediately (first file only)
                    async def startup_callback(cmd: str):
                        if file_path_str == file_paths[0]:
                            await websocket.send_json(
                                WebSocketMessage(
                                    type="status",
                                    message=f"LLM Command: {cmd}",
                                    data={"startup_command": cmd}
                                ).to_json()
                            )
                            await asyncio.sleep(0.1)  # Yield control to ensure message is sent
                    
                    # Wrap in task for cancellation support
                    async def generate_description_task():
                        return await vision_service.generate_description(
                            file_path,
                            metadata.type,
                            vision_model,
                            mmproj_file,
                            startup_callback,
                            dimension_callback,
                            keep_loaded=True  # Keep model loaded for batch processing
                        )
                    
                    current_task = asyncio.create_task(generate_description_task())
                    try:
                        thinking, description = await asyncio.wait_for(current_task, timeout=config.llm_timeout)
                    except asyncio.TimeoutError:
                        current_task.cancel()
                        raise RuntimeError(f"Description generation timed out after {config.llm_timeout} seconds")
                    current_task = None
                    
                    # Send thinking part if available
                    if thinking:
                        await websocket.send_json(
                            WebSocketMessage(
                                type="thinking",
                                message=f"Analysis for {file_path_str}",
                                data={"filename": file_path_str, "thinking": thinking}
                            ).to_json()
                        )
                    
                    # Send final description
                    await websocket.send_json(
                        WebSocketMessage(
                            type="result",
                            message=f"Description generated for {file_path_str}",
                            data={"filename": file_path_str, "description": description}
                        ).to_json()
                    )
                except Exception as desc_error:
                    error_details = {
                        "filename": file_path_str,
                        "error_type": type(desc_error).__name__,
                        "error_message": str(desc_error),
                        "traceback": traceback.format_exc(),
                        "vision_model": vision_model,
                        "mmproj_file": mmproj_file,
                        "file_type": metadata.type,
                        "file_exists": file_path.exists(),
                        "file_size": file_path.stat().st_size if file_path.exists() else None
                    }
                    await websocket.send_json(
                        WebSocketMessage(
                            type="error",
                            message=f"Failed to generate description for {file_path_str}: {type(desc_error).__name__}: {str(desc_error)}",
                            data=error_details
                        ).to_json()
                    )
                    continue
                
                await websocket.send_json(
                    WebSocketMessage(
                        type="status",
                        message="Description generation complete"
                    ).to_json()
                )
        finally:
            # Unload model after all files are processed or on error
            if vision_service:
                await vision_service.unload_model()
        
    except WebSocketDisconnect:
        # Client disconnected - cancel any running task
        if current_task and not current_task.done():
            current_task.cancel()
            try:
                await current_task
            except asyncio.CancelledError:
                pass
        
        # If using CLI mode, explicitly cancel the subprocess
        if llm_service and config.llm_mode == "cli":
            if hasattr(llm_service.current_backend, 'cancel_generation'):
                await llm_service.current_backend.cancel_generation()
        
        # Ensure model is unloaded on unexpected disconnect
        if vision_service:
            await vision_service.unload_model()
    except asyncio.CancelledError:
        # Task was cancelled (likely due to disconnect)
        if llm_service and config.llm_mode == "cli":
            if hasattr(llm_service.current_backend, 'cancel_generation'):
                await llm_service.current_backend.cancel_generation()
        
        if vision_service:
            await vision_service.unload_model()
    except asyncio.TimeoutError:
        # Task timed out
        if current_task and not current_task.done():
            current_task.cancel()
        
        if llm_service and config.llm_mode == "cli":
            if hasattr(llm_service.current_backend, 'cancel_generation'):
                await llm_service.current_backend.cancel_generation()
        
        await websocket.send_json(
            WebSocketMessage(
                type="error",
                message=f"Description generation timed out after {config.llm_timeout} seconds",
                data={"timeout": config.llm_timeout}
            ).to_json()
        )
        
        if vision_service:
            await vision_service.unload_model()
    except Exception as e:
        # Cancel task if running
        if current_task and not current_task.done():
            current_task.cancel()
            try:
                await current_task
            except asyncio.CancelledError:
                pass
        
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        await websocket.send_json(
            WebSocketMessage(
                type="error",
                message=f"WebSocket error in /describe: {type(e).__name__}: {str(e)}",
                data=error_details
            ).to_json()
        )
        
        if vision_service:
            await vision_service.unload_model()
    finally:
        # Final cleanup
        if current_task and not current_task.done():
            current_task.cancel()
        await websocket.close()


@router.websocket("/chat")
async def chat_ws(websocket: WebSocket):
    """Chat with LLM using RAG (WebSocket). Supports visual conversation when enabled."""
    await websocket.accept()
    
    # Track if we should cleanup on disconnect
    generation_task = None
    llm_service = None
    
    try:
        # Get configuration
        config = get_config()
        llm_service = get_llm_service()
        
        embedding_model = config.embedding_model
        
        metadata_store = get_metadata_store()
        rag_service = get_rag_service()
        
        # Check if metadata file has been updated and reload if necessary
        if await metadata_store.reload_if_modified():
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message="Storage metadata file was updated. Reloaded metadata."
                ).to_json()
            )
        
        # Try to load RAG (optional — chat works without it)
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
        
        # Load embedding model if RAG or knowledge storage needs it
        embedding_loaded = False
        if rag_available or config.enable_knowledge_storage:
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"Loading embedding model {embedding_model}..."
                ).to_json()
            )
            
            await llm_service.load_model(embedding_model)
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
                await asyncio.sleep(0.1)  # Yield control to ensure message is sent
        
        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message=f"Chat ready. Send your message."
            ).to_json()
        )
        
        # Single message handling - connection closes after response
        # Receive user message directly (no initial config needed)
        msg_data = await websocket.receive_json()
        
        user_message = msg_data.get("message")
        image_name = msg_data.get("image_name")  # Optional image name for visual chat
        provided_history = msg_data.get("history")  # Optional chat history (OpenAI format)
        
        # Determine which model to use based on visual chat setting AND whether image is provided
        # Vision models can be used for both text-only chat and vision tasks
        if config.enable_visual_chat and image_name:
            # Use vision model with image when visual chat is enabled and image is provided
            chat_model = config.vision_model
            mmproj_file = config.mmproj_model
            use_vision = True
        else:
            # Use configured chat model (which could be a vision model used for text-only chat)
            chat_model = config.chat_model
            # Use configured mmproj_model if set - Flutter sends the localMmprojFile 
            # (unique prefixed filename) when user selects a vision model for chat
            mmproj_file = config.mmproj_model if config.mmproj_model else None
            use_vision = False
        
        if not user_message:
            await websocket.send_json(
                WebSocketMessage(
                    type="error",
                    message="No message provided"
                ).to_json()
            )
            return
        
        # Use provided history if available, otherwise start fresh
        # History should be in OpenAI format: [{"role": "user"|"assistant", "content": "..."}]
        if provided_history is not None:
            if not isinstance(provided_history, list):
                await websocket.send_json(
                    WebSocketMessage(
                        type="error",
                        message="history parameter must be a list of message objects"
                    ).to_json()
                )
                return
            # Validate history format
            for msg in provided_history:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    await websocket.send_json(
                        WebSocketMessage(
                            type="error",
                            message="Each history item must be a dict with 'role' and 'content' keys"
                        ).to_json()
                    )
                    return
                if msg["role"] not in ["user", "assistant"]:
                    await websocket.send_json(
                        WebSocketMessage(
                            type="error",
                            message="History role must be 'user' or 'assistant'"
                        ).to_json()
                    )
                    return
            # Use provided history for this turn
            active_history = provided_history.copy()
        else:
            # Start with empty history
            active_history = []
        
        # Handle visual conversation with image
        image_base64 = None
        image_tags = []
        image_description = None
        if use_vision and image_name:
            # Get image metadata for tags and type detection
            image_metadata = metadata_store.get_metadata_by_filename(image_name)
            file_type_label = "PDF document" if (image_metadata and image_metadata.type == "pdf") else "video" if (image_metadata and image_metadata.type == "video") else "image"
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"Processing {file_type_label}: {image_name}..."
                ).to_json()
            )
            
            if image_metadata:
                image_tags = image_metadata.tags
                image_description = image_metadata.description
            
            # Load file using utility function (handles image, video, and PDF)
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
            else:
                # Calculate size for display
                import base64
                image_bytes_len = len(base64.b64decode(image_base64))
                
                # Prepare tag info message
                tag_info = f" (Tags: {', '.join(image_tags)})" if image_tags else ""
                await websocket.send_json(
                    WebSocketMessage(
                        type="status",
                        message=f"{file_type_label.capitalize()} loaded: {image_name} ({image_bytes_len / 1024:.1f} KB){tag_info}"
                    ).to_json()
                )
        
        # Add current user message to active history (text only for now, image handled separately in generation)
        active_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Perform RAG search if available
        context = ""
        if rag_available:
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message="Searching knowledge base..."
                ).to_json()
            )
            
            # Build search query including conversation history for better context
            # Format: user: "query1", assistant: "response1", user: "query2", ...
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
        
        # Search conversation knowledge base for relevant past facts
        knowledge_context = ""
        user_message_embedding = None
        knowledge_service = get_knowledge_service()
        if config.enable_knowledge_storage and embedding_loaded:
            try:
                # Embed user message once while the embedding model is already loaded
                user_message_embedding = await llm_service.embed(user_message)
                if knowledge_service.is_loaded():
                    relevant_facts = knowledge_service.select_knowledge(
                        user_message_embedding,
                        token_budget=config.max_knowledge_tokens,
                        min_relevance=config.min_knowledge_relevance,
                    )
                    if relevant_facts:
                        fact_lines = [f"- {f['message']}" for f in relevant_facts]
                        knowledge_context = "\n\nRelevant information from previous conversations:\n" + "\n".join(fact_lines)
                        await websocket.send_json(
                            WebSocketMessage(
                                type="status",
                                message=f"Found {len(relevant_facts)} relevant facts from conversation history"
                            ).to_json()
                        )
            except Exception as e:
                # Non-fatal: knowledge retrieval failure shouldn't block chat
                await websocket.send_json(
                    WebSocketMessage(
                        type="status",
                        message=f"Knowledge retrieval skipped: {e}"
                    ).to_json()
                )
        
        # Append knowledge context to RAG context
        context = context + knowledge_context
        
        # Now load the chat/vision model for generation
        model_type = "vision" if use_vision else "chat"
        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message=f"Loading {model_type} model {chat_model}..." + (f" with mmproj: {mmproj_file}" if use_vision and mmproj_file else "")
            ).to_json()
        )
        
        # Load model with mmproj if vision mode
        if use_vision and mmproj_file:
            await llm_service.load_model(chat_model, mmproj=mmproj_file)
        else:
            await llm_service.load_model(chat_model)
        
        # Send chat/vision model startup command
        chat_startup_cmd = llm_service.get_startup_command()
        if chat_startup_cmd:
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"Chat Model Command: {chat_startup_cmd}",
                    data={"chat_startup_command": chat_startup_cmd}
                ).to_json()
            )
            await asyncio.sleep(0.1)  # Yield control to ensure message is sent
        
        # Generate response
        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message="Generating response..."
            ).to_json()
        )
        
        response_text = ""
        
        # Create async generator wrapper to handle cancellation
        async def generate_response():
            nonlocal response_text
            try:
                # Use vision generation if image is provided and visual chat is enabled
                if use_vision and image_base64:
                    # For vision models with image, use generate_vision
                    # Build image context with tags and description
                    image_context_parts = []
                    if image_tags:
                        image_context_parts.append(f"Image Tags: {', '.join(image_tags)}")
                    if image_description:
                        image_context_parts.append(f"Image Description: {image_description}")
                    
                    image_context = "\n".join(image_context_parts) if image_context_parts else ""
                    
                    # Get current date
                    current_date = datetime.now().strftime("%B %d, %Y")
                    
                    # Combine system prompt, RAG context, and image context
                    system_content = f"{config.chat_system_prompt}\n\nCurrent Date: {current_date}\n\n{context}"
                    if image_context:
                        system_content += f"\n\nImage Context:\n{image_context}"
                    
                    prompt = f"{system_content}\n\nUser: {user_message}\n\nAssistant:"
                    
                    # Convert base64 back to bytes for generate_vision
                    import base64
                    image_bytes = base64.b64decode(image_base64)
                    
                    response_text = await llm_service.generate_vision(image_bytes, prompt, mmproj_file)
                    
                    # Send as progress for consistency
                    await websocket.send_json(
                        WebSocketMessage(
                            type="progress",
                            message=response_text,
                            data={"partial_response": response_text}
                        ).to_json()
                    )
                else:
                    # Standard text-based chat
                    # Get current date
                    current_date = datetime.now().strftime("%B %d, %Y")
                    
                    # Create system message with configurable prompt and context
                    messages = [
                        {
                            "role": "system",
                            "content": f"{config.chat_system_prompt}\n\nCurrent Date: {current_date}\n\n{context}"
                        }
                    ] + active_history
                    
                    async for chunk in llm_service.generate(messages, stream=True):
                        response_text += chunk
                        await websocket.send_json(
                            WebSocketMessage(
                                type="progress",
                                message=chunk,
                                data={"partial_response": response_text}
                            ).to_json()
                        )
            except asyncio.CancelledError:
                # Handle graceful cancellation
                await websocket.send_json(
                    WebSocketMessage(
                        type="status",
                        message="Generation cancelled due to client disconnect"
                    ).to_json()
                )
                raise
        
        # Run generation in a task to allow cancellation
        generation_task = asyncio.create_task(generate_response())
        
        # Wait for generation to complete with timeout
        try:
            await asyncio.wait_for(generation_task, timeout=config.llm_timeout)
        except asyncio.TimeoutError:
            generation_task.cancel()
            await websocket.send_json(
                WebSocketMessage(
                    type="error",
                    message=f"Response generation timed out after {config.llm_timeout} seconds"
                ).to_json()
            )
            return
        
        # Sanitize and parse structured output: strip any <think> (internal reasoning) and
        # return only <conclusion> and <files> to clients. This prevents exposing chain-of-thought
        # even if the model produced it.
        import re

        # Remove any internal reasoning sections from model output
        response_sanitized = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)

        # Extract content from XML tags (conclusion and files only)
        conclusion_match = re.search(r'<conclusion>(.*?)</conclusion>', response_sanitized, re.DOTALL)
        files_match = re.search(r'<files>(.*?)</files>', response_sanitized, re.DOTALL)

        sent_any = False

        # Send conclusion section if present
        if conclusion_match:
            conclusion_content = conclusion_match.group(1).strip()
            await websocket.send_json(
                WebSocketMessage(
                    type="conclusion",
                    message=conclusion_content
                ).to_json()
            )
            sent_any = True

        # Send files section if present
        if files_match:
            files_content = files_match.group(1).strip()
            # Attempt to parse files into a list (support newline or comma separated lists)
            files_list = []
            if files_content:
                # Split by newlines, commas, or dashes commonly used in examples
                parts = re.split(r'\r?\n|,|\-', files_content)
                for p in parts:
                    p = p.strip()
                    if p:
                        # Remove any leading bullets
                        p = re.sub(r'^[\-*]\s*', '', p)
                        files_list.append(p)

            await websocket.send_json(
                WebSocketMessage(
                    type="files",
                    message=files_content,
                    data={"relevant_files": files_list or [f.fileName for f in relevant_files]}
                ).to_json()
            )
            sent_any = True

        # If no structured tags found, send the sanitized response as a normal result
        if not sent_any:
            # Remove any remaining XML-like tags to avoid leaking internal formats
            clean_response = re.sub(r'<.*?>', '', response_sanitized, flags=re.DOTALL).strip()
            await websocket.send_json(
                WebSocketMessage(
                    type="result",
                    message="Response complete",
                    data={
                        "response": clean_response,
                        "relevant_files": [f.fileName for f in relevant_files]
                    }
                ).to_json()
            )
        else:
            # Send completion status
            await websocket.send_json(
                WebSocketMessage(
                    type="result",
                    message="Response complete"
                ).to_json()
            )
        
        # Unload model and close connection automatically after response
        if llm_service:
            await llm_service.unload_model()
        
        # Store objective facts from conversation into knowledge base (async, non-blocking)
        if config.enable_knowledge_storage:
            try:
                knowledge_service = get_knowledge_service()
                knowledge_service._objectivity_threshold = config.objectivity_threshold
                
                # Check user message for objective content (only user messages enter facts)
                user_obj, user_score = knowledge_service.should_store(user_message)
                if user_obj:
                    # Pass the precomputed embedding so no model reload is needed
                    knowledge_service.add_fact(
                        user_message, "user", user_score,
                        embedding=user_message_embedding,
                    )
            except Exception as e:
                # Non-fatal: knowledge storage failure shouldn't break chat
                print(f"Knowledge storage error (non-fatal): {e}")
        
    except WebSocketDisconnect:
        # Client disconnected - cancel any running generation
        if generation_task and not generation_task.done():
            generation_task.cancel()
            try:
                await generation_task
            except asyncio.CancelledError:
                pass
        
        # If using CLI mode, explicitly cancel the subprocess
        if llm_service and config.llm_mode == "cli":
            if hasattr(llm_service.current_backend, 'cancel_generation'):
                await llm_service.current_backend.cancel_generation()
        
        # Clean up
        if llm_service:
            await llm_service.unload_model()
    except asyncio.CancelledError:
        # Generation was cancelled (likely due to disconnect)
        if llm_service and config.llm_mode == "cli":
            if hasattr(llm_service.current_backend, 'cancel_generation'):
                await llm_service.current_backend.cancel_generation()
        
        if llm_service:
            await llm_service.unload_model()
    except Exception as e:
        # Cancel generation task if it's running
        if generation_task and not generation_task.done():
            generation_task.cancel()
            try:
                await generation_task
            except asyncio.CancelledError:
                pass
        
        # Get startup command for debugging
        startup_command = llm_service.get_startup_command() if llm_service else None
        
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "chat_model": msg_data.get("chat_model") if 'msg_data' in locals() else None,
            "startup_command": startup_command
        }
        await websocket.send_json(
            WebSocketMessage(
                type="error",
                message=f"Chat error: {type(e).__name__}: {str(e)}",
                data=error_details
            ).to_json()
        )
        # Clean up
        if llm_service:
            await llm_service.unload_model()
    finally:
        # Ensure cleanup happens even if error during error handling
        if generation_task and not generation_task.done():
            generation_task.cancel()
        await websocket.close()


@router.websocket("/cloud-chat")
async def cloud_chat_ws(websocket: WebSocket):
    """Cloud chat endpoint - provides RAG context and system prompt for external LLM calls."""
    await websocket.accept()
    
    try:
        # Get configuration
        config = get_config()
        llm_service = get_llm_service()
        
        embedding_model = config.embedding_model
        
        metadata_store = get_metadata_store()
        rag_service = get_rag_service()
        
        # Check if metadata file has been updated and reload if necessary
        if await metadata_store.reload_if_modified():
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message="Storage metadata file was updated. Reloaded metadata."
                ).to_json()
            )
        
        # Ensure RAG is loaded
        if not rag_service.is_loaded():
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message="Loading RAG database..."
                ).to_json()
            )
            if not rag_service.load_rag(metadata_store):
                raise ValueError("RAG not available. Generate RAG first.")
        
        # Load embedding model for RAG search
        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message=f"Loading embedding model {embedding_model}..."
            ).to_json()
        )
        
        await llm_service.load_model(embedding_model)
        
        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message="Ready to provide RAG context. Send your message."
            ).to_json()
        )
        
        # Receive user message
        msg_data = await websocket.receive_json()
        
        user_message = msg_data.get("message")
        image_name = msg_data.get("image_name")  # Optional image for visual context
        provided_history = msg_data.get("history")  # Optional chat history
        
        if not user_message:
            await websocket.send_json(
                WebSocketMessage(
                    type="error",
                    message="No message provided"
                ).to_json()
            )
            return
        
        # Validate history format if provided
        if provided_history is not None:
            if not isinstance(provided_history, list):
                await websocket.send_json(
                    WebSocketMessage(
                        type="error",
                        message="history parameter must be a list of message objects"
                    ).to_json()
                )
                return
            for msg in provided_history:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    await websocket.send_json(
                        WebSocketMessage(
                            type="error",
                            message="Each history item must be a dict with 'role' and 'content' keys"
                        ).to_json()
                    )
                    return
        
        # Handle image context if provided
        image_base64 = None
        image_tags = []
        image_description = None
        if image_name:
            # Get file metadata for type detection
            image_metadata = metadata_store.get_metadata_by_filename(image_name)
            file_type_label = "PDF document" if (image_metadata and image_metadata.type == "pdf") else "video" if (image_metadata and image_metadata.type == "video") else "image"
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"Loading {file_type_label} context: {image_name}..."
                ).to_json()
            )
            if image_metadata:
                image_tags = image_metadata.tags
                image_description = image_metadata.description
            
            # Load file (handles image, video, and PDF)
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
                return
        
        # Perform RAG search
        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message="Searching knowledge base..."
            ).to_json()
        )
        
        # Build search query with history if provided
        if provided_history:
            # Format history for embedding: "user: query1, assistant: response1, user: query2, ..."
            history_text = ", ".join([
                f"{msg['role']}: \"{msg['content']}\""
                for msg in provided_history
            ])
            search_query = f"{history_text}, user: \"{user_message}\""
        else:
            search_query = user_message
        
        # Search RAG
        relevant_files = await rag_service.search(
            search_query,
            k=config.top_k
        )
        
        # Build context from relevant files using shared helper function
        rag_context, file_list = build_rag_context_from_results(relevant_files)
        
        # Build image context if available
        image_context = None
        if image_name and (image_tags or image_description):
            image_context = {
                "image_name": image_name,
                "tags": image_tags,
                "description": image_description,
                "image_base64": image_base64
            }
        
        # Send the complete context package
        await websocket.send_json(
            WebSocketMessage(
                type="result",
                message="RAG context retrieved successfully",
                data={
                    "system_prompt": config.chat_system_prompt,
                    "rag_context": rag_context,
                    "relevant_files": file_list,
                    "file_details": [
                        {
                            "fileName": f.fileName,
                            "type": f.type,
                            "tags": f.tags,
                            "description": f.description,
                            "creationTime": f.creationTime
                        } for f in relevant_files
                    ],
                    "image_context": image_context,
                    "user_message": user_message,
                    "history": provided_history
                }
            ).to_json()
        )
        
        # Unload embedding model
        await llm_service.unload_model()
        
    except WebSocketDisconnect:
        # Client disconnected
        if llm_service:
            await llm_service.unload_model()
    except Exception as e:
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        await websocket.send_json(
            WebSocketMessage(
                type="error",
                message=f"Cloud chat error: {type(e).__name__}: {str(e)}",
                data=error_details
            ).to_json()
        )
        if llm_service:
            await llm_service.unload_model()
    finally:
        await websocket.close()


@router.websocket("/deep-chat")
async def deep_chat_ws(websocket: WebSocket):
    """
    Deep Chat WebSocket endpoint with multi-round thinking and RAG access.
    
    The LLM can perform multiple rounds of thinking and has access to functions
    that allow querying both media RAG and fact RAG. For the client calling this
    websocket, it acts the same as the normal chat websocket API.
    
    Number of rounds is controlled by the chat_rounds config parameter.
    """
    await websocket.accept()
    
    # Track if we should cleanup on disconnect
    generation_task = None
    llm_service = None
    
    try:
        # Get configuration and services
        config = get_config()
        llm_service = get_llm_service()
        rag_service = get_rag_service()
        knowledge_service = get_knowledge_service()
        metadata_store = get_metadata_store()
        
        # Prepare chat session (load RAG and embedding models)
        rag_available, embedding_loaded = await chat_helpers.prepare_chat_session(
            websocket, metadata_store
        )
        
        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message=f"Deep Chat ready. Send your message."
            ).to_json()
        )
        
        # Receive user message
        msg_data = await websocket.receive_json()
        
        user_message = msg_data.get("message")
        image_name = msg_data.get("image_name")  # Optional image name for visual chat
        provided_history = msg_data.get("history")  # Optional chat history (OpenAI format)
        
        # Determine which model to use
        if config.enable_visual_chat and image_name:
            chat_model = config.vision_model
            mmproj_file = config.mmproj_model
            use_vision = True
        else:
            chat_model = config.chat_model
            mmproj_file = config.mmproj_model if config.mmproj_model else None
            use_vision = False
        
        if not user_message:
            await websocket.send_json(
                WebSocketMessage(
                    type="error",
                    message="No message provided"
                ).to_json()
            )
            return
        
        # Validate and setup history
        active_history = await chat_helpers.validate_and_setup_history(websocket, provided_history)
        if active_history is None:
            return  # Error already sent
        
        # Handle visual conversation with image
        image_base64 = None
        image_tags = []
        image_description = None
        if use_vision and image_name:
            image_base64, image_tags, image_description, error = await chat_helpers.load_image_for_chat(
                websocket, image_name, metadata_store
            )
            if error:
                return  # Error already sent
        
        # Add current user message to active history
        active_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Gather initial context (limited initial RAG search)
        initial_context = await chat_helpers.gather_initial_context(
            websocket, user_message, active_history, rag_available, embedding_loaded
        )
        
        # Now load the chat model for generation
        model_type = "vision" if use_vision else "chat"
        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message=f"Loading {model_type} model {chat_model}..." + (f" with mmproj: {mmproj_file}" if use_vision and mmproj_file else "")
            ).to_json()
        )
        
        if use_vision and mmproj_file:
            await llm_service.load_model(chat_model, mmproj=mmproj_file)
        else:
            await llm_service.load_model(chat_model)
        
        # Send chat model startup command
        chat_startup_cmd = llm_service.get_startup_command()
        if chat_startup_cmd:
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"Chat Model Command: {chat_startup_cmd}",
                    data={"chat_startup_command": chat_startup_cmd}
                ).to_json()
            )
            await asyncio.sleep(0.1)
        
        # ===== DEEP CHAT MULTI-ROUND THINKING =====
        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message=f"Starting deep thinking ({config.chat_rounds} rounds)..."
            ).to_json()
        )
        
        # Define RAG query functions available to the LLM
        async def query_media_rag_func(query: str, k: int = 5) -> str:
            """Query the media RAG (images, videos, etc.) and return relevant file information."""
            if not rag_available:
                return "Media RAG is not available."
            
            try:
                # Switch to embedding model temporarily
                await llm_service.load_model(config.embedding_model)
                relevant_files = await rag_service.search(query, k=k)
                # Switch back to chat model
                if use_vision and mmproj_file:
                    await llm_service.load_model(chat_model, mmproj=mmproj_file)
                else:
                    await llm_service.load_model(chat_model)
                
                context, _ = chat_helpers.build_rag_context_from_results(relevant_files)
                return context
            except Exception as e:
                return f"Error querying media RAG: {str(e)}"
        
        async def query_fact_rag_func(query: str, token_budget: int = 1000, min_relevance: float = 0.3) -> str:
            """Query the conversation fact RAG and return relevant historical facts."""
            if not config.enable_knowledge_storage or not knowledge_service.is_loaded():
                return "Fact RAG is not available."
            
            try:
                # Switch to embedding model temporarily
                await llm_service.load_model(config.embedding_model)
                query_embedding = await llm_service.embed(query)
                # Switch back to chat model
                if use_vision and mmproj_file:
                    await llm_service.load_model(chat_model, mmproj=mmproj_file)
                else:
                    await llm_service.load_model(chat_model)
                
                relevant_facts = knowledge_service.select_knowledge(
                    query_embedding,
                    token_budget=token_budget,
                    min_relevance=min_relevance,
                )
                
                if relevant_facts:
                    fact_lines = [f"- {f['message']}" for f in relevant_facts]
                    return "\n".join(fact_lines)
                else:
                    return "No relevant facts found."
            except Exception as e:
                return f"Error querying fact RAG: {str(e)}"
        
        # Multi-round thinking loop
        thinking_history = []
        all_round_responses = []  # Collect raw output from every round
        final_response = ""
        final_files = []
        
        for round_num in range(config.chat_rounds):
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"Thinking round {round_num + 1}/{config.chat_rounds}..."
                ).to_json()
            )
            
            # Build messages for this round
            current_date = datetime.now().strftime("%B %d, %Y")
            
            # Prepare round context
            round_context = f"Current Date: {current_date}\n\n"
            
            # Include initial context if available (only in first round)
            if round_num == 0 and initial_context:
                round_context += f"{initial_context}\n\n"
            
            if thinking_history:
                round_context += "Previous thinking rounds:\n" + "\n".join(thinking_history) + "\n\n"
            
            # On the last round, inject a mandatory conclusion instruction
            is_last_round = (round_num == config.chat_rounds - 1)
            system_content = f"{config.deep_chat_system_prompt}\n\n{round_context}"
            if is_last_round:
                system_content += ("\n\n*** THIS IS YOUR FINAL ROUND. You MUST respond with "
                                   "<conclusion>your answer</conclusion> and "
                                   "<files>relevant files or empty</files> tags NOW. "
                                   "Do NOT use <think> tags. Do NOT call any functions. "
                                   "Provide your final answer immediately. ***")
            
            messages = [
                {
                    "role": "system",
                    "content": system_content
                }
            ] + active_history
            
            # Generate response for this round
            round_response = ""
            
            async def generate_round():
                nonlocal round_response
                try:
                    if use_vision and image_base64:
                        # Vision mode
                        image_context_parts = []
                        if image_tags:
                            image_context_parts.append(f"Image Tags: {', '.join(image_tags)}")
                        if image_description:
                            image_context_parts.append(f"Image Description: {image_description}")
                        
                        image_context = "\n".join(image_context_parts) if image_context_parts else ""
                        
                        vision_system = f"{config.deep_chat_system_prompt}\n\n{current_date}\n\n{round_context}"
                        if image_context:
                            vision_system += f"\n\nImage Context:\n{image_context}"
                        if is_last_round:
                            vision_system += ("\n\n*** THIS IS YOUR FINAL ROUND. You MUST respond with "
                                              "<conclusion>your answer</conclusion> and "
                                              "<files>relevant files or empty</files> tags NOW. "
                                              "Do NOT use <think> tags. Do NOT call any functions. "
                                              "Provide your final answer immediately. ***")
                        
                        prompt = f"{vision_system}\n\nUser: {user_message}\n\nAssistant:"
                        
                        import base64
                        image_bytes = base64.b64decode(image_base64)
                        
                        round_response = await llm_service.generate_vision(image_bytes, prompt, mmproj_file)
                    else:
                        # Text mode - streaming
                        async for chunk in llm_service.generate(messages, stream=True):
                            round_response += chunk
                except asyncio.CancelledError:
                    raise
            
            generation_task = asyncio.create_task(generate_round())
            
            try:
                await asyncio.wait_for(generation_task, timeout=config.llm_timeout)
            except asyncio.TimeoutError:
                generation_task.cancel()
                await websocket.send_json(
                    WebSocketMessage(
                        type="error",
                        message=f"Response generation timed out after {config.llm_timeout} seconds"
                    ).to_json()
                )
                return
            
            # Store raw round output
            all_round_responses.append(f"--- Round {round_num + 1} ---\n{round_response}")
            
            # Parse response for completion or function calls
            # Check for conclusion (final answer)
            conclusion_match = re.search(r'<conclusion>(.*?)</conclusion>', round_response, re.DOTALL)
            files_match = re.search(r'<files>(.*?)</files>', round_response, re.DOTALL)
            
            if conclusion_match:
                final_response = conclusion_match.group(1).strip()
                if files_match:
                    files_content = files_match.group(1).strip()
                    if files_content:
                        # Parse file list (comma or newline separated)
                        final_files = [f.strip().lstrip('- ') for f in re.split(r'[,\n]', files_content) if f.strip()]
                thinking_history.append(f"Round {round_num + 1}: Reached final conclusion")
                break
            
            # Extract and execute function calls
            function_calls = re.findall(r'<function_call>(.*?)</function_call>', round_response, re.DOTALL)
            
            if function_calls:
                function_results = []
                
                for fc in function_calls:
                    name_match = re.search(r'<name>(.*?)</name>', fc, re.DOTALL)
                    args_match = re.search(r'<arguments>(.*?)</arguments>', fc, re.DOTALL)
                    
                    if name_match and args_match:
                        func_name = name_match.group(1).strip()
                        args_json = args_match.group(1).strip()
                        
                        try:
                            args = json.loads(args_json)
                            
                            if func_name == "query_media_rag":
                                result = await query_media_rag_func(**args)
                                function_results.append(f"query_media_rag({args}): {result}")
                            elif func_name == "query_fact_rag":
                                result = await query_fact_rag_func(**args)
                                function_results.append(f"query_fact_rag({args}): {result}")
                            else:
                                function_results.append(f"Unknown function: {func_name}")
                        except Exception as e:
                            function_results.append(f"Error executing {func_name}: {str(e)}")
                
                # Add function results to thinking history
                thinking_history.append(f"Round {round_num + 1}: Called {len(function_results)} function(s)")
                
                # Add function results to active history for next round
                active_history.append({
                    "role": "assistant",
                    "content": f"Function results: {'; '.join(function_results)}"
                })
            else:
                # No function calls and no conclusion - treat as intermediate thinking
                thinking_history.append(f"Round {round_num + 1}: Thinking...")
        
        # Build full_response from all rounds
        full_response_text = "\n\n".join(all_round_responses)
        
        # If no conclusion was found during the loop, extract from last response
        if not final_response:
            # Sanitize: remove any <think> blocks first
            response_sanitized = re.sub(r'<think>.*?</think>', '', round_response, flags=re.DOTALL)
            
            # Try to extract conclusion
            conclusion_match = re.search(r'<conclusion>(.*?)</conclusion>', response_sanitized, re.DOTALL)
            files_match = re.search(r'<files>(.*?)</files>', response_sanitized, re.DOTALL)
            
            if conclusion_match:
                final_response = conclusion_match.group(1).strip()
            else:
                # Fall back to cleaned raw text (strip remaining XML tags)
                final_response = re.sub(r'<.*?>', '', response_sanitized, flags=re.DOTALL).strip()
            
            if files_match:
                files_content = files_match.group(1).strip()
                if files_content:
                    parts = re.split(r'\r?\n|,', files_content)
                    for p in parts:
                        p = re.sub(r'^[\-*]\s*', '', p.strip())
                        if p:
                            final_files.append(p)
        
        # Send conclusion message (matching regular chat format)
        await websocket.send_json(
            WebSocketMessage(
                type="conclusion",
                message=final_response
            ).to_json()
        )
        
        # Send files message (matching regular chat format)
        files_raw = ", ".join(final_files) if final_files else ""
        await websocket.send_json(
            WebSocketMessage(
                type="files",
                message=files_raw,
                data={"relevant_files": final_files}
            ).to_json()
        )
        
        # Send full_response with complete raw output from all rounds
        await websocket.send_json(
            WebSocketMessage(
                type="full_response",
                message=full_response_text,
                data={
                    "thinking_rounds": len(thinking_history),
                    "round_count": len(all_round_responses)
                }
            ).to_json()
        )
        
        # Send completion
        await websocket.send_json(
            WebSocketMessage(
                type="result",
                message="Response complete",
                data={
                    "thinking_rounds": len(thinking_history),
                    "relevant_files": final_files
                }
            ).to_json()
        )
        
        # Unload model
        if llm_service:
            await llm_service.unload_model()
        
        # Store objective facts from conversation into knowledge base
        await chat_helpers.store_objective_facts(
            user_message,
            None,  # We don't have the embedding readily available here
            embedding_loaded
        )
        
    except WebSocketDisconnect:
        if generation_task and not generation_task.done():
            generation_task.cancel()
            try:
                await generation_task
            except asyncio.CancelledError:
                pass
        
        if llm_service and config.llm_mode == "cli":
            if hasattr(llm_service.current_backend, 'cancel_generation'):
                await llm_service.current_backend.cancel_generation()
        
        if llm_service:
            await llm_service.unload_model()
    except asyncio.CancelledError:
        if llm_service and config.llm_mode == "cli":
            if hasattr(llm_service.current_backend, 'cancel_generation'):
                await llm_service.current_backend.cancel_generation()
        
        if llm_service:
            await llm_service.unload_model()
    except Exception as e:
        if generation_task and not generation_task.done():
            generation_task.cancel()
            try:
                await generation_task
            except asyncio.CancelledError:
                pass
        
        startup_command = llm_service.get_startup_command() if llm_service else None
        
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "chat_model": msg_data.get("chat_model") if 'msg_data' in locals() else None,
            "startup_command": startup_command
        }
        await websocket.send_json(
            WebSocketMessage(
                type="error",
                message=f"Deep chat error: {type(e).__name__}: {str(e)}",
                data=error_details
            ).to_json()
        )
        
        if llm_service:
            await llm_service.unload_model()
    finally:
        if generation_task and not generation_task.done():
            generation_task.cancel()
        await websocket.close()



@router.post("/kill", response_model=StatusResponse)
async def kill_server():
    """Kill the server and all associated processes (llama-server, llama-cli, etc.)."""
    try:
        # Get all services to clean up
        llm_service = get_llm_service()
        embedding_service = get_embedding_service()
        vision_service = get_vision_service()
        
        # Unload all models and kill any running processes
        async def cleanup_all():
            try:
                await llm_service.unload_model()
            except:
                pass
            
            try:
                await embedding_service.unload_model()
            except:
                pass
            
            try:
                await vision_service.unload_model()
            except:
                pass
        
        await cleanup_all()
        
        # Schedule server shutdown after response is sent
        async def shutdown():
            await asyncio.sleep(0.5)  # Give time for response to be sent
            
            # Kill any remaining llama processes
            try:
                os.system("pkill -9 llama-server")
                os.system("pkill -9 llama-cli")
                os.system("pkill -9 llama")
            except:
                pass
            
            # Exit the application
            os.kill(os.getpid(), signal.SIGTERM)
        
        # Schedule shutdown
        asyncio.create_task(shutdown())
        
        return StatusResponse(
            status="success",
            message="Server shutdown initiated. All processes will be terminated."
        )
    except Exception as e:
        return StatusResponse(
            status="error",
            message=f"Error during shutdown: {str(e)}"
        )


@router.post("/detect-faces")
async def detect_faces(request: dict):
    """
    Detect and identify faces in images.
    
    Request body:
    {
        "file_paths": ["image1.jpg", "image2.png"],
        "similarity_threshold": 0.6  // optional, default 0.6
    }
    
    Response:
    {
        "results": {
            "image1.jpg": [
                {
                    "face_id": "face_001",
                    "bbox": [x, y, width, height],
                    "confidence": 0.99,
                    "is_new": false
                }
            ]
        }
    }
    """
    try:
        file_paths = request.get("file_paths", [])
        similarity_threshold = request.get("similarity_threshold", 0.5)
        
        if not file_paths:
            raise HTTPException(status_code=400, detail="file_paths is required")
        
        metadata_store = get_metadata_store()
        face_service = get_face_service()
        
        # Check if metadata file has been updated and reload if necessary
        await metadata_store.reload_if_modified()
        
        results = {}
        
        for file_path_str in file_paths:
            # Get metadata
            metadata = metadata_store.get_metadata_by_filename(file_path_str)
            if not metadata:
                results[file_path_str] = {"error": f"Metadata not found for {file_path_str}"}
                continue
            
            # Get full path
            file_path = metadata_store.get_file_path(file_path_str)
            
            if not file_path.exists():
                results[file_path_str] = {"error": f"File not found: {file_path}"}
                continue
            
            # Detect and identify faces
            try:
                faces = await face_service.detect_and_identify_faces(
                    file_path,
                    metadata_store,
                    similarity_threshold
                )
                results[file_path_str] = faces
            except Exception as e:
                results[file_path_str] = {"error": str(e)}
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get-face-crop")
async def get_face_crop(request: dict):
    """
    Get cropped image of a specific face.
    
    Request body:
    {
        "image_name": "photo.jpg",
        "face_id": "face_001",
        "padding": 20  // optional, default 20
    }
    
    Response:
    {
        "face_crop_base64": "base64_encoded_image_data",
        "face_id": "face_001",
        "image_name": "photo.jpg"
    }
    """
    try:
        image_name = request.get("image_name")
        face_id = request.get("face_id")
        padding = request.get("padding", 20)
        
        if not image_name or not face_id:
            raise HTTPException(
                status_code=400, 
                detail="image_name and face_id are required"
            )
        
        metadata_store = get_metadata_store()
        face_service = get_face_service()
        
        # Check if metadata file has been updated and reload if necessary
        await metadata_store.reload_if_modified()
        
        # Get metadata
        metadata = metadata_store.get_metadata_by_filename(image_name)
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Metadata not found for {image_name}"
            )
        
        # Get full path
        image_path = metadata_store.get_file_path(image_name)
        
        if not image_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {image_path}"
            )
        
        # Get face crop
        face_crop_base64 = await face_service.get_face_crop_base64(
            image_path,
            face_id,
            metadata_store,
            padding
        )
        
        if face_crop_base64 is None:
            raise HTTPException(
                status_code=404,
                detail=f"Face {face_id} not found in {image_name}"
            )
        
        return {
            "face_crop_base64": face_crop_base64,
            "face_id": face_id,
            "image_name": image_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rename-face-id")
async def rename_face_id(request: dict):
    """
    Rename a face ID in the face embeddings database.
    
    Request body:
    {
        "old_face_id": "face_001",
        "new_face_id": "john_doe"
    }
    
    Response:
    {
        "status": "success",
        "message": "Face ID renamed from face_001 to john_doe",
        "old_face_id": "face_001",
        "new_face_id": "john_doe"
    }
    """
    try:
        old_face_id = request.get("old_face_id")
        new_face_id = request.get("new_face_id")
        
        if not old_face_id or not new_face_id:
            raise HTTPException(
                status_code=400,
                detail="old_face_id and new_face_id are required"
            )
        
        metadata_store = get_metadata_store()
        face_service = get_face_service()
        
        # Check if metadata file has been updated and reload if necessary
        await metadata_store.reload_if_modified()
        
        # Attempt to rename
        success = face_service.rename_face_id(
            old_face_id,
            new_face_id,
            metadata_store
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Face ID '{old_face_id}' not found"
            )
        
        return {
            "status": "success",
            "message": f"Face ID renamed from {old_face_id} to {new_face_id}",
            "old_face_id": old_face_id,
            "new_face_id": new_face_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
