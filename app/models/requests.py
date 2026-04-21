"""Request models for API endpoints."""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class ConfigUpdateRequest(BaseModel):
    """Request to update configuration."""
    reduced_embedding_size: Optional[int] = None
    chat_rounds: Optional[int] = None

    image_quality: Optional[float] = Field(None, ge=0.0, le=1.0, description="Image scale multiplier (0.0-1.0)")
    llm_mode: Optional[Literal["server", "cli"]] = None
    top_k: Optional[int] = None
    recency_bias: Optional[float] = None
    enable_visual_chat: Optional[bool] = None
    chat_model: Optional[str] = None
    embedding_model: Optional[str] = None
    vision_model: Optional[str] = None
    mmproj_model: Optional[str] = None
    chat_system_prompt: Optional[str] = None
    tag_prompt: Optional[str] = None
    describe_prompt: Optional[str] = None
    vision_binary: Optional[Literal["auto", "llama-mtmd-cli", "llama-qwen2vl-cli"]] = None
    backend: Optional[Literal["server", "cli"]] = None
    model_timeout: Optional[int] = None
    llm_timeout: Optional[int] = Field(None, ge=10, le=3600, description="Timeout for LLM operations in seconds")
    llm_params: Optional[Dict[str, Any]] = None
    binary_config: Optional[str] = Field(None, description="Binary configuration folder name (e.g., 'llama-mac-arm64', 'llama-win-vulkan-x64')")
    model_directory: Optional[str] = Field(None, description="Custom model directory path (absolute path)")
    tool_history_max_tags: Optional[int] = Field(None, ge=1, description="Number of tags kept when truncating scoped_rag_search results in tool call history")
    tool_history_max_results: Optional[int] = Field(None, ge=1, description="Number of results kept when truncating other MCP tool results in tool call history")
    max_tags_per_scope: Optional[int] = Field(None, ge=1, description="Maximum unique tags returned by get_scoped_tags per call")
    max_dates_per_scope: Optional[int] = Field(None, ge=1, description="Maximum date ranges returned by get_scoped_dates per call")


class StorageMetadataRequest(BaseModel):
    """Request to set storage metadata path."""
    path: str = Field(..., description="Path to storage-metadata.json file")
    bookmark: Optional[str] = Field(
        None, 
        description="Base64-encoded security-scoped bookmark for macOS sandbox access"
    )


class ModelDirectoryRequest(BaseModel):
    """Request to set model directory path with optional bookmark for sandbox access."""
    path: str = Field(..., description="Absolute path to model directory")
    bookmark: Optional[str] = Field(
        None, 
        description="Base64-encoded minimal bookmark for macOS sandbox access"
    )


class TagRequest(BaseModel):
    """Request to generate tags for files."""
    file_paths: List[str] = Field(..., description="List of file paths to tag")


class DescribeRequest(BaseModel):
    """Request to generate descriptions for files."""
    file_paths: List[str] = Field(..., description="List of file paths to describe")


class ChatMessage(BaseModel):
    """Chat message in OpenAI format."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatRequest(BaseModel):
    """Chat request."""
    messages: List[ChatMessage]
    stream: bool = Field(default=False, description="Stream responses")


class AvailableModelsRequest(BaseModel):
    """Request to get available models for a specific task type."""
    task_type: Optional[Literal["vision", "chat", "embedding"]] = Field(
        None, 
        description="Filter by task type (vision, chat, embedding). If None, returns all available models."
    )


class DownloadModelsRequest(BaseModel):
    """Request to download models from Hugging Face."""
    model_ids: List[str] = Field(
        ..., 
        description="List of model IDs from model_options to download (e.g., ['qwen_3', 'gemma3_4b_q4_k_m'])"
    )
    force_redownload: bool = Field(
        False, 
        description="Force re-download even if model already exists"
    )
    download_location: Optional[str] = Field(
        None,
        description="Custom download location (absolute path). If None, uses configured model_directory or default saved_llm location"
    )


class ModelOptionData(BaseModel):
    """Individual model option data from Flutter app."""
    model_id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Human-readable model name")
    type: Literal["chat", "vision", "embedding"] = Field(..., description="Model type")
    model_file: str = Field(..., description="Model filename (e.g., 'model.gguf')")
    mmproj_file: Optional[str] = Field(None, description="MMProj file for vision models")
    repo_id: str = Field(..., description="Hugging Face repository ID")
    is_default: bool = Field(False, description="Whether this is a default bundled model")
    llm_params: Optional[Dict[str, Any]] = Field(None, description="Custom LLM parameters")


class SetModelOptionsRequest(BaseModel):
    """Request to set model options from the Flutter app.
    
    This endpoint receives model configurations from the Flutter client,
    which now manages persistent model options locally and syncs them
    to the server on startup.
    """
    models: List[ModelOptionData] = Field(
        ..., 
        description="List of model options to set on the server"
    )


class CompactConversationsRequest(BaseModel):
    """Request to compact (summarize) conversations."""
    count: int = Field(..., ge=1, description="Number of conversations to compact")
    force_recompact: bool = Field(False, description="Re-compact already-compacted conversations")
    chat_model: Optional[str] = Field(None, description="Chat model override for summarization")
    embedding_model: Optional[str] = Field(None, description="Embedding model override")

