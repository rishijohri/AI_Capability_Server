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


class StorageMetadataRequest(BaseModel):
    """Request to set storage metadata path."""
    path: str = Field(..., description="Path to storage-metadata.json file")


class TagRequest(BaseModel):
    """Request to generate tags for files."""
    file_paths: List[str] = Field(..., description="List of file paths to tag")


class DescribeRequest(BaseModel):
    """Request to generate descriptions for files."""
    file_paths: List[str] = Field(..., description="List of file paths to describe")


class RegenerateEmbeddingsRequest(BaseModel):
    """Request to regenerate embeddings for specific files."""
    data: List[Dict[str, Any]] = Field(..., description="List of file metadata objects to regenerate embeddings for")


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
