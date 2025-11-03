"""Response models for API endpoints."""

from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel, Field


class ConfigResponse(BaseModel):
    """Configuration response."""
    reduced_embedding_size: Optional[int]
    chat_rounds: int
    image_quality: float
    llm_mode: str
    top_k: int
    recency_bias: float
    enable_visual_chat: bool
    chat_model: str
    embedding_model: str
    vision_model: str
    mmproj_model: str
    chat_system_prompt: str
    tag_prompt: str
    describe_prompt: str
    vision_binary: Optional[str] = None
    backend: Optional[str] = None
    model_timeout: Optional[int] = None
    llm_params: Dict[str, Any]
    rag_directory_name: str
    storage_metadata_path: Optional[str]


class StatusResponse(BaseModel):
    """Generic status response."""
    status: Literal["success", "error", "info"]
    message: str
    data: Optional[Dict[str, Any]] = None


class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: Literal["status", "progress", "result", "error", "confirmation_needed", "thinking", "conclusion", "files"]
    message: str
    data: Optional[Dict[str, Any]] = None
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON dict."""
        return self.model_dump()
