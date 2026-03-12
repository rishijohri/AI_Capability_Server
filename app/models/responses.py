"""Response models for API endpoints."""

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class ConfigResponse(BaseModel):
    """Configuration response."""
    reduced_embedding_size: Optional[int]
    chat_rounds: int
    chat_mode: str = Field(default="rag", description="Chat mode: 'rag' or 'mcp'")
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
    llm_timeout: int
    llm_params: Dict[str, Any]
    rag_directory_name: str
    storage_metadata_path: Optional[str]
    model_directory: Optional[str] = Field(None, description="Custom model directory path")
    binary_config: str = Field(..., description="Selected binary configuration folder")
    system_info: Dict[str, str] = Field(..., description="Detected system information")
    available_binary_configs: list[str] = Field(..., description="List of available binary configurations")


class StatusResponse(BaseModel):
    """Generic status response."""
    status: Literal["success", "error", "info"]
    message: str
    data: Optional[Dict[str, Any]] = None


class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: Literal["status", "progress", "result", "error", "confirmation_needed", "thinking", "conclusion", "files", "full_response"]
    message: str
    data: Optional[Dict[str, Any]] = None
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON dict."""
        return self.model_dump()


class ModelInfo(BaseModel):
    """Information about an available model."""
    name: str = Field(..., description="Model identifier")
    type: Literal["vision", "chat", "embedding"] = Field(..., description="Model task type")
    model_file: str = Field(..., description="Model filename")
    model_exists: bool = Field(..., description="Whether model file exists in model directory")
    mmproj_file: Optional[str] = Field(None, description="MMProj file for vision models")
    mmproj_exists: Optional[bool] = Field(None, description="Whether MMProj file exists (for vision models)")
    llm_params: Optional[Dict[str, Any]] = Field(None, description="Model-specific LLM parameters")


class AvailableModelsResponse(BaseModel):
    """Response with available models."""
    models: list[ModelInfo] = Field(..., description="List of available models")
    total_count: int = Field(..., description="Total number of models matching criteria")
    task_type: Optional[str] = Field(None, description="Filtered task type, if any")


class DownloadStatus(BaseModel):
    """Download status for a single file."""
    filename: str = Field(..., description="Name of the file being downloaded")
    status: Literal["pending", "downloading", "completed", "failed", "skipped"] = Field(..., description="Download status")
    error: Optional[str] = Field(None, description="Error message if failed")
    bytes_downloaded: Optional[int] = Field(None, description="Bytes downloaded so far")
    total_bytes: Optional[int] = Field(None, description="Total bytes to download")
    

class DownloadModelsResponse(BaseModel):
    """Response for model download operation."""
    model_id: str = Field(..., description="Model ID being processed")
    files: List[DownloadStatus] = Field(..., description="Status of files being downloaded")
    overall_status: Literal["pending", "downloading", "completed", "failed", "partial"] = Field(..., description="Overall download status for this model")


class ModelOption(BaseModel):
    """Information about a model that can be downloaded."""
    model_id: str = Field(..., description="Model identifier (key in model_options)")
    name: str = Field(..., description="Model name")
    type: Literal["vision", "chat", "embedding"] = Field(..., description="Model task type")
    model_file: str = Field(..., description="Model filename")
    mmproj_file: Optional[str] = Field(None, description="MMProj file for vision models")
    repo_id: str = Field(..., description="Hugging Face repository ID")
    repo_id_configured: bool = Field(..., description="Whether repo_id is configured (non-empty)")
    llm_params: Optional[Dict[str, Any]] = Field(None, description="Model-specific LLM parameters")


class ModelOptionsResponse(BaseModel):
    """Response with all downloadable models."""
    models: List[ModelOption] = Field(..., description="List of all models in model_options")
    total_count: int = Field(..., description="Total number of models")
    configured_count: int = Field(..., description="Number of models with repo_id configured")
    task_type: Optional[str] = Field(None, description="Filtered task type, if any")
