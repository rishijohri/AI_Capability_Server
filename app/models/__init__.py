"""Data models for AI Server."""

from .metadata import FileMetadata, MetadataStore
from app.models.requests import (
    ConfigUpdateRequest,
    StorageMetadataRequest,
    ModelDirectoryRequest,
    TagRequest,
    DescribeRequest,
    ChatMessage,
    ChatRequest,
    AvailableModelsRequest,
    DownloadModelsRequest,
    SetModelOptionsRequest
)
from .responses import (
    ConfigResponse,
    StatusResponse,
    WebSocketMessage,
    ModelInfo,
    AvailableModelsResponse,
    DownloadStatus,
    DownloadModelsResponse,
    ModelOption,
    ModelOptionsResponse
)

__all__ = [
    "FileMetadata",
    "MetadataStore",
    "ConfigUpdateRequest",
    "StorageMetadataRequest",
    "ModelDirectoryRequest",
    "TagRequest",
    "DescribeRequest",
    "ChatMessage",
    "ChatRequest",
    "AvailableModelsRequest",
    "DownloadModelsRequest",
    "SetModelOptionsRequest",
    "ConfigResponse",
    "StatusResponse",
    "WebSocketMessage",
    "ModelInfo",
    "AvailableModelsResponse",
    "DownloadStatus",
    "DownloadModelsResponse",
    "ModelOption",
    "ModelOptionsResponse"
]
