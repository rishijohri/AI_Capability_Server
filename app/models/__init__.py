"""Data models for AI Server."""

from .metadata import FileMetadata, MetadataStore
from app.models.requests import (
    ConfigUpdateRequest,
    StorageMetadataRequest,
    TagRequest,
    DescribeRequest,
    ChatMessage,
    ChatRequest,
    AvailableModelsRequest,
    DownloadModelsRequest
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
    "TagRequest",
    "DescribeRequest",
    "ChatMessage",
    "ChatRequest",
    "AvailableModelsRequest",
    "DownloadModelsRequest",
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
