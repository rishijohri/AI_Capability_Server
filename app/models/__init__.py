"""Data models for AI Server."""

from .metadata import FileMetadata, MetadataStore
from app.models.requests import (
    ConfigUpdateRequest,
    StorageMetadataRequest,
    TagRequest,
    DescribeRequest,
    ChatMessage,
    ChatRequest,
    AvailableModelsRequest
)
from .responses import (
    ConfigResponse,
    StatusResponse,
    WebSocketMessage,
    ModelInfo,
    AvailableModelsResponse
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
    "ConfigResponse",
    "StatusResponse",
    "WebSocketMessage",
    "ModelInfo",
    "AvailableModelsResponse"
]
