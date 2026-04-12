"""Unified RAG search result type for dual-type retrieval (files + conversations)."""

from dataclasses import dataclass, field
from typing import Literal, Optional

from app.models.metadata import FileMetadata


@dataclass
class RAGResult:
    """A single result from RAG search — wraps either a file or a compacted conversation."""

    source: Literal["file", "conversation"]
    identifier: str  # fileName for files, conversationId for conversations

    # File-specific (populated when source == "file")
    file_metadata: Optional[FileMetadata] = None

    # Conversation-specific (populated when source == "conversation")
    summary: Optional[str] = None
    compacted_at: Optional[str] = None
