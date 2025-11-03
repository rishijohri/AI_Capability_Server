"""Metadata models for file storage."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import json
from pathlib import Path


class FileMetadata(BaseModel):
    """Metadata for a single file."""
    fileName: str
    deviceId: str
    deviceName: str
    uploadTime: str
    tags: List[str] = Field(default_factory=list)
    aspectRatio: float
    type: str  # "video" or "image"
    aiModel: Optional[str] = None
    creationTime: str
    description: Optional[str] = None
    descriptionState: Optional[str] = None
    
    def get_creation_datetime(self) -> datetime:
        """Parse creation time to datetime."""
        return datetime.fromisoformat(self.creationTime.replace('Z', '+00:00'))
    
    def get_upload_datetime(self) -> datetime:
        """Parse upload time to datetime."""
        return datetime.fromisoformat(self.uploadTime.replace('Z', '+00:00'))
    
    def to_text_representation(self) -> str:
        """Convert metadata to text for embedding generation."""
        parts = [
            f"File: {self.fileName}",
            f"Type: {self.type}",
            f"Created: {self.creationTime}",
            f"Tags: {', '.join(self.tags)}" if self.tags else "",
            f"Description: {self.description}" if self.description else ""
        ]
        return "\n".join(p for p in parts if p)


class MetadataStore:
    """Store for managing file metadata."""
    
    def __init__(self, metadata_path: str):
        """Initialize metadata store."""
        self.metadata_path = Path(metadata_path)
        self.metadata: List[FileMetadata] = []
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load metadata from file."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.metadata = [FileMetadata(**item) for item in data]
    
    def save_metadata(self) -> None:
        """Save metadata to file (disabled - read-only mode)."""
        # No-op: Project never modifies storage_metadata.json
        pass
    
    def get_all_metadata(self) -> List[FileMetadata]:
        """Get all metadata."""
        return self.metadata
    
    def get_metadata_by_filename(self, filename: str) -> Optional[FileMetadata]:
        """Get metadata for a specific file."""
        for metadata in self.metadata:
            if metadata.fileName == filename:
                return metadata
        return None
    
    def update_metadata(self, filename: str, updates: Dict[str, Any]) -> bool:
        """Update metadata for a specific file (disabled - read-only mode)."""
        # No-op: Project never modifies storage_metadata.json
        # Always return False to indicate no update was made
        return False
    
    def get_file_path(self, filename: str) -> Path:
        """Get full path to a file relative to metadata location."""
        return self.metadata_path.parent / filename
    
    def reload(self) -> None:
        """Reload metadata from file."""
        self._load_metadata()
