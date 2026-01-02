"""Metadata models for file storage.

File Storage Structure:
    All photos and videos must be stored in a 'files' subdirectory at the same level
    as the storage_metadata.json file:
    
    /path/to/your/data/
    ├── storage_metadata.json    # Metadata file
    ├── rag/                     # RAG database (auto-generated)
    └── files/                   # All media files
        ├── photo1.jpg
        ├── video1.mp4
        └── ...
    
    The fileName field in storage_metadata.json should contain only the filename
    (e.g., "photo1.jpg"), not the full path. The MetadataStore.get_file_path()
    method automatically prepends "files/" to resolve the full path.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import json
from pathlib import Path
import asyncio


class FileMetadata(BaseModel):
    """Metadata for a single file.
    
    This model accepts extra fields that may be present in the storage_metadata.json
    file, ensuring forward compatibility with new metadata properties.
    """
    model_config = {"extra": "allow"}  # Allow extra fields
    
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
        """Convert metadata to text for embedding generation.
        
        Includes all metadata fields, including extra/unknown fields.
        """
        # Start with standard fields
        parts = [
            f"File: {self.fileName}",
            f"Type: {self.type}",
            f"Created: {self.creationTime}",
            f"Tags: {', '.join(self.tags)}" if self.tags else "",
            f"Description: {self.description}" if self.description else ""
        ]
        
        # Add any extra fields stored in model_extra (Pydantic v2)
        if hasattr(self, '__pydantic_extra__') and self.__pydantic_extra__:
            for field_name, field_value in self.__pydantic_extra__.items():
                if field_value is not None:
                    # Format extra fields for text representation
                    if isinstance(field_value, list):
                        parts.append(f"{field_name}: {', '.join(str(v) for v in field_value)}")
                    elif isinstance(field_value, dict):
                        parts.append(f"{field_name}: {json.dumps(field_value)}")
                    else:
                        parts.append(f"{field_name}: {field_value}")
        
        return "\n".join(p for p in parts if p)


class MetadataStore:
    """Store for managing file metadata."""
    
    def __init__(self, metadata_path: str):
        """Initialize metadata store."""
        self.metadata_path = Path(metadata_path)
        self.metadata: List[FileMetadata] = []
        self._last_modified_time: Optional[float] = None
        self._reload_lock = asyncio.Lock()
        self.identified_properties: Dict[str, str] = {}  # Track property types
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load metadata from file and identify all properties."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        # Track file modification time
        self._last_modified_time = self.metadata_path.stat().st_mtime
        
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.metadata = [FileMetadata(**item) for item in data]
            
            # Analyze all properties across all entries
            self._identify_properties(data)
    
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
        """Get full path to a file relative to metadata location.
        
        Files are expected to be in a 'files' subdirectory at the same level
        as the storage_metadata.json file.
        """
        return self.metadata_path.parent / "files" / filename
    
    def has_been_modified(self) -> bool:
        """Check if the metadata file has been modified since last load."""
        if not self.metadata_path.exists():
            return False
        
        current_mtime = self.metadata_path.stat().st_mtime
        return current_mtime > self._last_modified_time if self._last_modified_time else True
    
    async def reload_if_modified(self) -> bool:
        """Reload metadata if file has been modified. Returns True if reloaded.
        
        Thread-safe: Uses asyncio lock to prevent concurrent reloads.
        """
        async with self._reload_lock:
            if self.has_been_modified():
                self._load_metadata()
                return True
            return False
    
    def reload(self) -> None:
        """Reload metadata from file."""
        self._load_metadata()
    
    def _identify_properties(self, data: List[Dict[str, Any]]) -> None:
        """Identify all properties and their types from metadata entries."""
        self.identified_properties = {}
        
        for entry in data:
            for key, value in entry.items():
                if key not in self.identified_properties:
                    # Determine type
                    if value is None:
                        self.identified_properties[key] = "null"
                    elif isinstance(value, bool):
                        self.identified_properties[key] = "boolean"
                    elif isinstance(value, int):
                        self.identified_properties[key] = "integer"
                    elif isinstance(value, float):
                        self.identified_properties[key] = "number"
                    elif isinstance(value, str):
                        self.identified_properties[key] = "string"
                    elif isinstance(value, list):
                        if value and isinstance(value[0], str):
                            self.identified_properties[key] = "array[string]"
                        else:
                            self.identified_properties[key] = "array"
                    elif isinstance(value, dict):
                        self.identified_properties[key] = "object"
                    else:
                        self.identified_properties[key] = "unknown"
    
    def get_identified_properties(self) -> Dict[str, str]:
        """Get dictionary of identified property names and their types."""
        return self.identified_properties.copy()
