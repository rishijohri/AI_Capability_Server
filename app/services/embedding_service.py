"""Embedding generation service."""

from typing import List, Dict, Optional
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

from app.config import get_config
from app.models import FileMetadata, MetadataStore
from app.services.llm_service import get_llm_service


class EmbeddingService:
    """Service for generating and managing embeddings."""
    
    def __init__(self):
        """Initialize embedding service."""
        self.embeddings: Dict[str, List[float]] = {}
        self.pca_model: Optional[PCA] = None
        self.original_dim: Optional[int] = None
    
    async def generate_embeddings(
        self,
        metadata_store: MetadataStore,
        embedding_model: str,
        progress_callback=None,
        force_regen: bool = False
    ) -> Dict[str, List[float]]:
        """
        Generate embeddings for all files in metadata store.
        
        Args:
            metadata_store: Metadata store containing file information
            embedding_model: Name of embedding model to use
            progress_callback: Optional callback for progress updates
            force_regen: If True, regenerate all embeddings. If False, only generate for new files
            
        Returns:
            Dictionary mapping filenames to embedding vectors
        """
        llm_service = get_llm_service()
        
        # Get all metadata
        all_metadata = metadata_store.get_all_metadata()
        
        # Determine which files need embeddings
        if force_regen:
            # Regenerate all
            files_to_process = all_metadata
            if progress_callback:
                await progress_callback(0, 0, "Force regeneration enabled - processing all files")
        else:
            # Only process files without embeddings
            files_to_process = [
                metadata for metadata in all_metadata
                if metadata.fileName not in self.embeddings
            ]
            if progress_callback:
                existing_count = len(all_metadata) - len(files_to_process)
                await progress_callback(0, 0, f"Found {existing_count} existing embeddings, processing {len(files_to_process)} new files")
        
        if not files_to_process:
            if progress_callback:
                await progress_callback(0, 0, "All files already have embeddings")
            return self.embeddings
        
        # Load embedding model
        await llm_service.load_model(embedding_model)
        
        # Get and report startup command
        startup_cmd = llm_service.get_startup_command()
        if progress_callback and startup_cmd:
            await progress_callback(0, 0, f"LLM Command: {startup_cmd}")
        
        embeddings = self.embeddings.copy() if not force_regen else {}
        total = len(files_to_process)
        
        for idx, metadata in enumerate(files_to_process):
            # Generate text representation
            text = metadata.to_text_representation()
            
            # Generate embedding
            embedding = await llm_service.embed(text)
            embeddings[metadata.fileName] = embedding
            
            # Progress callback
            if progress_callback:
                await progress_callback(idx + 1, total, metadata.fileName)
        
        # Unload model after generation
        await llm_service.unload_model()
        
        # Store embeddings
        self.embeddings = embeddings
        self.original_dim = len(next(iter(embeddings.values()))) if embeddings else None
        
        # Save to file
        config = get_config()
        rag_dir = config.get_rag_directory()
        if rag_dir:
            rag_dir.mkdir(exist_ok=True)
            embeddings_file = rag_dir / "embeddings.json"
            with open(embeddings_file, 'w') as f:
                json.dump(embeddings, f, indent=2)
        
        return embeddings
    
    def load_embeddings(self, embeddings_path: Optional[Path] = None) -> bool:
        """
        Load embeddings from file.
        
        Args:
            embeddings_path: Optional path to embeddings file
            
        Returns:
            True if loaded successfully
        """
        if embeddings_path is None:
            config = get_config()
            rag_dir = config.get_rag_directory()
            if not rag_dir:
                return False
            embeddings_path = rag_dir / "embeddings.json"
        
        if not embeddings_path.exists():
            return False
        
        with open(embeddings_path, 'r') as f:
            self.embeddings = json.load(f)
        
        if self.embeddings:
            self.original_dim = len(next(iter(self.embeddings.values())))
        
        return True
    
    def reduce_embeddings(self, embeddings: Dict[str, List[float]], target_dim: int) -> Dict[str, List[float]]:
        """
        Reduce embedding dimensionality using PCA.
        
        Args:
            embeddings: Dictionary mapping filenames to embedding vectors
            target_dim: Target dimensionality
            
        Returns:
            Dictionary mapping filenames to reduced embeddings
        """
        if not embeddings:
            raise ValueError("No embeddings provided")
        
        # Convert to numpy array
        filenames = list(embeddings.keys())
        embeddings_matrix = np.array([embeddings[fn] for fn in filenames])
        
        # Apply PCA
        self.pca_model = PCA(n_components=target_dim)
        reduced_embeddings_matrix = self.pca_model.fit_transform(embeddings_matrix)
        
        # Convert back to dictionary
        reduced_embeddings = {
            fn: reduced_embeddings_matrix[i].tolist()
            for i, fn in enumerate(filenames)
        }
        
        # Update service embeddings
        self.embeddings = reduced_embeddings
        
        # Save to file
        config = get_config()
        rag_dir = config.get_rag_directory()
        if rag_dir:
            rag_dir.mkdir(exist_ok=True)
            embeddings_file = rag_dir / "embeddings.json"
            with open(embeddings_file, 'w') as f:
                json.dump(reduced_embeddings, f, indent=2)
        
        return reduced_embeddings
    
    def reduce_single_embedding(self, embedding: List[float]) -> List[float]:
        """
        Reduce a single embedding using trained PCA model.
        
        Args:
            embedding: Original embedding vector
            
        Returns:
            Reduced embedding vector
        """
        if self.pca_model is None:
            raise ValueError("PCA model not trained. Call reduce_embeddings first.")
        
        embedding_array = np.array(embedding).reshape(1, -1)
        reduced = self.pca_model.transform(embedding_array)
        return reduced[0].tolist()
    
    def get_embedding(self, filename: str) -> Optional[List[float]]:
        """Get embedding for a specific file."""
        return self.embeddings.get(filename)
    
    def get_all_embeddings(self) -> Dict[str, List[float]]:
        """Get all embeddings."""
        return self.embeddings
    
    async def generate_embeddings_for_files(
        self,
        files_to_process: List[FileMetadata],
        embedding_model: str,
        progress_callback=None,
        error_callback=None
    ) -> Dict[str, List[float]]:
        """
        Generate embeddings for a specific list of files.
        
        Args:
            files_to_process: List of file metadata to generate embeddings for
            embedding_model: Name of embedding model to use
            progress_callback: Optional callback for progress updates
            error_callback: Optional callback for error notifications
            
        Returns:
            Dictionary mapping filenames to embedding vectors
        """
        llm_service = get_llm_service()
        
        if not files_to_process:
            return self.embeddings
        
        # Load embedding model
        await llm_service.load_model(embedding_model)
        
        # Get and report startup command
        startup_cmd = llm_service.get_startup_command()
        if progress_callback and startup_cmd:
            await progress_callback(0, 0, f"LLM Command: {startup_cmd}")
        
        # Load existing embeddings
        embeddings = self.embeddings.copy()
        total = len(files_to_process)
        
        for idx, metadata in enumerate(files_to_process):
            max_retries = 2  # Retry once if server disconnects
            retry_count = 0
            success = False
            
            while retry_count <= max_retries and not success:
                try:
                    # Generate text representation
                    text = metadata.to_text_representation()
                    
                    # Generate embedding
                    embedding = await llm_service.embed(text)
                    embeddings[metadata.fileName] = embedding
                    success = True
                    
                    # Progress callback
                    if progress_callback:
                        await progress_callback(idx + 1, total, metadata.fileName)
                except Exception as e:
                    error_msg = str(e)
                    retry_count += 1
                    
                    # Check if it's a server disconnection/connection error
                    is_connection_error = any(keyword in error_msg.lower() for keyword in [
                        'server disconnected', 'connection', 'not running', 
                        'timeout', 'cannot connect', 'connection refused'
                    ])
                    
                    if is_connection_error and retry_count <= max_retries:
                        # Try to restart the model and retry
                        if progress_callback:
                            await progress_callback(0, 0, f"Server disconnected, restarting model (attempt {retry_count}/{max_retries})...")
                        
                        try:
                            # Unload and reload the model
                            await llm_service.unload_model()
                            await llm_service.load_model(embedding_model)
                            
                            # Report successful restart
                            if progress_callback:
                                await progress_callback(0, 0, "Model restarted successfully, retrying...")
                        except Exception as restart_error:
                            # If restart fails, notify and move to next file
                            if error_callback:
                                await error_callback(metadata.fileName, f"Failed to restart model: {str(restart_error)}")
                            break
                    else:
                        # Non-connection error or max retries reached
                        if error_callback:
                            if retry_count > max_retries:
                                await error_callback(metadata.fileName, f"Max retries exceeded: {error_msg}")
                            else:
                                await error_callback(metadata.fileName, error_msg)
                        break
        
        # Unload model after generation
        await llm_service.unload_model()
        
        # Store embeddings
        self.embeddings = embeddings
        if embeddings:
            self.original_dim = len(next(iter(embeddings.values())))
        
        # Apply PCA reduction if configured
        config = get_config()
        if config.reduced_embedding_size and embeddings:
            embeddings = self.reduce_embeddings(embeddings, config.reduced_embedding_size)
        else:
            # Save to file without PCA
            rag_dir = config.get_rag_directory()
            if rag_dir:
                rag_dir.mkdir(exist_ok=True)
                embeddings_file = rag_dir / "embeddings.json"
                with open(embeddings_file, 'w') as f:
                    json.dump(embeddings, f, indent=2)
        
        return embeddings


# Global embedding service instance
_embedding_service = EmbeddingService()


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    return _embedding_service
