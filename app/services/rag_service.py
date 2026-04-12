"""RAG service using FAISS for vector database."""

from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import pickle
from abc import ABC, abstractmethod
from datetime import datetime

try:
    import faiss
except ImportError:
    faiss = None

from app.config import get_config
from app.models import FileMetadata, MetadataStore
from app.models.rag_result import RAGResult
from app.services.embedding_service import get_embedding_service


class VectorDB(ABC):
    """Abstract base class for vector database implementations."""
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add vectors to the database."""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> Tuple[List[str], List[float]]:
        """Search for k nearest neighbors."""
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save database to file."""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Load database from file."""
        pass


class FAISSVectorDB(VectorDB):
    """FAISS implementation of vector database."""
    
    def __init__(self, dimension: int):
        """Initialize FAISS index."""
        if faiss is None:
            raise ImportError("FAISS library not installed")
        
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.id_map: List[str] = []
    
    def add_vectors(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Add vectors to FAISS index."""
        vectors = vectors.astype('float32')
        self.index.add(vectors)
        self.id_map.extend(ids)
    
    def search(self, query_vector: np.ndarray, k: int) -> Tuple[List[str], List[float]]:
        """Search for k nearest neighbors."""
        query_vector = query_vector.astype('float32').reshape(1, -1)
        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: query has {query_vector.shape[1]} dimensions "
                f"but FAISS index expects {self.dimension}. "
                f"Rebuild the RAG with the current embedding model."
            )
        distances, indices = self.index.search(query_vector, k)
        
        result_ids = [self.id_map[idx] for idx in indices[0] if idx < len(self.id_map)]
        result_distances = distances[0].tolist()
        
        return result_ids, result_distances
    
    def save(self, path: Path) -> None:
        """Save FAISS index and ID map."""
        # Save index
        faiss.write_index(self.index, str(path))
        
        # Save ID map
        id_map_path = path.parent / f"{path.stem}_idmap.pkl"
        with open(id_map_path, 'wb') as f:
            pickle.dump(self.id_map, f)
    
    def load(self, path: Path) -> None:
        """Load FAISS index and ID map."""
        # Load index
        self.index = faiss.read_index(str(path))
        self.dimension = self.index.d
        
        # Load ID map
        id_map_path = path.parent / f"{path.stem}_idmap.pkl"
        with open(id_map_path, 'rb') as f:
            self.id_map = pickle.load(f)


class RAGService:
    """Service for RAG (Retrieval Augmented Generation)."""
    
    def __init__(self):
        """Initialize RAG service."""
        self.vector_db: Optional[VectorDB] = None
        self.metadata_store: Optional[MetadataStore] = None
        self.use_reduced_embeddings = False
    
    async def build_rag(
        self,
        metadata_store: MetadataStore,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Build RAG database from embeddings.
        
        Args:
            metadata_store: Metadata store
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing:
                - success: bool
                - mismatched_files: List of filenames with mismatched embedding dimensions
                - majority_dimension: The dimension used for the RAG
                - removed_count: Number of files removed due to mismatch
        """
        config = get_config()
        embedding_service = get_embedding_service()
        
        # Get embeddings
        embeddings = embedding_service.get_all_embeddings()
        if not embeddings:
            raise ValueError("No embeddings available. Generate embeddings first.")
        
        # Check for dimension mismatches and find majority dimension
        dimension_counts = {}
        file_dimensions = {}
        
        for filename, embedding in embeddings.items():
            dim = len(embedding)
            file_dimensions[filename] = dim
            dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
        
        # Find majority dimension
        if len(dimension_counts) > 1:
            majority_dimension = max(dimension_counts, key=dimension_counts.get)
            if progress_callback:
                await progress_callback(
                    f"Warning: Found embeddings with different dimensions. "
                    f"Using majority dimension: {majority_dimension} "
                    f"({dimension_counts[majority_dimension]} files)"
                )
        else:
            majority_dimension = list(dimension_counts.keys())[0]
        
        # Filter out mismatched embeddings
        mismatched_files = []
        matched_embeddings = {}
        
        for filename, embedding in embeddings.items():
            if file_dimensions[filename] == majority_dimension:
                matched_embeddings[filename] = embedding
            else:
                mismatched_files.append(filename)
                if progress_callback:
                    await progress_callback(
                        f"Removing {filename}: dimension {file_dimensions[filename]} "
                        f"(expected {majority_dimension})"
                    )
        
        if not matched_embeddings:
            raise ValueError("No embeddings with matching dimensions found.")
        
        # Use matched embeddings for RAG
        embeddings = matched_embeddings
        original_dim = majority_dimension
        
        # Check if we need to reduce dimensions
        target_dim = config.reduced_embedding_size
        
        if target_dim and target_dim < original_dim:
            # Reduce embeddings
            if progress_callback:
                await progress_callback("Reducing embedding dimensions...")
            embeddings = embedding_service.reduce_embeddings(embeddings, target_dim)
            self.use_reduced_embeddings = True
            dimension = target_dim
        else:
            dimension = original_dim
            self.use_reduced_embeddings = False
        
        # Create vector database
        if progress_callback:
            await progress_callback("Creating FAISS index...")
        
        self.vector_db = FAISSVectorDB(dimension)
        
        # Prepare vectors and IDs
        filenames = list(embeddings.keys())
        vectors = np.array([embeddings[fn] for fn in filenames])
        
        # Add to database
        self.vector_db.add_vectors(vectors, filenames)
        self.metadata_store = metadata_store
        
        # --- Merge conversation embeddings into the same index ---
        conversation_count = 0
        try:
            from app.services.conversation_compaction_service import get_conversation_compaction_service
            compaction_service = get_conversation_compaction_service()
            if not compaction_service.is_loaded():
                compaction_service.initialize()
                compaction_service.load()
            
            conv_embeddings = compaction_service.get_all_embeddings()
            if conv_embeddings:
                if progress_callback:
                    await progress_callback(f"Adding {len(conv_embeddings)} conversation embedding(s) to RAG index...")
                
                conv_ids = []
                conv_vecs = []
                for cid, emb in conv_embeddings.items():
                    emb_dim = len(emb)
                    if emb_dim == dimension:
                        conv_ids.append(f"conv:{cid}")
                        conv_vecs.append(emb)
                    elif self.use_reduced_embeddings and emb_dim == original_dim:
                        reduced = embedding_service.reduce_single_embedding(emb)
                        conv_ids.append(f"conv:{cid}")
                        conv_vecs.append(reduced)
                    else:
                        if progress_callback:
                            await progress_callback(
                                f"Skipping conversation {cid}: dimension {emb_dim} "
                                f"(expected {dimension})"
                            )
                
                if conv_vecs:
                    conv_array = np.array(conv_vecs)
                    self.vector_db.add_vectors(conv_array, conv_ids)
                    conversation_count = len(conv_ids)
                    if progress_callback:
                        await progress_callback(f"Added {conversation_count} conversation embedding(s) to index")
        except Exception as e:
            if progress_callback:
                await progress_callback(f"Conversation embeddings skipped: {e}")
        
        # Save to file
        rag_dir = config.get_rag_directory()
        if rag_dir:
            rag_dir.mkdir(exist_ok=True)
            index_path = rag_dir / "faiss_index.bin"
            self.vector_db.save(index_path)
            
            if progress_callback:
                await progress_callback(f"RAG database saved to {index_path}")
        
        # Return result with mismatched files info
        return {
            "success": True,
            "mismatched_files": mismatched_files,
            "majority_dimension": majority_dimension,
            "removed_count": len(mismatched_files),
            "total_indexed": len(matched_embeddings) + conversation_count,
            "conversation_count": conversation_count
        }
    
    def load_rag(self, metadata_store: MetadataStore) -> Dict[str, Any]:
        """
        Load RAG database from file.
        
        Args:
            metadata_store: Metadata store
            
        Returns:
            Dictionary containing:
                - success: bool
                - dimension: The dimension of loaded RAG
                - indexed_files: Number of files in the index
        """
        config = get_config()
        rag_dir = config.get_rag_directory()
        
        if not rag_dir or not rag_dir.exists():
            return {"success": False, "error": "RAG directory not found"}
        
        index_path = rag_dir / "faiss_index.bin"
        if not index_path.exists():
            return {"success": False, "error": "RAG index not found"}
        
        try:
            # Determine dimension from config
            target_dim = config.reduced_embedding_size
            embedding_service = get_embedding_service()
            
            # Load embeddings to determine if they were reduced
            if not embedding_service.embeddings:
                embedding_service.load_embeddings()
            
            if target_dim and embedding_service.original_dim:
                self.use_reduced_embeddings = target_dim < embedding_service.original_dim
                dimension = target_dim if self.use_reduced_embeddings else embedding_service.original_dim
            else:
                dimension = embedding_service.original_dim or 768  # Default
                self.use_reduced_embeddings = False
            
            # Create and load vector database
            self.vector_db = FAISSVectorDB(dimension)
            self.vector_db.load(index_path)
            self.metadata_store = metadata_store
            
            return {
                "success": True,
                "dimension": dimension,
                "indexed_files": len(self.vector_db.id_map)
            }
        except Exception as e:
            print(f"Error loading RAG database: {e}")
            return {"success": False, "error": str(e)}
    
    async def search(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RAGResult]:
        """
        Search for relevant files and conversations using RAG.
        
        Args:
            query: Search query
            k: Number of results (None for config default)
            filters: Optional filters (keywords, etc.) — applied to file results only
            
        Returns:
            List of RAGResult (files and/or conversations)
        """
        if not self.vector_db or not self.metadata_store:
            raise ValueError("RAG not loaded. Call load_rag or build_rag first.")
        
        config = get_config()
        k = k or config.top_k
        
        # Generate query embedding using the currently loaded embedding model
        embedding_service = get_embedding_service()
        from app.services.llm_service import get_llm_service
        llm_service = get_llm_service()
        
        query_embedding = await llm_service.embed(query)
        
        # Reduce if necessary
        if self.use_reduced_embeddings:
            query_embedding = embedding_service.reduce_single_embedding(query_embedding)
        
        # Search vector database
        query_vector = np.array(query_embedding)
        ids, distances = self.vector_db.search(query_vector, k * 2)  # Get more for filtering
        
        # Lazy-load compaction service for conversation lookups
        compaction_service = None
        
        # Retrieve metadata and apply filters
        file_results: List[RAGResult] = []
        conv_results: List[RAGResult] = []
        
        for identifier, distance in zip(ids, distances):
            if identifier.startswith("conv:"):
                # --- Conversation result ---
                conv_id = identifier[5:]  # strip "conv:" prefix
                if compaction_service is None:
                    from app.services.conversation_compaction_service import get_conversation_compaction_service
                    compaction_service = get_conversation_compaction_service()
                
                summary = compaction_service.get_summary(conv_id)
                if summary is not None:
                    conv_results.append(RAGResult(
                        source="conversation",
                        identifier=conv_id,
                        summary=summary,
                        compacted_at=compaction_service.get_compacted_at(conv_id),
                    ))
            else:
                # --- File result ---
                metadata = self.metadata_store.get_metadata_by_filename(identifier)
                if metadata:
                    # Apply keyword filters (files only)
                    if filters and "keywords" in filters:
                        keywords = filters["keywords"]
                        text_to_search = " ".join(metadata.tags).lower()
                        if metadata.description:
                            text_to_search += " " + metadata.description.lower()
                        if not any(kw.lower() in text_to_search for kw in keywords):
                            continue
                    
                    file_results.append(RAGResult(
                        source="file",
                        identifier=identifier,
                        file_metadata=metadata,
                    ))
        
        # Apply recency bias only to file results
        if config.recency_bias > 1.0 and file_results:
            file_results = self._apply_recency_bias_rag(file_results, config.recency_bias)
        
        # Merge: conversation results first (maintain FAISS ranking), then file results
        combined = conv_results + file_results
        return combined[:k]
    
    def _apply_recency_bias(
        self,
        results: List[FileMetadata],
        bias_factor: float
    ) -> List[FileMetadata]:
        """
        Apply recency bias to search results.
        
        Args:
            results: Original search results
            bias_factor: Recency bias factor (>1.0 favors recent files)
            
        Returns:
            Re-ranked results
        """
        if not results:
            return results
        
        # Get timestamps
        now = datetime.now()
        scored_results = []
        
        for idx, metadata in enumerate(results):
            creation_time = metadata.get_creation_datetime()
            age_days = (now - creation_time).days
            
            # Calculate recency score (newer = higher score)
            recency_score = 1.0 / (1.0 + age_days / 365.0)  # Normalize by year
            
            # Combine with position score (earlier in results = better)
            position_score = 1.0 / (idx + 1)
            
            # Apply bias
            combined_score = position_score * (recency_score ** bias_factor)
            
            scored_results.append((combined_score, metadata))
        
        # Sort by combined score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return [metadata for _, metadata in scored_results]
    
    def _apply_recency_bias_rag(
        self,
        results: List[RAGResult],
        bias_factor: float
    ) -> List[RAGResult]:
        """Apply recency bias to file-type RAGResults only."""
        if not results:
            return results
        
        now = datetime.now()
        scored = []
        
        for idx, r in enumerate(results):
            if r.source == "file" and r.file_metadata:
                creation_time = r.file_metadata.get_creation_datetime()
                age_days = (now - creation_time).days
                recency_score = 1.0 / (1.0 + age_days / 365.0)
                position_score = 1.0 / (idx + 1)
                combined = position_score * (recency_score ** bias_factor)
            else:
                # Non-file results keep position-based score
                combined = 1.0 / (idx + 1)
            scored.append((combined, r))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored]
    
    def is_loaded(self) -> bool:
        """Check if RAG is loaded."""
        return self.vector_db is not None and self.metadata_store is not None


# Global RAG service instance
_rag_service = RAGService()


def get_rag_service() -> RAGService:
    """Get the global RAG service instance."""
    return _rag_service
