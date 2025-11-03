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
    ) -> None:
        """
        Build RAG database from embeddings.
        
        Args:
            metadata_store: Metadata store
            progress_callback: Optional callback for progress updates
        """
        config = get_config()
        embedding_service = get_embedding_service()
        
        # Get embeddings
        embeddings = embedding_service.get_all_embeddings()
        if not embeddings:
            raise ValueError("No embeddings available. Generate embeddings first.")
        
        # Check if we need to reduce dimensions
        original_dim = len(next(iter(embeddings.values())))
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
        
        # Save to file
        rag_dir = config.get_rag_directory()
        if rag_dir:
            rag_dir.mkdir(exist_ok=True)
            index_path = rag_dir / "faiss_index.bin"
            self.vector_db.save(index_path)
            
            if progress_callback:
                await progress_callback(f"RAG database saved to {index_path}")
    
    def load_rag(self, metadata_store: MetadataStore) -> bool:
        """
        Load RAG database from file.
        
        Args:
            metadata_store: Metadata store
            
        Returns:
            True if loaded successfully
        """
        config = get_config()
        rag_dir = config.get_rag_directory()
        
        if not rag_dir or not rag_dir.exists():
            return False
        
        index_path = rag_dir / "faiss_index.bin"
        if not index_path.exists():
            return False
        
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
            
            return True
        except Exception as e:
            print(f"Error loading RAG database: {e}")
            return False
    
    async def search(
        self,
        query: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[FileMetadata]:
        """
        Search for relevant files using RAG.
        
        Args:
            query: Search query
            k: Number of results (None for config default)
            filters: Optional filters (keywords, etc.)
            
        Returns:
            List of relevant file metadata
        """
        if not self.vector_db or not self.metadata_store:
            raise ValueError("RAG not loaded. Call load_rag or build_rag first.")
        
        config = get_config()
        k = k or config.top_k
        
        # Generate query embedding using the currently loaded embedding model
        # The embedding model should already be loaded by the caller
        embedding_service = get_embedding_service()
        
        # Use llm_service to generate embedding
        from app.services.llm_service import get_llm_service
        llm_service = get_llm_service()
        
        # Generate embedding for query using currently loaded model
        query_embedding = await llm_service.embed(query)
        
        # Reduce if necessary
        if self.use_reduced_embeddings:
            query_embedding = embedding_service.reduce_single_embedding(query_embedding)
        
        # Search vector database
        query_vector = np.array(query_embedding)
        filenames, distances = self.vector_db.search(query_vector, k * 2)  # Get more for filtering
        
        # Retrieve metadata and apply filters
        results = []
        for filename, distance in zip(filenames, distances):
            metadata = self.metadata_store.get_metadata_by_filename(filename)
            if metadata:
                # Apply keyword filters if specified
                if filters and "keywords" in filters:
                    keywords = filters["keywords"]
                    # Check if any keyword is in tags or description
                    text_to_search = " ".join(metadata.tags).lower()
                    if metadata.description:
                        text_to_search += " " + metadata.description.lower()
                    
                    if not any(kw.lower() in text_to_search for kw in keywords):
                        continue
                
                results.append(metadata)
                
                if len(results) >= k:
                    break
        
        # Apply recency bias
        if config.recency_bias > 1.0:
            results = self._apply_recency_bias(results, config.recency_bias)
        
        return results[:k]
    
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
    
    def is_loaded(self) -> bool:
        """Check if RAG is loaded."""
        return self.vector_db is not None and self.metadata_store is not None


# Global RAG service instance
_rag_service = RAGService()


def get_rag_service() -> RAGService:
    """Get the global RAG service instance."""
    return _rag_service
