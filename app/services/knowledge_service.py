"""Knowledge service for storing and retrieving factual conversation content."""

import json
import os
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from app.config import get_config
from app.utils.objectivity_detector import is_objective


# File names within the rag directory
FACTS_FILE = "conversation_facts.jsonl"
FACTS_EMBEDDINGS_FILE = "conversation_facts_embeddings.json"
FACTS_INDEX_FILE = "conversation_facts_index.faiss"
FACTS_INDEX_IDMAP_FILE = "conversation_facts_index_idmap.pkl"


class KnowledgeService:
    """Service for storing objective/factual messages and retrieving them via semantic search."""

    def __init__(self):
        self.facts: Dict[str, dict] = {}  # fact_id -> fact record
        self.embeddings: Dict[str, List[float]] = {}  # fact_id -> embedding vector
        self.faiss_index: Optional[Any] = None
        self.faiss_id_map: List[str] = []  # FAISS index position -> fact_id
        self._next_id: int = 1
        self._pending_facts: List[str] = []  # fact IDs awaiting embedding
        self._objectivity_threshold: float = 0.4

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _rag_dir(self) -> Optional[Path]:
        config = get_config()
        return config.get_rag_directory()

    def _facts_path(self) -> Optional[Path]:
        d = self._rag_dir()
        return d / FACTS_FILE if d else None

    def _embeddings_path(self) -> Optional[Path]:
        d = self._rag_dir()
        return d / FACTS_EMBEDDINGS_FILE if d else None

    def _index_path(self) -> Optional[Path]:
        d = self._rag_dir()
        return d / FACTS_INDEX_FILE if d else None

    # ------------------------------------------------------------------
    # Init / Load / Save
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create empty conversation facts files in the rag directory."""
        rag_dir = self._rag_dir()
        if not rag_dir:
            return
        rag_dir.mkdir(parents=True, exist_ok=True)

        facts_path = self._facts_path()
        if facts_path and not facts_path.exists():
            facts_path.touch()

        emb_path = self._embeddings_path()
        if emb_path and not emb_path.exists():
            emb_path.write_text("{}")

    def load(self) -> bool:
        """Load existing conversation facts and embeddings from disk.

        Returns True if facts were loaded successfully.
        """
        facts_path = self._facts_path()
        if not facts_path or not facts_path.exists():
            return False

        # Load facts
        self.facts = {}
        max_id = 0
        with open(facts_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                fid = record["id"]
                self.facts[fid] = record
                numeric = int(fid.split("_")[1]) if "_" in fid else 0
                if numeric > max_id:
                    max_id = numeric
        self._next_id = max_id + 1

        # Load embeddings
        emb_path = self._embeddings_path()
        if emb_path and emb_path.exists():
            with open(emb_path, "r") as f:
                self.embeddings = json.load(f)

        # Load FAISS index
        self._load_faiss_index()

        return True

    def _load_faiss_index(self) -> None:
        """Load FAISS index from disk if available."""
        if faiss is None:
            return
        idx_path = self._index_path()
        if not idx_path or not idx_path.exists():
            return

        import pickle
        idmap_path = idx_path.parent / FACTS_INDEX_IDMAP_FILE
        if not idmap_path.exists():
            return

        self.faiss_index = faiss.read_index(str(idx_path))
        with open(idmap_path, "rb") as f:
            self.faiss_id_map = pickle.load(f)

    def _save_facts_append(self, record: dict) -> None:
        """Append a single fact record to the JSONL file."""
        facts_path = self._facts_path()
        if not facts_path:
            return
        facts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(facts_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _save_embeddings(self) -> None:
        """Save all embeddings to disk."""
        emb_path = self._embeddings_path()
        if not emb_path:
            return
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        with open(emb_path, "w") as f:
            json.dump(self.embeddings, f)

    def _save_faiss_index(self) -> None:
        """Save FAISS index and id map to disk."""
        if faiss is None or self.faiss_index is None:
            return
        idx_path = self._index_path()
        if not idx_path:
            return

        import pickle
        faiss.write_index(self.faiss_index, str(idx_path))
        idmap_path = idx_path.parent / FACTS_INDEX_IDMAP_FILE
        with open(idmap_path, "wb") as f:
            pickle.dump(self.faiss_id_map, f)

    # ------------------------------------------------------------------
    # Knowledge storage
    # ------------------------------------------------------------------

    def should_store(self, text: str) -> tuple[bool, float]:
        """Check if a message is objective enough to store.

        Returns (should_store, subjectivity_score).
        """
        return is_objective(text, self._objectivity_threshold)

    def add_fact(
        self,
        message: str,
        role: str,
        subjectivity_score: float,
        embedding: Optional[List[float]] = None,
    ) -> Optional[str]:
        """Store a factual message and return its ID, or None if not stored.

        Args:
            message: The text to store.
            role: Message role ("user").
            subjectivity_score: Score from objectivity detector.
            embedding: Pre-computed embedding vector. When provided the fact is
                       indexed immediately, avoiding a costly model reload.
        """
        fact_id = f"fact_{self._next_id}"
        self._next_id += 1

        record = {
            "id": fact_id,
            "message": message,
            "role": role,
            "timestamp": datetime.utcnow().isoformat(),
            "subjectivity_score": round(subjectivity_score, 4),
        }

        self.facts[fact_id] = record
        self._save_facts_append(record)

        if embedding is not None:
            # Store embedding directly and rebuild index
            self.embeddings[fact_id] = embedding
            self._save_embeddings()
            self._rebuild_faiss_index()
        else:
            # Queue for later embedding
            self._pending_facts.append(fact_id)

        return fact_id

    # ------------------------------------------------------------------
    # Embedding & indexing
    # ------------------------------------------------------------------

    async def embed_pending_facts(self) -> int:
        """Generate embeddings for facts that haven't been embedded yet.

        Requires that an embedding model is already loaded via LLMService.
        Returns the number of newly embedded facts.
        """
        if not self._pending_facts:
            return 0

        from app.services.llm_service import get_llm_service
        llm_service = get_llm_service()

        newly_embedded = 0
        for fact_id in list(self._pending_facts):
            record = self.facts.get(fact_id)
            if not record:
                continue
            embedding = await llm_service.embed(record["message"])
            self.embeddings[fact_id] = embedding
            newly_embedded += 1

        self._pending_facts.clear()
        self._save_embeddings()
        self._rebuild_faiss_index()
        return newly_embedded

    def _rebuild_faiss_index(self) -> None:
        """Rebuild the FAISS index from all stored embeddings."""
        if faiss is None or not self.embeddings:
            return

        # Determine dimension from first embedding
        first_emb = next(iter(self.embeddings.values()))
        dimension = len(first_emb)

        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_id_map = []

        ids = sorted(self.embeddings.keys(), key=lambda x: int(x.split("_")[1]) if "_" in x else 0)
        vectors = np.array([self.embeddings[fid] for fid in ids], dtype="float32")

        self.faiss_index.add(vectors)
        self.faiss_id_map = ids

        self._save_faiss_index()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search_relevant_knowledge(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_relevance: float = 0.4,
    ) -> List[dict]:
        """Search for the most semantically relevant facts.

        Args:
            query_embedding: Embedding vector of the current user query.
            top_k: Maximum number of facts to return.
            min_relevance: Minimum cosine-similarity score (facts below are excluded).

        Returns:
            List of fact records sorted by relevance (highest first).
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []

        query_vec = np.array(query_embedding, dtype="float32").reshape(1, -1)
        distances, indices = self.faiss_index.search(query_vec, min(top_k * 2, self.faiss_index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.faiss_id_map):
                continue
            fact_id = self.faiss_id_map[idx]
            record = self.facts.get(fact_id)
            if not record:
                continue

            # Convert L2 distance to a similarity-like score (smaller distance = higher score)
            # Using 1/(1+d) as a simple bounded similarity proxy
            similarity = 1.0 / (1.0 + float(dist))

            if similarity < min_relevance:
                continue

            results.append({**record, "_similarity": round(similarity, 4)})

        # Sort by similarity descending
        results.sort(key=lambda r: r["_similarity"], reverse=True)
        return results[:top_k]

    def select_knowledge(
        self,
        query_embedding: List[float],
        token_budget: int = 2000,
        top_k: int = 10,
        min_relevance: float = 0.4,
    ) -> List[dict]:
        """Select the most relevant facts that fit within a token budget.

        Token estimation: ~4 characters per token (rough English approximation).

        Args:
            query_embedding: Embedding vector of the current query.
            token_budget: Maximum tokens allowed for knowledge context.
            top_k: Max candidates to consider.
            min_relevance: Minimum similarity threshold.

        Returns:
            List of fact records within budget, ordered by relevance.
        """
        candidates = self.search_relevant_knowledge(query_embedding, top_k, min_relevance)

        selected = []
        used_tokens = 0
        for fact in candidates:
            msg = fact["message"]
            estimated_tokens = len(msg) // 4 + 1
            if used_tokens + estimated_tokens > token_budget:
                continue
            used_tokens += estimated_tokens
            selected.append(fact)

        return selected

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return knowledge base statistics."""
        return {
            "total_facts": len(self.facts),
            "total_embeddings": len(self.embeddings),
            "pending_embeddings": len(self._pending_facts),
            "faiss_indexed": self.faiss_index.ntotal if self.faiss_index else 0,
        }

    def is_loaded(self) -> bool:
        """Return True if knowledge data has been loaded."""
        return bool(self.facts) or (self._facts_path() and self._facts_path().exists())


# Global singleton
_knowledge_service = KnowledgeService()


def get_knowledge_service() -> KnowledgeService:
    """Get the global KnowledgeService instance."""
    return _knowledge_service
