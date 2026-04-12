"""Conversation compaction service — batch-summarizes conversations and stores embeddings."""

import json
from typing import Dict, List, Optional, Set
from pathlib import Path
from datetime import datetime

from app.config import get_config


# ---------------------------------------------------------------------------
# Tag extraction
# ---------------------------------------------------------------------------

def extract_tags_from_summary(summary: str) -> List[str]:
    """Extract keyword tags from a compacted conversation summary using TextBlob noun phrases.

    Uses TextBlob's built-in noun phrase extractor which handles stopwords and
    grammar internally — no hardcoded patterns or regex needed.
    Returns up to 20 unique lowercase tags.
    """
    try:
        from textblob import TextBlob
        blob = TextBlob(summary)
        # dict preserves insertion order and deduplicates
        seen = dict.fromkeys(phrase.lower() for phrase in blob.noun_phrases)
        return list(seen)[:20]
    except Exception:
        return []

# File name within the rag directory
CONVERSATION_EMBEDDINGS_FILE = "conversation_embeddings_map.json"


class ConversationCompactionService:
    """Stores compacted conversation summaries and their embeddings.

    Embeddings are NOT kept in a separate FAISS index.  Instead they are
    persisted to ``conversation_embeddings_map.json`` in the RAG folder and
    merged into the **main** FAISS index during ``/generate-rag``.
    """

    def __init__(self):
        # conversationId -> {"summary": str, "embedding": List[float], "compactedAt": str}
        self._data: Dict[str, dict] = {}
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _rag_dir(self) -> Optional[Path]:
        config = get_config()
        return config.get_rag_directory()

    def _file_path(self) -> Optional[Path]:
        d = self._rag_dir()
        return d / CONVERSATION_EMBEDDINGS_FILE if d else None

    def initialize(self) -> None:
        """Create an empty JSON file in the RAG directory if it does not exist."""
        fp = self._file_path()
        if fp and not fp.exists():
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text("{}")

    def load(self) -> bool:
        """Load conversation embeddings from disk.  Returns True on success."""
        fp = self._file_path()
        if not fp or not fp.exists():
            self._data = {}
            self._loaded = True
            return False
        try:
            with open(fp, "r") as f:
                self._data = json.load(f)
            self._loaded = True
            return True
        except Exception as e:
            print(f"Failed to load conversation embeddings: {e}")
            self._data = {}
            self._loaded = True
            return False

    def save(self) -> None:
        """Persist current data to disk."""
        fp = self._file_path()
        if not fp:
            return
        fp.parent.mkdir(parents=True, exist_ok=True)
        with open(fp, "w") as f:
            json.dump(self._data, f, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        return self._loaded

    def get_compacted_ids(self) -> Set[str]:
        """Return the set of already-compacted conversation IDs."""
        return set(self._data.keys())

    def add_compacted_conversation(
        self, conversation_id: str, summary: str, embedding: List[float],
        tags: List[str] = []
    ) -> None:
        """Store a compacted conversation (summary + embedding + tags).

        If ``tags`` is empty, tags are auto-extracted from the summary via
        ``extract_tags_from_summary`` so the JSON always contains tag data.
        """
        resolved_tags = tags if tags else extract_tags_from_summary(summary)
        self._data[conversation_id] = {
            "summary": summary,
            "embedding": embedding,
            "compactedAt": datetime.now().isoformat(),
            "tags": resolved_tags,
        }

    def get_all_embeddings(self) -> Dict[str, List[float]]:
        """Return ``{conversationId: embedding}`` for all compacted conversations."""
        return {cid: entry["embedding"] for cid, entry in self._data.items() if "embedding" in entry}

    def get_tags(self, conversation_id: str) -> List[str]:
        """Return the stored tags for a conversation, or [] if not found."""
        return self._data.get(conversation_id, {}).get("tags", [])

    def backfill_missing_tags(self) -> int:
        """Backfill tags for any stored conversation that is missing them.

        Iterates all entries, computes tags from the summary where ``tags`` is
        absent or empty, and updates in-place.  Does NOT save — the caller
        must call ``save()`` if the return value is > 0.

        Returns:
            Number of entries that were updated.
        """
        updated = 0
        for entry in self._data.values():
            if not entry.get("tags"):
                tags = extract_tags_from_summary(entry.get("summary", ""))
                entry["tags"] = tags
                updated += 1
        return updated

    def get_summary(self, conversation_id: str) -> Optional[str]:
        """Look up the summary for a given conversation ID."""
        entry = self._data.get(conversation_id)
        return entry["summary"] if entry else None

    def get_compacted_at(self, conversation_id: str) -> Optional[str]:
        """Look up the compaction timestamp for a conversation."""
        entry = self._data.get(conversation_id)
        return entry.get("compactedAt") if entry else None

    def get_all_data(self) -> Dict[str, dict]:
        """Return the full internal data dict (read-only convenience)."""
        return self._data


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_compaction_service = ConversationCompactionService()


def get_conversation_compaction_service() -> ConversationCompactionService:
    """Get the global ConversationCompactionService instance."""
    return _compaction_service
