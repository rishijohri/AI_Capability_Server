"""MCP Filters — Shared filtering functions used by mcp_tools.py and deep_chat_handler.py.

These pure-Python helpers work on FileMetadata (or any duck-typed object with
``tags`` + ``creationTime`` attributes) and ConversationCandidate objects.
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.models.metadata import FileMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ConversationCandidate
# ---------------------------------------------------------------------------

@dataclass
class ConversationCandidate:
    """Represents a compacted conversation as a filterable candidate.

    Has a ``tags`` attribute so it flows through ``_filter_by_tags()`` without
    modification.
    """
    conv_id: str
    summary: str
    tags: List[str]
    compacted_at: str


# ---------------------------------------------------------------------------
# Date filter
# ---------------------------------------------------------------------------

def filter_by_date(
    all_meta: List[FileMetadata],
    start_date: str,
    end_date: str,
) -> List[FileMetadata]:
    """Filter files within date range (inclusive, end+1 day tolerance).

    Args:
        all_meta: List of FileMetadata objects.
        start_date: ISO date string ``YYYY-MM-DD``.
        end_date: ISO date string ``YYYY-MM-DD``.

    Returns:
        Filtered list of FileMetadata. Empty list if dates are invalid.
    """
    try:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date) + timedelta(days=1)
    except ValueError:
        return []

    matches: List[FileMetadata] = []
    for meta in all_meta:
        try:
            ct = meta.creationTime
            if not ct:
                continue
            file_dt = datetime.fromisoformat(ct.replace("Z", "+00:00")).replace(tzinfo=None)
            if start_dt <= file_dt < end_dt:
                matches.append(meta)
        except (ValueError, AttributeError):
            continue
    return matches


# ---------------------------------------------------------------------------
# Tag filter
# ---------------------------------------------------------------------------

def filter_by_tags(
    candidates: List[Any],
    tags: List[str],
    min_matches: int = 1,
    strict: bool = False,
) -> List[Any]:
    """Filter candidates that have at least min_matches matching tags.

    Works via duck-typing — both FileMetadata and ConversationCandidate expose
    a ``tags`` attribute.  Results are sorted descending by match count.

    Args:
        candidates: Iterable of objects with a ``tags`` attribute.
        tags: Required tag substrings (case-insensitive).
        min_matches: Minimum number of tag matches required (default 1).
            Overridden to ``len(tags)`` when ``strict=True``.
        strict: When True, ALL tags must match (AND logic).  When False
            (default), ANY single tag match is sufficient (OR logic) but
            results are still ranked by descending match count.

    Returns:
        Filtered and sorted list.
    """
    if not tags:
        return candidates

    effective_min = len(tags) if strict else min_matches
    needles = [t.lower() for t in tags]
    scored: List[Tuple[int, Any]] = []
    for c in candidates:
        c_tags_lower = [t.lower() for t in c.tags]
        match_count = sum(
            1 for needle in needles
            if any(needle in ct for ct in c_tags_lower)
        )
        if match_count >= effective_min:
            scored.append((match_count, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


# ---------------------------------------------------------------------------
# Tag extraction from candidate set
# ---------------------------------------------------------------------------

def get_tags_from_candidates(candidates: List[Any]) -> List[str]:
    """Return unique tags from a candidate set, sorted descending by frequency.

    Args:
        candidates: Iterable of objects with a ``tags`` attribute.

    Returns:
        Sorted list of unique lowercase tag strings.
    """
    tag_counter: Counter = Counter()
    for c in candidates:
        for tag in c.tags:
            tag_counter[tag.lower()] = tag_counter.get(tag.lower(), 0) + 1
    return [tag for tag, _ in tag_counter.most_common()]


# ---------------------------------------------------------------------------
# Scoped RAG — temporary FAISS index from filtered candidates
# ---------------------------------------------------------------------------

async def scoped_rag_search(
    filtered_candidates: List[Any],
    query: str,
    k: int,
    llm_service,
    embedding_model: str,
    metadata_store,
    compaction_service=None,
) -> List[Any]:
    """Build a temporary FAISS index from filtered candidates and perform a semantic search.

    File embeddings are loaded from disk (already indexed); conversation
    embeddings come from the compaction service.  Only the *query* embedding
    is freshly generated — no re-embedding of candidates occurs.

    Args:
        filtered_candidates: Pre-filtered FileMetadata / ConversationCandidate objects.
        query: Natural language query string.
        k: Number of top results to return.
        llm_service: LLM service instance for generating the query embedding.
        embedding_model: Model filename to use for query embedding.
        metadata_store: Metadata store (unused here, kept for API consistency).
        compaction_service: Optional conversation compaction service.

    Returns:
        Top-k candidates sorted by relevance.  Falls back to first-k if FAISS
        is unavailable or no embeddings are found.
    """
    try:
        import faiss
    except ImportError:
        logger.warning("FAISS not available — returning first-k candidates")
        return filtered_candidates[:k]

    from app.services.embedding_service import get_embedding_service
    embedding_service = get_embedding_service()

    if not embedding_service.embeddings:
        embedding_service.load_embeddings()

    stored_conv_embeddings: Dict[str, Any] = (
        compaction_service.get_all_embeddings() if compaction_service is not None else {}
    )

    candidate_embeddings: List[Tuple[str, List[float], Any]] = []
    for c in filtered_candidates:
        if isinstance(c, ConversationCandidate):
            emb = stored_conv_embeddings.get(c.conv_id)
            if emb is not None:
                candidate_embeddings.append((f"conv:{c.conv_id}", emb, c))
        else:
            emb = embedding_service.get_embedding(c.fileName)
            if emb is not None:
                candidate_embeddings.append((c.fileName, emb, c))

    if not candidate_embeddings:
        logger.warning("No embeddings found for filtered candidates")
        return filtered_candidates[:k]

    vectors = np.array([emb for _, emb, _ in candidate_embeddings], dtype="float32")
    dimension = vectors.shape[1]

    temp_index = faiss.IndexFlatL2(dimension)
    temp_index.add(vectors)

    logger.info(
        f"Scoped FAISS: {len(candidate_embeddings)} vectors (files+convs), dim={dimension}"
    )

    await llm_service.load_model(embedding_model)
    query_embedding = await llm_service.embed(query)
    query_vector = np.array(query_embedding, dtype="float32").reshape(1, -1)

    if embedding_service.pca_model is not None and query_vector.shape[1] != dimension:
        query_vector = np.array(
            embedding_service.reduce_single_embedding(query_embedding),
            dtype="float32",
        ).reshape(1, -1)

    actual_k = min(k, len(candidate_embeddings))
    distances, indices = temp_index.search(query_vector, actual_k)

    results: List[Any] = []
    for idx in indices[0]:
        if 0 <= idx < len(candidate_embeddings):
            results.append(candidate_embeddings[idx][2])

    file_count = sum(1 for r in results if not isinstance(r, ConversationCandidate))
    conv_count = sum(1 for r in results if isinstance(r, ConversationCandidate))
    logger.info(
        f"Scoped RAG: {len(results)} results ({file_count} files, {conv_count} convs)"
    )
    return results
