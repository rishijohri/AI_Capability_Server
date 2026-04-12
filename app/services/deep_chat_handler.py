"""Deep Chat Handler — Deterministic Filter Pipeline with iterative LLM refinement.

Architecture (3 LLM calls max):

    LLM Call 1 — Initial Extraction:
        Input:  user question + top MAX_INITIAL_TAGS library tags + date range
        Output: START_DATE, END_DATE, TAGS (initial picks from real tags)

    Python — Apply initial date filter, extract tags from filtered set

    LLM Call 2 — Refinement:
        Input:  file count + tags within filtered set
        Output: refined TAGS + SATISFIED:yes/no

    Python — Apply refined tag filter, RAG if needed

    LLM Call 3 — Answer Synthesis (streamed):
        Input:  user question + top-K file metadata
        Output: <conclusion> + <files>
"""

import asyncio
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from app.models.responses import WebSocketMessage
from app.models.metadata import FileMetadata
from app.services.conversation_compaction_service import (
    get_conversation_compaction_service,
)
from app.services.deep_chat_prompts import (
    build_extraction_prompt,
    build_refinement_prompt,
    build_synthesis_prompt,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_CONTEXT_FILES = 8         # Max files to feed to synthesis LLM
MAX_INITIAL_TAGS = 200        # Tags for Call 1 (must fit in 6500 ctx with prompt + output)
MAX_REFINEMENT_ROUNDS = 2     # Max LLM refinement iterations
MAX_FILE_CONTEXT_CHARS = 400  # Per-file metadata chars in synthesis prompt


# ---------------------------------------------------------------------------
# ConversationCandidate — duck-typed companion to FileMetadata for tag filtering
# ---------------------------------------------------------------------------

@dataclass
class ConversationCandidate:
    """Represents a compacted conversation as a filterable candidate.

    Has a ``tags`` attribute so it flows through ``_filter_by_tags()`` without
    any modification to that function.
    """
    conv_id: str
    summary: str
    tags: List[str]
    compacted_at: str


# ---------------------------------------------------------------------------
# Helpers: Library context
# ---------------------------------------------------------------------------

def _get_library_tags_and_dates(
    metadata_store,
    compaction_service=None,
) -> Tuple[List[str], int, List[str], Tuple[str, str]]:
    """Get tags/dates from files and (optionally) conversation keywords.

    Returns:
        (top_file_tags, total_unique_file_tags, conv_tags, (min_date, max_date))
    """
    all_meta = metadata_store.get_all_metadata()

    tag_counter: Counter = Counter()
    dates: List[datetime] = []

    for meta in all_meta:
        for tag in meta.tags:
            tag_counter[tag.lower()] = tag_counter.get(tag.lower(), 0) + 1
        try:
            dt = datetime.fromisoformat(meta.creationTime.replace('Z', '+00:00'))
            dates.append(dt)
        except (ValueError, AttributeError):
            pass

    # All unique tags sorted by frequency
    all_tags_sorted = [tag for tag, _ in tag_counter.most_common()]
    total_unique = len(all_tags_sorted)
    top_tags = all_tags_sorted[:MAX_INITIAL_TAGS]

    # Conversation keyword tags (from all compacted conversations)
    conv_tag_counter: Counter = Counter()
    if compaction_service is not None:
        for entry in compaction_service.get_all_data().values():
            for tag in entry.get("tags", []):
                conv_tag_counter[tag.lower()] += 1
    conv_tags = [tag for tag, _ in conv_tag_counter.most_common()]

    # Date range
    if dates:
        min_date = min(dates).strftime("%Y-%m-%d")
        max_date = max(dates).strftime("%Y-%m-%d")
    else:
        min_date = "unknown"
        max_date = "unknown"

    return top_tags, total_unique, conv_tags, (min_date, max_date)


def _get_tags_from_candidates(candidates: List[Any]) -> List[str]:
    """Extract all unique tags from a set of candidates (FileMetadata or ConversationCandidate),
    sorted by frequency.
    """
    tag_counter: Counter = Counter()
    for c in candidates:
        for tag in c.tags:
            tag_counter[tag.lower()] = tag_counter.get(tag.lower(), 0) + 1
    return [tag for tag, _ in tag_counter.most_common()]


# ---------------------------------------------------------------------------
# LLM Call 1 / Call 2: Parsing helpers
# ---------------------------------------------------------------------------


def _parse_line_params(text: str) -> Dict[str, Any]:
    """Parse strict key:value lines from LLM output."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    result = {
        "filter_order": "date_first",
        "start_date": None,
        "end_date": None,
        "tags": [],
        "rag_query": "",
        "satisfied": False,
        "plan": "one_step",
    }

    for line in text.split("\n"):
        line = line.strip()
        key_val = line.split(":", 1) if ":" in line else None
        if not key_val or len(key_val) < 2:
            continue

        key = key_val[0].strip().upper()
        val = key_val[1].strip()

        if key == "FILTER_ORDER":
            if "tags" in val.lower():
                result["filter_order"] = "tags_first"
            else:
                result["filter_order"] = "date_first"
        elif key == "START_DATE":
            if val.lower() != "none" and re.match(r'\d{4}-\d{2}-\d{2}', val):
                result["start_date"] = re.match(r'\d{4}-\d{2}-\d{2}', val).group()
        elif key == "END_DATE":
            if val.lower() != "none" and re.match(r'\d{4}-\d{2}-\d{2}', val):
                result["end_date"] = re.match(r'\d{4}-\d{2}-\d{2}', val).group()
        elif key == "TAGS":
            if val.lower() != "none":
                result["tags"] = [t.strip().lower() for t in val.split(",") if t.strip()]
        elif key == "RAG_QUERY":
            if val.lower() != "none":
                result["rag_query"] = val.strip('"')
        elif key == "SATISFIED":
            result["satisfied"] = val.lower().startswith("yes")
        elif key == "PLAN":
            result["plan"] = "two_step" if "two" in val.lower() else "one_step"

    return result


def _fallback_extraction(question: str) -> Dict[str, Any]:
    """Regex fallback when LLM extraction fails."""
    result = {"filter_order": "date_first", "start_date": None, "end_date": None, "tags": [], "rag_query": question, "satisfied": True}
    q_lower = question.lower()

    month_map = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
    }
    date_patterns = [
        (r'(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})', 'dmy'),
        (r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})', 'mdy'),
        (r'(\d{4})-(\d{2})-(\d{2})', 'iso'),
    ]
    for pattern, fmt in date_patterns:
        m = re.search(pattern, q_lower)
        if m:
            try:
                if fmt == 'dmy':
                    d = datetime(int(m.group(3)), month_map[m.group(2)], int(m.group(1)))
                elif fmt == 'mdy':
                    d = datetime(int(m.group(3)), month_map[m.group(1)], int(m.group(2)))
                else:
                    d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                ds = d.strftime('%Y-%m-%d')
                result["start_date"] = ds
                result["end_date"] = ds
            except ValueError:
                pass
            break

    topic_map = {
        'eat': ['food', 'meal', 'restaurant', 'dinner', 'lunch', 'breakfast'],
        'ate': ['food', 'meal', 'restaurant', 'dinner', 'lunch', 'breakfast'],
        'food': ['food', 'meal', 'restaurant'],
        'trip': ['travel', 'vacation', 'tourism'],
        'travel': ['travel', 'vacation', 'tourism'],
        'birthday': ['birthday', 'celebration', 'party'],
    }
    for keyword, tags in topic_map.items():
        if keyword in q_lower:
            result["tags"].extend(tags)
    result["tags"] = list(set(result["tags"]))
    return result


# ---------------------------------------------------------------------------
# LLM Call 2: Refinement (prompt built via deep_chat_prompts.build_refinement_prompt)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Filtering functions (pure Python)
# ---------------------------------------------------------------------------

def _filter_by_date(
    all_meta: List[FileMetadata],
    start_date: str,
    end_date: str,
) -> List[FileMetadata]:
    """Filter files within date range (inclusive, ±timezone tolerance)."""
    try:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date) + timedelta(days=1)
    except ValueError:
        return []

    matches = []
    for meta in all_meta:
        try:
            ct = meta.creationTime
            if not ct:
                continue
            file_dt = datetime.fromisoformat(ct.replace('Z', '+00:00')).replace(tzinfo=None)
            if start_dt <= file_dt < end_dt:
                matches.append(meta)
        except (ValueError, AttributeError):
            continue
    return matches


def _filter_by_tags(
    candidates: List[Any],
    tags: List[str],
    min_matches: int = 1,
) -> List[Any]:
    """Filter candidates (FileMetadata or ConversationCandidate) with at least
    min_matches matching tags.  Sorted descending by match count.
    Works via duck typing — both types expose a ``tags`` attribute.
    """
    if not tags:
        return candidates

    needles = [t.lower() for t in tags]
    scored = []
    for c in candidates:
        c_tags_lower = [t.lower() for t in c.tags]
        match_count = sum(
            1 for needle in needles
            if any(needle in ct for ct in c_tags_lower)
        )
        if match_count >= min_matches:
            scored.append((match_count, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]

# ---------------------------------------------------------------------------
# Scoped RAG: Temporary FAISS from filtered files
# ---------------------------------------------------------------------------

async def _scoped_rag_search(
    filtered_candidates: List[Any],
    query: str,
    k: int,
    llm_service,
    embedding_model: str,
    metadata_store,
    compaction_service=None,
) -> List[Any]:
    """Build a temporary FAISS index from filtered candidates (files + conversations) and search it.

    File embeddings are loaded from the embedding service (already on disk).
    Conversation embeddings are loaded from the compaction service (already stored).
    Only the query embedding is generated, so no re-embedding of candidates occurs.
    """
    try:
        import faiss
    except ImportError:
        logger.warning("FAISS not available, skipping scoped RAG")
        return filtered_candidates[:k]

    from app.services.embedding_service import get_embedding_service
    embedding_service = get_embedding_service()

    # Ensure file embeddings are loaded
    if not embedding_service.embeddings:
        embedding_service.load_embeddings()

    # Collect embeddings for all candidates
    # identifier -> (embedding, candidate)
    candidate_embeddings: List[Tuple[str, List[float], Any]] = []

    stored_conv_embeddings = (
        compaction_service.get_all_embeddings() if compaction_service is not None else {}
    )

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

    # Build temporary FAISS index
    identifiers = [ident for ident, _, _ in candidate_embeddings]
    vectors = np.array([emb for _, emb, _ in candidate_embeddings], dtype='float32')
    dimension = vectors.shape[1]

    temp_index = faiss.IndexFlatL2(dimension)
    temp_index.add(vectors)

    logger.info(f"Scoped FAISS: {len(identifiers)} vectors (files+convs), dim={dimension}")

    # Generate query embedding
    await llm_service.load_model(embedding_model)
    query_embedding = await llm_service.embed(query)
    query_vector = np.array(query_embedding, dtype='float32').reshape(1, -1)

    # Reduce dimensions if the RAG was built with reduced embeddings
    if embedding_service.pca_model is not None and query_vector.shape[1] != dimension:
        query_vector = np.array(
            embedding_service.reduce_single_embedding(query_embedding),
            dtype='float32'
        ).reshape(1, -1)

    # Search
    actual_k = min(k, len(identifiers))
    distances, indices = temp_index.search(query_vector, actual_k)

    # Map back to original candidates
    results: List[Any] = []
    for idx in indices[0]:
        if 0 <= idx < len(candidate_embeddings):
            results.append(candidate_embeddings[idx][2])

    file_count = sum(1 for r in results if isinstance(r, FileMetadata))
    conv_count = sum(1 for r in results if isinstance(r, ConversationCandidate))
    logger.info(f"Scoped RAG returned {len(results)} candidates: {file_count} files, {conv_count} conversations")
    return results


# ---------------------------------------------------------------------------
# Phase 3: Synthesis (prompts live in deep_chat_prompts.py)
# ---------------------------------------------------------------------------


def _format_file_for_context(meta: FileMetadata) -> str:
    """Format a single file's metadata for the synthesis prompt."""
    parts = [f"• {meta.fileName}"]
    if meta.creationTime:
        parts.append(f"  Date: {meta.creationTime[:10]}")
    if meta.tags:
        parts.append(f"  Tags: {', '.join(meta.tags[:12])}")
    if meta.description:
        desc = meta.description[:200] + "..." if len(meta.description) > 200 else meta.description
        parts.append(f"  Desc: {desc}")
    if meta.type:
        parts.append(f"  Type: {meta.type}")
    text = "\n".join(parts)
    return text[:MAX_FILE_CONTEXT_CHARS]

def _format_conversation_for_context(c: ConversationCandidate) -> str:
    """Format a compacted conversation candidate for the synthesis prompt."""
    parts = [f"\u2022 [Conversation {c.conv_id}]"]
    if c.compacted_at:
        parts.append(f"  Compacted: {c.compacted_at[:10]}")
    if c.tags:
        parts.append(f"  Keywords: {', '.join(c.tags[:12])}")
    if c.summary:
        summary_preview = c.summary[:300] + "..." if len(c.summary) > 300 else c.summary
        parts.append(f"  Facts: {summary_preview}")
    text = "\n".join(parts)
    return text[:MAX_FILE_CONTEXT_CHARS]


def _format_candidate_for_context(candidate: Any) -> str:
    """Dispatch to the appropriate formatter based on candidate type."""
    if isinstance(candidate, ConversationCandidate):
        return _format_conversation_for_context(candidate)
    return _format_file_for_context(candidate)

def _auto_extract_files(text: str, candidates: List[Any]) -> List[str]:
    """Auto-extract mentioned identifiers, fallback to all candidates if none explicitly mentioned."""
    ids_list = []
    for c in candidates:
        identifier = c.conv_id if isinstance(c, ConversationCandidate) else c.fileName
        if identifier in text:
            ids_list.append(identifier)

    # If the model didn't explicitly mention identifiers but answered based on context,
    # return all candidate identifiers as the relevant set.
    if not ids_list and candidates:
        ids_list = [
            c.conv_id if isinstance(c, ConversationCandidate) else c.fileName
            for c in candidates
        ]

    return ids_list


# ---------------------------------------------------------------------------
# Short Extraction request parsing
# ---------------------------------------------------------------------------

def _parse_short_extraction_request(text: str) -> Optional[Dict[str, Any]]:
    """Detect SHORT_EXTRACTION in synthesis output and extract the follow-up search params.

    Returns a dict with keys (start_date, end_date, tags, rag_query) if the
    keyword is present, otherwise None.
    """
    if "SHORT_EXTRACTION" not in text:
        return None

    result: Dict[str, Any] = {
        "start_date": None,
        "end_date": None,
        "tags": [],
        "rag_query": "",
        "insight": "",
    }

    for line in text.split("\n"):
        line = line.strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().upper()
        val = val.strip()

        if key == "START_DATE":
            if val.lower() != "none" and re.match(r'\d{4}-\d{2}-\d{2}', val):
                result["start_date"] = re.match(r'\d{4}-\d{2}-\d{2}', val).group()
        elif key == "END_DATE":
            if val.lower() != "none" and re.match(r'\d{4}-\d{2}-\d{2}', val):
                result["end_date"] = re.match(r'\d{4}-\d{2}-\d{2}', val).group()
        elif key == "TAGS":
            if val.lower() != "none":
                result["tags"] = [t.strip().lower() for t in val.split(",") if t.strip()]
        elif key == "RAG_QUERY":
            if val.lower() != "none":
                result["rag_query"] = val.strip('"')
        elif key == "INSIGHT":
            if val.lower() != "none":
                result["insight"] = val.strip('"')

    return result

def _strip_search_directives(text: str) -> str:
    """Remove SHORT_EXTRACTION directive lines from synthesis output,
    preserving the human-readable explanation text.
    """
    directive_prefixes = (
        "SHORT_EXTRACTION",
        "START_DATE:",
        "END_DATE:",
        "TAGS:",
        "RAG_QUERY:",
        "INSIGHT:",
    )
    clean = []
    for line in text.split("\n"):
        stripped = line.strip()
        if any(stripped.upper().startswith(p.upper()) for p in directive_prefixes):
            continue
        clean.append(line)
    return "\n".join(clean).strip()


# ---------------------------------------------------------------------------
# LLM helper: call and parse
# ---------------------------------------------------------------------------

async def _llm_call(llm_service, messages: List[Dict[str, Any]]) -> str:
    """Single non-streaming LLM call, returns full text."""
    raw = ""
    async for chunk in llm_service.generate(messages, stream=False):
        raw += chunk
    return raw


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

async def run_deep_chat(
    websocket,
    user_message: str,
    active_history: List[Dict[str, Any]],
    config,
    metadata_store,
    rag_service,
    llm_service,
    face_service,
    rag_available: bool,
    embedding_loaded: bool,
    use_vision: bool,
    image_base64: Optional[str],
    image_name: Optional[str],
    image_tags: List[str],
    image_description: Optional[str],
    chat_model: str,
    vision_model: Optional[str],
    embedding_model: str,
    mmproj_file: Optional[str],
):
    """Deep Chat — Iterative Filter Pipeline.

    LLM Call 1: Extract dates + initial tags (sees top 1000 library tags)
    Python:     Filter by dates → show LLM tags within filtered set
    LLM Call 2: Refine tags from filtered set, decide if satisfied
    Python:     Apply refined filters, RAG if needed
    LLM Call 3: Answer from top-K file metadata (streamed)
    """
    await websocket.send_json(
        WebSocketMessage(type="status", message="Deep Chat: Iterative Filter Pipeline").to_json()
    )

    # ------------------------------------------------------------------
    # Inject attached image context into the user's message if present
    # ------------------------------------------------------------------
    if image_name and (image_tags or image_description):
        image_context = "[Attached image context: "
        if image_description:
            image_context += f"Description '{image_description}'. "
        if image_tags:
            image_context += f"Tags '{', '.join(image_tags)}'"
        image_context += "]\n\n"
        user_message = image_context + user_message

    # ------------------------------------------------------------------
    # Get library context (tags + dates) and conversation candidates
    # ------------------------------------------------------------------
    compaction_service = get_conversation_compaction_service()
    if not compaction_service.is_loaded():
        compaction_service.load()

    top_tags, total_tags, conv_tags, date_range = _get_library_tags_and_dates(
        metadata_store, compaction_service
    )
    all_meta = metadata_store.get_all_metadata()

    # Build conversation candidates from all compacted conversations
    conversation_candidates: List[ConversationCandidate] = [
        ConversationCandidate(
            conv_id=cid,
            summary=d.get("summary", ""),
            tags=d.get("tags", []),
            compacted_at=d.get("compactedAt", ""),
        )
        for cid, d in compaction_service.get_all_data().items()
        if d.get("embedding")  # only include conversations that have been embedded
    ]

    logger.info(
        f"Library: {len(all_meta)} files, {total_tags} unique tags, "
        f"dates {date_range[0]} to {date_range[1]}, "
        f"{len(conversation_candidates)} compacted conversation(s)"
    )

    # ==================================================================
    # LLM CALL 1: Initial parameter extraction
    # ==================================================================
    await websocket.send_json(
        WebSocketMessage(type="thinking", message="Understanding your question...").to_json()
    )

    extraction_prompt = build_extraction_prompt(top_tags, total_tags, date_range, conv_tags)
    messages = [
        {"role": "system", "content": extraction_prompt},
        {"role": "user", "content": user_message},
    ]

    try:
        raw = await _llm_call(llm_service, messages)
        logger.info(f"Extraction raw ({len(raw)} chars): {raw[:500]}")
        params = _parse_line_params(raw)

        has_dates = params["start_date"] is not None
        has_tags = len(params["tags"]) > 0

        if not has_dates and not has_tags:
            logger.warning("LLM extraction empty — using fallback")
            params = _fallback_extraction(user_message)
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}", exc_info=True)
        params = _fallback_extraction(user_message)

    has_dates = params["start_date"] is not None
    has_tags = len(params["tags"]) > 0

    status_parts = []
    if has_dates:
        status_parts.append(f"dates: {params['start_date']} to {params['end_date']}")
    if has_tags:
        status_parts.append(f"tags: {', '.join(params['tags'][:8])}")

    await websocket.send_json(
        WebSocketMessage(
            type="status",
            message=f"Extracted: {'; '.join(status_parts) if status_parts else 'general search'}"
        ).to_json()
    )

    # ==================================================================
    # PYTHON: Initial date filter
    # ==================================================================
    date_filtered = []
    if has_dates:
        date_filtered = _filter_by_date(all_meta, params["start_date"], params["end_date"])
        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message=f"Date filter: {len(date_filtered)} files found"
            ).to_json()
        )

        # Progressive relaxation: expand ±2 days if nothing found
        if not date_filtered:
            try:
                start_dt = datetime.fromisoformat(params["start_date"])
                exp_start = (start_dt - timedelta(days=2)).strftime("%Y-%m-%d")
                exp_end = (start_dt + timedelta(days=2)).strftime("%Y-%m-%d")
                date_filtered = _filter_by_date(all_meta, exp_start, exp_end)
                if date_filtered:
                    await websocket.send_json(
                        WebSocketMessage(
                            type="status",
                            message=f"Expanded ±2 days: {len(date_filtered)} files"
                        ).to_json()
                    )
            except ValueError:
                pass

    # Working set: date-filtered if we have dates, otherwise full library
    working_set = date_filtered if date_filtered else all_meta

    # Conversation candidates participate in tag filtering only (no date filter)
    conv_working_set: List[ConversationCandidate] = list(conversation_candidates)

    # ==================================================================
    # LLM CALL 2: Refinement — show tags from combined filtered set
    # ==================================================================
    filtered_tags = _get_tags_from_candidates(working_set + conv_working_set)
    total_filtered_tags = len(filtered_tags)

    for refinement_round in range(MAX_REFINEMENT_ROUNDS):
        refinement_prompt = build_refinement_prompt(
            file_count=len(working_set),
            filtered_tags=filtered_tags,
            total_filtered_tags=total_filtered_tags,
            current_params=params,
            conv_count=len(conv_working_set),
        )

        ref_messages = [
            {"role": "system", "content": refinement_prompt},
            {"role": "user", "content": user_message},
        ]

        await websocket.send_json(
            WebSocketMessage(
                type="thinking",
                message=f"Refining search (round {refinement_round + 1})..."
            ).to_json()
        )

        try:
            ref_raw = await _llm_call(llm_service, ref_messages)
            logger.info(f"Refinement raw ({len(ref_raw)} chars): {ref_raw[:400]}")
            ref_params = _parse_line_params(ref_raw)

            # Update tags if LLM provided new ones
            if ref_params["tags"]:
                params["tags"] = ref_params["tags"]
            if ref_params["rag_query"]:
                params["rag_query"] = ref_params["rag_query"]

            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"Refined tags: {', '.join(params['tags'][:8])}"
                    + (f" | RAG: {params['rag_query'][:50]}" if params.get('rag_query') else "")
                ).to_json()
            )

            # If satisfied, stop refining
            if ref_params["satisfied"]:
                logger.info(f"LLM satisfied after round {refinement_round + 1}")
                break

            # Apply refined tags and re-filter for next round
            if ref_params["tags"]:
                tag_filtered = _filter_by_tags(working_set, ref_params["tags"])
                if tag_filtered:
                    working_set = tag_filtered

                # Apply same tags to conversation candidates (parallel lane)
                conv_tag_filtered = _filter_by_tags(conv_working_set, ref_params["tags"])
                if conv_tag_filtered:
                    conv_working_set = conv_tag_filtered

                filtered_tags = _get_tags_from_candidates(working_set + conv_working_set)
                total_filtered_tags = len(filtered_tags)

        except Exception as e:
            logger.error(f"Refinement failed: {e}", exc_info=True)
            break

    # ==================================================================
    # PYTHON: Final filtering + optional RAG refinement
    # ==================================================================
    # The working_set already contains the strict intersection of Date and Tag filters.
    # conv_working_set contains conversations that survived tag filtering.
    # Merge both lanes before semantic ranking.
    candidates: List[Any] = list(working_set) + list(conv_working_set)

    # If we have too many candidates, or we have a specific RAG query, we rank them semantically
    rag_query = params.get("rag_query", user_message)

    if rag_query and len(candidates) > 1:
        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message=f"Semantic ranking of {len(working_set)} file(s) + {len(conv_working_set)} conversation(s)..."
            ).to_json()
        )

        candidates = await _scoped_rag_search(
            filtered_candidates=candidates,
            query=rag_query,
            k=MAX_CONTEXT_FILES,
            llm_service=llm_service,
            embedding_model=embedding_model,
            metadata_store=metadata_store,
            compaction_service=compaction_service,
        )
        # Restore chat model after embedding model was used
        await llm_service.load_model(chat_model)

    # Final cap
    candidates = candidates[:MAX_CONTEXT_FILES]

    await websocket.send_json(
        WebSocketMessage(
            type="status",
            message=f"Final: {len(candidates)} files selected for answer"
        ).to_json()
    )

    # ==================================================================
    # LLM CALL 3+: Answer Synthesis with Short Extraction loop
    # ==================================================================
    max_short_extractions = getattr(config, 'chat_rounds', 2)
    remaining_extractions = max_short_extractions

    if candidates:
        file_context = "\n\n".join(_format_candidate_for_context(m) for m in candidates)
    else:
        file_context = "No files or conversations matched the search criteria."

    # Ensure appropriate model is loaded for synthesis
    if image_name and vision_model and mmproj_file:
        # Load vision model with mmproj for image-aware synthesis
        await llm_service.load_model(vision_model, mmproj=mmproj_file)
    else:
        await llm_service.load_model(chat_model)

    # Build messages for synthesis. If an image was attached, pass the
    # raw base64 image along with the RAG file context.
    if image_name and image_base64:
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            {"type": "text", "text": f"{user_message}\n\n=== FILE DATA ===\n{file_context}\n=== END ==="}
        ]
    else:
        user_content = f"{user_message}\n\n=== FILE DATA ===\n{file_context}\n=== END ==="

    synthesis_messages = [
        {"role": "system", "content": build_synthesis_prompt(remaining_extractions)},
        {"role": "user", "content": user_content},
    ]

    logger.info(f"Synthesis: {len(candidates)} files, ~{len(file_context)} chars of context")

    await websocket.send_json(
        WebSocketMessage(
            type="thinking",
            message=f"Composing answer from {len(candidates)} files..."
        ).to_json()
    )

    # Use non-streaming to guarantee we get full response before ws timeout
    full_response = ""
    try:
        async for chunk in llm_service.generate(synthesis_messages, stream=False):
            full_response += chunk
    except Exception as e:
        logger.error(f"Synthesis failed: {e}", exc_info=True)

    logger.info(f"Synthesis raw ({len(full_response)} chars): {full_response[:300]}")

    # ==================================================================
    # Short Extraction loop
    # ==================================================================
    collected_insights: List[str] = []
    for extraction_round in range(max_short_extractions):
        se_request = _parse_short_extraction_request(full_response)
        if not se_request:
            break  # AI is satisfied — no further extraction needed

        remaining_extractions -= 1
        logger.info(f"Short Extraction {extraction_round + 1}/{max_short_extractions}: {se_request}")
        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message=f"Short Extraction {extraction_round + 1}/{max_short_extractions}: searching for additional context..."
            ).to_json()
        )

        # Build candidate pool with AI-requested params
        # Conversations are exempt from date filtering — only files are date-filtered
        se_all_meta: List[FileMetadata] = list(all_meta)
        if se_request["start_date"] and se_request["end_date"]:
            se_date_filtered = _filter_by_date(
                all_meta, se_request["start_date"], se_request["end_date"]
            )
            if se_date_filtered:
                se_all_meta = se_date_filtered
                await websocket.send_json(
                    WebSocketMessage(
                        type="status",
                        message=f"Short Extraction date filter: {len(se_all_meta)} file(s) "
                                f"({se_request['start_date']} to {se_request['end_date']})"
                    ).to_json()
                )

        se_file_candidates: List[Any] = list(se_all_meta)
        # Conversation candidates always participate without date filtering
        se_conv_candidates: List[ConversationCandidate] = list(conversation_candidates)

        if se_request["tags"]:
            tag_filtered_files = _filter_by_tags(se_file_candidates, se_request["tags"])
            if tag_filtered_files:
                se_file_candidates = tag_filtered_files
            tag_filtered_convs = _filter_by_tags(se_conv_candidates, se_request["tags"])
            if tag_filtered_convs:
                se_conv_candidates = tag_filtered_convs

        se_candidates: List[Any] = se_file_candidates + se_conv_candidates
        se_query = se_request["rag_query"] or user_message

        if se_candidates:
            await websocket.send_json(
                WebSocketMessage(
                    type="status",
                    message=f"Semantic ranking of {len(se_file_candidates)} file(s) + "
                            f"{len(se_conv_candidates)} conversation(s)..."
                ).to_json()
            )
            se_candidates = await _scoped_rag_search(
                filtered_candidates=se_candidates,
                query=se_query,
                k=MAX_CONTEXT_FILES,
                llm_service=llm_service,
                embedding_model=embedding_model,
                metadata_store=metadata_store,
                compaction_service=compaction_service,
            )
            await llm_service.load_model(chat_model)

        # Merge new results with existing candidates (dedup, cap)
        seen_ids: Set[str] = {
            (c.conv_id if isinstance(c, ConversationCandidate) else c.fileName)
            for c in candidates
        }
        for c in se_candidates:
            cid = c.conv_id if isinstance(c, ConversationCandidate) else c.fileName
            if cid not in seen_ids:
                candidates.append(c)
                seen_ids.add(cid)
        candidates = candidates[:MAX_CONTEXT_FILES]

        # Re-synthesize with expanded data, carrying forward the insight
        await websocket.send_json(
            WebSocketMessage(
                type="thinking",
                message=f"Re-analyzing with {len(candidates)} candidate(s) "
                        f"(extraction {extraction_round + 1}/{max_short_extractions})..."
            ).to_json()
        )

        file_context = "\n\n".join(_format_candidate_for_context(m) for m in candidates)

        # Accumulate insights across rounds
        insight_text = se_request.get("insight", "")
        if insight_text:
            collected_insights.append(f"Round {extraction_round + 1}: {insight_text}")
        insight_block = ""
        if collected_insights:
            insight_block = "\n\n=== INSIGHTS FROM PREVIOUS ROUNDS ===\n" + "\n".join(collected_insights) + "\n=== END INSIGHTS ==="

        base_text = f"{user_message}{insight_block}\n\n=== FILE DATA ===\n{file_context}\n=== END ==="
        if image_name and image_base64:
            user_content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text", "text": base_text}
            ]
        else:
            user_content = base_text

        synthesis_messages = [
            {"role": "system", "content": build_synthesis_prompt(remaining_extractions)},
            {"role": "user", "content": user_content},
        ]

        full_response = ""
        try:
            async for chunk in llm_service.generate(synthesis_messages, stream=False):
                full_response += chunk
        except Exception as e:
            logger.error(f"Short Extraction synthesis {extraction_round + 1} failed: {e}", exc_info=True)
            break

        logger.info(
            f"Short Extraction {extraction_round + 1} synthesis raw "
            f"({len(full_response)} chars): {full_response[:300]}"
        )

    # Strip any leftover directives the AI may have emitted on the final pass
    full_response = _strip_search_directives(full_response)

    # Clean any leaked think/xml tags
    full_response = re.sub(r'<[^>]+>', '', full_response).strip()
    conclusion = full_response

    # If still no conclusion, use fallback
    if not conclusion:
        logger.warning(f"Empty conclusion generated")
        conclusion = "I couldn't find enough data to answer. Try specifying a date or topic more precisely."

    # Dynamically resolve identifier list from candidates
    files_list = _auto_extract_files(conclusion, candidates)
    
    # Always include the actively attached image if it was used in inference
    if image_name and image_name not in files_list:
        files_list.append(image_name)

    # Stream the conclusion to client in chunks for a natural feel
    chunk_size = 80
    for i in range(0, len(conclusion), chunk_size):
        await websocket.send_json(
            WebSocketMessage(type="progress", message=conclusion[i:i+chunk_size]).to_json()
        )
        await asyncio.sleep(0.02)

    # Send final results
    await websocket.send_json(
        WebSocketMessage(type="conclusion", message=conclusion).to_json()
    )

    if files_list:
        await websocket.send_json(
            WebSocketMessage(
                type="files",
                message=", ".join(files_list),
                data={"files": files_list}
            ).to_json()
        )

    await websocket.send_json(
        WebSocketMessage(
            type="full_response",
            message=conclusion,
            data={"tools_called": 3, "files": files_list}
        ).to_json()
    )
