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
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from app.models.responses import WebSocketMessage
from app.models.metadata import FileMetadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_CONTEXT_FILES = 8         # Max files to feed to synthesis LLM
MAX_INITIAL_TAGS = 200        # Tags for Call 1 (must fit in 6500 ctx with prompt + output)
MAX_REFINEMENT_TAGS = 400     # Tags for Call 2 (filtered set is smaller, can show more)
MAX_REFINEMENT_ROUNDS = 2     # Max LLM refinement iterations
MAX_FILE_CONTEXT_CHARS = 400  # Per-file metadata chars in synthesis prompt


# ---------------------------------------------------------------------------
# Helpers: Library context
# ---------------------------------------------------------------------------

def _get_library_tags_and_dates(metadata_store) -> Tuple[List[str], int, Tuple[str, str]]:
    """Get all tags (sorted by frequency) and date range from the library.

    Returns:
        (top_tags, total_unique_tags, (min_date, max_date))
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

    # Date range
    if dates:
        min_date = min(dates).strftime("%Y-%m-%d")
        max_date = max(dates).strftime("%Y-%m-%d")
    else:
        min_date = "unknown"
        max_date = "unknown"

    return top_tags, total_unique, (min_date, max_date)


def _get_tags_from_files(files: List[FileMetadata]) -> List[str]:
    """Extract all unique tags from a set of files, sorted by frequency."""
    tag_counter: Counter = Counter()
    for meta in files:
        for tag in meta.tags:
            tag_counter[tag.lower()] = tag_counter.get(tag.lower(), 0) + 1
    return [tag for tag, _ in tag_counter.most_common()]


# ---------------------------------------------------------------------------
# LLM Call 1: Initial Parameter Extraction
# ---------------------------------------------------------------------------

def _build_extraction_prompt(
    top_tags: List[str],
    total_tags: int,
    date_range: Tuple[str, str],
) -> str:
    """Build extraction prompt with real library data."""
    tags_str = ", ".join(top_tags)
    omitted = total_tags - len(top_tags)
    omitted_note = f"\n({omitted} additional less-common tags not shown)" if omitted > 0 else ""

    return f"""/no_think
Extract search parameters from the user's question about their media library.

LIBRARY TAGS ({len(top_tags)} shown, {total_tags} total): {tags_str}{omitted_note}
LIBRARY DATE RANGE: {date_range[0]} to {date_range[1]}

RESPOND EXACTLY:
FILTER_ORDER:date_first or tags_first
START_DATE:YYYY-MM-DD
END_DATE:YYYY-MM-DD
TAGS:tag1,tag2,tag3
RAG_QUERY:semantic search phrase

RULES:
- FILTER_ORDER: Choose date_first when question mentions a specific date/time. Choose tags_first when question is about a topic without specific date.
- Dates: YYYY-MM-DD. Same date for both if asking about one day. Use none if no date mentioned.
- TAGS: Pick from the LIBRARY TAGS above. Choose 3-10 that relate to the question topic.
- RAG_QUERY: A descriptive search phrase to find relevant files semantically.
- Use none if not applicable.
- Start response with FILTER_ORDER: immediately."""


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
# LLM Call 2: Refinement
# ---------------------------------------------------------------------------

def _build_refinement_prompt(
    file_count: int,
    filtered_tags: List[str],
    total_filtered_tags: int,
    current_params: Dict[str, Any],
) -> str:
    """Build refinement prompt showing tags within the filtered file set."""
    tags_str = ", ".join(filtered_tags[:MAX_REFINEMENT_TAGS])
    omitted = total_filtered_tags - min(len(filtered_tags), MAX_REFINEMENT_TAGS)
    omitted_note = f"\n({omitted} additional tags not shown)" if omitted > 0 else ""

    current_tags_str = ", ".join(current_params["tags"]) if current_params["tags"] else "none"
    date_info = ""
    if current_params["start_date"]:
        date_info = f"Date range: {current_params['start_date']} to {current_params['end_date']}"
    else:
        date_info = "Date range: not specified"

    return f"""/no_think
You filtered the media library and found {file_count} files.
{date_info}
Current selected tags: {current_tags_str}

TAGS IN FILTERED FILES ({min(len(filtered_tags), MAX_REFINEMENT_TAGS)} shown, {total_filtered_tags} total): {tags_str}{omitted_note}

Are you satisfied with the current filter to answer the user's question, or do you want to adjust?

RESPOND EXACTLY:
TAGS:tag1,tag2,tag3
RAG_QUERY:semantic search query for the topic
SATISFIED:yes or no

RULES:
- TAGS: Pick from the TAGS IN FILTERED FILES. Choose the most relevant ones (3-10).
- RAG_QUERY: A descriptive phrase to semantically search within the filtered files.
- SATISFIED:yes if the file set looks good, no if you want to filter more.
- Start response with TAGS: immediately."""


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
    files: List[FileMetadata],
    tags: List[str],
    min_matches: int = 1,
) -> List[FileMetadata]:
    """Filter files with at least min_matches matching tags. Sorted by match count."""
    if not tags:
        return files

    needles = [t.lower() for t in tags]
    scored = []
    for meta in files:
        meta_tags_lower = [t.lower() for t in meta.tags]
        match_count = sum(
            1 for needle in needles
            if any(needle in mt for mt in meta_tags_lower)
        )
        if match_count >= min_matches:
            scored.append((match_count, meta))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [meta for _, meta in scored]


def _merge_candidates(
    date_tag_files: List[FileMetadata],
    date_only_files: List[FileMetadata],
    tag_only_files: List[FileMetadata],
    max_files: int,
) -> List[FileMetadata]:
    """Merge with priority: date+tags > date-only > tags-only."""
    seen: Set[str] = set()
    result: List[FileMetadata] = []

    for source in [date_tag_files, date_only_files, tag_only_files]:
        for meta in source:
            if meta.fileName not in seen and len(result) < max_files:
                seen.add(meta.fileName)
                result.append(meta)
    return result


# ---------------------------------------------------------------------------
# Scoped RAG: Temporary FAISS from filtered files
# ---------------------------------------------------------------------------

async def _scoped_rag_search(
    filtered_files: List[FileMetadata],
    query: str,
    k: int,
    llm_service,
    embedding_model: str,
    metadata_store,
) -> List[FileMetadata]:
    """Build a temporary FAISS index from filtered files and search it.

    Uses existing embeddings from disk — only loads the embedding model
    to generate the query embedding. File embeddings are NOT regenerated.
    """
    try:
        import faiss
    except ImportError:
        logger.warning("FAISS not available, skipping scoped RAG")
        return filtered_files[:k]

    from app.services.embedding_service import get_embedding_service
    embedding_service = get_embedding_service()

    # Ensure embeddings are loaded
    if not embedding_service.embeddings:
        embedding_service.load_embeddings()

    if not embedding_service.embeddings:
        logger.warning("No embeddings available for scoped RAG")
        return filtered_files[:k]

    # Collect embeddings for filtered files only
    file_embeddings: List[Tuple[str, List[float]]] = []
    for meta in filtered_files:
        emb = embedding_service.get_embedding(meta.fileName)
        if emb is not None:
            file_embeddings.append((meta.fileName, emb))

    if not file_embeddings:
        logger.warning("No embeddings found for filtered files")
        return filtered_files[:k]

    # Build temporary FAISS index
    filenames = [fn for fn, _ in file_embeddings]
    vectors = np.array([emb for _, emb in file_embeddings], dtype='float32')
    dimension = vectors.shape[1]

    temp_index = faiss.IndexFlatL2(dimension)
    temp_index.add(vectors)

    logger.info(f"Scoped FAISS: {len(filenames)} vectors, dim={dimension}")

    # Generate query embedding (requires embedding model)
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
    actual_k = min(k, len(filenames))
    distances, indices = temp_index.search(query_vector, actual_k)

    # Map back to FileMetadata
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(filenames):
            fn = filenames[idx]
            meta = metadata_store.get_metadata_by_filename(fn)
            if meta:
                results.append(meta)

    logger.info(f"Scoped RAG returned {len(results)} files: {[r.fileName for r in results]}")
    return results


# ---------------------------------------------------------------------------
# Phase 3: Synthesis
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = """/no_think
You are Persona. Answer the user's question using ONLY the file data below.

RULES:
- Start your answer immediately. Do NOT use <think> tags.
- Reference specific file names, dates, tags, and descriptions from the data if helpful.
- If data is insufficient, say what you found and what's missing."""


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


def _auto_extract_files(text: str, candidates: List[FileMetadata]) -> List[str]:
    """Auto-extract mentioned filenames, fallback to all candidates if none explicitly mentioned."""
    files_list = []
    for c in candidates:
        if c.fileName in text:
            files_list.append(c.fileName)
            
    # If the model didn't explicitly mention the filenames but answered based on them, 
    # just return all the candidate files as the relevant set.
    if not files_list and candidates:
        files_list = [c.fileName for c in candidates]
        
    return files_list


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
    knowledge_service,
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
    # Get library context (tags + dates) — pure Python, fast
    # ------------------------------------------------------------------
    top_tags, total_tags, date_range = _get_library_tags_and_dates(metadata_store)
    all_meta = metadata_store.get_all_metadata()

    logger.info(f"Library: {len(all_meta)} files, {total_tags} unique tags, "
                f"dates {date_range[0]} to {date_range[1]}")

    # ==================================================================
    # LLM CALL 1: Initial parameter extraction
    # ==================================================================
    await websocket.send_json(
        WebSocketMessage(type="thinking", message="Understanding your question...").to_json()
    )

    extraction_prompt = _build_extraction_prompt(top_tags, total_tags, date_range)
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

    # ==================================================================
    # LLM CALL 2: Refinement — show tags from filtered set
    # ==================================================================
    filtered_tags = _get_tags_from_files(working_set)
    total_filtered_tags = len(filtered_tags)

    for refinement_round in range(MAX_REFINEMENT_ROUNDS):
        refinement_prompt = _build_refinement_prompt(
            file_count=len(working_set),
            filtered_tags=filtered_tags,
            total_filtered_tags=total_filtered_tags,
            current_params=params,
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
                    filtered_tags = _get_tags_from_files(working_set)
                    total_filtered_tags = len(filtered_tags)

        except Exception as e:
            logger.error(f"Refinement failed: {e}", exc_info=True)
            break

    # ==================================================================
    # PYTHON: Final filtering + optional RAG refinement
    # ==================================================================
    # The working_set already contains the strict intersection of Date and Tag filters
    # based on the iterative refinement. We do not mix in out-of-bounds files.
    candidates = working_set

    # If we have too many candidates, or we have a specific RAG query, we rank them semantically
    rag_query = params.get("rag_query", user_message)

    if rag_query and len(candidates) > 1:
        await websocket.send_json(
            WebSocketMessage(
                type="status",
                message=f"Semantic ranking of {len(candidates)} filtered files..."
            ).to_json()
        )

        candidates = await _scoped_rag_search(
            filtered_files=candidates,
            query=rag_query,
            k=MAX_CONTEXT_FILES,
            llm_service=llm_service,
            embedding_model=embedding_model,
            metadata_store=metadata_store,
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
    # LLM CALL 3: Answer Synthesis
    # ==================================================================
    if candidates:
        file_context = "\n\n".join(_format_file_for_context(m) for m in candidates)
    else:
        file_context = "No files matched the search criteria."

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
        {"role": "system", "content": SYNTHESIS_PROMPT},
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

    # The model's response continues after our "<conclusion>\n" prefill,
    # so wrap it back for parsing
    full_response = full_response

    # Clean any leaked think/xml tags
    full_response = re.sub(r'<[^>]+>', '', full_response).strip()
    conclusion = full_response

    # If still no conclusion, use fallback
    if not conclusion:
        logger.warning(f"Empty conclusion generated")
        conclusion = "I couldn't find enough data to answer. Try specifying a date or topic more precisely."

    # Dynamically resolve file list from candidates
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
