"""Deep Chat Prompts — All prompt templates and prompt-builder functions.

Separated from deep_chat_handler.py so logic and prompts are independently
maintainable.  This file contains **only** prompt strings, the constants they
reference, and thin builder functions that assemble them.
"""

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants used exclusively by prompts
# ---------------------------------------------------------------------------
MAX_REFINEMENT_TAGS = 400     # Tags shown to LLM in refinement prompt


# ---------------------------------------------------------------------------
# LLM Call 1: Initial Parameter Extraction
# ---------------------------------------------------------------------------

EXTRACTION_TEMPLATE = """/no_think
Extract search parameters from the user's question about their media library.

LIBRARY TAGS ({tag_count} shown, {total_tags} total): {tags_str}{omitted_note}{conv_block}
LIBRARY DATE RANGE: {min_date} to {max_date}

RESPOND EXACTLY:
FILTER_ORDER:date_first or tags_first
START_DATE:YYYY-MM-DD
END_DATE:YYYY-MM-DD
TAGS:tag1,tag2,tag3
RAG_QUERY:semantic search phrase
PLAN:one_step or two_step

RULES:
- FILTER_ORDER: Choose date_first when question mentions a specific date/time. Choose tags_first when question is about a topic without specific date.
- Dates: YYYY-MM-DD. Same date for both if asking about one day. Use none if no date mentioned.
- TAGS: Pick from LIBRARY TAGS or CONVERSATION KEYWORDS above. Choose 3-10 that relate to the question topic.
- RAG_QUERY: A descriptive search phrase to find relevant files semantically.
- PLAN: one_step for most direct questions. two_step if the question requires a chain — e.g. first find an event to establish a date, then search for related items around that date.
- Use none if not applicable.
- Start response with FILTER_ORDER: immediately."""


def build_extraction_prompt(
    top_tags: List[str],
    total_tags: int,
    date_range: Tuple[str, str],
    conv_tags: Optional[List[str]] = None,
) -> str:
    """Assemble the extraction prompt with real library data."""
    tags_str = ", ".join(top_tags)
    omitted = total_tags - len(top_tags)
    omitted_note = f"\n({omitted} additional less-common tags not shown)" if omitted > 0 else ""

    conv_block = ""
    if conv_tags:
        conv_tags_str = ", ".join(conv_tags[:100])
        conv_block = f"\nCONVERSATION KEYWORDS ({len(conv_tags)} keywords from chat history): {conv_tags_str}"

    return EXTRACTION_TEMPLATE.format(
        tag_count=len(top_tags),
        total_tags=total_tags,
        tags_str=tags_str,
        omitted_note=omitted_note,
        conv_block=conv_block,
        min_date=date_range[0],
        max_date=date_range[1],
    )


# ---------------------------------------------------------------------------
# LLM Call 2: Refinement
# ---------------------------------------------------------------------------

REFINEMENT_TEMPLATE = """/no_think
You filtered the media library and found {file_count} file(s){conv_info}.
{date_info}
Current selected tags: {current_tags_str}

TAGS IN FILTERED CANDIDATES ({shown_count} shown, {total_filtered_tags} total): {tags_str}{omitted_note}

Are you satisfied with the current filter to answer the user's question, or do you want to adjust?

RESPOND EXACTLY:
TAGS:tag1,tag2,tag3
RAG_QUERY:semantic search query for the topic
SATISFIED:yes or no

RULES:
- TAGS: Pick from the TAGS IN FILTERED CANDIDATES. Choose the most relevant ones (3-10).
- RAG_QUERY: A descriptive phrase to semantically search within the filtered files.
- SATISFIED:yes if the file set looks good, no if you want to filter more.
- Start response with TAGS: immediately."""


def build_refinement_prompt(
    file_count: int,
    filtered_tags: List[str],
    total_filtered_tags: int,
    current_params: Dict[str, Any],
    conv_count: int = 0,
) -> str:
    """Assemble the refinement prompt showing tags within the filtered set."""
    shown_count = min(len(filtered_tags), MAX_REFINEMENT_TAGS)
    tags_str = ", ".join(filtered_tags[:MAX_REFINEMENT_TAGS])
    omitted = total_filtered_tags - shown_count
    omitted_note = f"\n({omitted} additional tags not shown)" if omitted > 0 else ""

    current_tags_str = ", ".join(current_params["tags"]) if current_params["tags"] else "none"

    if current_params["start_date"]:
        date_info = f"Date range: {current_params['start_date']} to {current_params['end_date']}"
    else:
        date_info = "Date range: not specified"

    conv_info = f" + {conv_count} conversation(s)" if conv_count > 0 else ""

    return REFINEMENT_TEMPLATE.format(
        file_count=file_count,
        conv_info=conv_info,
        date_info=date_info,
        current_tags_str=current_tags_str,
        shown_count=shown_count,
        total_filtered_tags=total_filtered_tags,
        tags_str=tags_str,
        omitted_note=omitted_note,
    )


# ---------------------------------------------------------------------------
# LLM Call 3: Answer Synthesis (with Short Extraction awareness)
# ---------------------------------------------------------------------------

SYNTHESIS_TEMPLATE = """/no_think
You are Persona. Answer the user's question using ONLY the data below.
Context may include media files and past conversation summaries (labeled 'Conversation').

RULES:
- Start your answer immediately. Do NOT use <think> tags.
- Reference specific file names, dates, tags, descriptions, or conversation facts from the data if helpful.
- If the data IS SUFFICIENT, answer directly and completely.{short_extraction_block}"""

_SHORT_EXTRACTION_AVAILABLE = """
- If the data is INSUFFICIENT to answer confidently, you may request a Short Extraction — a targeted follow-up search with different parameters. You have {remaining} Short Extraction(s) remaining.
  To request one, output EXACTLY this block first (before any explanation):

SHORT_EXTRACTION
START_DATE:YYYY-MM-DD or none
END_DATE:YYYY-MM-DD or none
TAGS:tag1,tag2,tag3 or none
RAG_QUERY:focused phrase for the missing information
INSIGHT:one sentence summarizing what you learned so far from this extraction

  Then on the next line explain what specific information is still missing.
  INSIGHT is important — it carries your findings forward to the next search round."""

_SHORT_EXTRACTION_EXHAUSTED = """
- This is the final answer. Do NOT request further searches. If data is still insufficient, say what you found and what is missing."""


def build_synthesis_prompt(remaining_short_extractions: int = 0) -> str:
    """Assemble the synthesis prompt.

    When ``remaining_short_extractions > 0`` the AI is told it can trigger a
    Short Extraction cycle.  When 0, it must answer with whatever data it has.
    """
    if remaining_short_extractions > 0:
        block = _SHORT_EXTRACTION_AVAILABLE.format(remaining=remaining_short_extractions)
    else:
        block = _SHORT_EXTRACTION_EXHAUSTED

    return SYNTHESIS_TEMPLATE.format(short_extraction_block=block)
