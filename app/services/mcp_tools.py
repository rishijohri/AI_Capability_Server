"""MCP Tool Registry — 4 focused tools for deep-chat OpenAI tool-calling loop.

Tools:
    get_scoped_tags      — List top-M tags from a date/tag-scoped subset of the library.
    scoped_rag_search    — Semantic search within a date/tag-scoped subset.
    get_scoped_dates     — List contiguous date ranges that have matching files.
    get_conversation_rag — Semantic search across compacted conversation summaries.

Each tool result is a plain string.  The handler appends a budget annotation
"[Tool calls remaining: N]" before returning the result to the LLM.
"""

import asyncio
import logging
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from app.models.metadata import FileMetadata
from app.services.mcp_filters import (
    ConversationCandidate,
    filter_by_date,
    filter_by_tags,
    get_tags_from_candidates,
    scoped_rag_search,
)

logger = logging.getLogger(__name__)

# Maximum characters in a single tool result
MAX_RESULT_CHARS = 3000



def _truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (truncated)"


# ---------------------------------------------------------------------------
# MCPToolRegistry
# ---------------------------------------------------------------------------

class MCPToolRegistry:
    """Provides OpenAI-format tool definitions and executes MCP tool calls."""

    def __init__(
        self,
        metadata_store,
        llm_service=None,
        embedding_model: str = "",
        chat_model: str = "",
        compaction_service=None,
        config=None,
    ):
        self.metadata_store = metadata_store
        self.llm_service = llm_service
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.compaction_service = compaction_service
        self.config = config

        # ---------------------------------------------------------------------------
        # Tool registry — single source of truth for both schemas and handlers.
        # To add a new MCP tool: implement a handler method below, then append one
        # entry here with its OpenAI-format schema and a reference to the handler.
        # Both /api/mcp and /api/deep-chat pick it up automatically.
        # ---------------------------------------------------------------------------
        self._registry: List[Dict[str, Any]] = [
            {
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "get_scoped_tags",
                        "description": (
                            "List the most common tags from files that match an optional date range "
                            "and/or tag filter. Use this first to understand what the library contains "
                            "within a scope before running a semantic search."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "start_date": {
                                    "type": "string",
                                    "description": "Inclusive start date (YYYY-MM-DD). Omit for no lower bound.",
                                },
                                "end_date": {
                                    "type": "string",
                                    "description": "Inclusive end date (YYYY-MM-DD). Omit for no upper bound.",
                                },
                                "min_tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Optional list of tags to pre-filter. Only files that match at least one of these tags are counted.",
                                },
                                "strict": {
                                    "type": "boolean",
                                    "description": "When true, a file must match ALL tags in min_tags (AND logic). When false (default), matching ANY single tag is sufficient (OR logic); results are ranked by number of matching tags.",
                                },
                                "top_m": {
                                    "type": "integer",
                                    "description": "Maximum number of tags to return (default 50).",
                                },
                            },
                            "required": [],
                        },
                    },
                },
                "handler": self._get_scoped_tags,
            },
            {
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "scoped_rag_search",
                        "description": (
                            "Semantic search for media files within a mandatory date and tag scope. "
                            "You MUST supply both a date range (start_date + end_date) AND at least "
                            "one tag in min_tags before calling this tool. Use get_scoped_dates and "
                            "get_scoped_tags first to discover valid values. Returns the top-K most "
                            "relevant files for the query within that scope."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Natural language search query describing what to find.",
                                },
                                "start_date": {
                                    "type": "string",
                                    "description": "Inclusive start date (YYYY-MM-DD). Required — obtain from get_scoped_dates first.",
                                },
                                "end_date": {
                                    "type": "string",
                                    "description": "Inclusive end date (YYYY-MM-DD). Required — obtain from get_scoped_dates first.",
                                },
                                "min_tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Required tag filter. Provide at least one tag — obtain from get_scoped_tags first.",
                                },
                                "strict": {
                                    "type": "boolean",
                                    "description": "When true, files must match ALL tags in min_tags (AND logic). When false (default), matching ANY single tag is sufficient (OR logic); results are ranked by number of matching tags.",
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "Number of results to return (default from config top_k).",
                                },
                            },
                            "required": ["query", "start_date", "end_date", "min_tags"],
                        },
                    },
                },
                "handler": self._scoped_rag_search,
            },
            {
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "get_scoped_dates",
                        "description": (
                            "List the contiguous date ranges that contain matching files, sorted by "
                            "file count. Use this to understand the temporal distribution of the "
                            "library or a scoped subset. Helps identify when events happened."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "start_date": {
                                    "type": "string",
                                    "description": "Restrict to files on or after this date (YYYY-MM-DD). Omit for no lower bound.",
                                },
                                "end_date": {
                                    "type": "string",
                                    "description": "Restrict to files on or before this date (YYYY-MM-DD). Omit for no upper bound.",
                                },
                                "min_tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Optional tag filter applied before date grouping.",
                                },
                                "strict": {
                                    "type": "boolean",
                                    "description": "When true, files must match ALL tags in min_tags (AND logic). When false (default), matching ANY single tag is sufficient (OR logic); results are ranked by number of matching tags.",
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "Maximum number of date ranges to return (default 10).",
                                },
                            },
                            "required": [],
                        },
                    },
                },
                "handler": self._get_scoped_dates,
            },
            {
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "get_conversation_rag",
                        "description": (
                            "Semantic search across compacted past conversation summaries. "
                            "Use this to find relevant facts, preferences, or events that were "
                            "discussed in previous conversations but may not appear as media files."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Natural language query describing what to find in past conversations.",
                                },
                                "top_n": {
                                    "type": "integer",
                                    "description": "Number of conversation snippets to return (default 5).",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                },
                "handler": self._get_conversation_rag,
            },
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tool_definitions(self, only_rag: bool = False) -> List[Dict[str, Any]]:
        """Return OpenAI-format tool definitions.

        Args:
            only_rag: When True, return only ``scoped_rag_search`` (used when
                      budget == 1 to force a final targeted search).
        """
        if only_rag:
            return [
                e["schema"] for e in self._registry
                if e["schema"]["function"]["name"] == "scoped_rag_search"
            ]
        return [e["schema"] for e in self._registry]

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool by name and return its string result.

        Args:
            tool_name: Name of the tool to call.
            arguments: Parsed JSON arguments dict.

        Returns:
            Plain-text result string (truncated to MAX_RESULT_CHARS).
        """
        entry = next(
            (e for e in self._registry if e["schema"]["function"]["name"] == tool_name),
            None,
        )
        if entry is None:
            return f"Unknown tool: {tool_name}"
        try:
            result = await entry["handler"](**arguments)
            return _truncate(result)
        except Exception as exc:
            logger.error(f"Tool {tool_name} failed: {exc}", exc_info=True)
            return f"Tool error ({tool_name}): {exc}"

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def _get_scoped_tags(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_tags: Optional[List[str]] = None,
        strict: bool = False,
        top_m: Optional[int] = None,
    ) -> str:
        """Implement get_scoped_tags."""
        effective_top_m: int = (
            top_m
            if top_m is not None
            else (getattr(self.config, "max_tags_per_scope", 50) if self.config else 50)
        )

        all_meta: List[FileMetadata] = self.metadata_store.get_all_metadata()
        candidates: List[Any] = list(all_meta)

        if start_date and end_date:
            date_filtered = filter_by_date(all_meta, start_date, end_date)
            if date_filtered:
                candidates = date_filtered
            else:
                return f"No files found between {start_date} and {end_date}."
        elif start_date:
            date_filtered = filter_by_date(all_meta, start_date, start_date)
            if date_filtered:
                candidates = date_filtered

        if min_tags:
            tag_filtered = filter_by_tags(candidates, min_tags, strict=strict)
            if tag_filtered:
                candidates = tag_filtered

        if not candidates:
            return "No files matched the specified scope."

        tag_counter: Counter = Counter()
        for c in candidates:
            for tag in c.tags:
                tag_counter[tag.lower()] += 1

        top_tags = tag_counter.most_common(effective_top_m)
        if not top_tags:
            return f"Found {len(candidates)} file(s) in scope but none have tags."

        lines = [f"Top {len(top_tags)} tags from {len(candidates)} file(s) in scope:"]
        for tag, count in top_tags:
            lines.append(f"  {tag}: {count}")
        return "\n".join(lines)

    async def _scoped_rag_search(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_tags: Optional[List[str]] = None,
        strict: bool = False,
        top_k: Optional[int] = None,
    ) -> str:
        """Implement scoped_rag_search."""
        effective_top_k: int = (
            top_k
            if top_k is not None
            else (getattr(self.config, "top_k", 5) if self.config else 5)
        )

        all_meta: List[FileMetadata] = self.metadata_store.get_all_metadata()
        candidates: List[Any] = list(all_meta)

        if start_date and end_date:
            date_filtered = filter_by_date(all_meta, start_date, end_date)
            if date_filtered:
                candidates = date_filtered
            else:
                return f"No files found between {start_date} and {end_date}."
        elif start_date:
            date_filtered = filter_by_date(all_meta, start_date, start_date)
            if date_filtered:
                candidates = date_filtered

        if min_tags:
            tag_filtered = filter_by_tags(candidates, min_tags, strict=strict)
            if tag_filtered:
                candidates = tag_filtered

        if not candidates:
            return "No files matched the specified scope."

        logger.info(f"scoped_rag_search: {len(candidates)} candidates, query='{query[:80]}'")

        results = await scoped_rag_search(
            filtered_candidates=candidates,
            query=query,
            k=effective_top_k,
            llm_service=self.llm_service,
            embedding_model=self.embedding_model,
            metadata_store=self.metadata_store,
            compaction_service=self.compaction_service,
        )

        if self.chat_model and self.llm_service:
            await self.llm_service.load_model(self.chat_model)

        if not results:
            return "No matching files found for the query."

        lines = [f"Found {len(results)} file(s) matching '{query}':"]
        for r in results:
            if isinstance(r, ConversationCandidate):
                lines.append(f"\n[Conversation {r.conv_id}]")
                if r.compacted_at:
                    lines.append(f"  Compacted: {r.compacted_at[:10]}")
                if r.tags:
                    lines.append(f"  Keywords: {', '.join(r.tags[:10])}")
                if r.summary:
                    preview = r.summary[:250] + "..." if len(r.summary) > 250 else r.summary
                    lines.append(f"  Summary: {preview}")
            else:
                lines.append(f"\n- {r.fileName}")
                if r.creationTime:
                    lines.append(f"  Date: {r.creationTime[:10]}")
                if r.tags:
                    lines.append(f"  Tags: {', '.join(r.tags[:12])}")
                if r.description:
                    desc = r.description[:200] + "..." if len(r.description) > 200 else r.description
                    lines.append(f"  Description: {desc}")
                if r.type:
                    lines.append(f"  Type: {r.type}")

        return "\n".join(lines)

    async def _get_scoped_dates(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_tags: Optional[List[str]] = None,
        strict: bool = False,
        top_k: Optional[int] = None,
    ) -> str:
        """Implement get_scoped_dates."""
        effective_top_k: int = (
            top_k
            if top_k is not None
            else (getattr(self.config, "max_dates_per_scope", 10) if self.config else 10)
        )

        all_meta: List[FileMetadata] = self.metadata_store.get_all_metadata()
        candidates: List[Any] = list(all_meta)

        if start_date and end_date:
            date_filtered = filter_by_date(all_meta, start_date, end_date)
            if date_filtered:
                candidates = date_filtered

        if min_tags:
            tag_filtered = filter_by_tags(candidates, min_tags, strict=strict)
            if tag_filtered:
                candidates = tag_filtered

        if not candidates:
            return "No files matched the specified scope."

        dated: List[datetime] = []
        for c in candidates:
            try:
                ct = c.creationTime
                if ct:
                    dt = datetime.fromisoformat(ct.replace("Z", "+00:00")).replace(tzinfo=None)
                    dated.append(dt)
            except (ValueError, AttributeError):
                continue

        if not dated:
            return f"Found {len(candidates)} file(s) but none have parseable dates."

        dated.sort()

        ranges: List[Dict[str, Any]] = []
        range_start = dated[0].date()
        range_end = dated[0].date()
        count = 1

        for dt in dated[1:]:
            d = dt.date()
            gap = (d - range_end).days
            if gap <= 1:
                range_end = d
                count += 1
            else:
                ranges.append({"start": str(range_start), "end": str(range_end), "file_count": count})
                range_start = d
                range_end = d
                count = 1

        ranges.append({"start": str(range_start), "end": str(range_end), "file_count": count})
        ranges.sort(key=lambda r: r["file_count"], reverse=True)
        top_ranges = ranges[:effective_top_k]

        lines = [
            f"Top {len(top_ranges)} date range(s) by file count "
            f"(from {len(candidates)} matching files):"
        ]
        for r in top_ranges:
            if r["start"] == r["end"]:
                lines.append(f"  {r['start']} — {r['file_count']} file(s)")
            else:
                lines.append(f"  {r['start']} to {r['end']} — {r['file_count']} file(s)")
        return "\n".join(lines)

    async def _get_conversation_rag(
        self,
        query: str,
        top_n: Optional[int] = None,
    ) -> str:
        """Implement get_conversation_rag."""
        effective_top_n: int = (
            top_n
            if top_n is not None
            else (getattr(self.config, "top_k", 5) if self.config else 5)
        )

        if self.compaction_service is None:
            return "Conversation compaction service is not available."

        if not self.compaction_service.is_loaded():
            self.compaction_service.load()

        all_data = self.compaction_service.get_all_data()
        if not all_data:
            return "No compacted conversations found."

        candidates: List[ConversationCandidate] = [
            ConversationCandidate(
                conv_id=cid,
                summary=d.get("summary", ""),
                tags=d.get("tags", []),
                compacted_at=d.get("compactedAt", ""),
            )
            for cid, d in all_data.items()
            if d.get("embedding")
        ]

        if not candidates:
            return "No embedded conversations available for search."

        results = await scoped_rag_search(
            filtered_candidates=candidates,
            query=query,
            k=effective_top_n,
            llm_service=self.llm_service,
            embedding_model=self.embedding_model,
            metadata_store=self.metadata_store,
            compaction_service=self.compaction_service,
        )

        if self.chat_model and self.llm_service:
            await self.llm_service.load_model(self.chat_model)

        if not results:
            return "No relevant past conversations found."

        lines = [f"Found {len(results)} relevant past conversation(s):"]
        for r in results:
            if isinstance(r, ConversationCandidate):
                lines.append(f"\n[Conversation {r.conv_id}]")
                if r.compacted_at:
                    lines.append(f"  Compacted: {r.compacted_at[:10]}")
                if r.tags:
                    lines.append(f"  Keywords: {', '.join(r.tags[:10])}")
                if r.summary:
                    preview = r.summary[:300] + "..." if len(r.summary) > 300 else r.summary
                    lines.append(f"  Summary: {preview}")

        return "\n".join(lines)

