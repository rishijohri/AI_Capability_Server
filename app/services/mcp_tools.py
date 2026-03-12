"""MCP Tool Registry — OpenAI-format tool definitions and execution for MCP chat mode."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import Counter

from app.models.metadata import MetadataStore

logger = logging.getLogger(__name__)

# Maximum characters in a tool result to protect context window
MAX_RESULT_CHARS = 8000


def _truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (truncated)"


def _format_file_entry(meta) -> str:
    """Format a single FileMetadata into a concise text block."""
    lines = [f"- {meta.fileName}"]
    if hasattr(meta, 'type') and meta.type:
        lines[0] += f"  [{meta.type}]"
    if hasattr(meta, 'creationTime') and meta.creationTime:
        lines.append(f"  Created: {meta.creationTime}")
    if meta.tags:
        lines.append(f"  Tags: {', '.join(meta.tags)}")
    desc = getattr(meta, 'description', None)
    if desc:
        short = desc[:200] + "..." if len(desc) > 200 else desc
        lines.append(f"  Description: {short}")
    return "\n".join(lines)


class MCPToolRegistry:
    """Provides OpenAI-format tool definitions and executes tool calls against local services."""

    def __init__(
        self,
        metadata_store: MetadataStore,
        rag_service=None,
        knowledge_service=None,
        llm_service=None,
        face_service=None,
        embedding_loaded: bool = False,
        rag_available: bool = False,
    ):
        self.metadata_store = metadata_store
        self.rag_service = rag_service
        self.knowledge_service = knowledge_service
        self.llm_service = llm_service
        self.face_service = face_service
        self.embedding_loaded = embedding_loaded
        self.rag_available = rag_available

        # Map tool name → handler
        self._handlers = {
            "search_media": self._search_media,
            "search_by_person": self._search_by_person,
            "search_by_location": self._search_by_location,
            "search_by_tags": self._search_by_tags,
            "search_by_date_range": self._search_by_date_range,
            "get_file_info": self._get_file_info,
            "list_known_people": self._list_known_people,
            "list_known_locations": self._list_known_locations,
            "get_library_stats": self._get_library_stats,
            "search_knowledge": self._search_knowledge,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return OpenAI-format tool definitions list."""
        tools: List[Dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "search_media",
                    "description": "Semantic search across the media library using natural language. Finds files whose tags/descriptions are semantically similar to the query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Natural language search query"},
                            "count": {"type": "integer", "description": "Number of results (default 5, max 20)", "default": 5},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_by_person",
                    "description": "Find all files containing a specific person (matched by person: tag prefix).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "person_name": {"type": "string", "description": "Person name or face ID to search for"},
                        },
                        "required": ["person_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_by_location",
                    "description": "Find files from a specific city or country (matched by city: or country: tag prefix).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City or country name"},
                            "location_type": {
                                "type": "string",
                                "enum": ["city", "country", "any"],
                                "description": "Filter by city, country, or any location type (default: any)",
                                "default": "any",
                            },
                        },
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_by_tags",
                    "description": "Find files matching specific tag keywords (substring match on tags). Can require all tags to match or any.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of tag keywords to search for",
                            },
                            "match_all": {
                                "type": "boolean",
                                "description": "If true, all tags must match; if false, any tag matches (default: false)",
                                "default": False,
                            },
                        },
                        "required": ["tags"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_by_date_range",
                    "description": "Find files created within a date range.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_date": {"type": "string", "description": "Start date in ISO format (e.g. 2024-01-01)"},
                            "end_date": {"type": "string", "description": "End date in ISO format (e.g. 2024-12-31)"},
                        },
                        "required": ["start_date", "end_date"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_file_info",
                    "description": "Get detailed metadata for a specific file including all tags, description, dates, and extra fields.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Exact filename to look up"},
                        },
                        "required": ["filename"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_known_people",
                    "description": "List all known people appearing in the library (from person: tags), with file counts.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_known_locations",
                    "description": "List all known cities and countries in the library (from city: and country: tags), with file counts.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_library_stats",
                    "description": "Get overall statistics about the media library: total files, type breakdown, date range, top tags/people/locations.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge",
                    "description": "Search the conversation knowledge base for previously discussed facts or information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query for knowledge base"},
                            "token_budget": {"type": "integer", "description": "Max tokens of results (default 1000)", "default": 1000},
                        },
                        "required": ["query"],
                    },
                },
            },
        ]
        return tools

    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool by name with given arguments. Returns result text."""
        handler = self._handlers.get(name)
        if handler is None:
            return f"Error: Unknown tool '{name}'"
        try:
            result = await handler(arguments)
            return _truncate(result)
        except Exception as e:
            logger.error(f"MCP tool '{name}' failed: {e}", exc_info=True)
            return f"Error executing {name}: {str(e)}"

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def _search_media(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        count = min(int(args.get("count", 5)), 20)

        if not self.rag_service or not self.rag_available:
            return "Media search is not available (RAG not loaded)."

        results = await self.rag_service.search(query, k=count)
        if not results:
            return f"No files found matching '{query}'."

        lines = [f"Found {len(results)} file(s) matching '{query}':"]
        for meta in results:
            lines.append(_format_file_entry(meta))
        return "\n".join(lines)

    async def _search_by_person(self, args: Dict[str, Any]) -> str:
        person_name = args.get("person_name", "").strip()
        if not person_name:
            return "Error: person_name is required."

        all_meta = self.metadata_store.get_all_metadata()
        needle = person_name.lower()
        matches = []
        for meta in all_meta:
            for tag in meta.tags:
                tag_lower = tag.lower()
                if tag_lower.startswith("person:") and needle in tag_lower[len("person:"):]:
                    matches.append(meta)
                    break

        if not matches:
            return f"No files found for person '{person_name}'."

        # Sort by creation date descending
        matches.sort(key=lambda m: m.creationTime or "", reverse=True)
        lines = [f"Found {len(matches)} file(s) with person '{person_name}':"]
        for meta in matches[:30]:
            lines.append(_format_file_entry(meta))
        if len(matches) > 30:
            lines.append(f"... and {len(matches) - 30} more files.")
        return "\n".join(lines)

    async def _search_by_location(self, args: Dict[str, Any]) -> str:
        location = args.get("location", "").strip()
        loc_type = args.get("location_type", "any").lower()
        if not location:
            return "Error: location is required."

        all_meta = self.metadata_store.get_all_metadata()
        needle = location.lower()
        matches = []
        for meta in all_meta:
            for tag in meta.tags:
                tag_lower = tag.lower()
                if loc_type == "city" and tag_lower.startswith("city:") and needle in tag_lower[len("city:"):]:
                    matches.append(meta)
                    break
                elif loc_type == "country" and tag_lower.startswith("country:") and needle in tag_lower[len("country:"):]:
                    matches.append(meta)
                    break
                elif loc_type == "any":
                    if (tag_lower.startswith("city:") and needle in tag_lower[len("city:"):]) or \
                       (tag_lower.startswith("country:") and needle in tag_lower[len("country:"):]):
                        matches.append(meta)
                        break

        if not matches:
            return f"No files found for location '{location}'."

        matches.sort(key=lambda m: m.creationTime or "", reverse=True)
        lines = [f"Found {len(matches)} file(s) from '{location}':"]
        for meta in matches[:30]:
            lines.append(_format_file_entry(meta))
        if len(matches) > 30:
            lines.append(f"... and {len(matches) - 30} more files.")
        return "\n".join(lines)

    async def _search_by_tags(self, args: Dict[str, Any]) -> str:
        tags = args.get("tags", [])
        match_all = args.get("match_all", False)
        if not tags:
            return "Error: tags list is required."

        all_meta = self.metadata_store.get_all_metadata()
        needles = [t.lower() for t in tags]
        matches = []
        for meta in all_meta:
            meta_tags_lower = [t.lower() for t in meta.tags]
            if match_all:
                if all(any(n in mt for mt in meta_tags_lower) for n in needles):
                    matches.append(meta)
            else:
                if any(any(n in mt for mt in meta_tags_lower) for n in needles):
                    matches.append(meta)

        if not matches:
            return f"No files found matching tags: {', '.join(tags)}."

        matches.sort(key=lambda m: m.creationTime or "", reverse=True)
        lines = [f"Found {len(matches)} file(s) matching tags [{', '.join(tags)}] (match_all={match_all}):"]
        for meta in matches[:30]:
            lines.append(_format_file_entry(meta))
        if len(matches) > 30:
            lines.append(f"... and {len(matches) - 30} more files.")
        return "\n".join(lines)

    async def _search_by_date_range(self, args: Dict[str, Any]) -> str:
        start_str = args.get("start_date", "")
        end_str = args.get("end_date", "")
        if not start_str or not end_str:
            return "Error: start_date and end_date are required."

        try:
            start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        except ValueError:
            return f"Error: Invalid start_date format '{start_str}'. Use ISO format (e.g. 2024-01-01)."
        try:
            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        except ValueError:
            return f"Error: Invalid end_date format '{end_str}'. Use ISO format (e.g. 2024-12-31)."

        all_meta = self.metadata_store.get_all_metadata()
        matches = []
        for meta in all_meta:
            ct = meta.creationTime
            if not ct:
                continue
            try:
                file_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            # Compare as naive if timezones are mixed
            if start_dt.tzinfo and not file_dt.tzinfo:
                start_cmp = start_dt.replace(tzinfo=None)
                end_cmp = end_dt.replace(tzinfo=None)
            elif file_dt.tzinfo and not start_dt.tzinfo:
                file_dt = file_dt.replace(tzinfo=None)
                start_cmp = start_dt
                end_cmp = end_dt
            else:
                start_cmp = start_dt
                end_cmp = end_dt
            if start_cmp <= file_dt <= end_cmp:
                matches.append(meta)

        if not matches:
            return f"No files found between {start_str} and {end_str}."

        matches.sort(key=lambda m: m.creationTime or "")
        lines = [f"Found {len(matches)} file(s) between {start_str} and {end_str}:"]
        for meta in matches[:30]:
            lines.append(_format_file_entry(meta))
        if len(matches) > 30:
            lines.append(f"... and {len(matches) - 30} more files.")
        return "\n".join(lines)

    async def _get_file_info(self, args: Dict[str, Any]) -> str:
        filename = args.get("filename", "").strip()
        if not filename:
            return "Error: filename is required."

        meta = self.metadata_store.get_metadata_by_filename(filename)
        if meta is None:
            return f"File '{filename}' not found in the library."

        return meta.to_text_representation()

    async def _list_known_people(self, _args: Dict[str, Any]) -> str:
        all_meta = self.metadata_store.get_all_metadata()
        person_counter: Counter = Counter()
        for meta in all_meta:
            for tag in meta.tags:
                if tag.lower().startswith("person:"):
                    name = tag[len("person:"):]
                    person_counter[name] += 1

        if not person_counter:
            return "No known people found in the library."

        lines = [f"Known people ({len(person_counter)}):"]
        for name, count in person_counter.most_common():
            lines.append(f"- {name}: {count} file(s)")
        return "\n".join(lines)

    async def _list_known_locations(self, _args: Dict[str, Any]) -> str:
        all_meta = self.metadata_store.get_all_metadata()
        city_counter: Counter = Counter()
        country_counter: Counter = Counter()
        for meta in all_meta:
            for tag in meta.tags:
                tag_lower = tag.lower()
                if tag_lower.startswith("city:"):
                    city_counter[tag[len("city:"):]] += 1
                elif tag_lower.startswith("country:"):
                    country_counter[tag[len("country:"):]] += 1

        if not city_counter and not country_counter:
            return "No known locations found in the library."

        lines = []
        if country_counter:
            lines.append(f"Countries ({len(country_counter)}):")
            for name, count in country_counter.most_common():
                lines.append(f"- {name}: {count} file(s)")
        if city_counter:
            lines.append(f"\nCities ({len(city_counter)}):")
            for name, count in city_counter.most_common():
                lines.append(f"- {name}: {count} file(s)")
        return "\n".join(lines)

    async def _get_library_stats(self, _args: Dict[str, Any]) -> str:
        all_meta = self.metadata_store.get_all_metadata()
        if not all_meta:
            return "Library is empty — no files found."

        total = len(all_meta)
        type_counter: Counter = Counter()
        tag_counter: Counter = Counter()
        person_set: set = set()
        location_set: set = set()
        dates = []

        for meta in all_meta:
            file_type = getattr(meta, 'type', 'unknown') or 'unknown'
            type_counter[file_type] += 1
            for tag in meta.tags:
                tag_lower = tag.lower()
                if tag_lower.startswith("person:"):
                    person_set.add(tag[len("person:"):])
                elif tag_lower.startswith("city:") or tag_lower.startswith("country:"):
                    location_set.add(tag)
                else:
                    tag_counter[tag] += 1
            ct = meta.creationTime
            if ct:
                try:
                    dates.append(datetime.fromisoformat(ct.replace("Z", "+00:00")))
                except (ValueError, AttributeError):
                    pass

        lines = [f"Library Statistics:"]
        lines.append(f"Total files: {total}")
        lines.append(f"By type: {', '.join(f'{t}: {c}' for t, c in type_counter.most_common())}")
        if dates:
            lines.append(f"Date range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}")
        lines.append(f"Known people: {len(person_set)}")
        if person_set:
            lines.append(f"  Names: {', '.join(sorted(person_set)[:20])}")
        lines.append(f"Known locations: {len(location_set)}")
        if location_set:
            lines.append(f"  Places: {', '.join(sorted(location_set)[:20])}")
        if tag_counter:
            top_tags = tag_counter.most_common(15)
            lines.append(f"Top tags: {', '.join(f'{t}({c})' for t, c in top_tags)}")
        return "\n".join(lines)

    async def _search_knowledge(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        token_budget = int(args.get("token_budget", 1000))
        if not query:
            return "Error: query is required."

        if not self.knowledge_service:
            return "Knowledge service is not available."

        if not self.llm_service:
            return "LLM service is not available (needed for embedding query)."

        try:
            # Generate embedding for the query
            embedding = await self.llm_service.embed(query)
            facts = self.knowledge_service.select_knowledge(
                query_embedding=embedding,
                token_budget=token_budget,
            )
        except Exception as e:
            return f"Knowledge search failed: {str(e)}"

        if not facts:
            return f"No relevant knowledge found for '{query}'."

        lines = [f"Found {len(facts)} relevant fact(s):"]
        for fact in facts:
            role = fact.get("role", "unknown")
            message = fact.get("message", "")
            ts = fact.get("timestamp", "")
            lines.append(f"- [{role}] ({ts}): {message}")
        return "\n".join(lines)
