# Deep Chat Refactor Summary

## Overview
The deep-chat endpoint has been refactored to follow the established coding patterns in the codebase:

## Changes Made

### 1. Added `deep_chat_system_prompt` to settings.py ✅
**Location:** `/app/config/settings.py` (after `describe_prompt`)

The Deep Thinking system prompt is now stored in settings.py alongside other prompts (chat_system_prompt, tag_prompt, describe_prompt), following the established pattern.

### 2. Created Chat Helper Functions ✅
**Location:** `/app/utils/chat_helpers.py` (new file)

Helper functions extracted to reduce code duplication between `/chat` and `/deep-chat`:

- `prepare_chat_session(websocket, metadata_store)` - Load RAG and embedding models
- `validate_and_setup_history(websocket, provided_history)` - Validate conversation history format
- `load_image_for_chat(websocket, image_name, metadata_store)` -load and process images for visual chat
- `gather_initial_context(...)` - Gather limited initial RAG context for Deep Chat
- `search_rag_for_context(...)` - Perform comprehensive RAG search for regular chat
- `store_objective_facts(...)` - Store objective facts into knowledge base
- `build_rag_context_from_results(...)` - Format RAG search results

### 3. Refactored Deep Chat Endpoint ✅
**Status:** Code ready but NOT YET APPLIED to routes.py

**Refactored version location:** `/app/api/deep_chat_refactored.py`

#### Key Improvements:
- **Reduced from ~620 lines to ~450 lines** (27% reduction)
- Uses `config.deep_chat_system_prompt` instead of hardcoded 60-line prompt
- Uses helper functions instead of duplicating initialization logic
- Cleaner, more maintainable code structure
- Follows established pattern (prompts in settings.py, common logic in utils/)

#### What Changed:
1. **Session preparation:** Now uses `chat_helpers.prepare_chat_session()` instead of inline RAG/embedding loading (~80 lines → 3 lines)
2. **History validation:** Now uses `chat_helpers.validate_and_setup_history()` (~30 lines → 2 lines)
3. **Image loading:** Now uses `chat_helpers.load_image_for_chat()` (~40 lines → 5 lines)
4. **Initial context:** Now uses `chat_helpers.gather_initial_context()` (~50 lines → 3 lines)
5. **System prompt:** Now uses `config.deep_chat_system_prompt` (~60 lines → 1 line reference)
6. **Fact storage:** Now uses `chat_helpers.store_objective_facts()` (~20 lines → 3 lines)
7. **RAG context building:** Now uses `chat_helpers.build_rag_context_from_results()` (shared function)

#### Function Calling:
- Still supports `query_media_rag` and `query_fact_rag` functions
- Updated function signature for `query_fact_rag` to accept `token_budget` and `min_relevance` parameters (not just `k`)
- Uses helper to format RAG results

#### Output Format:
- Changed from `<final_answer>` tags to `<conclusion>` + `<files>` tags to match regular chat format
- Extracts and returns file list in the response

### 4. Updated Imports  ✅
**Location:** `/app/utils/__init__.py` and `/app/api/routes.py`

- Added `chat_helpers` to `/app/utils/__init__.py` exports
- Added `from app.utils import chat_helpers` to `/app/api/routes.py`

## Next Steps

### To Apply the Refactoring:

Replace the current `/deep-chat` endpoint in `/app/api/routes.py` (lines 2383-3004) with the refactored version from `/app/api/deep_chat_refactored.py`.

**Important:** Change the function name from `deep_chat_websocket_handler` to `deep_chat_ws` when copying to maintain the existing route handler name.

The refactored version is at: `/app/api/deep_chat_refactored.py`

Simply:
1. Open `/app/api/deep_chat_refactored.py`
2. Copy the entire function body (starting from `async def`)
3. Replace the existing `deep_chat_ws` function in `/app/api/routes.py` (lines 2383-3004)
4. Rename `deep_chat_websocket_handler` → `deep_chat_ws`
5. Delete `/app/api/deep_chat_refactored.py` after applying

## Benefits

- ✅ **Code reusability:** Common logic shared between chat endpoints
- ✅ **Maintainability:** Changes to initialization logic only need to be made once
- ✅ **Consistency:** Prompts follow established pattern (stored in settings.py)
- ✅ **Readability:** Deep chat endpoint is now ~27% shorter and easier to understand
- ✅ **Testability:** Helper functions can be tested independently

## Testing Recommendations

After applying the refactor:
1. Test `/api/deep-chat` with text-only queries
2. Test `/api/deep-chat` with image queries
3. Test function calling (both query_media_rag and query_fact_rag)
4. Test multi-round thinking behavior
5. Verify initial context gathering
6. Verify final response format with files list

## Files Modified

- ✅ `/app/config/settings.py` - Added `deep_chat_system_prompt` field
- ✅ `/app/utils/chat_helpers.py` - Created with 7 helper functions
- ✅ `/app/utils/__init__.py` - Added chat_helpers export
- ✅ `/app/api/routes.py` - Added import for chat_helpers
- ⏳ `/app/api/routes.py` - Deep chat endpoint replacement PENDING
- 📄 `/app/api/deep_chat_refactored.py` - Temporary file with refactored code (delete after applying)

## Architectural Compliance

✅ System prompts in `settings.py` (following chat_system_prompt, tag_prompt, describe_prompt pattern)
✅ Common logic in helper functions (chat_helpers.py in utils/)
✅ No code duplication between endpoints
✅ Follows existing coding patterns in the codebase
