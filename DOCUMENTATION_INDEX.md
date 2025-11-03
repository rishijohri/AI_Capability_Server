# Documentation Index

Complete documentation for the AI Server project.

## 📖 Core Documentation

### [README_AI_SERVER.md](README_AI_SERVER.md)
**Main project documentation** covering:
- Features overview
- Project structure
- Installation and setup
- Running the server
- Building with PyInstaller
- API endpoints (overview)
- Configuration options
- Architecture and modularity
- File metadata format
- Basic examples
- Troubleshooting

**Start here** if you're new to the project.

---

### [API_REFERENCE.md](API_REFERENCE.md)
**Complete API documentation** with:
- Full REST API specification
- Detailed WebSocket protocols
- Request/response examples for all endpoints
- Error handling patterns
- Complete workflow examples
- Python and JavaScript code samples
- Best practices

**Use this** for detailed API integration and development.

---

### [QUICKSTART.md](QUICKSTART.md)
**5-minute setup guide** including:
- Prerequisites checklist
- Step-by-step installation
- First-time configuration
- Quick test procedures
- Common issues and solutions

**Best for** getting the server running quickly.

---

### [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)
**Practical examples** with your specific models:
- Real commands using your actual model files
- Python WebSocket client examples
- cURL commands for REST endpoints
- Complete workflow demonstrations
- Performance notes
- Tips and troubleshooting

**Best for** learning by example with your exact setup.

---

### [MODEL_CONFIG.md](MODEL_CONFIG.md)
**Model-specific configuration** covering:
- Your installed models and their purposes
- Recommended model parameters
- Binary selection guide
- LLM backend modes (server vs CLI)
- Backend behavior for different task types
- Model capabilities and limitations
- Performance characteristics

**Best for** understanding which model to use for what task.

---

### [LLM_BACKEND_IMPLEMENTATION.md](LLM_BACKEND_IMPLEMENTATION.md)
**Technical backend documentation** covering:
- Backend architecture (server vs CLI mode)
- LLM parameter application across all operations
- Task routing logic (chat, embeddings, vision)
- Vision binary specialization (llama-qwen2vl-cli, llama-mtmd-cli)
- Complete parameter verification
- Configuration examples

**Best for** understanding how the server handles LLM operations internally.

---

## 🚀 Getting Started Path

**If you're new, follow this order:**

1. **[QUICKSTART.md](QUICKSTART.md)** → Get the server running (5 minutes)
2. **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** → Try basic operations with your models (15 minutes)
3. **[API_REFERENCE.md](API_REFERENCE.md)** → Learn the full API (as needed)
4. **[README_AI_SERVER.md](README_AI_SERVER.md)** → Understand architecture and advanced topics
5. **[MODEL_CONFIG.md](MODEL_CONFIG.md)** → Optimize for your use case

---

## 📚 By Use Case

### I want to...

**...get started quickly**
→ [QUICKSTART.md](QUICKSTART.md)

**...integrate the API into my app**
→ [API_REFERENCE.md](API_REFERENCE.md)

**...see working code examples**
→ [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)

**...understand the architecture**
→ [README_AI_SERVER.md](README_AI_SERVER.md)

**...optimize model performance**
→ [MODEL_CONFIG.md](MODEL_CONFIG.md)

**...understand LLM backend behavior**
→ [LLM_BACKEND_IMPLEMENTATION.md](LLM_BACKEND_IMPLEMENTATION.md)

**...verify parameter application**
→ [LLM_BACKEND_IMPLEMENTATION.md](LLM_BACKEND_IMPLEMENTATION.md#verification)

**...troubleshoot an issue**
→ [README_AI_SERVER.md](README_AI_SERVER.md#troubleshooting) or [QUICKSTART.md](QUICKSTART.md#common-issues)

**...change configuration**
→ [API_REFERENCE.md](API_REFERENCE.md#post-config)

**...replace a component (Vector DB, LLM backend)**
→ [README_AI_SERVER.md](README_AI_SERVER.md#architecture--modularity)

---

## 📊 Quick Reference

### Endpoints

| Endpoint | Type | Purpose | Documentation |
|----------|------|---------|---------------|
| `/api/config` | GET/POST | View/update configuration | [API Reference](API_REFERENCE.md#get-config) |
| `/api/set-storage-metadata` | POST | Set metadata file | [API Reference](API_REFERENCE.md#post-set-storage-metadata) |
| `/api/load-rag` | POST | Load RAG database | [API Reference](API_REFERENCE.md#post-load-rag) |
| `/api/generate-embeddings` | WebSocket | Generate embeddings | [API Reference](API_REFERENCE.md#ws-generate-embeddings) |
| `/api/generate-rag` | WebSocket | Build RAG database | [API Reference](API_REFERENCE.md#ws-generate-rag) |
| `/api/tag` | WebSocket | Generate image/video tags | [API Reference](API_REFERENCE.md#ws-tag) |
| `/api/describe` | WebSocket | Generate descriptions | [API Reference](API_REFERENCE.md#ws-describe) |
| `/api/chat` | WebSocket | Chat with RAG context | [API Reference](API_REFERENCE.md#ws-chat) |

### Configuration Fields

| Field | Type | Editable | Documentation |
|-------|------|----------|---------------|
| `chat_model` | string | ✅ | [Model Config](MODEL_CONFIG.md) |
| `embedding_model` | string | ✅ | [Model Config](MODEL_CONFIG.md) |
| `vision_model` | string | ✅ | [Model Config](MODEL_CONFIG.md) |
| `reduced_embedding_size` | int/null | ✅ | [API Reference](API_REFERENCE.md#get-config) |
| `image_quality` | string | ✅ | [API Reference](API_REFERENCE.md#get-config) |
| `top_k` | int | ✅ | [API Reference](API_REFERENCE.md#get-config) |
| `recency_bias` | float | ✅ | [API Reference](API_REFERENCE.md#get-config) |
| `llm_params` | object | ✅ | [API Reference](API_REFERENCE.md#get-config) |

See [API_REFERENCE.md](API_REFERENCE.md#get-config) for complete field descriptions.

---

## 🔧 Additional Files

### [requirements.txt](requirements.txt)
Python package dependencies

### [ai_server.spec](ai_server.spec)
PyInstaller specification for building executable

### [run_server.py](run_server.py)
Server startup script

### [example_client.py](example_client.py)
Sample WebSocket client implementation

### [validate.py](validate.py)
Setup validation script - run before first use

### [sample_metadata.json](sample_metadata.json)
Example metadata file structure

---

## 💡 Tips

### For API Development
1. Start with [API_REFERENCE.md](API_REFERENCE.md) - most comprehensive
2. Use [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) for copy-paste ready code
3. Reference the [complete workflow example](API_REFERENCE.md#complete-workflow-example)

### For Deployment
1. Follow [QUICKSTART.md](QUICKSTART.md) for initial setup
2. Use [README_AI_SERVER.md](README_AI_SERVER.md#building-standalone-executable) for PyInstaller build
3. Check [troubleshooting section](README_AI_SERVER.md#troubleshooting) for common issues

### For Understanding the Codebase
1. Read [README_AI_SERVER.md](README_AI_SERVER.md#project-structure) for file organization
2. Review [README_AI_SERVER.md](README_AI_SERVER.md#architecture--modularity) for design patterns
3. Check inline code documentation in source files

---

## 🆘 Getting Help

**Problem:** Server won't start  
**Solution:** Run `python validate.py` to check setup

**Problem:** Model not found  
**Solution:** Check [MODEL_CONFIG.md](MODEL_CONFIG.md) for correct filenames

**Problem:** API returning errors  
**Solution:** See [API_REFERENCE.md](API_REFERENCE.md#error-handling)

**Problem:** WebSocket connection issues  
**Solution:** Review [API_REFERENCE.md](API_REFERENCE.md#connection-example)

**Problem:** Performance issues  
**Solution:** Check [MODEL_CONFIG.md](MODEL_CONFIG.md) for optimization tips

---

## 📝 Documentation Version

Last Updated: October 23, 2025  
Project Version: 1.0

---

## 🤝 Contributing

When updating documentation:
- Keep examples working and tested
- Update this index when adding new docs
- Maintain consistency in formatting
- Include practical, copy-paste ready examples
- Cross-reference related sections

---

**Need more help?** All documents include detailed examples and troubleshooting sections.
