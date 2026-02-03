# Available Models API Endpoint

## Overview

The `/api/available-models` endpoint provides a list of models available in your system, filtered by task type, and includes information about whether the required model files actually exist in the `model/` directory.

## Endpoint

**GET** `/api/available-models`

### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `task_type` | string | No | Filter models by task type: `vision`, `chat`, or `embedding`. If not provided, returns all models. |

## Response Format

```json
{
  "models": [
    {
      "name": "qwen_2.5_vl",
      "type": "vision",
      "model_file": "Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf",
      "model_exists": true,
      "mmproj_file": "mmproj-F16.gguf",
      "mmproj_exists": true,
      "llm_params": {
        "ctx_size": 12192,
        "n_gpu_layers": 99,
        "top_k": 40,
        "top_p": 0.9,
        "temp": 0.35,
        "presence_penalty": 0.2,
        "microstat": 2,
        "batch_size": 8192,
        "ubatch_size": 1024
      }
    }
  ],
  "total_count": 1,
  "task_type": "vision"
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `models` | array | List of model information objects |
| `total_count` | integer | Total number of models matching the criteria |
| `task_type` | string/null | The task type filter applied, or null if none |

### Model Object Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Model identifier/name |
| `type` | string | Task type: `vision`, `chat`, or `embedding` |
| `model_file` | string | Filename of the model in the `model/` directory |
| `model_exists` | boolean | **true** if model file exists, **false** otherwise |
| `mmproj_file` | string/null | MMProj filename (for vision models only) |
| `mmproj_exists` | boolean/null | **true** if MMProj file exists (for vision models) |
| `llm_params` | object/null | Model-specific LLM parameters |

## Usage Examples

### Get All Available Models

```bash
curl http://localhost:8000/api/available-models
```

```python
import requests

response = requests.get("http://localhost:8000/api/available-models")
models = response.json()

print(f"Total models: {models['total_count']}")
for model in models['models']:
    status = "✓" if model['model_exists'] else "✗"
    print(f"{status} {model['name']} ({model['type']})")
```

### Get Vision Models Only

```bash
curl "http://localhost:8000/api/available-models?task_type=vision"
```

```python
import requests

response = requests.get(
    "http://localhost:8000/api/available-models",
    params={"task_type": "vision"}
)
models = response.json()

for model in models['models']:
    if model['model_exists']:
        print(f"✓ {model['name']}: {model['model_file']}")
        if model.get('mmproj_file'):
            mmproj_status = "✓" if model['mmproj_exists'] else "✗"
            print(f"  MMProj: {mmproj_status} {model['mmproj_file']}")
```

### Get Chat Models Only

```bash
curl "http://localhost:8000/api/available-models?task_type=chat"
```

```python
import requests

response = requests.get(
    "http://localhost:8000/api/available-models",
    params={"task_type": "chat"}
)
models = response.json()

# Filter to only available (existing) models
available = [m for m in models['models'] if m['model_exists']]
print(f"Available chat models: {len(available)}/{models['total_count']}")
```

### Get Embedding Models Only

```bash
curl "http://localhost:8000/api/available-models?task_type=embedding"
```

```python
import requests

response = requests.get(
    "http://localhost:8000/api/available-models",
    params={"task_type": "embedding"}
)
models = response.json()
```

## Use Cases

### 1. Check Model Availability Before Operations

```python
import requests

def get_available_vision_models():
    """Get list of vision models that are ready to use."""
    response = requests.get(
        "http://localhost:8000/api/available-models",
        params={"task_type": "vision"}
    )
    models = response.json()['models']
    
    # Filter to only models with both model and mmproj files
    return [
        m for m in models 
        if m['model_exists'] and (
            not m.get('mmproj_file') or m.get('mmproj_exists')
        )
    ]

available_models = get_available_vision_models()
if available_models:
    print(f"Can use: {available_models[0]['name']}")
else:
    print("No vision models available - please download models")
```

### 2. Dynamic UI Model Selection

```python
import requests

def populate_model_dropdown(task_type):
    """Get models for a dropdown menu in a UI."""
    response = requests.get(
        "http://localhost:8000/api/available-models",
        params={"task_type": task_type}
    )
    models = response.json()['models']
    
    # Return only models that exist
    return [
        {
            'label': f"{m['name']} ({m['model_file']})",
            'value': m['name'],
            'available': m['model_exists']
        }
        for m in models
    ]

# For a vision task UI
vision_models = populate_model_dropdown("vision")
```

### 3. Validate Configuration

```python
import requests

def validate_model_config(config):
    """Validate that configured models actually exist."""
    response = requests.get("http://localhost:8000/api/available-models")
    all_models = {m['name']: m for m in response.json()['models']}
    
    issues = []
    
    # Check vision model
    vision_model = config.get('vision_model')
    if vision_model and vision_model in all_models:
        model = all_models[vision_model]
        if not model['model_exists']:
            issues.append(f"Vision model file missing: {model['model_file']}")
        if model.get('mmproj_file') and not model.get('mmproj_exists'):
            issues.append(f"MMProj file missing: {model['mmproj_file']}")
    
    # Check chat model
    chat_model = config.get('chat_model')
    if chat_model and chat_model in all_models:
        if not all_models[chat_model]['model_exists']:
            issues.append(f"Chat model file missing: {all_models[chat_model]['model_file']}")
    
    return issues

# Usage
config = {"vision_model": "qwen_2.5_vl", "chat_model": "qwen_3"}
issues = validate_model_config(config)
if issues:
    print("Configuration issues:")
    for issue in issues:
        print(f"  - {issue}")
```

### 4. Setup Validation Script

```python
import requests

def check_system_readiness():
    """Check which models are ready to use."""
    response = requests.get("http://localhost:8000/api/available-models")
    models_by_type = {}
    
    for model in response.json()['models']:
        task_type = model['type']
        if task_type not in models_by_type:
            models_by_type[task_type] = {'available': 0, 'missing': 0}
        
        if model['model_exists']:
            # For vision, also check mmproj
            if task_type == 'vision' and model.get('mmproj_file'):
                if model.get('mmproj_exists'):
                    models_by_type[task_type]['available'] += 1
                else:
                    models_by_type[task_type]['missing'] += 1
            else:
                models_by_type[task_type]['available'] += 1
        else:
            models_by_type[task_type]['missing'] += 1
    
    print("System Readiness:")
    for task_type, counts in models_by_type.items():
        total = counts['available'] + counts['missing']
        status = "✓" if counts['available'] > 0 else "✗"
        print(f"  {status} {task_type}: {counts['available']}/{total} available")

check_system_readiness()
```

## Error Responses

### Invalid Task Type

**Request:**
```bash
curl "http://localhost:8000/api/available-models?task_type=invalid"
```

**Response:** `400 Bad Request`
```json
{
  "detail": "Invalid task_type 'invalid'. Must be one of: vision, chat, embedding"
}
```

## Implementation Details

### Model Definition Location

Models are defined in `app/config/settings.py` in the `model_options` dictionary:

```python
model_options = {
    "qwen_2.5_vl": {
        "name": "qwen_2.5_vl",
        "type": "vision",
        "model_file": "Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf",
        "mmproj_file": "mmproj-F16.gguf",
        "llm_params": { ... }
    },
    "qwen_3": {
        "name": "qwen_3",
        "type": "chat",
        "model_file": "Qwen3-8B-Q4_K_M.gguf",
        "llm_params": { ... }
    },
    # ... more models
}
```

### File Existence Check

The endpoint checks if files exist in the `model/` directory:
- For all models: Checks `model_file` exists
- For vision models: Also checks `mmproj_file` exists (if specified)

### Adding New Models

To add a new model, update `model_options` in `app/config/settings.py`:

```python
model_options = {
    # ... existing models
    "my_new_model": {
        "name": "my_new_model",
        "type": "chat",  # or "vision" or "embedding"
        "model_file": "my-model.gguf",
        "llm_params": {
            "ctx_size": 8192,
            "temp": 0.7,
            # ... other parameters
        }
    }
}
```

The new model will automatically appear in the API response.

## Benefits

1. **Pre-flight Checks**: Verify models exist before attempting operations
2. **Dynamic UI**: Build UIs that adapt to available models
3. **Setup Validation**: Help users identify missing model files
4. **Configuration Management**: Validate configurations before saving
5. **Troubleshooting**: Quickly identify which models are missing

## Notes

- This endpoint is **read-only** and does not modify any configuration
- Model existence is checked on each request (not cached)
- Missing models are included in the response with `model_exists: false`
- The endpoint does not load or validate model formats, only checks file existence
- For vision models, both model and mmproj files must exist to be considered fully available
