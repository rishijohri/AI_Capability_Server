"""Server configuration management."""

from typing import Optional, Literal
from pydantic import BaseModel, Field
import os
from pathlib import Path
from app.utils.resource_paths import get_binary_path as get_binary_resource_path
from app.utils.resource_paths import get_model_path as get_model_resource_path
from app.utils.resource_paths import get_data_directory


class LLMParams(BaseModel):
    """LLM execution parameters."""
    ctx_size: int = Field(default=6500, description="Context size")
    temp: float = Field(default=0.35, description="Temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling")
    top_k: int = Field(default=40, description="Top-k sampling")
    presence_penalty: float = Field(default=0.2, description="Presence penalty")
    mirostat: int = Field(default=0, description="Mirostat sampling")
    batch_size: int = Field(default=1024, description="Batch size")
    ubatch_size: int = Field(default=512, description="Micro-batch size")
    n_gpu_layers: int = Field(default=999, description="Number of GPU layers to offload")

model_options = [
        {
            "name": "qwen_2.5_vl",
            "type": "vision",
            "model_file": "Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf",
            "mmproj_file": "mmproj-F16.gguf",
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
        },
        {
            "name": "qwen_3",
            "type": "chat",
            "model_file": "Qwen3-8B-Q4_K_M.gguf",
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
            
        },
        {
            "name": "qwen_3_embedding",
            "type": "embedding",
            "model_file":"qwen3-embedding-8b-q4_k_m.gguf",
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
        },
        {
            "name": "MiniCPM4.5_KM",
            "type": "vision",
            # "model_file": "ggml-model-Q4_0.gguf",
            "model_file": "ggml-model-Q4_K_M.gguf",
            "mmproj_file": "mmproj-model-f16-ggml.gguf",
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
        },
        {
            "name": "MiniCPM4.5_KS",
            "type": "vision",
            # "model_file": "ggml-model-Q4_0.gguf",
            "model_file": "MiniCPM4-Q4_K_S.gguf",
            "mmproj_file": "MiniCPM4_mmproj-model-f16.gguf",
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
        },
        {
            "name": "granite4_micro",
            "type": "chat",
            "model_file": "granite4-7b-q4k.gguf",
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
        },
        {
            "name": "granite4_350m",
            "type": "chat",
            "model_file": "granite-4.0-350m-UD-Q6_K_XL.gguf",
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
        },
        {
            "name": "llava_phi3_4k",
            "type": "vision",
            "model_file": "llava-phi3-mini-Q4_K_M.gguf",
            "mmproj_file": "llava-phi3-mini-mmproj-f16.gguf",
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
]

class ServerConfig(BaseModel):
    """AI Server configuration."""
    
    

    # Editable configurations
    reduced_embedding_size: Optional[int] = Field(
        default=None, 
        description="Reduced embedding size for RAG (None means original)"
    )
    chat_rounds: int = Field(
        default=3, 
        ge=1, 
        le=10,
        description="Number of rounds for user chat query"
    )
    image_quality: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Image quality multiplier (0.0-1.0). 1.0 = original dimensions, <1.0 = scale dimensions"
    )
    llm_mode: Literal["server", "cli"] = Field(
        default="server",
        description="Use llama-server or llama-cli"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Top K for RAG retrievals"
    )
    recency_bias: float = Field(
        default=1.0,
        ge=0.1,
        description="Recency bias for file retrieval (1.0=no bias, >1.0=favor recent)"
    )
    enable_visual_chat: bool = Field(
        default=True,
        description="Enable visual conversation mode (uses vision model for chat with images)"
    )
    llm_timeout: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Timeout in seconds for LLM operations (10-3600 seconds)"
    )
    
    # Model file names
    chat_model: str = Field(
        default="Qwen3-8B-Q4_K_M.gguf",
        description="Chat/conversation model filename"
    )
    embedding_model: str = Field(
        default="qwen3-embedding-8b-q4_k_m.gguf",
        description="Embedding model filename"
    )
    vision_model: str = Field(
        default = "gemma-3-4b-it-Q4_K_M.gguf",
        description="Vision model filename"
    )
    mmproj_model: str = Field(
        default="gemma_3_mmproj-F16.gguf", # MiniCPM4.5
        description="MMProj model filename for vision"
    )
    
    # System prompts for different tasks
    chat_system_prompt: str = Field(
        default="""You are Persona, a helpful AI assistant. Provide concise, factual answers and DO NOT reveal internal chain-of-thought or reasoning processes.

REQUIRED OUTPUT FORMAT:
- Include ONLY the following two XML sections in your final output, and nothing else:
  1) <conclusion>...</conclusion>  -- your final answer (brief, direct)
  2) <files>...</files>           -- newline or comma-separated relevant file references (empty if none)

FORMAT RULES:
- Do NOT include any <think> or internal reasoning tags.
- If no files are relevant, include an empty <files></files> section.
- Keep the content in <conclusion> short and focused; avoid step-by-step reasoning.

EXAMPLE WITH FILES:
<conclusion>
The image shows a man in a museum exhibit interacting with flutes while listening to an audio guide.
</conclusion>

<files>
- exhibits/flutes_gallery.jpg
</files>

EXAMPLE WITHOUT FILES:
<conclusion>
The scene depicts a quiet museum visitor examining musical instruments while listening to an audio guide.
</conclusion>

<files>
</files>

Remember: Only <conclusion> and <files> are allowed in the final model output; do not output reasoning or any extra tags.""",
        description="System prompt for chat conversations"
    )
    tag_prompt: str = Field(
        default="""Analyze this image and generate descriptive tags.

STRICT FORMAT REQUIREMENTS:
1. Wrap your analysis process in <think></think> tags
2. Wrap ONLY comma-separated tags in <conclusion></conclusion> tags

EXAMPLE RESPONSE FORMAT:
<think>
I can see several elements in this image... [your analysis here]
</think>

<conclusion>
woman, smiling, outdoor, sunny day, park, trees, casual clothing, happy
</conclusion>

RULES FOR CONCLUSION:
- Output ONLY comma-separated keywords
- NO sentences, explanations, or extra punctuation
- Include: objects, people, activities, setting, mood, colors, visible text

Remember: The conclusion section must contain ONLY comma-separated tags.""",
        description="Prompt template for generating tags"
    )
    describe_prompt: str = Field(
        default="""Analyze and describe this image in detail.

STRICT FORMAT REQUIREMENTS:
1. Wrap your analysis process in <think></think> tags
2. Wrap your final description in <conclusion></conclusion> tags

EXAMPLE RESPONSE FORMAT:
<think>
Let me examine the key elements... [your analysis here]
</think>

<conclusion>
The image shows a woman smiling outdoors in a park on a sunny day. She is wearing casual clothing and appears happy. Several trees are visible in the background.
</conclusion>

RULES FOR CONCLUSION:
- Provide a detailed, coherent description
- Include: what you see, setting, people/objects, colors, mood, visible text
- Write in complete sentences

Remember: Both sections are required.""",
        description="Prompt template for generating descriptions"
    )
    
    # LLM Parameters
    llm_params: LLMParams = Field(default_factory=LLMParams)
    
    # Read-only configurations
    rag_directory_name: str = Field(
        default="rag",
        description="RAG directory name (read-only)"
    )
    
    # Runtime state
    storage_metadata_path: Optional[str] = Field(
        default=None,
        description="Path to storage-metadata.json file"
    )
    
    # Binary and model paths (relative to project root)
    binary_dir: str = Field(
        default="binary",
        description="Directory containing llama binaries"
    )
    model_dir: str = Field(
        default="model",
        description="Directory containing model files"
    )
    face_models_dir: Optional[str] = Field(
        default=None,
        description="Directory containing InsightFace models (None = use bundled model/ directory)"
    )
    
    # Vision binary preference
    vision_binary: Literal["auto", "llama-mtmd-cli", "llama-qwen2vl-cli"] = Field(
        default="auto",
        description="Vision binary to use (auto=detect from model name)"
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "reduced_embedding_size": 512,
                "chat_rounds": 3,
                "image_quality": 0.75,
                "llm_mode": "server",
                "top_k": 5,
                "recency_bias": 1.5,
                "enable_visual_chat": True,
                "chat_model": "Qwen3-8B-Q4_K_M.gguf",
                "embedding_model": "qwen3-embedding-8b-q4_k_m.gguf",
                "vision_model": "Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf",
                "mmproj_model": "mmproj-F16.gguf",
                "chat_system_prompt": "You are Persona, a helpful AI assistant...",
                "tag_prompt": "List relevant tags for this image...",
                "describe_prompt": "Describe this image in detail..."
            }
        }
    
    def get_image_scale(self) -> float:
        """Get image scale multiplier based on image quality setting.
        
        Returns:
            float: Scale multiplier (1.0 = original, <1.0 = reduced)
        """
        return self.image_quality
    
    def get_rag_directory(self) -> Optional[Path]:
        """Get full path to RAG directory."""
        if not self.storage_metadata_path:
            return None
        metadata_path = Path(self.storage_metadata_path)
        return metadata_path.parent / self.rag_directory_name
    
    def get_binary_path(self, binary_name: str) -> Path:
        """Get absolute path to a specific binary.
        
        Uses PyInstaller-compatible resource path resolution.
        """
        return get_binary_resource_path(binary_name)
    
    def get_model_path(self, model_name: str) -> Path:
        """Get absolute path to a specific model.
        
        Uses PyInstaller-compatible resource path resolution.
        """
        return get_model_resource_path(model_name)
    
    def get_vision_binary(self, model_name: str) -> str:
        """Determine which vision binary to use based on model name."""
        if self.vision_binary != "auto":
            return self.vision_binary
        
        # Auto-detect based on model name
        model_lower = model_name.lower()
        # if "qwen2.5-vl" in model_lower or "ggml" in model_lower:
        return "llama-mtmd-cli"
        # else:
        #     return "llama-cli"


# Global configuration instance
_config = ServerConfig()


def get_config() -> ServerConfig:
    """Get the global configuration instance."""
    return _config


def update_config(**kwargs) -> ServerConfig:
    """Update the global configuration."""
    global _config
    
    # Filter out read-only fields
    read_only_fields = {"rag_directory_name", "storage_metadata_path", "binary_dir", "model_dir"}
    editable_kwargs = {k: v for k, v in kwargs.items() if k not in read_only_fields}
    
    # Handle nested llm_params update
    if "llm_params" in editable_kwargs:
        current_llm_params = _config.llm_params.model_dump()
        current_llm_params.update(editable_kwargs["llm_params"])
        editable_kwargs["llm_params"] = LLMParams(**current_llm_params)
    
    # Update configuration
    for key, value in editable_kwargs.items():
        if hasattr(_config, key):
            setattr(_config, key, value)
    
    return _config


def set_storage_metadata_path(path: str) -> None:
    """Set the storage metadata path (internal use only)."""
    global _config
    _config.storage_metadata_path = path
