"""Server configuration management."""

from typing import Optional, Literal
from pydantic import BaseModel, Field
import os
from pathlib import Path
from app.utils.resource_paths import get_binary_path as get_binary_resource_path
from app.utils.resource_paths import get_model_path as get_model_resource_path
from app.utils.resource_paths import get_data_directory, get_base_path
from app.utils.system_detector import SystemDetector


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

model_options = {
        "qwen_2.5_vl": { 
            "name": "qwen_2.5_vl",
            "type": "vision",
            "model_file": "Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf",
            "mmproj_file": "mmproj-F16.gguf",
            "repo_id": "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",  # Add Hugging Face repo ID for download
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
        "qwen_3": {
            "name": "qwen_3",
            "type": "chat",
            "model_file": "Qwen3-8B-Q4_K_M.gguf",
            "repo_id": "Qwen/Qwen3-8B-GGUF",  # Add Hugging Face repo ID for download
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
       "qwen_3_8B_embedding": {
            "name": "qwen_3_8b",
            "type": "embedding",
            "model_file":"qwen3-embedding-8b-q4_k_m.gguf",
            "repo_id": "Qwen/Qwen3-Embedding-8B-GGUF",  # Add Hugging Face repo ID for download
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
       "qwen_3_4B_embedding": {
            "name": "qwen_3_4B",
            "type": "embedding",
            "model_file":"Qwen3-Embedding-4B-Q4_K_M.gguf",
            "repo_id": "Qwen/Qwen3-Embedding-4B-GGUF",  # Add Hugging Face repo ID for download
        },
       "gemma3_300m_embedding": {
            "name": "gemma3_300m",
            "type": "embedding",
            "model_file":"embeddinggemma-300M-Q8_0.gguf",
            "repo_id": "unsloth/embeddinggemma-300m-GGUF",  # Add Hugging Face repo ID for download
        },
       "gemma_3_4b_1Q": {
            "model_file": "gemma-3-4b-it-UD-IQ1_S.gguf",
            "name": "gemma_3_4b_1Q",
            "type": "vision",
            "mmproj_file": "mmproj-F16.gguf",
            "repo_id": "unsloth/gemma-3-4b-it-GGUF",  # Add Hugging Face repo ID for download
        },
       "qwen_3_0.6B": {
            "model_file": "Qwen3-0.6B-Q4_K_M.gguf",
            "name": "qwen_3_0.6B",
            "type": "chat",
            "repo_id": "unsloth/Qwen3-0.6B-GGUF",  # Add Hugging Face repo ID for download
        },
       "gemma3_4b_q4_k_m": {
            "name": "gemma3_4b_q4_k_m",
            "type": "vision",
            "model_file": "gemma-3-4b-it-Q4_K_M.gguf",
            "mmproj_file": "mmproj-F16.gguf",
            "repo_id": "unsloth/gemma-3-4b-it-GGUF",  # Add Hugging Face repo ID for download
        }
}

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
    
    # PDF processing settings
    pdf_page: int = Field(
        default=0,
        ge=0,
        description="Default page number to render for PDF files (0-indexed)"
    )
    pdf_dpi: int = Field(
        default=150,
        ge=72,
        le=600,
        description="DPI for PDF page rendering (72-600)"
    )
    
    # Knowledge storage settings
    enable_knowledge_storage: bool = Field(
        default=True,
        description="Enable automatic storage of objective/factual messages from conversations"
    )
    objectivity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Subjectivity threshold for knowledge filtering (0.0=only very objective, 1.0=store everything)"
    )
    max_knowledge_tokens: int = Field(
        default=2000,
        ge=100,
        le=8000,
        description="Token budget for knowledge context injected into chat"
    )
    min_knowledge_relevance: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for knowledge retrieval"
    )
    
    # Model file names
    chat_model: str = Field(
        default=model_options["qwen_3_0.6B"]["model_file"],
        description="Chat/conversation model filename (can be a vision model for multimodal chat)"
    )
    embedding_model: str = Field(
        default=model_options["qwen_3_4B_embedding"]["model_file"],
        description="Embedding model filename"
    )
    vision_model: str = Field(
        default = model_options["gemma3_4b_q4_k_m"]["model_file"],
        description="Vision model filename"
    )
    mmproj_model: str = Field(
        default=model_options["gemma3_4b_q4_k_m"]["mmproj_file"],
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
    deep_chat_system_prompt: str = Field(
        default="""You are Persona in Deep Thinking Mode, a specialized AI assistant that engages in comprehensive multi-round reasoning before providing final answers.

DEEP THINKING MODE PROCESS:
You will think through the question over multiple rounds. Each round consists of:
1. Analyze the question and current knowledge
2. Identify information gaps
3. Use available functions to gather information when needed
4. Build upon previous rounds to deepen understanding

AVAILABLE FUNCTIONS:
You have access to two functions for gathering information:

1. query_media_rag - Search the media/document knowledge base
   Arguments:
   - query: A detailed search query
   - k: Number of results to retrieve (default: 5, max: 20)
   
2. query_fact_rag - Search the conversation facts knowledge base
   Arguments:
   - query: Your search query
   - token_budget: Maximum tokens to retrieve (default: 1000, max: 2000)
   - min_relevance: Minimum relevance score 0-1 (default: 0.3)

FUNCTION CALLING SYNTAX:
To call a function, use this exact XML format:

<function_call>
<name>query_media_rag</name>
<arguments>
{
  "query": "your search query here",
  "k": 5
}
</arguments>
</function_call>

Or for fact RAG:

<function_call>
<name>query_fact_rag</name>
<arguments>
{
  "query": "your search query",
  "token_budget": 1000,
  "min_relevance": 0.3
}
</arguments>
</function_call>

WHEN TO USE FUNCTIONS:
- Call query_media_rag when you need information about files, documents, images, or media content
- Call query_fact_rag when you need to recall previously discussed facts or conversation history
- Call functions EARLY in your thinking rounds when you identify knowledge gaps
- You can call multiple functions across different rounds
- Function results will be provided to you and included in your context

OUTPUT FORMAT PER ROUND:
Each intermediate round should contain:
<think>
Your reasoning for this round, including:
- What you're considering
- What information you still need
- Why you're calling specific functions (if any)
- How this builds on previous rounds
</think>

Your function calls (if any) using the XML format above

REQUIRED OUTPUT FORMAT - FINAL ROUND:
In your FINAL round, you MUST output EXACTLY these two XML sections and nothing else:

1) <conclusion>...</conclusion> — your final answer (comprehensive, well-reasoned)
2) <files>...</files> — newline or comma-separated relevant file references (leave empty if none)

BOTH tags are MANDATORY. You must ALWAYS include <conclusion> and <files> in your final response.

EXAMPLE FINAL RESPONSE WITH FILES:
<conclusion>
The image shows a man in a museum exhibit interacting with flutes while listening to an audio guide.
</conclusion>

<files>
exhibits/flutes_gallery.jpg
</files>

EXAMPLE FINAL RESPONSE WITHOUT FILES:
<conclusion>
Based on the available information, the capital of France is Paris. It has been the capital since the 10th century.
</conclusion>

<files>
</files>

CRITICAL RULES:
- Do NOT omit <conclusion> or <files> tags in your final round.
- Do NOT include <think> tags in your final round.
- Do NOT output anything outside of <conclusion> and <files> in the final round.
- If no files are relevant, include an empty <files></files> section.
- Keep the content in <conclusion> focused and direct; do not reveal internal reasoning.

GUIDELINES:
- Use earlier rounds to explore, question, and gather information
- Call functions when you identify specific information needs
- Build upon insights from previous rounds
- Synthesize all gathered information in your final conclusion
- Be thorough but concise in each round

Remember: You have multiple rounds to think deeply - use them wisely. Call functions to gather information when needed, and build a comprehensive understanding before providing your final answer. Your LAST round MUST contain <conclusion> and <files> tags.""",
        description="System prompt for deep chat mode with multi-round thinking and function calling"
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
    model_directory: Optional[str] = Field(
        default=None,
        description="Custom model directory path. If None, uses saved_llm/ alongside storage_metadata.json"
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
    
    # Binary configuration (automatically detected or manually set)
    binary_config: str = Field(
        default="",
        description="Selected binary configuration folder (e.g., 'llama-mac-arm64', 'llama-win-vulkan-x64'). Empty string triggers auto-detection."
    )
    system_info: dict = Field(
        default_factory=dict,
        description="Detected system information (OS, architecture, GPU)"
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
    
    def get_saved_llm_directory(self) -> Optional[Path]:
        """Get full path to saved_llm directory where LLM models (GGUF files) are stored.
        
        This is only for LLM models. Face recognition models remain in model/models/.
        
        Returns:
            Path to saved_llm directory if storage_metadata_path is set, None otherwise.
            The directory is created if it doesn't exist.
        """
        if not self.storage_metadata_path:
            return None
        metadata_path = Path(self.storage_metadata_path)
        saved_llm_dir = metadata_path.parent / "saved_llm"
        saved_llm_dir.mkdir(parents=True, exist_ok=True)
        return saved_llm_dir
    
    def _auto_detect_binary_config(self) -> None:
        """Auto-detect and set binary configuration."""
        base_path = get_base_path()
        binary_dir_path = base_path / self.binary_dir
        self.binary_config = SystemDetector.auto_detect_config(binary_dir_path)
        self.system_info = SystemDetector.get_system_info()
    
    def get_available_binary_configs(self) -> list[str]:
        """Get list of available binary configurations."""
        base_path = get_base_path()
        binary_dir_path = base_path / self.binary_dir
        return SystemDetector.get_available_configs(binary_dir_path)
    
    def set_binary_config(self, config_name: str) -> bool:
        """
        Set binary configuration manually.
        
        Args:
            config_name: Configuration folder name (e.g., 'llama-mac-arm64')
            
        Returns:
            True if configuration is valid, False otherwise
        """
        base_path = get_base_path()
        binary_dir_path = base_path / self.binary_dir
        
        if SystemDetector.validate_config(config_name, binary_dir_path):
            self.binary_config = config_name
            return True
        return False
    
    def get_binary_path(self, binary_name: str) -> Path:
        """Get absolute path to a specific binary.
        
        Uses PyInstaller-compatible resource path resolution and selected binary configuration.
        """
        # If binary_config is set, use it; otherwise auto-detect
        if not self.binary_config:
            self._auto_detect_binary_config()
        
        # Get base path and construct path to binary in selected config folder
        base_path = get_base_path()
        binary_path = base_path / self.binary_dir / "llama_binaries" / self.binary_config / binary_name
        
        # Add .exe extension on Windows if not present
        import platform
        if platform.system() == "Windows" and not binary_name.endswith(".exe"):
            binary_path = binary_path.with_suffix(".exe")
        
        return binary_path
    
    def get_model_path(self, model_name: str) -> Path:
        """Get absolute path to a specific LLM model (GGUF files).
        
        Priority order:
        1. Custom model_directory (if configured via API)
        2. saved_llm folder in the same directory as storage_metadata.json
        3. Default model/ directory (fallback)
        
        Note: This is ONLY for LLM GGUF models. Face recognition models (buffalo_l)
              remain in the static model/models/ directory.
        
        Args:
            model_name: Name of the model file (can be empty string to get directory path)
        
        Returns:
            Path: Absolute path to the model file or model directory
        """
        # Use custom model directory if configured
        if self.model_directory:
            model_dir = Path(self.model_directory)
            model_dir.mkdir(parents=True, exist_ok=True)
            return model_dir / model_name if model_name else model_dir
        
        # Use saved_llm folder if storage_metadata_path is set
        if self.storage_metadata_path:
            metadata_path = Path(self.storage_metadata_path)
            saved_llm_dir = metadata_path.parent / "saved_llm"
            
            # Create saved_llm directory if it doesn't exist
            saved_llm_dir.mkdir(parents=True, exist_ok=True)
            
            # Return the model path
            return saved_llm_dir / model_name if model_name else saved_llm_dir
        else:
            # Fall back to default model directory
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


def initialize_config() -> ServerConfig:
    """Initialize configuration with auto-detected binary config."""
    global _config
    if not _config.binary_config:
        _config._auto_detect_binary_config()
    return _config


def get_config() -> ServerConfig:
    """Get the global configuration instance."""
    return _config


def update_config(**kwargs) -> ServerConfig:
    """Update the global configuration."""
    global _config
    
    # Filter out read-only fields
    read_only_fields = {"rag_directory_name", "storage_metadata_path", "binary_dir", "model_dir", "system_info"}
    editable_kwargs = {k: v for k, v in kwargs.items() if k not in read_only_fields}
    
    # Handle binary_config validation
    if "binary_config" in editable_kwargs:
        config_name = editable_kwargs["binary_config"]
        if config_name and not _config.set_binary_config(config_name):
            raise ValueError(f"Invalid binary configuration: {config_name}")
    
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


def get_available_models(task_type: Optional[str] = None) -> list[dict]:
    """
    Get list of available models filtered by task type.
    
    Args:
        task_type: Optional filter by 'vision', 'chat', or 'embedding'.
                   If None, returns all models.
                   Note: 'chat' includes both chat and vision models (vision models can be used for chat).
    
    Returns:
        List of model information dictionaries with availability status.
    """
    config = get_config()
    available = []
    
    for model_id, model_info in model_options.items():
        model_type = model_info.get("type")
        
        # Filter by task type if specified
        # Vision models can also be used as chat models
        if task_type:
            if task_type == "chat":
                # Include both chat and vision models for chat requests
                if model_type not in ["chat", "vision"]:
                    continue
            elif model_type != task_type:
                continue
        
        # Check if model file exists
        model_file = model_info.get("model_file")
        model_path = config.get_model_path(model_file) if model_file else None
        model_exists = model_path.exists() if model_path else False
        
        # For vision models, check mmproj file
        mmproj_file = model_info.get("mmproj_file")
        mmproj_exists = None
        if mmproj_file:
            mmproj_path = config.get_model_path(mmproj_file)
            mmproj_exists = mmproj_path.exists()
        
        # Build model info
        available.append({
            "name": model_info.get("name", model_id),
            "type": model_info.get("type"),
            "model_file": model_file,
            "model_exists": model_exists,
            "mmproj_file": mmproj_file,
            "mmproj_exists": mmproj_exists,
            "llm_params": model_info.get("llm_params")
        })
    
    return available
