"""LLM service for interacting with llama-server and llama-cli."""

from typing import List, Dict, Any, Optional, AsyncGenerator
from pathlib import Path
import asyncio
import json
import platform
import subprocess
import aiohttp
from abc import ABC, abstractmethod

from app.config import get_config
from app.utils import get_process_manager

# Global lock to ensure only one model is active at a time
_model_lock = asyncio.Lock()

# Windows NTSTATUS 0xC0000135 — the OS could not locate a required DLL.
# GPU-accelerated llama.cpp builds link against external runtime DLLs
# (e.g. vulkan-1.dll, CUDA/HIP/SYCL runtimes).  When those DLLs are absent
# the process exits immediately with this code.
# Python exposes the raw unsigned DWORD, so it appears as 3221225781.
_WIN_STATUS_DLL_NOT_FOUND = 0xC0000135  # 3221225781

_WIN_DLL_NOT_FOUND_HINT = (
    "\nHint: Exit code 0xC0000135 (STATUS_DLL_NOT_FOUND) means "
    "the llama-server binary could not find a required DLL "
    "(e.g. vulkan-1.dll, CUDA/HIP/SYCL runtime DLLs). "
    "The selected binary configuration may require GPU runtime "
    "libraries that are not installed on this machine. "
    "Install the required GPU drivers/runtime or ensure the "
    "CPU-only binary configuration is selected instead."
)

# On Windows, suppress console window popups for spawned subprocesses.
# CREATE_NO_WINDOW prevents console creation; STARTUPINFO+SW_HIDE hides
# any window the child process may try to show.
# IMPORTANT: Returns a FRESH STARTUPINFO each call because subprocess.Popen
# mutates the object in-place (adds STARTF_USESTDHANDLES, sets pipe handles).
def _win_subprocess_kwargs() -> Dict[str, Any]:
    """Return kwargs for subprocess creation that suppress console windows on Windows."""
    if platform.system() != "Windows":
        return {}
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = subprocess.SW_HIDE
    return {
        "creationflags": subprocess.CREATE_NO_WINDOW,
        "startupinfo": si,
    }


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    async def start(self, model_path: Path, **kwargs) -> None:
        """Start the LLM backend."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the LLM backend."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate response from LLM."""
        pass
    
    async def generate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response with OpenAI-format tool calling.
        
        Returns the full choices[0].message dict which may contain
        'content', 'tool_calls', or both.
        """
        raise NotImplementedError("Tool calling not supported by this backend")
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        pass
    
    @abstractmethod
    async def generate_vision(
        self,
        image_bytes: bytes,
        prompt: str,
        mmproj_file: Optional[str] = None
    ) -> str:
        """Generate response from vision model with image input."""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if backend is running."""
        pass


class LlamaServerBackend(LLMBackend):
    """Llama-server backend implementation."""
    
    def __init__(self):
        """Initialize llama-server backend."""
        self.process_manager = get_process_manager()
        self.base_url = "http://localhost:8100"
        self.process_name = "llama-server"
        self._model_loaded = False
        self.startup_command: Optional[List[str]] = None
    
    async def start(self, model_path: Path, **kwargs) -> None:
        """Start llama-server."""
        config = get_config()
        binary_path = config.get_binary_path("llama-server")
        
        if not binary_path.exists():
            raise FileNotFoundError(f"llama-server binary not found: {binary_path}")
        
        # Kill any existing llama-server processes
        await self.process_manager.kill_existing_binary_processes("llama-server")
        
        # Build command
        command = [
            str(binary_path),
            "--model", str(model_path),
            "--ctx-size", str(config.llm_params.ctx_size),
            "--batch-size", str(config.llm_params.batch_size),
            "--ubatch-size", str(config.llm_params.ubatch_size),
            "--n-gpu-layers", str(config.llm_params.n_gpu_layers),
            "--port", "8100",
            "--host", "127.0.0.1",
            "--embeddings",  # Enable embeddings support
            "--jinja"  # Enable Jinja templates for proper tool-calling support
        ]
        
        # Add mmproj if provided
        if "mmproj" in kwargs:
            mmproj_path = config.get_model_path(kwargs["mmproj"])
            if mmproj_path.exists():
                command.extend(["--mmproj", str(mmproj_path)])

        # Enable thinking mode if requested (deep chat only)
        if kwargs.get("enable_thinking", False):
            command.extend(["--chat-template-kwargs", '{"enable_thinking":true}'])

        # Add any additional arguments
        for key, value in kwargs.items():
            if key.startswith("--") and key != "--mmproj":
                command.extend([key, str(value)])
        
        # Store command for reference
        self.startup_command = command
        
        # Start server
        await self.process_manager.start_process(
            self.process_name,
            command
        )
        
        # Wait for server to be ready
        await self._wait_for_server()
        self._model_loaded = True
    
    def get_startup_command(self) -> Optional[str]:
        """Get the command used to start llama-server."""
        if self.startup_command:
            return " ".join(self.startup_command)
        return None
    
    async def _wait_for_server(self, timeout: int = 60) -> None:
        """Wait for server to be ready.
        
        Polls the /health endpoint and checks that the subprocess is still alive.
        If the process dies before becoming healthy, the actual stderr is surfaced.
        """
        start_time = asyncio.get_event_loop().time()
        
        while True:
            # --- Check if the process crashed before becoming healthy ---
            proc = self.process_manager.active_processes.get(self.process_name)
            if proc is not None and proc.poll() is not None:
                # Process exited — collect stderr for diagnostics
                exit_code = proc.returncode
                stderr_output = ""
                try:
                    _, stderr_output = proc.communicate(timeout=5)
                except Exception:
                    stderr_output = "(unable to read stderr)"

                # Provide a targeted explanation for the missing-DLL failure
                # that is commonly seen with GPU-accelerated builds on Windows.
                extra_hint = ""
                if platform.system() == "Windows":
                    exit_code_u32 = exit_code & 0xFFFFFFFF
                    if exit_code_u32 == _WIN_STATUS_DLL_NOT_FOUND:
                        extra_hint = _WIN_DLL_NOT_FOUND_HINT

                raise RuntimeError(
                    f"llama-server exited immediately with code {exit_code}.\n"
                    f"Command: {self.get_startup_command()}\n"
                    f"Stderr: {stderr_output[-2000:] if stderr_output else '(empty)'}"
                    f"{extra_hint}"
                )
            
            # --- Poll health endpoint ---
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/health") as response:
                        if response.status == 200:
                            return
            except Exception:
                pass
            
            if asyncio.get_event_loop().time() - start_time > timeout:
                # Timeout — try to capture any output from the still-running process
                stderr_hint = ""
                if proc is not None:
                    try:
                        proc.kill()
                        _, stderr_hint = proc.communicate(timeout=5)
                    except Exception:
                        pass
                raise TimeoutError(
                    f"Llama-server failed to start within {timeout}s.\n"
                    f"Command: {self.get_startup_command()}\n"
                    f"Stderr: {stderr_hint[-2000:] if stderr_hint else '(empty)'}"
                )
            
            await asyncio.sleep(1)
    
    async def stop(self) -> None:
        """Stop llama-server."""
        await self.process_manager.kill_process(self.process_name)
        self._model_loaded = False
    
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate response from llama-server."""
        if not self.is_running():
            raise RuntimeError("Llama-server is not running")
        
        config = get_config()
        
        payload = {
            "messages": messages,
            "stream": stream,
            "temperature": config.llm_params.temp,
            "top_p": config.llm_params.top_p,
            "top_k": config.llm_params.top_k,
            "presence_penalty": config.llm_params.presence_penalty,
            "mirostat": config.llm_params.mirostat,
            **kwargs
        }
        
        # Create timeout (5 minutes for generation)
        timeout = aiohttp.ClientTimeout(total=300)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            ) as response:
                if stream:
                    async for line in response.content:
                        line_text = line.decode('utf-8').strip()
                        if line_text.startswith("data: "):
                            data_str = line_text[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
                else:
                    data = await response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0]["message"].get("content", "")
                        if content:
                            yield content
                    else:
                        import logging
                        logging.getLogger("app.services.llm_service").error(f"Generate returned no choices: {data}")
    
    async def generate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response with OpenAI-format tool calling via llama-server.
        
        Non-streaming — tool call responses are structured JSON.
        Returns the full choices[0].message dict.
        """
        if not self.is_running():
            raise RuntimeError("Llama-server is not running")
        
        config = get_config()
        
        payload = {
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "stream": False,
            "temperature": config.llm_params.temp,
            "top_p": config.llm_params.top_p,
            "top_k": config.llm_params.top_k,
            "presence_penalty": config.llm_params.presence_penalty,
            "mirostat": config.llm_params.mirostat,
            **kwargs
        }
        
        timeout = aiohttp.ClientTimeout(total=300)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            ) as response:
                data = await response.json()
                if "error" in data:
                    error_msg = data.get("error", {})
                    if isinstance(error_msg, dict):
                        error_text = error_msg.get("message", str(error_msg))
                    else:
                        error_text = str(error_msg)
                    raise RuntimeError(f"Llama-server tool calling error: {error_text}")
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]
                raise RuntimeError(f"Unexpected tool calling response: {data}")
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding using llama-server."""
        if not self.is_running():
            raise RuntimeError("Llama-server is not running")
        
        # Create timeout (60 seconds for embeddings)
        timeout = aiohttp.ClientTimeout(total=60)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.base_url}/v1/embeddings",
                json={"input": text}
            ) as response:
                data = await response.json()
                
                # Check for error response from llama-server
                if "error" in data:
                    error_msg = data.get("error", {})
                    if isinstance(error_msg, dict):
                        error_text = error_msg.get("message", str(error_msg))
                    else:
                        error_text = str(error_msg)
                    raise RuntimeError(f"Llama-server error: {error_text}")
                
                # Handle different response formats
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0]["embedding"]
                elif "embedding" in data:
                    return data["embedding"]
                elif "embeddings" in data and len(data["embeddings"]) > 0:
                    return data["embeddings"][0]
                else:
                    raise ValueError(f"Unexpected embedding response format: {list(data.keys())}. Response: {data}")
    
    async def generate_vision(
        self,
        image_bytes: bytes,
        prompt: str,
        mmproj_file: Optional[str] = None
    ) -> str:
        """Generate response from vision model using llama-server with base64 image."""
        if not self.is_running():
            raise RuntimeError("Llama-server is not running")
        
        import base64
        
        # Convert image to base64 data URL
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{base64_image}"
        
        config = get_config()
        
        # Build multimodal message payload
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url}
                        }
                    ]
                }
            ],
            "slot_id": -1,  # Use ephemeral slot to avoid cache storage
            "temperature": config.llm_params.temp,
            "top_p": config.llm_params.top_p,
            "top_k": config.llm_params.top_k,
            "presence_penalty": config.llm_params.presence_penalty,
            "mirostat": config.llm_params.mirostat
        }
        
        # Create timeout for vision requests (5 minutes)
        timeout = aiohttp.ClientTimeout(total=300)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            ) as response:
                data = await response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                raise RuntimeError(f"Vision generation failed: {data}")
    
    def is_running(self) -> bool:
        """Check if llama-server is running."""
        return self.process_manager.is_process_running(self.process_name) and self._model_loaded


class LlamaCLIBackend(LLMBackend):
    """Llama-cli backend implementation."""
    
    def __init__(self):
        """Initialize llama-cli backend."""
        self.process_manager = get_process_manager()
        self.model_path: Optional[Path] = None
        self._binary_path: Optional[Path] = None
        self.startup_command: Optional[List[str]] = None
        self._embed_binary_path: Optional[Path] = None
        self._mmproj_path: Optional[Path] = None
        self._vision_binary_path: Optional[Path] = None
        self._current_process: Optional[asyncio.subprocess.Process] = None
        self._process_lock = asyncio.Lock()
    async def start(self, model_path: Path, **kwargs) -> None:
        """Prepare llama-cli (no persistent process)."""
        config = get_config()
        self._vision_binary_path = config.get_binary_path("llama-mtmd-cli")
        self._binary_path = config.get_binary_path("llama-cli")
        self._embed_binary_path = config.get_binary_path("llama-embedding")  # Use llama-cli for embeddings
        if not self._binary_path.exists():
            raise FileNotFoundError(f"llama-cli binary not found: {self._binary_path}")
        
        # Kill any existing llama-cli processes
        await self.process_manager.kill_existing_binary_processes("llama-cli")
        
        self.model_path = model_path
        
        # Store mmproj if provided
        if "mmproj" in kwargs:
            mmproj_path = config.get_model_path(kwargs["mmproj"])
            if mmproj_path.exists():
                self._mmproj_path = mmproj_path
        
        # Build example startup command
        command = [
            str(self._binary_path),
            "--model", str(model_path),
            "--ctx-size", str(config.llm_params.ctx_size),
            "--temp", str(config.llm_params.temp),
            "--top-p", str(config.llm_params.top_p),
            "--top-k", str(config.llm_params.top_k),
            "--presence-penalty", str(config.llm_params.presence_penalty),
            "--mirostat", str(config.llm_params.mirostat),
            "--batch-size", str(config.llm_params.batch_size),
            "--ubatch-size", str(config.llm_params.ubatch_size),
            "--n-gpu-layers", str(config.llm_params.n_gpu_layers)
        ]
        print("Running LLM CLI Command", command)
        # Add mmproj if stored
        if self._mmproj_path:
            command.extend(["--mmproj", str(self._mmproj_path)])
        
        self.startup_command = command
    
    def get_startup_command(self) -> Optional[str]:
        """Get the example command for llama-cli."""
        if self.startup_command:
            return " ".join(self.startup_command)
        return None
    
    async def stop(self) -> None:
        """Stop CLI backend and cancel any running process."""
        await self.cancel_generation()
        self.model_path = None
    
    async def cancel_generation(self) -> None:
        """Cancel any running generation process."""
        async with self._process_lock:
            if self._current_process is not None:
                try:
                    # Try graceful termination first
                    self._current_process.terminate()
                    try:
                        await asyncio.wait_for(self._current_process.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        # Force kill if graceful termination times out
                        self._current_process.kill()
                        await self._current_process.wait()
                except ProcessLookupError:
                    # Process already terminated
                    pass
                except Exception as e:
                    print(f"Error cancelling CLI process: {e}")
                finally:
                    self._current_process = None
    
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate response using llama-cli."""
        if not self.model_path:
            raise RuntimeError("Model not loaded")
        
        config = get_config()
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        # Build command
        command = [
            str(self._binary_path),
            "--model", str(self.model_path),
            "--prompt", prompt,
            "--ctx-size", str(config.llm_params.ctx_size),
            "--temp", str(config.llm_params.temp),
            "--top-p", str(config.llm_params.top_p),
            "--top-k", str(config.llm_params.top_k),
            "--presence-penalty", str(config.llm_params.presence_penalty),
            "--mirostat", str(config.llm_params.mirostat),
            "--batch-size", str(config.llm_params.batch_size),
            "--ubatch-size", str(config.llm_params.ubatch_size),
            "--n-gpu-layers", str(config.llm_params.n_gpu_layers)
        ]
        
        # Add mmproj if stored
        if self._mmproj_path:
            command.extend(["--mmproj", str(self._mmproj_path)])
        
        # Run process with tracking
        async with self._process_lock:
            self._current_process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **_win_subprocess_kwargs()
            )
            process = self._current_process
        
        try:
            # Read output
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                # Check if it was cancelled (negative return codes indicate signals)
                if process.returncode < 0:
                    raise asyncio.CancelledError("Generation was cancelled")
                raise RuntimeError(f"llama-cli failed: {stderr.decode()}")
            
            # Yield full response
            yield stdout.decode().strip()
        finally:
            # Clean up process reference
            async with self._process_lock:
                if self._current_process == process:
                    self._current_process = None
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding using llama-cli with embedding flag."""
        if not self.model_path:
            raise RuntimeError("Model not loaded")
        # Ensure embed binary path is available (start() normally sets this)
        config = get_config()
        if not self._embed_binary_path:
            self._embed_binary_path = config.get_binary_path("llama-cli")  # Use llama-cli for embeddings

        command = [
            str(self._embed_binary_path),
            "--model", str(self.model_path),
            # "--embedding",  # Add embedding flag
            "--prompt", text
        ]
        
        async with self._process_lock:
            self._current_process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **_win_subprocess_kwargs()
            )
            process = self._current_process
        
        try:
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                # Check if it was cancelled
                if process.returncode < 0:
                    raise asyncio.CancelledError("Embedding generation was cancelled")
                raise RuntimeError(f"llama-cli embedding failed: {stderr.decode()}")
        finally:
            async with self._process_lock:
                if self._current_process == process:
                    self._current_process = None

        # Parse embedding from output (support multiple formats)
        output = stdout.decode().strip()

        # 1) Try JSON first
        try:
            embedding = json.loads(output)
            # If top-level is a dict with 'data' or 'embedding', normalize it
            if isinstance(embedding, dict):
                if 'data' in embedding and len(embedding['data']) > 0:
                    return embedding['data'][0].get('embedding', embedding['data'][0])
                if 'embedding' in embedding:
                    return embedding['embedding']
                if 'embeddings' in embedding and len(embedding['embeddings']) > 0:
                    return embedding['embeddings'][0]
                # Unexpected dict shape; fallthrough to other parsing
            elif isinstance(embedding, list):
                return embedding
        except Exception:
            # Not JSON or different structure; try textual parsing below
            pass

        # 2) Try to find a JSON array substring anywhere in output (e.g., [0.1, 0.2])
        import re
        json_array_match = re.search(r'\[[\s\d+\-eE.,]+\]', output)
        if json_array_match:
            try:
                return json.loads(json_array_match.group(0))
            except Exception:
                # continue to other heuristics
                pass

        # 3) Look for lines like: 'embedding 0:  0.001452  0.001441  0.017390 ...'
        # Extract numbers from the first such line (or entire output if no prefix)
        # Find lines that contain the word 'embedding' and numbers
        for line in output.splitlines():
            if 'embedding' in line.lower():
                # Remove the leading 'embedding ...:' prefix
                parts = re.split(r'embedding\s*\d*\s*:\s*', line, flags=re.IGNORECASE)
                numeric_part = parts[-1] if parts else line
                nums = re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', numeric_part)
                if nums:
                    return [float(x) for x in nums]

        # 4) As a last resort, extract all floats from the entire output
        nums = re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', output)
        if nums:
            return [float(x) for x in nums]

        # If nothing parsed, raise with helpful diagnostics
        raise ValueError(f"Failed to parse embedding from output: {output}")
    
    async def generate_vision(
        self,
        image_bytes: bytes,
        prompt: str,
        mmproj_file: Optional[str] = None
    ) -> str:
        """Generate response from vision model using vision binaries."""
        if not self.model_path:
            raise RuntimeError("Model not loaded")
        
        config = get_config()
        
        # Determine which vision binary to use
        model_name = self.model_path.name
        binary_name = config.get_vision_binary(model_name)
        binary_path = config.get_binary_path(binary_name)
        
        if not binary_path.exists():
            raise FileNotFoundError(f"Vision binary not found: {binary_path}")
        
        # Save image to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = Path(tmp.name)
        
        try:
            # Build command with all LLM parameters
            command = [
                str(binary_path),
                "--model", str(self.model_path),
                "--image", str(tmp_path),
                "--prompt", prompt,
                "--ctx-size", str(config.llm_params.ctx_size),
                "--temp", str(config.llm_params.temp),
                "--top-p", str(config.llm_params.top_p),
                "--top-k", str(config.llm_params.top_k),
                "--presence-penalty", str(config.llm_params.presence_penalty),
                "--mirostat", str(config.llm_params.mirostat),
                "--batch-size", str(config.llm_params.batch_size),
                "--ubatch-size", str(config.llm_params.ubatch_size),
                "--n-gpu-layers", str(config.llm_params.n_gpu_layers)
            ]
            
            # Add mmproj if specified
            if mmproj_file:
                mmproj_path = config.get_model_path(mmproj_file)
                if mmproj_path.exists():
                    command.extend(["--mmproj", str(mmproj_path)])
            
            # Execute command with tracking
            async with self._process_lock:
                self._current_process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **_win_subprocess_kwargs()
                )
                process = self._current_process
            
            try:
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    # Check if it was cancelled
                    if process.returncode < 0:
                        raise asyncio.CancelledError("Vision generation was cancelled")
                    raise RuntimeError(f"Vision model failed: {stderr.decode()}")
                
                return stdout.decode().strip()
            finally:
                async with self._process_lock:
                    if self._current_process == process:
                        self._current_process = None
            
        finally:
            # Clean up temporary file
            tmp_path.unlink()
    
    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert OpenAI messages format to prompt string."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    def is_running(self) -> bool:
        """CLI is always 'ready' if model is set."""
        return self.model_path is not None


class LLMService:
    """Main LLM service that manages backend selection."""
    
    def __init__(self):
        """Initialize LLM service."""
        self.server_backend = LlamaServerBackend()
        self.cli_backend = LlamaCLIBackend()
        self.current_backend: Optional[LLMBackend] = None
        self.current_model: Optional[str] = None
        self.current_kwargs: Dict[str, Any] = {}  # Store kwargs for model restoration
    
    async def load_model(
        self,
        model_name: str,
        use_server: Optional[bool] = None,
        **kwargs
    ) -> None:
        """
        Load a model with specified backend.
        Ensures only one model is active at a time globally.
        
        Args:
            model_name: Name of model file in model directory
            use_server: Use server mode (True) or CLI mode (False), None for config default
            **kwargs: Additional arguments for model loading
        """
        async with _model_lock:
            # Stop current backend if running
            if self.current_backend:
                await self.current_backend.stop()
            
            config = get_config()
            model_path = config.get_model_path(model_name)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            # Select backend
            if use_server is None:
                use_server = config.llm_mode == "server"
            
            backend = self.server_backend if use_server else self.cli_backend
            
            # Start backend
            await backend.start(model_path, **kwargs)
            
            self.current_backend = backend
            self.current_model = model_name
            self.current_kwargs = kwargs  # Store kwargs for restoration
    
    async def unload_model(self) -> None:
        """Unload current model."""
        async with _model_lock:
            if self.current_backend:
                await self.current_backend.stop()
                self.current_backend = None
                self.current_model = None
    
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate response from current model."""
        if not self.current_backend:
            raise RuntimeError("No model loaded")
        
        async for chunk in self.current_backend.generate(messages, stream, **kwargs):
            yield chunk
    
    async def generate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response with tool calling from current model."""
        if not self.current_backend:
            raise RuntimeError("No model loaded")
        
        return await self.current_backend.generate_with_tools(messages, tools, tool_choice, **kwargs)
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding from current model."""
        if not self.current_backend:
            raise RuntimeError("No model loaded")
        
        return await self.current_backend.embed(text)
    
    async def generate_vision(
        self,
        image_bytes: bytes,
        prompt: str,
        mmproj_file: Optional[str] = None
    ) -> str:
        """Generate response from vision model with image input."""
        if not self.current_backend:
            raise RuntimeError("No model loaded")
        
        return await self.current_backend.generate_vision(image_bytes, prompt, mmproj_file)
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.current_backend is not None and self.current_backend.is_running()
    
    def get_current_model(self) -> Optional[str]:
        """Get name of currently loaded model."""
        return self.current_model
    
    def get_startup_command(self) -> Optional[str]:
        """Get the command used to start the current backend."""
        if self.current_backend and hasattr(self.current_backend, 'get_startup_command'):
            return self.current_backend.get_startup_command()
        return None


# Global LLM service instance
_llm_service = LLMService()


def get_llm_service() -> LLMService:
    """Get the global LLM service instance."""
    return _llm_service
