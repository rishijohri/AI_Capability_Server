"""Process manager for external binaries."""

import subprocess
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path
import signal
import psutil


class ProcessManager:
    """Manage external processes (llama-server, llama-cli, etc.)."""
    
    def __init__(self):
        """Initialize process manager."""
        self.active_processes: Dict[str, subprocess.Popen] = {}
    
    async def start_process(
        self,
        name: str,
        command: List[str],
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None
    ) -> subprocess.Popen:
        """
        Start a new process.
        
        Args:
            name: Unique identifier for the process
            command: Command and arguments to execute
            cwd: Working directory
            env: Environment variables
            
        Returns:
            Process handle
        """
        # Kill existing process with same name
        await self.kill_process(name)
        
        # Start new process
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            text=True
        )
        
        self.active_processes[name] = process
        return process
    
    async def kill_process(self, name: str, timeout: int = 5) -> bool:
        """
        Kill a process by name.
        
        Args:
            name: Process identifier
            timeout: Timeout in seconds
            
        Returns:
            True if process was killed, False if not found
        """
        if name not in self.active_processes:
            return False
        
        process = self.active_processes[name]
        
        try:
            # Try graceful termination first
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Force kill if timeout
                process.kill()
                process.wait()
            
            del self.active_processes[name]
            return True
            
        except Exception as e:
            print(f"Error killing process {name}: {e}")
            return False
    
    async def kill_all(self, timeout: int = 5) -> None:
        """Kill all active processes."""
        names = list(self.active_processes.keys())
        for name in names:
            await self.kill_process(name, timeout)
    
    def is_process_running(self, name: str) -> bool:
        """Check if a process is running."""
        if name not in self.active_processes:
            return False
        
        process = self.active_processes[name]
        return process.poll() is None
    
    async def wait_for_process(self, name: str, timeout: Optional[int] = None) -> Optional[int]:
        """
        Wait for a process to complete.
        
        Args:
            name: Process identifier
            timeout: Timeout in seconds
            
        Returns:
            Exit code, or None if timeout
        """
        if name not in self.active_processes:
            return None
        
        process = self.active_processes[name]
        
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
    
    def get_process_output(self, name: str) -> Optional[tuple[str, str]]:
        """
        Get stdout and stderr from a completed process.
        
        Args:
            name: Process identifier
            
        Returns:
            Tuple of (stdout, stderr), or None if process not found
        """
        if name not in self.active_processes:
            return None
        
        process = self.active_processes[name]
        
        if process.poll() is None:
            # Process still running
            return None
        
        stdout, stderr = process.communicate()
        return stdout, stderr
    
    async def kill_existing_binary_processes(self, binary_name: str) -> int:
        """
        Kill all system processes with the given binary name.
        
        Args:
            binary_name: Name of the binary (e.g., 'llama-server', 'llama-cli')
            
        Returns:
            Number of processes killed
        """
        killed_count = 0
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    # Check if process name matches
                    if proc.info['name'] == binary_name:
                        proc.kill()
                        killed_count += 1
                        continue
                    
                    # Also check command line for the binary name
                    cmdline = proc.info['cmdline']
                    if cmdline and any(binary_name in cmd for cmd in cmdline):
                        proc.kill()
                        killed_count += 1
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
        except Exception as e:
            print(f"Error killing processes for {binary_name}: {e}")
        
        # Give processes time to die
        if killed_count > 0:
            await asyncio.sleep(0.5)
        
        return killed_count


# Global process manager instance
_process_manager = ProcessManager()


def get_process_manager() -> ProcessManager:
    """Get the global process manager instance."""
    return _process_manager
