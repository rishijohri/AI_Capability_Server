#!/usr/bin/env python3
"""
Validation script to check AI Server setup.
Run this to verify all components are properly configured.
"""

import sys
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def check_mark(condition):
    """Return checkmark or X based on condition."""
    return f"{GREEN}✓{RESET}" if condition else f"{RED}✗{RESET}"


def print_header(text):
    """Print section header."""
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}")


def check_imports():
    """Check if all required imports work."""
    print_header("Checking Python Imports")
    
    imports = [
        ("FastAPI", "fastapi"),
        ("Uvicorn", "uvicorn"),
        ("WebSockets", "websockets"),
        ("Pillow", "PIL"),
        ("NumPy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("FAISS", "faiss"),
        ("Pydantic", "pydantic"),
        ("OpenCV", "cv2"),
        ("psutil", "psutil"),
        ("aiohttp", "aiohttp"),
    ]
    
    all_ok = True
    for name, module in imports:
        try:
            __import__(module)
            print(f"{check_mark(True)} {name}")
        except ImportError:
            print(f"{check_mark(False)} {name} - Not installed")
            all_ok = False
    
    return all_ok


def check_app_structure():
    """Check if app structure is correct."""
    print_header("Checking Application Structure")
    
    required_files = [
        "app/__init__.py",
        "app/main.py",
        "app/config/__init__.py",
        "app/config/settings.py",
        "app/models/__init__.py",
        "app/models/metadata.py",
        "app/models/requests.py",
        "app/models/responses.py",
        "app/services/__init__.py",
        "app/services/llm_service.py",
        "app/services/embedding_service.py",
        "app/services/rag_service.py",
        "app/services/vision_service.py",
        "app/api/__init__.py",
        "app/api/routes.py",
        "app/utils/__init__.py",
        "app/utils/image_processor.py",
        "app/utils/process_manager.py",
    ]
    
    all_ok = True
    for file_path in required_files:
        exists = Path(file_path).exists()
        print(f"{check_mark(exists)} {file_path}")
        all_ok = all_ok and exists
    
    return all_ok


def check_app_imports():
    """Check if app modules import correctly."""
    print_header("Checking Application Imports")
    
    modules = [
        ("Config", "app.config"),
        ("Models", "app.models"),
        ("Services", "app.services"),
        ("API", "app.api"),
        ("Utils", "app.utils"),
        ("Main", "app.main"),
    ]
    
    all_ok = True
    for name, module in modules:
        try:
            __import__(module)
            print(f"{check_mark(True)} {name} ({module})")
        except Exception as e:
            print(f"{check_mark(False)} {name} ({module}) - {str(e)}")
            all_ok = False
    
    return all_ok


def check_directories():
    """Check if required directories exist."""
    print_header("Checking Directory Structure")
    
    dirs = [
        ("Binary directory", "binary", True),
        ("Model directory", "model", True),
        ("Virtual environment", "venv", False),
    ]
    
    warnings = []
    for name, dir_path, required in dirs:
        exists = Path(dir_path).exists()
        if required and not exists:
            print(f"{check_mark(False)} {name} ({dir_path}) - Required but missing")
            warnings.append(f"Create {dir_path}/ directory")
        elif not exists:
            print(f"{YELLOW}⚠{RESET} {name} ({dir_path}) - Optional, not found")
        else:
            print(f"{check_mark(True)} {name} ({dir_path})")
    
    return warnings


def check_binaries():
    """Check if binaries exist."""
    print_header("Checking Binaries")
    
    import platform
    
    # Determine binary extension based on platform
    is_windows = platform.system() == "Windows"
    ext = ".exe" if is_windows else ""
    
    binaries = [
        f"binary/llama-server{ext}",
        f"binary/llama-cli{ext}",
        f"binary/llama-mtmd-cli{ext}",
        f"binary/llama-embedding{ext}",
    ]
    
    warnings = []
    found_count = 0
    for binary in binaries:
        path = Path(binary)
        exists = path.exists()
        if exists:
            found_count += 1
            if is_windows:
                # On Windows, .exe files are automatically executable
                print(f"{check_mark(True)} {binary}")
            else:
                # On Unix-like systems, check executable bit
                executable = path.stat().st_mode & 0o111
                if executable:
                    print(f"{check_mark(True)} {binary} (executable)")
                else:
                    print(f"{YELLOW}⚠{RESET} {binary} (not executable)")
                    warnings.append(f"Make {binary} executable: chmod +x {binary}")
        else:
            print(f"{YELLOW}⚠{RESET} {binary} - Not found (optional)")
    
    if found_count == 0:
        if is_windows:
            warnings.append("No binaries found in binary/ directory. Add .exe binaries for Windows.")
        else:
            warnings.append("No binaries found in binary/ directory. Add binaries and run: chmod +x binary/*")
    
    # Platform-specific note
    if is_windows:
        print(f"\n{BLUE}Note:{RESET} Windows detected - expecting .exe binaries")
    else:
        print(f"\n{BLUE}Note:{RESET} Unix-like system detected - expecting binaries without extensions")
    
    return warnings


def check_models():
    """Check if models exist."""
    print_header("Checking Models")
    
    model_dir = Path("model")
    if not model_dir.exists():
        print(f"{check_mark(False)} Model directory not found")
        return ["Create model/ directory and add .gguf model files"]
    
    # Check for .gguf files
    gguf_files = list(model_dir.glob("*.gguf"))
    
    if not gguf_files:
        print(f"{check_mark(False)} No .gguf model files found")
        return ["Add .gguf model files to model/ directory"]
    
    print(f"{GREEN}✓{RESET} Found {len(gguf_files)} model file(s):")
    
    # Categorize models
    embedding_models = []
    vision_models = []
    chat_models = []
    mmproj_files = []
    
    for model_file in gguf_files:
        name = model_file.name
        size_mb = model_file.stat().st_size / (1024 * 1024)
        
        if "mmproj" in name.lower():
            mmproj_files.append(name)
            print(f"  • {name} ({size_mb:.1f} MB) - MMProj")
        elif "embedding" in name.lower():
            embedding_models.append(name)
            print(f"  • {name} ({size_mb:.1f} MB) - Embedding")
        elif "vl" in name.lower() or "vision" in name.lower():
            vision_models.append(name)
            print(f"  • {name} ({size_mb:.1f} MB) - Vision")
        else:
            chat_models.append(name)
            print(f"  • {name} ({size_mb:.1f} MB) - Chat/Instruct")
    
    warnings = []
    if not embedding_models:
        warnings.append("No embedding model found (needed for RAG)")
    if not vision_models:
        warnings.append("No vision model found (needed for tag/describe)")
    if not chat_models:
        warnings.append("No chat model found (needed for chat endpoint)")
    if vision_models and not mmproj_files:
        warnings.append("Vision models found but no mmproj file (may be needed)")
    
    return warnings


def check_files():
    """Check if important files exist."""
    print_header("Checking Project Files")
    
    files = [
        ("Requirements file", "requirements.txt"),
        ("PyInstaller spec", "ai_server.spec"),
        ("Run script", "run_server.py"),
        ("Example client", "example_client.py"),
        ("Main README", "README_AI_SERVER.md"),
        ("Quick Start", "QUICKSTART.md"),
        ("Model Config", "MODEL_CONFIG.md"),
        ("Usage Examples", "USAGE_EXAMPLES.md"),
        ("API Reference", "API_REFERENCE.md"),
        ("Documentation Index", "DOCUMENTATION_INDEX.md"),
        ("Sample metadata", "sample_metadata.json"),
    ]
    
    all_ok = True
    for name, file_path in files:
        exists = Path(file_path).exists()
        print(f"{check_mark(exists)} {name} ({file_path})")
        all_ok = all_ok and exists
    
    return all_ok


def test_server_creation():
    """Test if server can be created."""
    print_header("Testing Server Creation")
    
    try:
        from app.main import create_app
        app = create_app()
        print(f"{check_mark(True)} Server creation successful")
        print(f"{check_mark(True)} FastAPI app instantiated")
        return True
    except Exception as e:
        print(f"{check_mark(False)} Server creation failed: {str(e)}")
        return False


def main():
    """Main validation function."""
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}AI Server Validation Script{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}")
    
    warnings = []
    
    # Run checks
    imports_ok = check_imports()
    structure_ok = check_app_structure()
    app_imports_ok = check_app_imports()
    dir_warnings = check_directories()
    binary_warnings = check_binaries()
    model_warnings = check_models()
    files_ok = check_files()
    server_ok = test_server_creation()
    
    warnings.extend(dir_warnings)
    warnings.extend(binary_warnings)
    warnings.extend(model_warnings)
    
    # Summary
    print_header("Validation Summary")
    
    checks = [
        ("Python dependencies", imports_ok),
        ("Application structure", structure_ok),
        ("Application imports", app_imports_ok),
        ("Project files", files_ok),
        ("Server creation", server_ok),
    ]
    
    all_passed = True
    for name, status in checks:
        print(f"{check_mark(status)} {name}")
        all_passed = all_passed and status
    
    # Warnings
    if warnings:
        print(f"\n{YELLOW}Warnings:{RESET}")
        for warning in warnings:
            print(f"  • {warning}")
    
    # Final status
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    if all_passed and not warnings:
        print(f"{GREEN}✓ All checks passed! Server is ready to run.{RESET}")
        print(f"\nTo start the server:")
        print(f"  python run_server.py")
    elif all_passed:
        print(f"{YELLOW}⚠ Core checks passed but some warnings exist.{RESET}")
        print(f"\nThe server will run, but you need to:")
        for warning in warnings:
            print(f"  • {warning}")
    else:
        print(f"{RED}✗ Some checks failed. Please fix the issues above.{RESET}")
        return 1
    
    print(f"{BLUE}{'=' * 60}{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
