"""Validation script for model download configuration."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.settings import model_options, get_config


def validate_model_configuration():
    """Validate that models are properly configured for downloading."""
    
    print("=" * 70)
    print("MODEL DOWNLOAD CONFIGURATION VALIDATOR")
    print("=" * 70)
    print()
    
    config = get_config()
    
    # Statistics
    total_models = len(model_options)
    configured_models = 0
    models_with_issues = []
    
    print(f"Found {total_models} models in model_options\n")
    
    for model_id, model_info in model_options.items():
        print(f"Checking: {model_id}")
        print(f"  Name: {model_info.get('name', 'N/A')}")
        print(f"  Type: {model_info.get('type', 'N/A')}")
        
        issues = []
        
        # Check model_file
        model_file = model_info.get("model_file")
        if not model_file:
            issues.append("❌ Missing 'model_file'")
        else:
            print(f"  Model File: {model_file}")
            model_path = config.get_model_path(model_file)
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"    ✅ File exists ({size_mb:.1f} MB)")
            else:
                print(f"    ⚠️  File not found (will need download)")
        
        # Check mmproj_file for vision models
        if model_info.get("type") == "vision":
            mmproj_file = model_info.get("mmproj_file")
            if not mmproj_file:
                issues.append("❌ Vision model missing 'mmproj_file'")
            else:
                print(f"  MMProj File: {mmproj_file}")
                mmproj_path = config.get_model_path(mmproj_file)
                if mmproj_path.exists():
                    size_mb = mmproj_path.stat().st_size / (1024 * 1024)
                    print(f"    ✅ File exists ({size_mb:.1f} MB)")
                else:
                    print(f"    ⚠️  File not found (will need download)")
        
        # Check repo_id
        repo_id = model_info.get("repo_id")
        if not repo_id:
            issues.append("❌ Missing 'repo_id' - cannot download")
            print(f"  Repo ID: ❌ NOT CONFIGURED")
        elif not repo_id.strip():
            issues.append("❌ Empty 'repo_id' - cannot download")
            print(f"  Repo ID: ❌ EMPTY")
        else:
            print(f"  Repo ID: ✅ {repo_id}")
            configured_models += 1
            
            # Validate repo_id format
            if "/" not in repo_id:
                issues.append("⚠️  Invalid repo_id format (should be 'username/repo-name')")
            elif repo_id.startswith("http"):
                issues.append("⚠️  repo_id should not include URL (just 'username/repo-name')")
        
        if issues:
            models_with_issues.append((model_id, issues))
            print("\n  Issues:")
            for issue in issues:
                print(f"    {issue}")
        
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total models: {total_models}")
    print(f"Models with repo_id configured: {configured_models}")
    print(f"Models ready for download: {configured_models - len(models_with_issues)}")
    print(f"Models with issues: {len(models_with_issues)}")
    print()
    
    if models_with_issues:
        print("MODELS WITH ISSUES:")
        print("-" * 70)
        for model_id, issues in models_with_issues:
            print(f"\n{model_id}:")
            for issue in issues:
                print(f"  {issue}")
        print()
        print("ACTION REQUIRED:")
        print("  1. Edit app/config/settings.py")
        print("  2. Add 'repo_id' field to models listed above")
        print("  3. Use format: 'username/repository-name'")
        print("  4. See REPO_ID_EXAMPLES.md for examples")
    else:
        print("✅ All models are properly configured!")
    
    print()
    
    # Check if huggingface_hub is installed
    print("=" * 70)
    print("DEPENDENCY CHECK")
    print("=" * 70)
    try:
        import huggingface_hub
        print(f"✅ huggingface_hub is installed (version {huggingface_hub.__version__})")
    except ImportError:
        print("❌ huggingface_hub is NOT installed")
        print("   Install it with: pip install huggingface_hub")
    print()
    
    # Check model directory
    print("=" * 70)
    print("MODEL DIRECTORY")
    print("=" * 70)
    model_dir = config.get_model_path("")
    print(f"Path: {model_dir}")
    if model_dir.exists():
        print("✅ Directory exists")
        
        # List existing model files
        model_files = list(model_dir.glob("*.gguf"))
        print(f"Found {len(model_files)} GGUF files:")
        for model_file in model_files[:10]:  # Show first 10
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  - {model_file.name} ({size_mb:.1f} MB)")
        if len(model_files) > 10:
            print(f"  ... and {len(model_files) - 10} more")
    else:
        print("⚠️  Directory does not exist (will be created on first download)")
    
    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print()
    
    return len(models_with_issues) == 0


if __name__ == "__main__":
    try:
        success = validate_model_configuration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
