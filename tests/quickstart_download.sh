#!/bin/bash

# Quick Start: Model Download Feature
# This script helps you get started with downloading models from Hugging Face

echo "=================================================="
echo "  AI Capability - Model Download Quick Start"
echo "=================================================="
echo ""

# Check if running from project root
if [ ! -f "run_server.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Example: ./tests/quickstart_download.sh"
    exit 1
fi

echo "Step 1: Installing dependencies..."
echo "-----------------------------------"
pip install huggingface_hub>=0.20.0
if [ $? -ne 0 ]; then
    echo "❌ Failed to install huggingface_hub"
    exit 1
fi
echo "✅ Dependencies installed"
echo ""

echo "Step 2: Validating configuration..."
echo "------------------------------------"
python tests/validate_download_config.py
if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  Configuration issues found!"
    echo ""
    echo "Please edit app/config/settings.py and add 'repo_id' fields to models."
    echo "See REPO_ID_EXAMPLES.md for examples."
    echo ""
    echo "Example:"
    echo "  \"qwen_3_0.6B\": {"
    echo "      \"model_file\": \"Qwen3-0.6B-Q4_K_M.gguf\","
    echo "      \"name\": \"qwen_3_0.6B\","
    echo "      \"type\": \"chat\","
    echo "      \"repo_id\": \"Qwen/Qwen3-0.6B-GGUF\"  # Add this line"
    echo "  }"
    echo ""
    echo "After adding repo_id values, run this script again."
    exit 1
fi
echo "✅ Configuration validated"
echo ""

echo "Step 3: Starting server (in background)..."
echo "-------------------------------------------"
# Check if server is already running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "ℹ️  Server already running on port 8000"
else
    python run_server.py > /tmp/ai_capability_server.log 2>&1 &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"
    echo "Waiting for server to start..."
    sleep 3
    
    # Check if server started successfully
    if ! lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
        echo "❌ Failed to start server"
        echo "Check logs at /tmp/ai_capability_server.log"
        exit 1
    fi
    echo "✅ Server started successfully"
fi
echo ""

echo "=================================================="
echo "  Ready to Download Models!"
echo "=================================================="
echo ""
echo "Option 1: Use the test script (Python)"
echo "  python tests/test_download_models.py"
echo ""
echo "Option 2: Use curl + websocat"
echo "  brew install websocat  # Install if needed"
echo "  echo '{\"model_ids\": [\"qwen_3_0.6B\"]}' | websocat ws://localhost:8000/api/download-models"
echo ""
echo "Option 3: Use Python interactive shell"
echo "  python -c '"
echo "import asyncio, json, websockets"
echo "async def dl():"
echo "    async with websockets.connect(\"ws://localhost:8000/api/download-models\") as ws:"
echo "        await ws.send(json.dumps({\"model_ids\": [\"qwen_3_0.6B\"]}))"
echo "        async for msg in ws: print(json.loads(msg)[\"message\"])"
echo "asyncio.run(dl())"
echo "'"
echo ""
echo "Documentation:"
echo "  - MODEL_DOWNLOAD_GUIDE.md - Complete guide"
echo "  - REPO_ID_EXAMPLES.md - Configuration examples"
echo "  - API_REFERENCE.md - API documentation"
echo ""
echo "Recommended first download: qwen_3_0.6B (~600MB)"
echo ""
