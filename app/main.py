"""Main application entry point for AI Server."""

import asyncio
import signal
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api import router
from app.utils import get_process_manager
from app.services import get_llm_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    print("AI Server starting...")
    
    yield
    
    # Shutdown
    print("AI Server shutting down...")
    
    # Clean up resources
    process_manager = get_process_manager()
    await process_manager.kill_all()
    
    llm_service = get_llm_service()
    await llm_service.unload_model()
    
    print("AI Server stopped.")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="AI Server",
        description="Backend AI Server with RAG, Vision, and Chat capabilities",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(router, prefix="/api")
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "AI Server",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "config": "/api/config",
                "set_storage_metadata": "/api/set-storage-metadata",
                "load_rag": "/api/load-rag",
                "generate_embeddings": "/api/generate-embeddings (WebSocket)",
                "generate_rag": "/api/generate-rag (WebSocket)",
                "tag": "/api/tag (WebSocket)",
                "describe": "/api/describe (WebSocket)",
                "chat": "/api/chat (WebSocket)"
            }
        }
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    return app


def main():
    """Main entry point."""
    app = create_app()
    
    # Configure server
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        access_log=True
    )
    
    server = uvicorn.Server(config)
    
    # Run server
    print("Starting AI Server on http://127.0.0.1:8000")
    print("API documentation available at http://127.0.0.1:8000/docs")
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
