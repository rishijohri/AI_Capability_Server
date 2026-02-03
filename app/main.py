"""Main application entry point for AI Server."""

import asyncio
import signal
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api import router
from app.utils import get_process_manager
from app.utils.logging_config import get_logger
from app.services import get_llm_service
from app.config import initialize_config

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("=" * 60)
    logger.info("AI SERVER STARTUP")
    logger.info("=" * 60)
    
    try:
        # Initialize configuration and detect system
        logger.info("Initializing configuration...")
        config = initialize_config()
        logger.info(f"System detected: {config.system_info.get('os', 'unknown')} "
                   f"({config.system_info.get('architecture', 'unknown')}, "
                   f"{config.system_info.get('gpu', 'cpu')})")
        logger.info(f"Selected binary configuration: {config.binary_config}")
        
        available_configs = config.get_available_binary_configs()
        if available_configs:
            logger.info(f"Available binary configurations: {', '.join(available_configs)}")
        else:
            logger.warning("No binary configurations found")
        
        logger.info("AI Server startup complete - ready to accept requests")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.critical(f"Startup failed: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("AI SERVER SHUTDOWN INITIATED")
    logger.info("=" * 60)
    
    try:
        # Clean up resources
        logger.info("Cleaning up process manager...")
        process_manager = get_process_manager()
        await process_manager.kill_all()
        logger.info("Process manager cleanup complete")
        
        logger.info("Unloading LLM models...")
        llm_service = get_llm_service()
        await llm_service.unload_model()
        logger.info("LLM models unloaded")
        
        logger.info("AI Server shutdown complete")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    logger.info("Creating FastAPI application...")
    
    app = FastAPI(
        title="AI Server",
        description="Backend AI Server with RAG, Vision, and Chat capabilities",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    logger.debug("Adding CORS middleware")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    logger.debug("Including API routes")
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
                "select_storage_metadata": "/api/select-storage-metadata",
                "load_rag": "/api/load-rag",
                "vector_embeddings": "/api/vector-embeddings (WebSocket)",
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
    try:
        logger.info("Creating application instance...")
        app = create_app()
        
        # Configure server
        logger.info("Configuring uvicorn server...")
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config)
        
        # Run server
        logger.info("=" * 60)
        logger.info("Starting AI Server on http://127.0.0.1:8000")
        logger.info("API documentation available at http://127.0.0.1:8000/docs")
        logger.info("=" * 60)
        
        try:
            server.run()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.critical(f"Server crashed: {e}", exc_info=True)
            raise
            
    except Exception as e:
        logger.critical(f"Fatal error in main(): {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

