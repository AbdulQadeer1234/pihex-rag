"""Main FastAPI application entry point"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.api.health import health_router
from app.config.config import Config
from app.api.ingest import ingest_router

from app.api.ask_api import ask_router

from app.utils.milvus_utils import setup_milvus_database, initialize_embeddings, get_vector_store
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup and shutdown events"""
        try:
            # Startup: Initialize required components
            logger.info("Starting application initialization...")
            
            # Step 1: Initialize Milvus database
            logger.info("Initializing Milvus database connection...")
            if not setup_milvus_database():
                raise RuntimeError("Failed to initialize Milvus database")
            logger.info("Milvus database connection established successfully")
                
            # Step 2: Initialize embedding model
            logger.info("Initializing embedding model...")
            embedding_model = initialize_embeddings()
            if not embedding_model:
                raise RuntimeError("Failed to initialize embedding model")
            logger.info("Embedding model initialized successfully")
            
            # Step 3: Initialize vector store singleton
            logger.info("Initializing vector store...")
            vector_store = await get_vector_store(force_reinit=True)
            if not vector_store:
                raise RuntimeError("Failed to initialize vector store")
            logger.info("Vector store initialized successfully")
            
            yield  # Application runs here
            
            # Shutdown: Cleanup resources
            logger.info("Shutting down application...")
            # Add cleanup code here if needed
            
        except Exception as e:
            logger.error(f"Application lifecycle error: {str(e)}")
            raise
    
    app = FastAPI(
        lifespan=lifespan,
        title="Pihex assessment API",
        description="API for Homevestors GenAI chat and document processing",
        version="1.0.0"
    )
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(
        ask_router,
        prefix="/api",
        tags=["chat"]
    )

    return app

app = create_app()

if __name__ == "__main__":
    try:
        logger.info(f"Starting server on port {Config.PORT}")
        uvicorn.run(
            "app.main:app", 
            host="0.0.0.0", 
            port=Config.PORT,
            reload=Config.DEBUG_MODE
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise
