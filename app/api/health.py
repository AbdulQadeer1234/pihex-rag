from fastapi import APIRouter

health_router = APIRouter()

@health_router.get("/")
async def root():
    """API health check endpoint"""
    return {"status": "healthy", "message": "LangGraph Agent API is running"}