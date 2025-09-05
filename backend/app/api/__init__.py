"""
API package initialization.
"""
from fastapi import APIRouter

# Create main API router
api_router = APIRouter()

# Import and include endpoint routers
from .endpoints import aneurysm

# Include the routers
api_router.include_router(aneurysm.router)
