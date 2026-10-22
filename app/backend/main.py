"""FastAPI main entrypoint configuring routes, middlewares, and lifecycle handlers."""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.api.v1.routers import admin, chat, documents, health
from app.backend.exceptions import AppException, app_exception_handler, generic_exception_handler
from app.backend.lifespan import lifespan
from app.backend.middleware import RequestTimingAndTraceMiddleware
from src.config.settings import settings

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hr_rag_backend")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="REST API for Enterprise HR Policy Retrieval-Augmented Generation.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 1. CORS Configuration (Allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Timing and Tracing Middleware
app.add_middleware(RequestTimingAndTraceMiddleware)

# 3. Exception Handlers
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# 4. Register V1 API Routers
app.include_router(health.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")


@app.get("/", tags=["Root"])
def root():
    """Root redirect endpoint pointing to interactive API documentation."""
    return {
        "project": settings.PROJECT_NAME,
        "docs": "/docs",
        "health": "/api/v1/health",
        "api_v1": "/api/v1",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.backend.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=True,
    )