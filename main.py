"""Root entry point for FastAPI server."""
from app.main import app, create_app

# Backward-compatible exports
__all__ = ["app", "create_app"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)