"""FastAPI app (backend.md §9.1). Local: `uvicorn app.main:app --port 8000`. Vercel: api/index.py imports `app`."""
from fastapi import FastAPI
from fastapi.responses import FileResponse

from app import errors
from app.config import WEB_DIST
from app.errors import ApiError
from app.routers import meta, products, search


def create_app() -> FastAPI:
    app = FastAPI(title="multimodal_search demo", version="0.1.0")
    errors.install(app)
    for r in (search.router, products.router, meta.router):
        app.include_router(r, prefix="/api")

    @app.get("/api/{path:path}", include_in_schema=False)
    def api_404(path: str):
        raise ApiError(404, "NOT_FOUND", f"no endpoint /api/{path}")

    if WEB_DIST.is_dir():  # local only; on Vercel the CDN serves web/dist
        root = WEB_DIST.resolve()

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            f = (root / path).resolve()
            return FileResponse(f if path and f.is_file() and f.is_relative_to(root) else root / "index.html")

    return app


app = create_app()
