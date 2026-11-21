"""One error shape for every failure (backend.md §9.6)."""
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

log = logging.getLogger("mms")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: list | None = None):
        self.status, self.code, self.message, self.details = status, code, message, details or []


def _body(code: str, message: str, details: list | None = None) -> dict:
    return {"error": {"code": code, "message": message, "details": details or []}}


def install(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api(_: Request, e: ApiError):
        return JSONResponse(_body(e.code, e.message, e.details), status_code=e.status)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, e: RequestValidationError):
        details = [{"path": ".".join(str(p) for p in err["loc"][1:]), "issue": err["type"]} for err in e.errors()]
        msg = "; ".join(f"{d['path']}: {err['msg']}" for d, err in zip(details, e.errors()))
        return JSONResponse(_body("VALIDATION_ERROR", msg, details), status_code=400)

    @app.exception_handler(HTTPException)
    async def _http(_: Request, e: HTTPException):
        code = "NOT_FOUND" if e.status_code == 404 else "HTTP_ERROR"
        return JSONResponse(_body(code, str(e.detail)), status_code=e.status_code)

    @app.exception_handler(Exception)
    async def _internal(request: Request, e: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(_body("INTERNAL", "internal error"), status_code=500)
