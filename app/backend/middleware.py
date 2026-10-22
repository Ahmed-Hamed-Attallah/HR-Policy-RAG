"""Custom FastAPI middleware for request tracing, latency profiling, and logging."""

import logging
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestTimingAndTraceMiddleware(BaseHTTPMiddleware):
    """Tracks latency and injects a deterministic correlation trace ID for each HTTP request."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        trace_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.time()

        # Attach request ID to request state for access inside endpoints
        request.state.trace_id = trace_id

        response = await call_next(request)

        process_time_ms = (time.time() - start_time) * 1000
        response.headers["X-Request-ID"] = trace_id
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"

        logger.info(
            f"[{trace_id}] {request.method} {request.url.path} "
            f"status={response.status_code} latency={process_time_ms:.2f}ms"
        )
        return response