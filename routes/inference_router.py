"""An OpenAI-compatible API for the models that are running, at one address
however many there are. Point an OpenAI client at `http://<this machine>:8000/v1`
and send a running model's `model_name` as `model`.

Requests go on to the model's llama-server as they are, and its replies come
back as they are, streamed or not. The app's own errors are in OpenAI's format,
so clients show their message."""

import json
import secrets
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.background import BackgroundTask

from services import settings
from services.inference_proxy import UpstreamUnavailableError, inference_proxy as proxy
from services.llama_servers import ModelLoadingError, ModelNotRunningError, llama_server_service as servers

# How long a client should wait before asking again for a model that's loading.
RETRY_AFTER_SECONDS = 5


class OpenAIError(Exception):
    """An error in the shape OpenAI clients read: `{"error": {"message", ...}}`."""

    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str = "invalid_request_error",
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.code = code
        self.headers = headers

    def response(self) -> JSONResponse:
        body = {"error": {"message": self.message, "type": self.error_type, "param": None, "code": self.code}}
        return JSONResponse(body, status_code=self.status_code, headers=self.headers)


async def openai_error_handler(_: Request, error: Exception) -> JSONResponse:
    if not isinstance(error, OpenAIError):
        raise error
    return error.response()


_bearer = HTTPBearer(auto_error=False, description="The `LOCALAI_API_KEY`, when one is set.")


async def require_api_key(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]) -> None:
    """With `LOCALAI_API_KEY` set, every request must send it as a bearer
    token, which is what OpenAI clients do with their `api_key`."""
    key = settings.api_key()
    if key is None:
        return
    if credentials is None or not secrets.compare_digest(credentials.credentials.encode(), key.encode()):
        raise OpenAIError(
            401, "Missing or wrong API key. Send it as `Authorization: Bearer <key>`.", code="invalid_api_key"
        )


router = APIRouter(dependencies=[Depends(require_api_key)])


def _body_docs(example: dict[str, Any]) -> dict[str, Any]:
    """The body is passed on unread, so the docs page is told what it looks like."""
    return {
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": {"type": "object"}, "example": example}},
        }
    }


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """The models that are running and ready. Each `id` is what to send as `model`."""
    ready = [server for server in servers.all_servers() if server.status == "ready"]
    return {
        "object": "list",
        "data": [
            {
                "id": server.model_name,
                "object": "model",
                "created": int((server.ready_at or server.created_at).timestamp()),
                "owned_by": "localai",
            }
            for server in ready
        ],
    }


@router.post(
    "/chat/completions",
    openapi_extra=_body_docs(
        {
            "model": "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": False,
        }
    ),
)
async def chat_completions(request: Request) -> StreamingResponse:
    """OpenAI's chat completions, from the running model named by `model`. With
    `"stream": true` the reply comes as server-sent events, token by token."""
    return await _forward(request, "/v1/chat/completions")


@router.post(
    "/completions",
    openapi_extra=_body_docs(
        {"model": "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M", "prompt": "Once upon a time", "max_tokens": 64}
    ),
)
async def completions(request: Request) -> StreamingResponse:
    """Plain text completion of a `prompt`, without the model's chat template."""
    return await _forward(request, "/v1/completions")


async def _forward(request: Request, path: str) -> StreamingResponse:
    body = await request.body()
    try:
        payload = json.loads(body)
    except ValueError as error:
        raise OpenAIError(400, f"The body isn't valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise OpenAIError(400, "The body must be a JSON object.")
    model = payload.get("model")
    if model is not None and not isinstance(model, str):
        raise OpenAIError(400, "`model` must be a string.")

    try:
        server = servers.find_running(model)
    except ModelNotRunningError as error:
        raise OpenAIError(404, str(error), code="model_not_found") from error
    except ModelLoadingError as error:
        raise OpenAIError(
            503, str(error), "server_error", "model_loading", headers={"Retry-After": str(RETRY_AFTER_SECONDS)}
        ) from error
    try:
        upstream = await proxy.open(server, path, body, request.headers)
    except UpstreamUnavailableError as error:
        raise OpenAIError(502, str(error), "server_error", "model_unavailable") from error

    # Its own errors, such as a prompt longer than the context, are in OpenAI's
    # format already, so they go back as they are too.
    headers = {"cache-control": upstream.headers["cache-control"]} if "cache-control" in upstream.headers else None
    return StreamingResponse(
        proxy.passthrough(upstream),
        status_code=upstream.status_code,
        headers=headers,
        media_type=upstream.headers.get("content-type"),
        background=BackgroundTask(upstream.aclose),
    )
