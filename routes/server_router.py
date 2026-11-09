from functools import partial

from anyio import to_thread
from fastapi import APIRouter, HTTPException, Query, status

from models.servers import ModelServer, ServerLogs
from schemas.servers import ModelServerResponse, ServerLaunchRequest, ServerLogsResponse
from services.llama_runtime import RuntimeConflictError, RuntimeNotFoundError
from services.llama_servers import (
    ServerConflictError,
    ServerNotFoundError,
    ServerStartError,
    llama_server_service as service,
)
from services.model_downloads import LocalModelBusyError, LocalModelChoiceError, LocalModelNotFoundError

router = APIRouter()


@router.post("", response_model=ModelServerResponse, status_code=status.HTTP_202_ACCEPTED)
async def launch_server(request: ServerLaunchRequest) -> ModelServer:
    """Starts llama-server on a downloaded model, and returns while it loads.
    Poll `GET /servers/{id}` until `status` is `ready`, or `failed` with the reason.

    Once it's ready, `base_url` is llama.cpp's own chat page, and its
    OpenAI-compatible API is at `base_url` + `/v1`, with `model_name` as the
    `model`. It only listens on this machine. Launching a model that's already
    running with the same settings returns that server."""
    launch = partial(
        service.launch,
        request.repo_id,
        request.quantization,
        request.vision_encoder,
        context_size=request.context_size,
        gpu_layers=request.gpu_layers,
        port=request.port,
        runtime_id=request.runtime_id,
    )
    try:
        return await to_thread.run_sync(launch)
    except (LocalModelNotFoundError, RuntimeNotFoundError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except LocalModelChoiceError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (LocalModelBusyError, RuntimeConflictError, ServerConflictError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ServerStartError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@router.get("", response_model=list[ModelServerResponse])
async def list_servers() -> list[ModelServer]:
    """Every server since the app started, newest first, including stopped ones."""
    return service.all_servers()


@router.get("/{server_id}", response_model=ModelServerResponse)
async def get_server(server_id: str) -> ModelServer:
    try:
        return service.get(server_id)
    except ServerNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/{server_id}/logs", response_model=ServerLogsResponse)
async def get_server_logs(server_id: str, lines: int = Query(default=200, ge=1, le=5000)) -> ServerLogs:
    """The end of what llama-server printed, which is where to look when one fails."""
    try:
        return await to_thread.run_sync(service.logs, server_id, lines)
    except ServerNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.delete("/{server_id}", response_model=ModelServerResponse)
async def stop_server(server_id: str) -> ModelServer:
    """Stops a server and waits until it has, so its memory is free again."""
    try:
        return await to_thread.run_sync(service.stop, server_id)
    except ServerNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
