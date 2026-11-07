from anyio import to_thread
from fastapi import APIRouter, HTTPException, Path, Query, Response, status

from models.downloads import ModelLibrary
from schemas.downloads import OPTION_PATTERN, REPO_ID_PATTERN, ModelLibraryResponse
from services.model_downloads import (
    LocalModelBusyError,
    LocalModelNotFoundError,
    model_download_service as service,
)

router = APIRouter()


@router.get("", response_model=ModelLibraryResponse)
async def get_library() -> ModelLibrary:
    """The models on this machine, with the path of each file to run. Anything
    unfinished is listed as `incomplete`, with how much of it is on disk."""
    return await to_thread.run_sync(service.library)


@router.delete("/{repo_id:path}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_local_model(
    repo_id: str = Path(pattern=REPO_ID_PATTERN, max_length=200),
    quantization: str | None = Query(default=None, pattern=OPTION_PATTERN, max_length=200),
    vision_encoder: str | None = Query(default=None, pattern=OPTION_PATTERN, max_length=200),
) -> Response:
    """Deletes a repo's files, or just the `quantization` or `vision_encoder` named.
    Anything half downloaded goes too. A repo that's downloading has to be cancelled first."""
    try:
        await to_thread.run_sync(service.delete, repo_id, quantization, vision_encoder)
    except LocalModelNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except LocalModelBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return Response(status_code=status.HTTP_204_NO_CONTENT)
