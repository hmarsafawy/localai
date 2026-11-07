from anyio import to_thread
from fastapi import APIRouter, HTTPException, status
from huggingface_hub.utils import HfHubHTTPError

from models.jobs import JobInfo
from routes.model_router import map_hf_error
from schemas.downloads import ModelDownloadRequest
from schemas.runtime import JobResponse
from services.model_downloads import (
    HubUnavailableError,
    ModelAccessError,
    ModelDownloadRequestError,
    NotEnoughSpaceError,
    model_download_service as service,
)

router = APIRouter()

_NO_SUCH_JOB = "No such download. Downloads don't survive a restart of the app, but starting it again resumes it."


@router.post("", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_download(request: ModelDownloadRequest) -> JobInfo:
    """Downloads a text model's quantization and vision encoder in the background
    and returns a job to poll. Only `repo_id` is needed: by default it picks the
    recommended quantization and, if the repo has one, its vision encoder.

    `quantization` and `vision_encoder` take an option `id` from
    `GET /models/{repo}/requirements`, or a quantization name such as `Q4_K_M`.
    `vision_encoder` can also be `none`. What's already on disk isn't fetched again,
    and repeating a request that's still running joins that download."""
    try:
        return await to_thread.run_sync(service.start, request.repo_id, request.quantization, request.vision_encoder)
    except HfHubHTTPError as error:
        raise map_hf_error(error) from error
    except ModelDownloadRequestError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ModelAccessError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except NotEnoughSpaceError as error:
        raise HTTPException(status_code=507, detail=str(error)) from error
    except HubUnavailableError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.get("", response_model=list[JobResponse])
async def list_downloads() -> list[JobInfo]:
    """Every download since the app started, newest first."""
    return service.jobs.all_jobs()


@router.get("/{job_id}", response_model=JobResponse)
async def get_download(job_id: str) -> JobInfo:
    job = service.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=_NO_SUCH_JOB)
    return job


@router.delete("/{job_id}", response_model=JobResponse)
async def cancel_download(job_id: str) -> JobInfo:
    """Asks a download to stop. What it already fetched is kept, so starting it
    again picks up from there. To reclaim the space, delete it from the library."""
    job = service.jobs.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=_NO_SUCH_JOB)
    return job
