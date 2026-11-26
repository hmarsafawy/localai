from anyio import to_thread
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from huggingface_hub.utils import HfHubHTTPError

from models.jobs import JobInfo
from routes.huggingface_router import map_hf_error
from routes.job_events import EVENT_STREAM_RESPONSES, job_events
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
    recommended quantization and, if the repo has one, its vision encoder. The
    recommendation is the biggest quantization that fits the GPU along with the
    vision encoder and `context_size` tokens of context (16,384 by default).

    `quantization` and `vision_encoder` take an option `id` from
    `GET /models/{repo}/requirements`, or a quantization name such as `Q4_K_M`.
    `vision_encoder` can also be `none`. What's already on disk isn't fetched again,
    and repeating a request that's still running joins that download.

    A repo with no GGUF files is converted: its safetensors are downloaded,
    converted to GGUF with the active llama.cpp runtime's converter (set up
    first if it isn't yet, about 1 GB once) and quantized, then deleted. That
    takes more disk, RAM and time than a download; the job's
    `details.conversion` says how much disk. `quantization` is then one of
    BF16, F16, Q8_0, Q6_K, Q5_K_M, Q5_K_S, Q4_K_M, Q4_K_S, Q4_0, Q3_K_L,
    Q3_K_M, Q3_K_S or Q2_K. A vision model's vision encoder is converted from
    the same download, as F16 unless `vision_encoder` is BF16, Q8_0 or `none`.
    Left to `auto`, one the converter can't convert is left out with a note,
    and the model is converted as a text model."""
    try:
        return await to_thread.run_sync(
            service.start, request.repo_id, request.quantization, request.vision_encoder, request.context_size
        )
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


@router.get("/{job_id}/events", response_class=StreamingResponse, responses=EVENT_STREAM_RESPONSES)
async def follow_download(job_id: str) -> StreamingResponse:
    """The download's progress as it happens, as server-sent events, instead of
    polling `GET /downloads/{job_id}`. It ends once the download has finished."""
    return job_events(service.jobs, job_id, _NO_SUCH_JOB)


@router.delete("/{job_id}", response_model=JobResponse)
async def cancel_download(job_id: str) -> JobInfo:
    """Asks a download to stop. What it already fetched is kept, so starting it
    again picks up from there. To reclaim the space, delete it from the library."""
    job = service.jobs.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=_NO_SUCH_JOB)
    return job
