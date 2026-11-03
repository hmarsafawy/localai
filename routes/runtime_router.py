from anyio import to_thread
from fastapi import APIRouter, Body, HTTPException, Query, Response, status

from models.jobs import JobInfo
from models.runtime import AvailableBuilds, InstalledRuntimes
from schemas.runtime import (
    VERSION_PATTERN,
    AvailableBuildsResponse,
    InstalledRuntimesResponse,
    JobResponse,
    RuntimeActivateRequest,
    RuntimeInstallRequest,
)
from services.llama_runtime import (
    GitHubRateLimitError,
    GitHubUnavailableError,
    RuntimeConflictError,
    RuntimeNotFoundError,
    llama_runtime_service as service,
)

router = APIRouter()


def _map_runtime_error(error: Exception) -> HTTPException:
    if isinstance(error, RuntimeNotFoundError):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, RuntimeConflictError):
        return HTTPException(status_code=409, detail=str(error))
    if isinstance(error, GitHubRateLimitError):
        return HTTPException(status_code=429, detail=str(error))
    return HTTPException(status_code=502, detail=str(error))


_RUNTIME_ERRORS = (RuntimeNotFoundError, RuntimeConflictError, GitHubRateLimitError, GitHubUnavailableError)


@router.get("", response_model=InstalledRuntimesResponse)
async def list_runtimes(refresh: bool = Query(default=False)) -> InstalledRuntimes:
    """Installed builds, which is active, and whether a newer one is out. Works
    offline. `refresh=true` re-tests every build on this machine."""
    return await to_thread.run_sync(service.list_installed, refresh)


@router.get("/available", response_model=AvailableBuildsResponse)
async def list_available_builds(
    version: str = Query(default="stable", pattern=VERSION_PATTERN),
) -> AvailableBuilds:
    """The builds of a release that can run on this OS and CPU, best first, each
    with the reason for its place. `version` is `stable`, `latest`, a stable
    version such as `v0.8.0`, or a build such as `b11960`."""
    try:
        return await to_thread.run_sync(service.available, version)
    except _RUNTIME_ERRORS as error:
        raise _map_runtime_error(error) from error


@router.post("/install", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def install_runtime(request: RuntimeInstallRequest | None = Body(default=None)) -> JobInfo:
    """Starts an install in the background and returns its job to poll. An empty
    request installs the stable release in the best build for this machine. With
    `backend=auto`, a build that can't see the GPU is dropped for the next best."""
    options = request or RuntimeInstallRequest()
    try:
        return await to_thread.run_sync(service.start_install, options.version, options.backend, options.activate)
    except _RUNTIME_ERRORS as error:
        raise _map_runtime_error(error) from error


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobInfo:
    job = service.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job. Jobs don't survive a restart of the app.")
    return job


@router.delete("/jobs/{job_id}", response_model=JobResponse)
async def cancel_job(job_id: str) -> JobInfo:
    """Asks a job to stop. What it already downloaded is kept, so installing the
    same build again picks up from there."""
    job = service.jobs.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job. Jobs don't survive a restart of the app.")
    return job


@router.put("/active", response_model=InstalledRuntimesResponse)
async def set_active_runtime(request: RuntimeActivateRequest) -> InstalledRuntimes:
    """Switches to an installed build, which must run on this machine."""
    try:
        return await to_thread.run_sync(service.set_active, request.id)
    except _RUNTIME_ERRORS as error:
        raise _map_runtime_error(error) from error


@router.delete("/{runtime_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_runtime(runtime_id: str) -> Response:
    """Removes an installed build. The active one has to be switched away from first."""
    try:
        await to_thread.run_sync(service.delete, runtime_id)
    except _RUNTIME_ERRORS as error:
        raise _map_runtime_error(error) from error
    return Response(status_code=status.HTTP_204_NO_CONTENT)
