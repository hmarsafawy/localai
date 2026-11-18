"""A background job's progress as it happens, as server-sent events, for any
kind of job. Each event's data is the job, as its `GET` route gives it. The
stream ends after the job finishes, with the job as it finished."""

from collections.abc import AsyncIterator
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from services.jobs import JobStore

# For the docs page, which otherwise expects JSON.
EVENT_STREAM_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Server-sent events, each with the job as its data.",
        "content": {
            "text/event-stream": {
                "example": 'data: {"id": "8c1d…", "status": "running", "bytes_done": 1048576, …}\n\n'
            }
        },
    }
}


def job_events(store: JobStore, job_id: str, not_found: str) -> StreamingResponse:
    if store.get(job_id) is None:
        raise HTTPException(status_code=404, detail=not_found)

    async def events() -> AsyncIterator[str]:
        async for job in store.follow(job_id):
            yield f"data: {job.model_dump_json()}\n\n"

    # Without these, a proxy in between could hold the events back.
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(events(), media_type="text/event-stream", headers=headers)
