from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class JobInfo(BaseModel):
    """A long-running background task, such as a download and install."""

    id: str
    kind: str
    # What the job is working on; a second request for it joins this job.
    key: str
    status: str = "queued"  # "queued" | "running" | "completed" | "failed" | "cancelled"
    stage: str | None = None
    # What the job was asked to do, once any defaults are resolved.
    details: dict[str, Any] = Field(default_factory=dict)
    current_file: str | None = None
    bytes_done: int = 0
    bytes_total: int | None = None
    speed_bytes_per_second: float | None = None
    cancel_requested: bool = False
    notes: list[str] = Field(default_factory=list)
    error: str | None = None
    result: dict[str, Any] | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
