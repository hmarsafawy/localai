"""Background jobs with progress and cancellation, plus the resumable download
they're mostly used for. Knows nothing about what's being downloaded, so model
downloads can use it the same way runtime installs do.

Jobs live in memory and don't survive a restart. A partial download does, as a
file on disk, so running the job again picks up where it stopped.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from models.jobs import JobInfo

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = ("queued", "running")
MAX_FINISHED_JOBS = 100
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
SPEED_SAMPLE_SECONDS = 0.5
DOWNLOAD_TIMEOUT = httpx.Timeout(30.0, read=60.0)


class JobCancelled(Exception):
    """Raised inside a job's work once someone has asked it to stop."""


class DownloadIntegrityError(Exception):
    """The downloaded file doesn't match its published checksum."""


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class _JobRecord:
    info: JobInfo
    cancel: threading.Event = field(default_factory=threading.Event)
    speed_sample_at: float = 0.0
    speed_sample_bytes: int = 0


class JobContext:
    """What a job's work uses to report progress and notice cancellation."""

    def __init__(self, store: JobStore, record: _JobRecord) -> None:
        self._store = store
        self._record = record

    def update(self, **fields: Any) -> None:
        with self._store.lock:
            for name, value in fields.items():
                setattr(self._record.info, name, value)

    def note(self, text: str) -> None:
        with self._store.lock:
            self._record.info.notes.append(text)

    def begin_transfer(self, bytes_total: int | None, bytes_done: int = 0) -> None:
        with self._store.lock:
            self._record.info.bytes_total = bytes_total
            self._record.info.bytes_done = bytes_done
            self._record.info.speed_bytes_per_second = None
            self._record.speed_sample_at = time.monotonic()
            self._record.speed_sample_bytes = bytes_done

    def progress(self, bytes_done: int) -> None:
        with self._store.lock:
            record = self._record
            record.info.bytes_done = bytes_done
            now = time.monotonic()
            elapsed = now - record.speed_sample_at
            if elapsed >= SPEED_SAMPLE_SECONDS:
                record.info.speed_bytes_per_second = (bytes_done - record.speed_sample_bytes) / elapsed
                record.speed_sample_at = now
                record.speed_sample_bytes = bytes_done

    def check_cancelled(self) -> None:
        if self._record.cancel.is_set():
            raise JobCancelled


class JobStore:
    def __init__(self, max_concurrent: int = 2) -> None:
        self.lock = threading.Lock()
        self._jobs: dict[str, _JobRecord] = {}
        self._slots = threading.BoundedSemaphore(max_concurrent)

    def submit(self, kind: str, key: str, work: Callable[[JobContext], dict[str, Any] | None]) -> JobInfo:
        """Starts `work` on a background thread, or returns the job already
        working on `key`. Daemon threads so a download can't block shutdown."""
        with self.lock:
            for record in self._jobs.values():
                if record.info.key == key and record.info.status in ACTIVE_STATUSES:
                    return record.info.model_copy(deep=True)
            record = _JobRecord(JobInfo(id=uuid.uuid4().hex, kind=kind, key=key, created_at=_now()))
            self._jobs[record.info.id] = record
            self._prune()
            snapshot = record.info.model_copy(deep=True)
        threading.Thread(target=self._run, args=(record, work), name=f"job-{kind}", daemon=True).start()
        return snapshot

    def get(self, job_id: str) -> JobInfo | None:
        with self.lock:
            record = self._jobs.get(job_id)
            return record.info.model_copy(deep=True) if record else None

    def cancel(self, job_id: str) -> JobInfo | None:
        """Asks a job to stop. It's a request: the job stops at its next check,
        so it's reported as `cancel_requested` until it has."""
        with self.lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            if record.info.status in ACTIVE_STATUSES:
                record.cancel.set()
                record.info.cancel_requested = True
            return record.info.model_copy(deep=True)

    def is_active(self, key: str) -> bool:
        with self.lock:
            return any(
                record.info.key == key and record.info.status in ACTIVE_STATUSES
                for record in self._jobs.values()
            )

    def _run(self, record: _JobRecord, work: Callable[[JobContext], dict[str, Any] | None]) -> None:
        # Polled rather than blocking so a queued job can be cancelled.
        while not self._slots.acquire(timeout=0.2):
            if record.cancel.is_set():
                self._finish(record, "cancelled")
                return
        try:
            if record.cancel.is_set():
                self._finish(record, "cancelled")
                return
            with self.lock:
                record.info.status = "running"
                record.info.started_at = _now()
            result = work(JobContext(self, record))
            self._finish(record, "completed", result=result)
        except JobCancelled:
            self._finish(record, "cancelled")
        except Exception as error:
            # The last stop before the thread dies: this is how the failure
            # reaches whoever is polling the job, whatever it was.
            logger.exception("Job %s (%s) failed", record.info.id, record.info.kind)
            self._finish(record, "failed", error=str(error) or type(error).__name__)
        finally:
            self._slots.release()

    def _finish(
        self, record: _JobRecord, status: str, result: dict[str, Any] | None = None, error: str | None = None
    ) -> None:
        with self.lock:
            record.info.status = status
            record.info.result = result
            record.info.error = error
            record.info.finished_at = _now()
            record.info.speed_bytes_per_second = None

    def _prune(self) -> None:
        finished = [
            job_id for job_id, record in self._jobs.items() if record.info.status not in ACTIVE_STATUSES
        ]
        for job_id in finished[: max(len(finished) - MAX_FINISHED_JOBS, 0)]:
            del self._jobs[job_id]


def _total_from_content_range(header: str | None) -> int | None:
    match = re.search(r"/(\d+)$", header or "")
    return int(match.group(1)) if match else None


def download_file(
    client: httpx.Client,
    url: str,
    destination: Path,
    context: JobContext,
    *,
    sha256: str | None = None,
    base_bytes: int = 0,
) -> None:
    """Downloads to `destination`, continuing from whatever's already there, and
    checks the whole file against `sha256` when one is given. `base_bytes` is
    what earlier files in the same job transferred, for its overall progress.
    A cancelled download's partial file is kept so it can resume."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}

    with client.stream("GET", url, headers=headers, follow_redirects=True, timeout=DOWNLOAD_TIMEOUT) as response:
        if response.status_code == 416:
            # Asking past the end is what a finished file looks like. Anything
            # else means the partial belongs to some other version of the file.
            if _total_from_content_range(response.headers.get("Content-Range")) != existing:
                destination.unlink()
                download_file(client, url, destination, context, sha256=sha256, base_bytes=base_bytes)
                return
        else:
            response.raise_for_status()
            if response.status_code != 206:
                existing = 0  # The server ignored the range and sent it all.
            with open(destination, "ab" if existing else "wb") as output:
                written = existing
                context.progress(base_bytes + written)
                for chunk in response.iter_bytes(DOWNLOAD_CHUNK_BYTES):
                    context.check_cancelled()
                    output.write(chunk)
                    written += len(chunk)
                    context.progress(base_bytes + written)

    if sha256 is not None:
        context.update(stage="verifying_checksum")
        actual = _sha256_of(destination, context)
        if actual.lower() != sha256.lower():
            destination.unlink()
            raise DownloadIntegrityError(
                f"{destination.name} failed its SHA-256 check (expected {sha256}, got {actual}). "
                "The file was removed; try again."
            )


def _sha256_of(path: Path, context: JobContext) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(DOWNLOAD_CHUNK_BYTES):
            context.check_cancelled()
            digest.update(chunk)
    return digest.hexdigest()
