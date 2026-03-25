from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from briefgpt_arxiv.models import IngestionJob
from briefgpt_arxiv.utils import utcnow_naive


class JobTracker:
    def __init__(self, session: Session) -> None:
        self.session = session

    def start(self, job_type: str, target_id: int) -> IngestionJob:
        attempt_count = self._next_attempt_count(job_type, target_id)
        job = IngestionJob(
            job_type=job_type,
            target_id=target_id,
            status="started",
            attempt_count=attempt_count,
        )
        self.session.add(job)
        self.session.flush()
        return job

    def finish(
        self,
        job: IngestionJob,
        *,
        status: str = "completed",
        error_message: str | None = None,
    ) -> IngestionJob:
        job.status = status
        job.error_message = error_message
        job.finished_at = utcnow_naive()
        self.session.add(job)
        return job

    def record_failure(self, job_type: str, target_id: int, error_message: str) -> IngestionJob:
        job = IngestionJob(
            job_type=job_type,
            target_id=target_id,
            status="failed",
            attempt_count=self._next_attempt_count(job_type, target_id),
            error_message=error_message,
            finished_at=utcnow_naive(),
        )
        self.session.add(job)
        self.session.flush()
        return job

    def _next_attempt_count(self, job_type: str, target_id: int) -> int:
        existing_max = self.session.scalar(
            select(func.max(IngestionJob.attempt_count)).where(
                IngestionJob.job_type == job_type,
                IngestionJob.target_id == target_id,
            )
        )
        return (existing_max or 0) + 1
