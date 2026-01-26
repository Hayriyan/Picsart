from celery import Celery
from app.config import settings

celery_app = Celery('lab_platform', broker=settings.REDIS_URL)

@celery_app.task(bind=True)
def run_submission_tests(self, submission_id: int, github_url: str):
    return {"submission_id": submission_id, "passed": False}
