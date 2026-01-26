from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/submissions", tags=["submissions"])

@router.post("/{lab_code}")
def submit_lab(lab_code: str, github_url: str):
    return {}
