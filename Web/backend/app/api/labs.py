from fastapi import APIRouter

router = APIRouter(prefix="/api/labs", tags=["labs"])

@router.get("")
def list_labs(phase: str = None):
    return []

@router.get("/{lab_code}")
def get_lab(lab_code: str):
    return {}
