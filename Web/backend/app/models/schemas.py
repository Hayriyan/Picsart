from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field
from enum import Enum

class Phase(str, Enum):
    PHASE_1 = "phase_1"
    PHASE_2 = "phase_2"
    PHASE_3 = "phase_3"

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    hashed_password: str
    name: str
    github_username: Optional[str] = None
    is_instructor: bool = False
    current_phase: Phase = Field(default=Phase.PHASE_1)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Lab(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    code: str = Field(index=True, unique=True)
    title: str
    description: str
    phase: Phase
    points: int
    is_mandatory: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Submission(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    lab_id: int = Field(foreign_key="lab.id")
    student_id: int = Field(foreign_key="user.id")
    github_url: str
    status: str = "queued"
    points_earned: int = 0
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
