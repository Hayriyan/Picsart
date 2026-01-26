from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr

router = APIRouter(prefix="/api/auth", tags=["auth"])

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str

@router.post("/register")
def register(user_data: UserRegister):
    return {"message": "Registration endpoint"}

@router.post("/login")
def login(email: EmailStr, password: str):
    return {"access_token": "token", "token_type": "bearer"}
