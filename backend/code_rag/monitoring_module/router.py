from fastapi import APIRouter
from sqlalchemy.ext.asyncio import AsyncSession
from database import SessionLocal
from models import PipelineLog

router = APIRouter()

@router.get("/status")
async def get_status():
    return {"status": "Monitoring работает"}

@router.post("/log")
async def save_log(status: str, message: str):
    async with SessionLocal() as session:
        log = PipelineLog(status=status, message=message)
        session.add(log)
        await session.commit()
    return {"message": "Сохранено"}