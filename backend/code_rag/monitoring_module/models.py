from sqlalchemy import Column, Integer, String, DateTime
from database import Base
import datetime

class PipelineLog(Base):
    __tablename__ = "pipeline_logs"

    id = Column(Integer, primary_key=True, index=True)
    status = Column(String)
    message = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)