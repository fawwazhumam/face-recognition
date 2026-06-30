from pydantic import BaseModel
from typing import Optional

class FacePredictionResponse(BaseModel):
    status: str
    match: bool
    user_id: Optional[str] = None
    name: Optional[str] = None
    confidence_score: Optional[float] = None
    message: Optional[str] = None

class FaceRegisterResponse(BaseModel):
    status: str
    message: str