"""POST /registry/promote request — mirrors contracts/promote_v1.json."""

from datetime import datetime
from pydantic import BaseModel


class PromoteRequest(BaseModel):
    model_uri: str
    approved_by: str
    approval_id: str
    investigation_id: str
    timestamp: datetime
