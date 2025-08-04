from typing import List
from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    question: str

class Source(BaseModel):
    doc: str
    snippet: str

class AnswerPayload(BaseModel):
    answer: str
    category: str
    confidence: float = Field(..., ge=0, le=1)
    sources: List[Source] = Field(default_factory=list)

    class Config:
        extra = "forbid"

class IngestRequest(BaseModel):
    document_name: Optional[str] = Field(None, alias="Document_Name", description="Name of the document to ingest.")

class IngestResponse(BaseModel):
    message: str = Field(..., description="Status message for the ingestion task.")
    filename: str = Field(..., description="Name of the ingested file.")
    task_id: Optional[str] = Field(None, description="Optional task ID for tracking.")

