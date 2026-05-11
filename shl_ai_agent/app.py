from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from services.conversation import process_conversation


app = FastAPI(title="SHL AI Assessment Recommender")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    response = process_conversation([msg.model_dump() for msg in request.messages])
    return response
