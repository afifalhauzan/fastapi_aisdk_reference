from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, Query, Request as FastAPIRequest
from fastapi.responses import StreamingResponse
import google.generativeai as genai
from .config import GEMINI_API_KEY, GEMINI_MODEL
from .utils.prompt import ClientMessage, convert_to_gemini_messages
from .utils.stream import patch_response_with_headers, stream_text
from .utils.tools import AVAILABLE_TOOLS, TOOL_DEFINITIONS

app = FastAPI()


class Request(BaseModel):
    messages: List[ClientMessage]


@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query('data')):
    print(f"\n[API] Incoming chat request - Protocol: {protocol}")
    print(f"[API] Message count: {len(request.messages)}")
    
    messages = request.messages
    gemini_messages, system_instruction = convert_to_gemini_messages(messages)

    print(f"[API] Using model: {GEMINI_MODEL}")
    genai.configure(api_key=GEMINI_API_KEY)
    client = genai.GenerativeModel(GEMINI_MODEL)
    response = StreamingResponse(
        stream_text(client, (gemini_messages, system_instruction), TOOL_DEFINITIONS, AVAILABLE_TOOLS, protocol),
        media_type="text/event-stream",
    )
    return patch_response_with_headers(response, protocol)
