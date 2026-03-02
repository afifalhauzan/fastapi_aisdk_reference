from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, Query, Request as FastAPIRequest
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import asyncio
import json
from .config import GEMINI_API_KEY, GEMINI_MODEL
from .utils.prompt import ClientMessage, convert_to_gemini_messages
from .utils.stream import patch_response_with_headers, stream_text
from .utils.tools import AVAILABLE_TOOLS, TOOL_DEFINITIONS

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


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
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        }
    )
    return patch_response_with_headers(response, protocol)


@app.get("/api/test-stream")
async def test_streaming():
    """Test endpoint to debug streaming behavior"""
    async def generate_test_stream():
        test_message = "This is a test of incremental streaming. Each word should appear one by one in the frontend."
        words = test_message.split()
        
        # Send start event
        yield f"data: {json.dumps({'type': 'start', 'messageId': 'test-123'})}\n\n"
        await asyncio.sleep(0.1)
        
        # Send text start event  
        yield f"data: {json.dumps({'type': 'text-start', 'id': 'text-1'})}\n\n"
        await asyncio.sleep(0.1)
        
        # Stream each word with delay
        for i, word in enumerate(words):
            text_delta = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'type': 'text-delta', 'id': 'text-1', 'delta': text_delta})}\n\n"
            print(f"[TEST] Sent word {i+1}/{len(words)}: '{text_delta}'")
            await asyncio.sleep(0.3)  # 300ms delay between words
        
        # Send end events
        yield f"data: {json.dumps({'type': 'text-end', 'id': 'text-1'})}\n\n"
        yield f"data: {json.dumps({'type': 'finish', 'messageMetadata': {'finishReason': 'stop'}})}\n\n"
        yield "data: [DONE]\n\n"
    
    response = StreamingResponse(
        generate_test_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive", 
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        }
    )
    return response
