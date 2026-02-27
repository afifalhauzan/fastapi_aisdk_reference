import json
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict

from .attachment import ClientAttachment


class ToolInvocationState(str, Enum):
    CALL = 'call'
    PARTIAL_CALL = 'partial-call'
    RESULT = 'result'

class ToolInvocation(BaseModel):
    state: ToolInvocationState
    toolCallId: str
    toolName: str
    args: Any
    result: Any


class ClientMessagePart(BaseModel):
    type: str
    text: Optional[str] = None
    contentType: Optional[str] = None
    url: Optional[str] = None
    data: Optional[Any] = None
    toolCallId: Optional[str] = None
    toolName: Optional[str] = None
    state: Optional[str] = None
    input: Optional[Any] = None
    output: Optional[Any] = None
    args: Optional[Any] = None

    model_config = ConfigDict(extra="allow")


class ClientMessage(BaseModel):
    role: str
    content: Optional[str] = None
    parts: Optional[List[ClientMessagePart]] = None
    experimental_attachments: Optional[List[ClientAttachment]] = None
    toolInvocations: Optional[List[ToolInvocation]] = None


def convert_to_gemini_messages(messages: List[ClientMessage]) -> List[dict]:
    gemini_messages = []

    system_instruction = None
    for message in messages:
        if message.role == 'system':
            system_instruction = message.content
            continue
            
        message_parts = []
        
        if message.content:
            message_parts.append({"text": message.content})
        
        if message.parts:
            for part in message.parts:
                if part.type == 'text' and part.text:
                    message_parts.append({"text": part.text})
                elif part.type == 'file' and part.contentType and part.contentType.startswith('image') and part.url:
                    # Handle image attachments for Gemini
                    message_parts.append({
                        "inline_data": {
                            "mime_type": part.contentType,
                            "data": part.url  # This might need base64 encoding depending on your setup
                        }
                    })
        
        if message.experimental_attachments:
            for attachment in message.experimental_attachments:
                if attachment.contentType.startswith('image'):
                    message_parts.append({
                        "inline_data": {
                            "mime_type": attachment.contentType,
                            "data": attachment.url
                        }
                    })
        
        # Map OpenAI roles to Gemini roles
        gemini_role = "user" if message.role in ["user", "system"] else "model"
        
        if message_parts:
            gemini_messages.append({
                "role": gemini_role,
                "parts": message_parts
            })
    
    return gemini_messages, system_instruction
