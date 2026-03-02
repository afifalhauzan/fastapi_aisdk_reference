import json
import traceback
import uuid
from typing import Any, Callable, Dict, Mapping, Sequence

from fastapi.responses import StreamingResponse
import google.generativeai as genai


def stream_text(
    model: genai.GenerativeModel,
    messages_and_system: tuple,
    tool_definitions: Sequence[Dict[str, Any]],
    available_tools: Mapping[str, Callable[..., Any]],
    protocol: str = "data",
):
    """Yield Server-Sent Events for a streaming chat completion."""
    try:
        messages, system_instruction = messages_and_system
        
        print(f"[STREAM] Starting stream with {len(messages)} messages")
        print(f"[STREAM] System instruction: {system_instruction[:100] + '...' if system_instruction and len(system_instruction) > 100 else system_instruction}")
        
        def format_sse(payload: dict) -> str:
            return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"

        message_id = f"msg-{uuid.uuid4().hex}"
        text_stream_id = "text-1"
        text_started = False
        text_finished = False
        finish_reason = None

        print(f"[STREAM] Message ID: {message_id}")
        yield format_sse({"type": "start", "messageId": message_id})

        # Convert tool definitions to Gemini format
        gemini_tools = None
        if tool_definitions:
            print(f"[TOOLS] Setting up {len(tool_definitions)} tools")
            gemini_tools = []
            for tool_def in tool_definitions:
                function_def = tool_def["function"]
                gemini_tools.append({
                    "function_declarations": [{
                        "name": function_def["name"],
                        "description": function_def["description"],
                        "parameters": function_def["parameters"]
                    }]
                })
        else:
            print("[TOOLS] No tools configured")

        # Create generation config
        generation_config = genai.GenerationConfig(
            temperature=0.7,
        )
        print(f"[CONFIG] Generation config: temperature=0.7")

        # Start chat without tools in start_chat
        chat = model.start_chat(history=[])
        print("[CHAT] Chat session started")
        
        chunk_count = 0

        # Send the conversation history and get streaming response
        if messages:
            # For Gemini, we need to send the last message and get a streaming response
            last_message = messages[-1]
            message_parts = last_message["parts"]
            
            # Convert parts to text for streaming
            message_text = ""
            for part in message_parts:
                if "text" in part:
                    message_text += part["text"]
            
            # Handle system instruction by prepending it
            if system_instruction:
                message_text = f"{system_instruction}\n\n{message_text}"
            
            print(f"[GEMINI] Sending message to Gemini: {message_text[:200]}{'...' if len(message_text) > 200 else ''}")
            print(f"[GEMINI] Tools enabled: {'Yes' if gemini_tools else 'No'}")
            
            # Pass tools to send_message instead
            response = chat.send_message(
                message_text,
                generation_config=generation_config,
                tools=gemini_tools,
                stream=True
            )
            
            print("[GEMINI] Starting stream processing...")

            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    if not text_started:
                        print("[TEXT] Text streaming started")
                        yield format_sse({"type": "text-start", "id": text_stream_id})
                        text_started = True
                    print(f"[TEXT] Text chunk {chunk_count}: {repr(chunk.text[:100])}")
                    yield format_sse(
                        {"type": "text-delta", "id": text_stream_id, "delta": chunk.text}
                    )
                
                # Handle function calls
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call'):
                                tool_call_id = f"call_{uuid.uuid4().hex}"
                                function_call = part.function_call
                                tool_name = function_call.name
                                
                                print(f"[TOOL] Function call detected: {tool_name}")
                                
                                yield format_sse({
                                    "type": "tool-input-start",
                                    "toolCallId": tool_call_id,
                                    "toolName": tool_name,
                                })
                                
                                # Handle function arguments safely
                                arguments = {}
                                if hasattr(function_call, 'args') and function_call.args is not None:
                                    arguments = dict(function_call.args)
                                
                                print(f"[TOOL] Function arguments: {arguments}")
                                
                                yield format_sse({
                                    "type": "tool-input-available",
                                    "toolCallId": tool_call_id,
                                    "toolName": tool_name,
                                    "input": arguments,
                                })
                                
                                # Execute the tool
                                tool_function = available_tools.get(tool_name)
                                if tool_function:
                                    print(f"[TOOL] Executing tool: {tool_name}")
                                    try:
                                        tool_result = tool_function(**arguments)
                                        print(f"[TOOL] Tool result: {str(tool_result)[:200]}{'...' if len(str(tool_result)) > 200 else ''}")
                                        yield format_sse({
                                            "type": "tool-output-available",
                                            "toolCallId": tool_call_id,
                                            "output": tool_result,
                                        })
                                    except Exception as error:
                                        print(f"[ERROR] Tool execution error: {str(error)}")
                                        yield format_sse({
                                            "type": "tool-output-error",
                                            "toolCallId": tool_call_id,
                                            "errorText": str(error),
                                        })
                                else:
                                    print(f"[ERROR] Tool not found: {tool_name}")
                                    yield format_sse({
                                        "type": "tool-output-error",
                                        "toolCallId": tool_call_id,
                                        "errorText": f"Tool '{tool_name}' not found.",
                                    })

        if text_started and not text_finished:
            print("[TEXT] Text streaming finished")
            yield format_sse({"type": "text-end", "id": text_stream_id})
            text_finished = True

        finish_metadata: Dict[str, Any] = {
            "finishReason": "stop"
        }
        
        print(f"[STREAM] Stream completed after {chunk_count} chunks")
        yield format_sse({"type": "finish", "messageMetadata": finish_metadata})
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        print(f"[ERROR] Stream error: {str(e)}")
        traceback.print_exc()
        raise


def patch_response_with_headers(
    response: StreamingResponse,
    protocol: str = "data",
) -> StreamingResponse:
    """Apply the standard streaming headers expected by the Vercel AI SDK."""

    response.headers["x-vercel-ai-ui-message-stream"] = "v1"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"

    if protocol:
        response.headers.setdefault("x-vercel-ai-protocol", protocol)

    return response
