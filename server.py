"""
Two-Step Gemini Voice Agent – Backend
======================================
Pipeline per user utterance
  1. Audio (WAV) → gemini-3.1-flash-lite-preview → plain transcript
  2. Transcript + conversation history → gemini-3.1-flash-lite-preview
     (with optional MCP tool-calling loop) → response text
  3. Response text → gemini-2.5-flash-preview-tts → streaming PCM audio

WebSocket protocol
  Client → Server
    {"type": "audio",    "data": "<base64-WAV>"}
    {"type": "barge_in"}          – cancel current TTS, ready to listen
    {"type": "ping"}

  Server → Client
    {"type": "status",        "message": "..."}
    {"type": "transcript",    "text": "..."}
    {"type": "response_text", "text": "..."}
    {"type": "audio_chunk",   "data": "<base64-PCM>", "mime_type": "audio/pcm;rate=24000"}
    {"type": "audio_end"}
    {"type": "error",         "message": "..."}
    {"type": "pong"}
"""

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
AUDIO_MODEL: str = os.environ.get("AUDIO_MODEL", "gemini-3.1-flash-lite-preview")
TTS_MODEL: str = os.environ.get("TTS_MODEL", "gemini-2.5-flash-preview-tts")
TTS_VOICE: str = os.environ.get("TTS_VOICE", "Aoede")
MCP_SERVER_URL: str = os.environ.get("MCP_SERVER_URL", "").rstrip("/")
PORT: int = int(os.environ.get("PORT", "8000"))

STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Two-Step Gemini Voice Agent")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/config")
async def get_config() -> dict:
    """Return non-sensitive runtime config to the frontend."""
    return {
        "audio_model": AUDIO_MODEL,
        "tts_model": TTS_MODEL,
        "has_mcp": bool(MCP_SERVER_URL),
    }


# ---------------------------------------------------------------------------
# MCP client (JSON-RPC over HTTP)
# ---------------------------------------------------------------------------
class MCPClient:
    """Minimal MCP client: list tools, call tools."""

    def __init__(self, server_url: str) -> None:
        self._url = server_url
        self._tools: list[dict] = []
        self._loaded = False

    async def load_tools(self) -> list[dict]:
        if self._loaded or not self._url:
            return self._tools
        try:
            async with httpx.AsyncClient(timeout=10.0) as http:
                resp = await http.post(
                    self._url,
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                    headers={"Content-Type": "application/json"},
                )
                data = resp.json()
                self._tools = data.get("result", {}).get("tools", [])
                self._loaded = True
                logger.info("MCP: loaded %d tools from %s", len(self._tools), self._url)
        except Exception as exc:
            logger.warning("MCP load_tools failed: %s", exc)
        return self._tools

    async def call_tool(self, name: str, arguments: dict) -> dict:
        if not self._url:
            return {"error": "MCP not configured"}
        try:
            async with httpx.AsyncClient(timeout=30.0) as http:
                resp = await http.post(
                    self._url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {"name": name, "arguments": arguments},
                    },
                    headers={"Content-Type": "application/json"},
                )
                return resp.json().get("result", {})
        except Exception as exc:
            logger.error("MCP call_tool(%s) failed: %s", name, exc)
            return {"error": str(exc)}

    def to_gemini_tools(self) -> list[types.Tool]:
        if not self._tools:
            return []
        decls = [
            types.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters=t.get("inputSchema", {"type": "object", "properties": {}}),
            )
            for t in self._tools
        ]
        return [types.Tool(function_declarations=decls)]


# ---------------------------------------------------------------------------
# Agent session  (one per WebSocket connection)
# ---------------------------------------------------------------------------
MAX_TOOL_ITERATIONS = 6   # cap agentic loop to prevent runaway tool-calling
MAX_HISTORY_TURNS   = 5   # number of prior exchange pairs kept in context

_SYSTEM_TRANSCRIBE = (
    "You are a speech transcription service. "
    "Listen to the audio and return ONLY the exact words spoken. "
    "Do not add commentary, formatting, or punctuation corrections. "
    "Return the raw transcription only."
)

_SYSTEM_AGENT = (
    "You are a helpful, friendly voice assistant. "
    "Respond concisely and naturally. "
    "Your response will be spoken aloud via text-to-speech, so avoid markdown, "
    "bullet lists, code blocks, or special characters. "
    "Use complete, natural spoken sentences."
)


class AgentSession:
    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._mcp = MCPClient(MCP_SERVER_URL)
        # Conversation history: list of (user_text, assistant_text)
        self._history: list[tuple[str, str]] = []
        self._tts_task: Optional[asyncio.Task] = None
        self._cancel_tts = asyncio.Event()

    # ------------------------------------------------------------------
    # Messaging helpers
    # ------------------------------------------------------------------
    async def _send(self, msg: dict) -> None:
        await self._ws.send_text(json.dumps(msg))

    async def _status(self, message: str) -> None:
        logger.info("Status → %s", message)
        await self._send({"type": "status", "message": message})

    # ------------------------------------------------------------------
    # Step 1 – Audio → Transcript
    # ------------------------------------------------------------------
    async def _transcribe(self, wav_bytes: bytes) -> str:
        """Send WAV audio to the Gemini audio model and return plain transcript."""
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                    types.Part.from_text("Transcribe the audio above."),
                ],
            )
        ]
        resp = await self._client.aio.models.generate_content(
            model=AUDIO_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_TRANSCRIBE,
                temperature=0.0,
            ),
        )
        return (resp.text or "").strip()

    # ------------------------------------------------------------------
    # Step 2 – Transcript → Agent response (with optional MCP tool calls)
    # ------------------------------------------------------------------
    async def _run_agent(self, user_text: str) -> str:
        """Run the agentic loop and return the final text response."""
        # Build context from recent history
        messages: list[types.Content] = []
        for u, a in self._history[-MAX_HISTORY_TURNS:]:
            messages.append(types.Content(role="user", parts=[types.Part.from_text(u)]))
            messages.append(types.Content(role="model", parts=[types.Part.from_text(a)]))
        messages.append(types.Content(role="user", parts=[types.Part.from_text(user_text)]))

        gemini_tools = self._mcp.to_gemini_tools()
        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_AGENT,
            tools=gemini_tools if gemini_tools else None,
            temperature=0.7,
        )

        last_resp = None
        for _iteration in range(MAX_TOOL_ITERATIONS):
            resp = await self._client.aio.models.generate_content(
                model=AUDIO_MODEL,
                contents=messages,
                config=config,
            )
            last_resp = resp

            candidate = resp.candidates[0] if resp.candidates else None
            if not candidate:
                break

            fn_calls = [
                p.function_call
                for p in candidate.content.parts
                if hasattr(p, "function_call") and p.function_call
            ]

            if not fn_calls:
                response_text = resp.text or ""
                # Persist to history
                self._history.append((user_text, response_text))
                return response_text

            # Execute tool calls via MCP
            messages.append(candidate.content)
            fn_responses: list[types.Part] = []
            for fc in fn_calls:
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}
                logger.info("Tool call: %s(%s)", tool_name, tool_args)
                await self._status(f"Using tool: {tool_name}…")
                result = await self._mcp.call_tool(tool_name, tool_args)
                fn_responses.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": result},
                    )
                )
            messages.append(types.Content(role="user", parts=fn_responses))

        fallback = (last_resp.text if last_resp else None) or "I couldn't complete that request."
        self._history.append((user_text, fallback))
        return fallback

    # ------------------------------------------------------------------
    # Step 3 – Response text → Streaming TTS
    # ------------------------------------------------------------------
    async def _stream_tts(self, text: str) -> None:
        """Convert text to speech via Gemini TTS and send PCM chunks to client."""
        self._cancel_tts.clear()
        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=TTS_VOICE)
                )
            ),
        )
        try:
            async for chunk in self._client.aio.models.generate_content_stream(
                model=TTS_MODEL,
                contents=[text],
                config=config,
            ):
                if self._cancel_tts.is_set():
                    logger.info("TTS cancelled (barge-in)")
                    break
                try:
                    part = chunk.candidates[0].content.parts[0]
                    if part.inline_data and part.inline_data.data:
                        await self._send(
                            {
                                "type": "audio_chunk",
                                "data": base64.b64encode(part.inline_data.data).decode(),
                                "mime_type": part.inline_data.mime_type or "audio/pcm;rate=24000",
                            }
                        )
                except (IndexError, AttributeError):
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("TTS error: %s", exc, exc_info=True)
            await self._send({"type": "error", "message": f"TTS error: {exc}"})
        finally:
            await self._send({"type": "audio_end"})

    # ------------------------------------------------------------------
    # Barge-in support
    # ------------------------------------------------------------------
    async def _cancel_tts_stream(self) -> None:
        self._cancel_tts.set()
        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()
            try:
                await self._tts_task
            except (asyncio.CancelledError, Exception):
                pass
        self._tts_task = None

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    async def _process_audio(self, wav_bytes: bytes) -> None:
        try:
            await self._cancel_tts_stream()

            # Step 1
            await self._status("Transcribing…")
            transcript = await self._transcribe(wav_bytes)
            if not transcript:
                await self._status("Could not understand audio – please try again.")
                return
            await self._send({"type": "transcript", "text": transcript})

            # Step 2
            await self._status("Thinking…")
            response = await self._run_agent(transcript)
            if not response:
                await self._status("No response generated – please try again.")
                return
            await self._send({"type": "response_text", "text": response})

            # Step 3
            await self._status("Speaking…")
            self._tts_task = asyncio.create_task(self._stream_tts(response))
            await self._tts_task

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Pipeline error: %s", exc, exc_info=True)
            await self._send({"type": "error", "message": str(exc)})
        finally:
            await self._status("Ready")

    # ------------------------------------------------------------------
    # WebSocket message loop
    # ------------------------------------------------------------------
    async def run(self) -> None:
        logger.info("WebSocket connected")
        await self._status("Initializing…")

        tools = await self._mcp.load_tools()
        suffix = f" ({len(tools)} MCP tools loaded)" if tools else ""
        await self._status(f"Ready{suffix}. Click the mic to start.")

        try:
            while True:
                raw = await self._ws.receive_text()
                msg: dict = json.loads(raw)
                mtype = msg.get("type")

                if mtype == "audio":
                    b64 = msg.get("data", "")
                    if b64:
                        wav_bytes = base64.b64decode(b64)
                        asyncio.create_task(self._process_audio(wav_bytes))

                elif mtype == "barge_in":
                    await self._cancel_tts_stream()
                    await self._send({"type": "audio_end"})
                    await self._status("Listening…")

                elif mtype == "ping":
                    await self._send({"type": "pong"})

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as exc:
            logger.error("WebSocket error: %s", exc, exc_info=True)
        finally:
            await self._cancel_tts_stream()


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session = AgentSession(websocket)
    await session.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=PORT, log_level="info")
