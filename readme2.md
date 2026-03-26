# Two-Step Gemini Voice Agent

A low-latency, full-duplex voice assistant that runs as a local Python server and is
accessible via any modern browser.

## Pipeline

```
Mic → RNNoise VAD → WAV (16 kHz)
           │
           ▼  WebSocket
    ┌──────────────────────────────────┐
    │  Step 1  audio/wav → Gemini      │
    │          gemini-3.1-flash-lite   │
    │          → plain transcript      │
    │                                  │
    │  Step 2  transcript + history    │
    │          → Gemini agent loop     │
    │            (MCP tool calls)      │
    │          → response text         │
    │                                  │
    │  Step 3  text → Gemini TTS       │
    │          → streaming PCM chunks  │
    └──────────────────────────────────┘
           │
           ▼  WebSocket (audio_chunk)
    Browser AudioContext → speakers
```

**Barge-in**: when the user starts speaking while TTS is playing the frontend
immediately sends a `barge_in` message, the backend cancels the TTS task, and
the browser stops playback.

## Stack

| Layer    | Tech                                             |
|----------|--------------------------------------------------|
| Frontend | HTML + CSS + vanilla JS (no build tools)         |
| VAD      | RNNoise-WASM (energy-based fallback if offline)  |
| Backend  | Python · FastAPI · Uvicorn                       |
| AI       | `google-genai` SDK                               |
| STT      | `gemini-3.1-flash-lite-preview` (multimodal)     |
| Agent    | `gemini-3.1-flash-lite-preview` + MCP tools      |
| TTS      | `gemini-2.5-flash-preview-tts` (streaming PCM)   |
| Tools    | Any JSON-RPC MCP server (optional)               |

## Quick Start

### 1 – Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 – Configuration

```bash
cp .env.example .env
# Edit .env – at minimum set GEMINI_API_KEY
```

| Variable        | Required | Default                          | Notes                        |
|-----------------|----------|----------------------------------|------------------------------|
| `GEMINI_API_KEY`| ✅       | —                                | Get at aistudio.google.com   |
| `AUDIO_MODEL`   | no       | `gemini-3.1-flash-lite-preview`  |                              |
| `TTS_MODEL`     | no       | `gemini-2.5-flash-preview-tts`   |                              |
| `TTS_VOICE`     | no       | `Aoede`                          | Kore, Charon, Fenrir, etc.   |
| `MCP_SERVER_URL`| no       | *(empty)*                        | JSON-RPC MCP endpoint        |
| `PORT`          | no       | `8000`                           |                              |

### 3 – Run

```bash
python server.py
```

Open **http://localhost:8000** in your browser.

To expose it from VS Code, open the *Ports* panel (⇧⌘P → "Forward a Port") and
forward port `8000`. VS Code will give you a public HTTPS URL.

## MCP Tool Calling

Set `MCP_SERVER_URL` to any MCP-compatible JSON-RPC server. The backend
discovers the tool list on connection and passes them to Gemini as function
declarations. When Gemini emits a function call the backend executes it via
`tools/call` and feeds the result back – this loops until Gemini produces a
plain-text response.

## Browser Requirements

Chrome/Edge 89+, Firefox 114+, Safari 16.4+ (Web Audio API + ES modules).
Microphone permission required.
