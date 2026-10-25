"""FastAPI Web Chat & Voice Interface for PERSON 2.

Features:
- Responsive chat interface with audio playback
- REST endpoint for text turns (/api/chat)
- Audio upload and voice message endpoint (/api/voice)
"""
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional

from .knowledge_base import KnowledgeBase
from .retriever import SemanticVectorRetriever
from .tools_adapter import ToolsAdapter
from .conversation_agent import ConversationAgent
from .voice_adapter import GnaniVoiceAdapter

app = FastAPI(title="ZentroMart Support Agent — PERSON 2 (Semantic Vector RAG)")

# Lazy globals
_agent: Optional[ConversationAgent] = None
_voice: Optional[GnaniVoiceAdapter] = None


def get_agent_and_voice():
    global _agent, _voice
    if _agent is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ds_dir = os.path.join(base_dir, 'Capstone_Project_Dataset_FTTT_2026')
        kb = KnowledgeBase(ds_dir)
        retriever = SemanticVectorRetriever(kb)
        tools = ToolsAdapter(ds_dir)
        _agent = ConversationAgent(kb, tools, retriever)
        _voice = GnaniVoiceAdapter(_agent)
    return _agent, _voice


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "web_user_1"


@app.get("/", response_class=HTMLResponse)
async def get_index():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ZentroMart AI Assistant — Semantic Vector Search</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #4f46e5;
                --primary-hover: #4338ca;
                --bg: #0f172a;
                --surface: #1e293b;
                --surface-border: #334155;
                --text: #f8fafc;
                --text-muted: #94a3b8;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
            body { background: var(--bg); color: var(--text); display: flex; flex-direction: column; height: 100vh; }
            header { background: var(--surface); border-bottom: 1px solid var(--surface-border); padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; }
            header h1 { font-size: 1.25rem; font-weight: 600; display: flex; align-items: center; gap: 0.5rem; }
            .badge { background: #312e81; color: #a5b4fc; padding: 0.25rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; }
            #chat-container { flex: 1; overflow-y: auto; padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; }
            .message { max-width: 75%; padding: 0.85rem 1.15rem; border-radius: 1rem; line-height: 1.5; font-size: 0.95rem; }
            .message.user { align-self: flex-end; background: var(--primary); color: #fff; border-bottom-right-radius: 0.2rem; }
            .message.assistant { align-self: flex-start; background: var(--surface); border: 1px solid var(--surface-border); border-bottom-left-radius: 0.2rem; }
            .message .meta { font-size: 0.7rem; color: var(--text-muted); margin-top: 0.4rem; }
            #input-bar { background: var(--surface); border-top: 1px solid var(--surface-border); padding: 1rem 1.5rem; display: flex; gap: 0.75rem; align-items: center; }
            input[type="text"] { flex: 1; background: #0f172a; border: 1px solid var(--surface-border); border-radius: 0.75rem; padding: 0.8rem 1.2rem; color: #fff; font-size: 0.95rem; outline: none; }
            input[type="text"]:focus { border-color: var(--primary); }
            button { background: var(--primary); color: #fff; border: none; border-radius: 0.75rem; padding: 0.8rem 1.4rem; font-weight: 500; cursor: pointer; transition: 0.2s; }
            button:hover { background: var(--primary-hover); }
        </style>
    </head>
    <body>
        <header>
            <h1>ZentroMart Support AI <span class="badge">PERSON 2: Vector Search</span></h1>
            <span style="font-size: 0.85rem; color: var(--text-muted);">Ref Date: 20 Oct 2026</span>
        </header>
        <div id="chat-container">
            <div class="message assistant">
                Hello! I am ZentroMart's AI customer support assistant. How can I help you today?
            </div>
        </div>
        <div id="input-bar">
            <input type="text" id="user-input" placeholder="Ask about orders, returns, warranty, or refunds..." onkeydown="if(event.key==='Enter') sendMessage()" />
            <button onclick="sendMessage()">Send</button>
        </div>
        <script>
            async function sendMessage() {
                const input = document.getElementById('user-input');
                const text = input.value.trim();
                if (!text) return;
                input.value = '';

                const chat = document.getElementById('chat-container');
                const uMsg = document.createElement('div');
                uMsg.className = 'message user';
                uMsg.innerText = text;
                chat.appendChild(uMsg);
                chat.scrollTop = chat.scrollHeight;

                try {
                    const res = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: text })
                    });
                    const data = await res.json();
                    const aMsg = document.createElement('div');
                    aMsg.className = 'message assistant';
                    aMsg.innerText = data.response;
                    if (data.evidence && data.evidence.length > 0) {
                        const meta = document.createElement('div');
                        meta.className = 'meta';
                        meta.innerText = 'Evidence: ' + data.evidence.join(', ');
                        aMsg.appendChild(meta);
                    }
                    chat.appendChild(aMsg);
                    chat.scrollTop = chat.scrollHeight;
                } catch(e) {
                    console.error(e);
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    agent, _ = get_agent_and_voice()
    turn = agent.process_turn(req.session_id, req.message)
    return {
        "response": turn["response"],
        "decision": turn.get("decision", "answer"),
        "evidence": turn.get("evidence", []),
        "tool_calls": turn.get("tool_calls", [])
    }


@app.post("/api/voice")
async def voice_endpoint(file: UploadFile = File(...)):
    agent, voice = get_agent_and_voice()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    turn = voice.process_voice_turn(tmp_path)
    return turn
