import os
import uuid
import json
import sqlite3
import difflib
from datetime import datetime
from typing import List, Optional, Tuple
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

# -----------------------------
# 1. ROBUST CONFIG & SETUP
# -----------------------------

# FORCE load .env from the current directory
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# DEBUG: Check if key is loaded
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("CRITICAL ERROR: GROQ_API_KEY not found.")
    print(f"Looking for .env at: {env_path.resolve()}")
    # If you are still stuck, you can UNCOMMENT the line below and paste the key directly (not recommended for production)
    # api_key = "gsk_..." 
else:
    print(f"SUCCESS: API Key loaded (starts with {api_key[:5]}...)")

# Database and File Paths
DB_PATH = "sessions.db"
FAQ_PATH = "faqs.json"
MODEL_NAME = "llama-3.1-8b-instant" 

# Configure Groq Client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

SYSTEM_PROMPT = """
You are a helpful AI customer support assistant for ACME Store.

INSTRUCTIONS:
1. Use the provided "Context from Knowledge Base" to answer user questions.
2. If the user says "Hi", "Hello", or engages in small talk, answer politely without using the context.
3. If the answer is NOT in the context and it's a specific product/policy question, OR if the user is angry, you MUST say exactly: "I will connect you to a human agent."
4. Keep answers concise and friendly.
"""

# -----------------------------
# 2. DATABASE HELPERS
# -----------------------------
def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            history TEXT NOT NULL,
            escalated INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()

def load_session(session_id: str) -> Tuple[List[dict], bool]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT history, escalated FROM sessions WHERE id = ?", (session_id,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return [], False
    history_json, escalated = row
    return json.loads(history_json), bool(escalated)

def save_session(session_id: str, history: List[dict], escalated: bool) -> None:
    now = datetime.utcnow().isoformat()
    history_json = json.dumps(history)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO sessions (id, history, escalated, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET 
            history=excluded.history,
            escalated=excluded.escalated,
            updated_at=excluded.updated_at
        """,
        (session_id, history_json, int(escalated), now, now),
    )
    conn.commit()
    conn.close()

# -----------------------------
# 3. FAQ LOGIC
# -----------------------------
FAQS: List[dict] = []

def load_faqs() -> None:
    global FAQS
    if not os.path.exists(FAQ_PATH):
        print(f"WARNING: {FAQ_PATH} not found. Creating empty list.")
        FAQS = []
        return
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        FAQS = json.load(f)

def best_faq_match(question: str) -> Tuple[Optional[dict], float]:
    best_item = None
    best_score = 0.0
    q = question.lower()

    for faq in FAQS:
        ref_q = faq["question"].lower()
        score = difflib.SequenceMatcher(None, q, ref_q).ratio()
        if score > best_score:
            best_score = score
            best_item = faq

    return best_item, best_score

# -----------------------------
# 4. API APP & ENDPOINTS
# -----------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    escalated: bool

app = FastAPI(title="AI Customer Support Bot (Free Version)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    init_db()
    load_faqs()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history, escalated = load_session(session_id)

    history.append({"role": "user", "content": req.message})

    # Get Context
    faq, score = best_faq_match(req.message)
    
    if faq and score > 0.3:
        faq_context = f"Context from Knowledge Base:\nQ: {faq['question']}\nA: {faq['answer']}"
    else:
        faq_context = "Context from Knowledge Base: No relevant FAQ found."

    # Build Prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": faq_context},
    ] + history[-5:] 

    reply = ""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=messages,
            temperature=0.3
        )
        reply = completion.choices[0].message.content.strip()

        if "connect you to a human" in reply.lower():
            escalated = True

    except Exception as e:
        print("AI Provider Error:", e)
        reply = "I'm having trouble connecting to the server. Please try again later."

    history.append({"role": "assistant", "content": reply})
    save_session(session_id, history, escalated)

    return ChatResponse(session_id=session_id, reply=reply, escalated=escalated)

@app.get("/summary/{session_id}")
async def summarize_session(session_id: str):
    history, _ = load_session(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found")

    text_log = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    
    summary_prompt = f"Summarize this support conversation and suggest next actions.\n\n{text_log}"
    
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": summary_prompt}]
    )
    
    return {"summary": completion.choices[0].message.content}

# Mount Frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
