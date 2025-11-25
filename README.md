# 🤖 AI Support Bot (FastAPI + Groq)

This project is an **AI-powered customer support bot** built with **FastAPI**.  
It uses:

- A **Groq LLM model** (`llama-3.1-8b-instant`) for answers  
- A **FAQ JSON file** as a small knowledge base  
- A **SQLite database** to store chat sessions and escalation status  
- A simple **frontend** served from the `static/` folder  

The bot acts as a support assistant for **ACME Store** and can either answer from FAQs or escalate to a human.

---

## ✨ Features

- Chat endpoint to talk with the bot (`/chat`)
- Uses **FAQ matching** with fuzzy similarity
- Remembers **chat history per session** in SQLite
- Can **escalate to human** when needed
- Summary endpoint to **summarize a full conversation** (`/summary/{session_id}`)
- Serves a static frontend from `/static` at `/`

---

## 🧱 Tech Stack

- **Backend:** FastAPI
- **Model Provider:** Groq (via OpenAI-compatible API)
- **Database:** SQLite (`sessions.db`)
- **Config:** `.env` file (GROQ_API_KEY)
- **Frontend:** Static files (HTML/CSS/JS) under `static/`

---

## 📁 Project Structure

```bash
ai-support-bot/
├── main.py          # FastAPI backend & logic
├── faqs.json        # FAQ knowledge base (Q/A pairs)
├── sessions.db      # SQLite database (auto-created)
├── .env             # Environment variables (GROQ_API_KEY)
└── static/          # Frontend files (index.html, JS, CSS, etc.)



