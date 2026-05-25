# 🤖 RAG ChatBot

> Upload any document and chat with it in **1–2 seconds** — powered by Groq's ultra-fast LPU inference.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama_3.1_8B-F55036?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-0467DF?style=for-the-badge)

---

## ✨ Features

- 📄 **Upload documents** — supports `.txt`, `.pdf`, and `.docx`
- ⚡ **1–2 second responses** via Groq's LPU hardware (200–500 tokens/sec)
- 🧠 **RAG pipeline** — retrieves only the most relevant chunks before answering
- 💬 **Chat history** — remembers the last 3 exchanges per session
- 🔍 **Source citations** — shows which document chunks were used to answer
- 🖥️ **Custom frontend** — clean browser-based chat UI, no terminal needed

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Llama 3.1 8B via [Groq API](https://console.groq.com) |
| **Orchestration** | LangChain |
| **Embeddings** | `all-MiniLM-L6-v2` (HuggingFace) |
| **Vector Store** | FAISS (local, persisted to disk) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Vanilla HTML / CSS / JavaScript |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/HameeedCodes932/RAG-ChatBot.git
cd RAG-ChatBot
```

### 2. Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free, no credit card)
3. Click **API Keys** → **Create API Key**
4. Copy the key — it looks like `gsk_xxxxxxxxxxxx`

### 3. Add your API key

Open `backend/config.py` and replace the placeholder:

```python
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
```

Or set it as an environment variable (recommended):

```bash
# Windows
set GROQ_API_KEY=gsk_xxxxxxxxxxxx

# Mac / Linux
export GROQ_API_KEY=gsk_xxxxxxxxxxxx
```

### 4. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 5. Run the server

```bash
uvicorn main:app --reload --port 8000
```

You should see:
```
⏳ Pre-loading embedding model...
⏳ Pre-loading FAISS index...
✅ Ready — Groq (Llama 3.1 8B) + FAISS loaded.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 6. Open the chatbot

Go to **http://localhost:8000** in your browser. That's it!

---

## 📁 Project Structure

```
RAG-ChatBot/
├── backend/
│   ├── main.py          # FastAPI app & API endpoints
│   ├── rag_chain.py     # LangChain RAG chain with Groq
│   ├── ingestion.py     # Document chunking & FAISS indexing
│   ├── config.py        # All settings (model, chunking, prompts)
│   └── requirements.txt
├── frontend/
│   ├── index.html       # Chat UI
│   ├── index.css        # Styles
│   └── index.js         # Frontend logic
├── data/
│   └── faiss_index/     # Persisted vector store (auto-generated)
├── .env.example         # Environment variable template
└── README.md
```

---

## ⚙️ Configuration

All settings are in `backend/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model to use |
| `CHUNK_SIZE` | `500` | Characters per document chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `max_tokens` | `256` | Max tokens in each response |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Check server + Groq connectivity |
| `POST` | `/api/upload` | Upload a document for ingestion |
| `POST` | `/api/chat` | Send a question, get an answer |
| `GET` | `/api/documents` | List all ingested documents |
| `POST` | `/api/clear-history` | Clear chat history for a session |

---

## 🛡️ Security Note

Never commit your Groq API key to GitHub. Use environment variables or a `.env` file and make sure `.env` is listed in `.gitignore`.

---

## 👨‍💻 Author

**Fazal Hameed**
GitHub: [@HameeedCodes932](https://github.com/HameeedCodes932)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
