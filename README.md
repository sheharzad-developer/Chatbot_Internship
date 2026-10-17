# 🤖 Chatbot Internship Project

<img width="1394" height="968" alt="image" src="https://github.com/user-attachments/assets/18e489cd-c5d7-49bd-9d3c-422f5d50c534" />


A fully local AI chatbot: a **local LLM via Ollama**, a **local SQLite database**, a web interface, chat history, and document knowledge (RAG). No cloud API keys required.

## 🌟 Features

- **🦙 Ollama**: Runs a local model (default `llama3.2`) — your data never leaves your machine
- **🗄️ SQLite**: Chat sessions and messages stored in a local file (`data/chatbot.db`)
- **🧠 Conversation Memory**: Earlier turns in a session are sent to the model as context
- **🏢 Multi-Tenant Agents**: Per-organization configs in `config/tenants/*.yaml`
- **📚 Document Knowledge**: RAG (Retrieval-Augmented Generation) system
- **🌐 Web UI**: Plain HTML/CSS/JavaScript frontend with markdown formatting
- **💬 Chat History**: Browse and restore past conversations

## 🏗️ Architecture

- **Backend**: FastAPI + SQLite + RAG System
- **Frontend**: Clean HTML/CSS/JavaScript (no frameworks!)
- **Database**: SQLite (built into Python — nothing to install)
- **LLM**: Ollama local server (`http://localhost:11434`)
- **Vector Search**: Sentence Transformers + scikit-learn

## 🚀 Quick Start

### 1️⃣ Install Ollama and a model
```bash
# Install from https://ollama.com, then:
ollama pull llama3.2
ollama serve   # skip if the Ollama app is already running
```

### 2️⃣ Setup Environment
```bash
git clone https://github.com/sheharzad-developer/Chatbot_Internship.git
cd Chatbot_Internship

python3 -m venv .venv
.venv/bin/pip install -r requirements-local.txt

# Optional — defaults work out of the box
cp .env.example .env
```

### 3️⃣ Start Backend
```bash
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001
```

### 4️⃣ Start Frontend
```bash
# In new terminal
.venv/bin/python run_frontend.py
```

### 5️⃣ Access Application
- **Frontend**: http://localhost:3001
- **Backend API**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs

## 🎯 Key Components

### Frontend (`frontend/`)
- `index.html` - Chat interface with markdown rendering
- `server.py` - Simple HTTP server with CORS support

### Backend (`main.py`)
- `/chat` or `/chat/ollama` - Chat with the local Ollama model (saved to SQLite)
- `/chat/fast` - Chat without saving history
- `/tenants/{id}/chat` - Tool-aware tenant agent
- `/chat/history` - Retrieve chat sessions
- `/chat/session/{id}` - Load or delete a specific conversation

### Agents (`agents/`)
- `ollama_agent.py` - Direct chat with the local model, with session history
- `tool_aware_agent.py` - Multi-tenant agent with tool calling

### Services (`services/`)
- `ollama_service.py` - Ollama HTTP client
- `rag_service.py` - Document search system
- `tavily_search.py` - Web search integration

### Database (`database/`)
- `sqlite_db.py` - SQLite chat store (`chat_sessions` + `chat_messages` tables)

## 🔧 Environment Variables

All optional — the defaults below are used when unset.

```bash
DATABASE_PATH=data/chatbot.db
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_TIMEOUT=120

# Optional integrations
TAVILY_API_KEY=your_tavily_key
LANGFUSE_SECRET_KEY=your_langfuse_secret
LANGFUSE_PUBLIC_KEY=your_langfuse_public
LANGFUSE_HOST=https://cloud.langfuse.com
```

To use a different model, `ollama pull <model>` and set `OLLAMA_MODEL` (direct chat) and `llm.model` in `config/tenants/*.yaml` (tenant chat).

## 🛠️ Development

### API Testing
```bash
# Chat with the local model
curl -X POST "http://localhost:8001/chat/ollama" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "user_id": "test"}'

# Health (shows database and Ollama status)
curl http://localhost:8001/health
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Submit a pull request

## 📄 License

This project is part of an internship program and is licensed under the MIT License.

## 🎯 Internship Goals Achieved

- ✅ Local LLM Integration (Ollama)
- ✅ FastAPI Backend Development  
- ✅ SQLite Database Management
- ✅ Modern Frontend Development
- ✅ Chat History Implementation
- ✅ RAG System Integration
- ✅ Production-Ready Deployment
- ✅ Beautiful UI/UX Design

---

**Built with ❤️ during Software Engineering Internship**
