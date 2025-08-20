# 🤖 Reflect Agent AI - Chatbot Internship Project

<img width="1394" height="968" alt="image" src="https://github.com/user-attachments/assets/18e489cd-c5d7-49bd-9d3c-422f5d50c534" />


A sophisticated AI chatbot application with dual agent support (Google Gemini & LangChain), beautiful web interface, chat history, and document knowledge management.

## 🌟 Features

- **🔮 Google Gemini Integration**: Direct Google GenAI API with your exact implementation
- **🤖 LangChain Agent**: Alternative AI agent with advanced reasoning
- **💬 Chat History**: Full conversation browsing and restoration
- **📚 Document Knowledge**: RAG (Retrieval-Augmented Generation) system
- **🌐 Beautiful UI**: Modern HTML frontend with ChatGPT-style formatting
- **📱 Responsive Design**: Works on desktop and mobile
- **⚡ Real-time**: Instant messaging with proper error handling
- **🔄 Agent Switching**: Compare responses between different AI models

## 🏗️ Architecture

- **Backend**: FastAPI + MongoDB + RAG System
- **Frontend**: Clean HTML/CSS/JavaScript (no frameworks!)
- **Database**: MongoDB for persistent chat history
- **Vector Search**: Sentence Transformers + scikit-learn
- **AI Models**: Google Gemini 2.5 Flash + LangChain agents

## 🚀 Quick Start

### 1️⃣ Setup Environment
```bash
# Clone repository
git clone https://github.com/sheharzad-developer/Chatbot_Internship.git
cd Chatbot_Internship

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Add your Google Gemini API key to .env
```

### 2️⃣ Start Backend
```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

### 3️⃣ Start Frontend
```bash
# In new terminal
python run_frontend.py
```

### 4️⃣ Access Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs

## 🎯 Key Components

### Frontend (`frontend/`)
- `index.html` - Beautiful chat interface with markdown rendering
- `server.py` - Simple HTTP server with CORS support

### Backend (`main.py`)
- `/chat/gemini` - Google Gemini API endpoint
- `/chat` - LangChain agent endpoint  
- `/chat/history` - Retrieve chat sessions
- `/chat/session/{id}` - Load specific conversation

### Agents (`agents/`)
- `gemini_agent.py` - Google GenAI integration
- `simple_agent.py` - LangChain-based agent
- `reflect_agent.py` - Advanced reasoning agent

### Services (`services/`)
- `gemini_service.py` - Direct Google GenAI client
- `rag_service.py` - Document search system
- `tavily_search.py` - Web search integration

## 🔧 Environment Variables

```bash
# Required
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional
MONGODB_URI=mongodb://localhost:27017/chatbot
TAVILY_API_KEY=your_tavily_key
LANGFUSE_SECRET_KEY=your_langfuse_secret
LANGFUSE_PUBLIC_KEY=your_langfuse_public  
LANGFUSE_HOST=https://cloud.langfuse.com
```

## 📁 Project Structure

```
Chatbot_Internship/
├── 🤖 agents/              # AI agent implementations
│   ├── gemini_agent.py     # Google Gemini integration
│   ├── simple_agent.py     # LangChain agent
│   └── reflect_agent.py    # Advanced reasoning
├── ⚙️ config/              # Configuration files
├── 🗄️ database/            # MongoDB connections
├── 🌐 frontend/            # HTML web interface
│   ├── index.html          # Main chat interface
│   └── server.py          # Frontend server
├── 📊 models/              # Pydantic data models
├── 🔧 services/            # External integrations
│   ├── gemini_service.py   # Google GenAI client
│   ├── rag_service.py      # Document search
│   └── tavily_search.py    # Web search
├── 📚 docs/                # Documentation
├── 🧪 examples/            # Code examples
├── 🧪 tests/              # Test files
├── 📄 main.py             # FastAPI backend
├── 🚀 run_frontend.py     # Frontend launcher
└── 📋 requirements.txt    # Dependencies
```

## 🎨 Features Showcase

### Dual Agent System
- **🔮 Gemini Agent**: Lightning-fast responses with Google's latest model
- **🤖 Simple Agent**: Robust LangChain-based reasoning

### Chat History
- **📜 Browse Sessions**: See all your past conversations
- **🔄 Continue Chats**: Click any session to resume
- **📅 Date Tracking**: Organized by creation date

### Beautiful Formatting
- **📝 Markdown Support**: Bold, italic, code blocks
- **• Bullet Points**: Clean list formatting
- **📏 Proper Spacing**: ChatGPT-style text rendering

## 🛠️ Development

### Running Tests
```bash
python -m pytest tests/
```

### API Testing
```bash
# Test Gemini endpoint
curl -X POST "http://localhost:8001/chat/gemini" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "user_id": "test"}'

# Test health
curl http://localhost:8001/health
```

## 📈 Monitoring

- **Health Check**: http://localhost:8001/health
- **System Stats**: Built-in monitoring dashboard
- **Chat Analytics**: Message count and session tracking

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Submit a pull request

## 📄 License

This project is part of an internship program and is licensed under the MIT License.

## 🎯 Internship Goals Achieved

- ✅ Google Gemini API Integration
- ✅ FastAPI Backend Development  
- ✅ MongoDB Database Management
- ✅ Modern Frontend Development
- ✅ Chat History Implementation
- ✅ RAG System Integration
- ✅ Production-Ready Deployment
- ✅ Beautiful UI/UX Design

---

**Built with ❤️ during Software Engineering Internship**
