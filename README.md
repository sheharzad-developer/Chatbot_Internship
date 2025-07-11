# Reflect Agent Chatbot

A sophisticated chatbot built with FastAPI featuring a Reflect Agent with conditional edges, RAG (Retrieval-Augmented Generation), Tavily web search, MongoDB chat history, Langfuse observability, and a **beautiful Gradio web interface**.

## Features

- 🤖 **Reflect Agent** with conditional edges using LangGraph
- 🔍 **RAG System** with document chunking and vector search (sklearn + Sentence Transformers)
- 🌐 **Tavily Web Search** for real-time information
- 💬 **Chat History** stored in MongoDB
- 📊 **Langfuse Integration** for observability and tracing
- 🚀 **FastAPI** with comprehensive REST API
- 🎨 **Gradio Frontend** - Beautiful web interface
- 🔒 **Environment Variables** for secure configuration

## Project Structure

```
chatbot/
├── agents/
│   ├── __init__.py
│   └── reflect_agent.py       # Main reflect agent with conditional edges
├── config/
│   ├── __init__.py
│   └── settings.py           # Environment configuration
├── database/
│   ├── __init__.py
│   └── mongodb.py           # MongoDB operations
├── frontend/
│   ├── __init__.py
│   └── gradio_app.py        # Gradio web interface
├── models/
│   ├── __init__.py
│   └── chat.py             # Pydantic models
├── services/
│   ├── __init__.py
│   ├── rag_service.py      # RAG with chunking and vector search
│   └── tavily_search.py    # Tavily web search service
├── main.py                 # FastAPI application
├── run.py                  # FastAPI startup script
├── run_frontend.py         # Gradio frontend launcher
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd chatbot
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys and credentials
   ```

## Usage

### 🎨 **Option 1: Web Interface (Recommended)**

```bash
# Start the beautiful Gradio web interface
python3 run_frontend.py
```

This will:
- ✅ Check if the backend is running (auto-start if needed)
- 🌐 Launch the web interface at http://localhost:7860
- 📱 Open your browser automatically

**Features of the Web Interface:**
- 💬 **Chat Tab**: Conversational interface with your reflect agent
- 📚 **Document Management**: Add documents and search RAG system
- 📊 **System Status**: Health checks and chat history
- 🎯 **User-Friendly**: No need for curl commands!

### ⚙️ **Option 2: API Only**

```bash
# 1. Start the backend API
python3 run.py
# or
python3 main.py

# 2. Access API documentation
# Visit: http://localhost:8000/docs
```

## Access Points

### 🎨 **Web Interface** (Main Access)
- **Gradio Frontend**: http://localhost:7860

### 🔧 **API Endpoints**
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Alternative Docs**: http://localhost:8000/redoc

## How It Works

### Reflect Agent with Conditional Edges

The agent uses LangGraph to implement a sophisticated reasoning flow:

1. **Think Node**: Analyzes the input and decides what action to take
2. **Conditional Edge**: Decides whether to act or provide a direct answer
3. **Act Node**: Performs web search or document search if needed
4. **Observe Node**: Processes the results
5. **Final Answer Node**: Generates the response

The agent **doesn't always follow** the thought→action→observation pattern. It uses conditional edges to:
- Skip actions when a direct answer is sufficient
- Limit iterations to prevent infinite loops
- Choose between web search and document search based on the query

### RAG System

- **Document Chunking**: Uses tiktoken to split documents into manageable chunks
- **Vector Embeddings**: Uses Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Search**: sklearn NearestNeighbors for fast similarity search
- **Persistence**: Saves index and metadata to disk

### Services Integration

- **Tavily**: Real-time web search for current information
- **MongoDB**: Persistent chat history storage
- **Langfuse**: Observability and tracing for debugging and monitoring

## Quick Test

### 🎨 **Via Web Interface:**
1. Run `python3 run_frontend.py`
2. Open http://localhost:7860
3. Start chatting in the web interface!

### 🔧 **Via API:**
```bash
# Chat with the bot
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What is the latest news about AI?"}'

# Add a document to RAG
curl -X POST "http://localhost:8000/rag/add-document" \
     -d "content=Your document content here&title=My Document"

# Search documents
curl -X GET "http://localhost:8000/rag/search?query=your search query"
```

## Development

The project follows best practices:

- **Modular Architecture**: Clean separation of concerns
- **Environment Variables**: No hardcoded secrets
- **Async/Await**: Efficient async operations
- **Error Handling**: Comprehensive error handling
- **Logging**: Structured logging throughout
- **Type Hints**: Full type annotation
- **Beautiful UI**: User-friendly Gradio interface
- **Documentation**: Comprehensive API docs

## Monitoring

The system includes observability through:

- **Langfuse Tracing**: Track agent reasoning and performance
- **Health Checks**: Monitor system status via web UI or API
- **Structured Logging**: Debug and monitor operations
- **RAG Statistics**: Track document and search metrics
- **Web Dashboard**: Real-time system status in Gradio

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. 