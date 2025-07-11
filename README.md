# Reflect Agent Chatbot

A sophisticated chatbot built with FastAPI featuring a Reflect Agent with conditional edges, RAG (Retrieval-Augmented Generation), Tavily web search, MongoDB chat history, and Langfuse observability.

## Features

- 🤖 **Reflect Agent** with conditional edges using LangGraph
- 🔍 **RAG System** with document chunking and vector search (FAISS + Sentence Transformers)
- 🌐 **Tavily Web Search** for real-time information
- 💬 **Chat History** stored in MongoDB
- 📊 **Langfuse Integration** for observability and tracing
- 🚀 **FastAPI** with comprehensive REST API
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
├── models/
│   ├── __init__.py
│   └── chat.py             # Pydantic models
├── services/
│   ├── __init__.py
│   ├── rag_service.py      # RAG with chunking and vector search
│   └── tavily_search.py    # Tavily web search service
├── main.py                 # FastAPI application
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
   python -m venv venv
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

## Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
# MongoDB Configuration
MONGODB_URL=your_mongodb_connection_string

# Tavily API Configuration
TAVILY_API_KEY=your_tavily_api_key

# Langfuse Configuration
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_HOST=https://cloud.langfuse.com

# Google AI Configuration
GOOGLE_AI_GENERATIVE=your_google_ai_api_key

# XAI Configuration (optional)
XAI_API_KEY=your_xai_api_key
```

## Usage

1. **Start the server:**
   ```bash
   python main.py
   ```
   
   Or with uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the API:**
   - API Documentation: http://localhost:8000/docs
   - API Base URL: http://localhost:8000

## API Endpoints

### Chat Endpoints

- `POST /chat` - Send a message to the chatbot
- `GET /chat/history` - Get chat history (all sessions or by user_id)
- `GET /chat/session/{session_id}` - Get specific chat session
- `DELETE /chat/session/{session_id}` - Delete chat session

### RAG Endpoints

- `POST /rag/add-document` - Add document to RAG system
- `POST /rag/upload-file` - Upload text file to RAG system
- `GET /rag/search` - Search documents in RAG system
- `GET /rag/stats` - Get RAG system statistics

### System Endpoints

- `GET /` - Root endpoint with basic info
- `GET /health` - Health check endpoint

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
- **Vector Search**: FAISS for fast similarity search
- **Persistence**: Saves index and metadata to disk

### Services Integration

- **Tavily**: Real-time web search for current information
- **MongoDB**: Persistent chat history storage
- **Langfuse**: Observability and tracing for debugging and monitoring

## Example Usage

### Chat with the bot:
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What is the latest news about AI?"}'
```

### Add a document to RAG:
```bash
curl -X POST "http://localhost:8000/rag/add-document" \
     -H "Content-Type: application/json" \
     -d '{"content": "Your document content here", "title": "My Document"}'
```

### Search documents:
```bash
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
- **Documentation**: Comprehensive API docs

## Monitoring

The system includes observability through:

- **Langfuse Tracing**: Track agent reasoning and performance
- **Health Checks**: Monitor system status
- **Structured Logging**: Debug and monitor operations
- **RAG Statistics**: Track document and search metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. 