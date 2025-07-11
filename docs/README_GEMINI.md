# Google Gemini API Integration

This project now includes direct Google Gemini API integration using the `google-genai` client.

## 📁 Project Structure

```
chatbot/
├── agents/
│   ├── gemini_agent.py          # Gemini agent implementation
│   └── simple_agent.py          # Original LangChain agent
├── services/
│   ├── gemini_service.py        # Direct Gemini API service
│   └── rag_service.py           # RAG service
├── examples/
│   └── gemini_example.py        # Standalone Gemini examples
├── tests/
│   └── test_gemini.py           # Integration tests
├── docs/
│   └── GEMINI_SETUP.md          # Setup guide
└── main.py                      # Main FastAPI application
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Environment Variables
Create a `.env` file in the project root:
```env
GOOGLE_AI_GENERATIVE=AIzaSyBY5IBp0zRBqtstrIWYOHvfWNZVku_8IwE
```

### 3. Run Examples
```bash
# Standalone examples with hardcoded API key
python examples/gemini_example.py

# Test integration with environment variables
python tests/test_gemini.py
```

### 4. Start the Web Server
```bash
python main.py
```

## 🔧 API Endpoints

- **Basic Chat:** `POST /chat` (Simple agent)
- **Gemini Chat:** `POST /chat/gemini` (Direct Gemini agent)
- **Agent Selection:** `POST /chat/agent?agent_type=gemini|simple`
- **Test Agents:** `GET /agents/test`
- **Health Check:** `GET /health`

## 📝 Usage Examples

### Basic Gemini Usage
```python
from services.gemini_service import GeminiService

service = GeminiService()
response = await service.generate_simple("Explain AI in simple terms")
print(response)
```

### Agent Usage
```python
from agents.gemini_agent import GeminiAgent

agent = GeminiAgent()
response = await agent.run("Hello, how are you?")
print(response)
```

### API Usage
```bash
# Test Gemini endpoint
curl -X POST "http://localhost:8000/chat/gemini" \
     -H "Content-Type: application/json" \
     -d '{"message": "Explain how AI works"}'

# Test with agent selection
curl -X POST "http://localhost:8000/chat/agent?agent_type=gemini" \
     -H "Content-Type: application/json" \
     -d '{"message": "What is machine learning?"}'
```

## 🧪 Testing

```bash
# Run integration tests
python tests/test_gemini.py

# Test API endpoints
curl http://localhost:8000/agents/test
```

## 🔧 Configuration

The Gemini service supports the following configuration options:

- **Model Selection:** `gemini-2.5-flash` (default)
- **Thinking Mode:** Disabled by default for faster responses
- **Error Handling:** Built-in retry and error logging
- **Environment Variables:** Uses `GOOGLE_AI_GENERATIVE` from `.env`

## 📖 Documentation

- **Setup Guide:** [`docs/GEMINI_SETUP.md`](./GEMINI_SETUP.md)
- **Examples:** [`examples/gemini_example.py`](../examples/gemini_example.py)
- **Tests:** [`tests/test_gemini.py`](../tests/test_gemini.py)

## 🔍 Troubleshooting

1. **API Key Issues:** Ensure `GOOGLE_AI_GENERATIVE` is set correctly
2. **Connection Problems:** Check internet connectivity and API quotas
3. **Import Errors:** Run `pip install google-genai`
4. **Service Errors:** Check logs in the console output

## 🆚 Comparison: Simple vs Gemini Agent

| Feature | Simple Agent | Gemini Agent |
|---------|-------------|--------------|
| Backend | LangChain + Gemini | Direct google-genai |
| Response Speed | Moderate | Fast |
| Configuration | LangChain config | Direct API config |
| Thinking Mode | Not configurable | Configurable |
| Error Handling | LangChain built-in | Custom implementation | 