# Google Gemini API Integration

This project now includes direct Google Gemini API integration using the `google-genai` client.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Environment Variables**
   Create a `.env` file in the project root with your Google AI API key:
   ```
   GOOGLE_AI_GENERATIVE=AIzaSyBY5IBp0zRBqtstrIWYOHvfWNZVku_8IwE
   ```

## Usage Examples

### 1. Basic Usage
```python
from google import genai
import os

os.environ['GEMINI_API_KEY'] = 'your-api-key'
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash", 
    contents="Explain how AI works in a few words"
)
print(response.text)
```

### 2. Advanced Usage with Thinking Disabled
```python
from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disables thinking
    ),
)
print(response.text)
```

## Available Components

1. **GeminiService** (`services/gemini_service.py`)
   - Direct Google Gemini API integration
   - Supports both basic and advanced configuration
   - Built-in error handling and logging

2. **GeminiAgent** (`agents/gemini_agent.py`)
   - Agent wrapper for the Gemini service
   - Compatible with existing chat infrastructure

3. **Test Scripts**
   - `test_gemini.py` - Test the integration with environment variables
   - `gemini_example.py` - Standalone examples with hardcoded API key

## Running Tests

```bash
# Test with environment variables
python test_gemini.py

# Run standalone examples
python gemini_example.py
```

## Integration with Existing Chat System

The new Gemini agent can be used alongside existing agents:

```python
from agents.gemini_agent import GeminiAgent

agent = GeminiAgent()
response = await agent.run("Hello, how are you?")
print(response)
``` 