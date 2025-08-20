# 🤖 Gradio Chatbot Interface

This is a Python Gradio version of the HTML chatbot interface. It provides the same functionality as the original HTML interface but using Gradio's modern UI components.

## Features

✨ **Core Features:**
- 💬 Interactive chat interface with both Simple and Gemini AI agents
- ⚡ Fast mode option (skip database for faster responses)
- 🔄 Real-time message exchange with backend API
- 📱 Responsive design that works on desktop and mobile

🛠️ **Management Features:**
- 💚 System health monitoring
- 📜 Chat history viewing
- 🗑️ Clear chat functionality
- 🚀 Agent switching (Simple/Gemini)

## Requirements

- Python 3.8+
- Backend chatbot API running on `localhost:8000`

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Make sure your backend is running:**
   - Ensure your chatbot backend API is running on `http://localhost:8000`
   - The backend should have the following endpoints available:
     - `/health` - System health check
     - `/chat` - Simple agent chat
     - `/chat/gemini` - Gemini agent chat  
     - `/chat/fast` - Fast mode chat
     - `/chat/history` - Chat history
     - `/chat/session/{id}` - Load specific chat session

## Usage

1. **Start the Gradio interface:**
   ```bash
   python gradio.py
   ```

2. **Access the interface:**
   - Open your browser and go to `http://localhost:7860`
   - Or use the public URL if share mode is enabled

3. **Using the interface:**
   - **Choose Agent:** Select between Simple or Gemini AI agents
   - **Fast Mode:** Toggle for faster responses (skips database)
   - **Chat:** Type messages in the input box and press Enter or click Send
   - **Health Check:** Click "System Health" to check backend status
   - **History:** Click "Chat History" to view recent conversations
   - **Clear:** Click "Clear Chat" to start a fresh conversation

## Configuration

You can modify these settings in the `gradio.py` file:

- **API_BASE:** Change the backend URL (default: `http://localhost:8000`)
- **Server Port:** Change the Gradio port (default: `7860`)
- **Share Mode:** Set `share=True` in `interface.launch()` for public access

## API Compatibility

This Gradio interface is fully compatible with the same backend API used by the HTML version. It makes the same HTTP requests and expects the same response format:

**Request Format:**
```json
{
    "message": "User message",
    "session_id": "session_uuid",
    "user_id": "gradio_user",
    "fast_mode": true/false
}
```

**Response Format:**
```json
{
    "response": "AI response",
    "session_id": "session_uuid"
}
```

## Differences from HTML Version

🔄 **Similarities:**
- Same core functionality
- Same API endpoints
- Same agent selection
- Same fast mode feature

✨ **Gradio Advantages:**
- Modern, responsive UI
- Better mobile support
- Built-in markdown rendering
- Automatic input validation
- Better error handling
- Easier to extend and modify

📱 **UI Differences:**
- Chat history is displayed in a status panel instead of sidebar
- More compact layout optimized for various screen sizes
- Built-in Gradio theming and styling

## Troubleshooting

**Common Issues:**

1. **"Connection Error" messages:**
   - Ensure backend is running on `localhost:8000`
   - Check if backend endpoints are accessible

2. **Gradio not starting:**
   - Install requirements: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

3. **Chat not working:**
   - Verify backend health using the "System Health" button
   - Check browser console for JavaScript errors

4. **Port conflicts:**
   - Change the port in `gradio.py` if 7860 is already in use
   - Update the `server_port` parameter in `interface.launch()`

## Development

To modify or extend the Gradio interface:

1. **Add new features:** Modify the `create_gradio_interface()` function
2. **Change styling:** Update the CSS in the `gr.Blocks` configuration
3. **Add new API calls:** Create new functions similar to `send_message()`
4. **Debug:** Enable verbose logging by setting `quiet=False` in launch

## License

This Gradio interface inherits the same license as the main chatbot project. 