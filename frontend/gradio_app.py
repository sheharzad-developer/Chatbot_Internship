import gradio as gr
import requests
import json
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI backend URL
API_BASE_URL = "http://localhost:8000"

class ChatbotInterface:
    def __init__(self):
        self.session_id = None
        self.chat_history = []
    
    def send_message(self, message: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
        """Send message to the chatbot and get response"""
        if not message.strip():
            return history, ""
        
        try:
            # Prepare request data
            request_data = {
                "message": message,
                "session_id": self.session_id,
                "user_id": "gradio_user"
            }
            
            # Send request to FastAPI backend
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                bot_response = result.get("response", "I apologize, but I couldn't generate a response.")
                self.session_id = result.get("session_id")
                
                # Update history
                history.append((message, bot_response))
                return history, ""
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                history.append((message, error_msg))
                return history, ""
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Connection error: {str(e)}"
            logger.error(error_msg)
            history.append((message, "❌ Sorry, I'm having trouble connecting to the server. Please make sure the FastAPI server is running."))
            return history, ""
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            history.append((message, "❌ An unexpected error occurred. Please try again."))
            return history, ""
    
    def get_health_status(self) -> str:
        """Get health status from the backend"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                return f"✅ **System Status: {health_data.get('status', 'unknown').upper()}**\n\n" + \
                       f"- MongoDB: {health_data.get('mongodb', 'unknown')}\n" + \
                       f"- RAG Documents: {health_data.get('rag_system', {}).get('total_documents', 0)}\n" + \
                       f"- Services: All operational"
            else:
                return f"❌ Health check failed: {response.status_code}"
        except Exception as e:
            return f"❌ Cannot connect to server: {str(e)}"
    
    def get_chat_history(self) -> str:
        """Get chat history from the backend"""
        try:
            response = requests.get(f"{API_BASE_URL}/chat/history", timeout=10)
            if response.status_code == 200:
                history_data = response.json()
                sessions = history_data.get("sessions", [])
                
                if not sessions:
                    return "No chat history found."
                
                history_text = f"**Found {len(sessions)} chat session(s):**\n\n"
                
                for i, session in enumerate(sessions[:5], 1):  # Show last 5 sessions
                    title = session.get("title", "Untitled Chat")
                    message_count = len(session.get("messages", []))
                    created_at = session.get("created_at", "")
                    
                    history_text += f"**{i}. {title}**\n"
                    history_text += f"   - Messages: {message_count}\n"
                    history_text += f"   - Created: {created_at[:19]}\n\n"
                
                return history_text
            else:
                return f"❌ Failed to get chat history: {response.status_code}"
        except Exception as e:
            return f"❌ Error getting chat history: {str(e)}"
    
    def search_documents(self, query: str) -> str:
        """Search documents using RAG"""
        if not query.strip():
            return "Please enter a search query."
        
        try:
            response = requests.get(
                f"{API_BASE_URL}/rag/search",
                params={"query": query, "top_k": 3},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                search_results = result.get("results", "No results found.")
                stats = result.get("stats", {})
                
                formatted_response = f"**Search Results for: '{query}'**\n\n"
                formatted_response += f"{search_results}\n\n"
                formatted_response += f"**Stats:** {stats.get('total_documents', 0)} documents, {stats.get('total_chunks', 0)} chunks"
                
                return formatted_response
            else:
                return f"❌ Search failed: {response.status_code}"
        except Exception as e:
            return f"❌ Search error: {str(e)}"
    
    def add_document(self, content: str, title: str) -> str:
        """Add a document to the RAG system"""
        if not content.strip():
            return "Please enter document content."
        
        try:
            data = {
                "content": content,
                "title": title or "Untitled Document"
            }
            
            response = requests.post(
                f"{API_BASE_URL}/rag/add-document",
                data=data,  # Using form data instead of JSON
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                stats = result.get("stats", {})
                return f"✅ Document added successfully!\n\n" + \
                       f"Total documents: {stats.get('total_documents', 0)}\n" + \
                       f"Total chunks: {stats.get('total_chunks', 0)}"
            else:
                return f"❌ Failed to add document: {response.status_code} - {response.text}"
        except Exception as e:
            return f"❌ Error adding document: {str(e)}"

# Initialize the chatbot interface
chatbot = ChatbotInterface()

# Create Gradio interface
def create_interface():
    # Custom CSS for ChatGPT-like styling
    custom_css = """
    /* Main container styling */
    .gradio-container {
        max-width: 100%;
        margin: 0;
        padding: 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        text-align: center;
        margin-bottom: 0;
    }
    
    /* Chat container - full height like ChatGPT */
    .chat-container {
        height: 70vh;
        background-color: #f7f7f8;
        border-radius: 8px;
        border: 1px solid #e5e5ea;
    }
    
    /* Message styling */
    .message {
        margin: 10px 0;
        padding: 12px 16px;
        border-radius: 18px;
        max-width: 80%;
        line-height: 1.4;
    }
    
    .user-message {
        background-color: #007bff;
        color: white;
        margin-left: auto;
        text-align: right;
    }
    
    .bot-message {
        background-color: #f1f3f5;
        color: #333;
        border: 1px solid #e5e5ea;
    }
    
    /* Input area styling */
    .input-container {
        padding: 20px;
        background-color: white;
        border-top: 1px solid #e5e5ea;
        position: sticky;
        bottom: 0;
    }
    
    /* Input box styling */
    .message-input {
        border-radius: 25px !important;
        border: 2px solid #e5e5ea !important;
        padding: 12px 20px !important;
        font-size: 16px !important;
        transition: border-color 0.2s ease !important;
    }
    
    .message-input:focus {
        border-color: #007bff !important;
        box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1) !important;
    }
    
    /* Send button styling */
    .send-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 25px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        transition: all 0.2s ease !important;
    }
    
    .send-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Sidebar styling */
    .sidebar {
        background-color: #f8f9fa;
        border-right: 1px solid #e5e5ea;
        height: 100vh;
        padding: 20px;
    }
    
    /* Tab styling */
    .tab-nav button {
        background-color: transparent !important;
        border: none !important;
        padding: 12px 24px !important;
        margin: 0 4px !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    .tab-nav button[aria-selected="true"] {
        background-color: #007bff !important;
        color: white !important;
    }
    
    /* Examples styling */
    .examples-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 12px;
        margin-top: 20px;
    }
    
    .example-item {
        background: white;
        border: 1px solid #e5e5ea;
        border-radius: 12px;
        padding: 16px;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: center;
    }
    
    .example-item:hover {
        border-color: #007bff;
        box-shadow: 0 2px 8px rgba(0, 123, 255, 0.1);
        transform: translateY(-2px);
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .gradio-container {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        
        .chat-container {
            background-color: #2d2d2d;
            border-color: #404040;
        }
        
        .bot-message {
            background-color: #404040;
            color: #ffffff;
            border-color: #555555;
        }
        
        .sidebar {
            background-color: #2d2d2d;
            border-color: #404040;
        }
    }
    """
    
    with gr.Blocks(
        title="🤖 Reflect Agent AI",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        # Header section with ChatGPT-like styling
        with gr.Row(elem_classes=["header-container"]):
            gr.HTML("""
            <div style="text-align: center; padding: 20px;">
                <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700; margin-bottom: 10px;">
                    🤖 Reflect Agent AI
                </h1>
                <p style="margin: 0; font-size: 1.1rem; opacity: 0.9;">
                    Intelligent Assistant with Conditional Reasoning, Web Search & Document Knowledge
                </p>
            </div>
            """)
        
        # Main layout with sidebar and chat
        with gr.Row():
            # Sidebar for navigation
            with gr.Column(scale=1, elem_classes=["sidebar"]):
                gr.HTML("""
                <div style="margin-bottom: 20px;">
                    <h3 style="margin: 0 0 10px 0; color: #666;">🚀 Features</h3>
                    <div style="font-size: 14px; line-height: 1.6; color: #888;">
                        <div style="margin: 8px 0;">🧠 Smart Reasoning</div>
                        <div style="margin: 8px 0;">🌐 Web Search</div>
                        <div style="margin: 8px 0;">📚 Document Search</div>
                        <div style="margin: 8px 0;">💬 Chat History</div>
                        <div style="margin: 8px 0;">📊 System Monitoring</div>
                    </div>
                </div>
                """)
                
                # Quick actions in sidebar
                with gr.Group():
                    gr.Markdown("### Quick Actions")
                    health_btn = gr.Button("💚 System Health", variant="secondary", size="sm")
                    history_btn = gr.Button("📜 Chat History", variant="secondary", size="sm")
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")
            
            # Main chat area
            with gr.Column(scale=4):
                # Chat interface
                chatbot_ui = gr.Chatbot(
                    label="",
                    height=600,
                    show_label=False,
                    container=True,
                    elem_classes=["chat-container"],
                    avatar_images=("🧑‍💻", "🤖"),
                    bubble_full_width=False
                )
                
                # Input area at bottom (ChatGPT style)
                with gr.Row(elem_classes=["input-container"]):
                    msg_input = gr.Textbox(
                        placeholder="Message Reflect Agent AI...",
                        label="",
                        lines=1,
                        scale=6,
                        elem_classes=["message-input"],
                        show_label=False
                    )
                    send_btn = gr.Button(
                        "Send", 
                        scale=1, 
                        variant="primary",
                        elem_classes=["send-button"]
                    )
                
                # Example prompts (ChatGPT style)
                with gr.Row():
                    with gr.Column():
                        gr.HTML("""
                        <div class="examples-container">
                            <div class="example-item" onclick="document.querySelector('textarea[placeholder*=\"Message\"]').value='What\\'s the latest in AI technology?'; document.querySelector('textarea[placeholder*=\"Message\"]').dispatchEvent(new Event('input', {bubbles: true}));">
                                <strong>🔬 Latest AI News</strong><br>
                                <small>Get current information about AI developments</small>
                            </div>
                            <div class="example-item" onclick="document.querySelector('textarea[placeholder*=\"Message\"]').value='Can you help me understand machine learning?'; document.querySelector('textarea[placeholder*=\"Message\"]').dispatchEvent(new Event('input', {bubbles: true}));">
                                <strong>🧠 Learn ML</strong><br>
                                <small>Understand machine learning concepts</small>
                            </div>
                            <div class="example-item" onclick="document.querySelector('textarea[placeholder*=\"Message\"]').value='Search for Python programming best practices'; document.querySelector('textarea[placeholder*=\"Message\"]').dispatchEvent(new Event('input', {bubbles: true}));">
                                <strong>💻 Python Tips</strong><br>
                                <small>Find programming best practices</small>
                            </div>
                            <div class="example-item" onclick="document.querySelector('textarea[placeholder*=\"Message\"]').value='What documents do you have access to?'; document.querySelector('textarea[placeholder*=\"Message\"]').dispatchEvent(new Event('input', {bubbles: true}));">
                                <strong>📚 Knowledge Base</strong><br>
                                <small>Explore available documents</small>
                            </div>
                        </div>
                        """)
        
        # Hidden tabs for additional features (accessible via sidebar)
        with gr.Tabs(visible=False) as hidden_tabs:
            # Document Management Tab
            with gr.TabItem("📚 Document Management"):
                gr.Markdown("### 📄 Add Documents to Knowledge Base")
                
                with gr.Row():
                    with gr.Column():
                        doc_title = gr.Textbox(
                            label="Document Title",
                            placeholder="Enter document title...",
                            elem_classes=["message-input"]
                        )
                        doc_content = gr.Textbox(
                            label="Document Content",
                            placeholder="Paste your document content here...",
                            lines=10,
                            elem_classes=["message-input"]
                        )
                        add_doc_btn = gr.Button("Add Document 📄", variant="primary", elem_classes=["send-button"])
                    
                    with gr.Column():
                        doc_result = gr.Textbox(
                            label="Result",
                            lines=10,
                            interactive=False
                        )
                
                gr.Markdown("### 🔍 Search Documents")
                with gr.Row():
                    search_input = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter your search query...",
                        scale=3,
                        elem_classes=["message-input"]
                    )
                    search_btn = gr.Button("Search 🔍", scale=1, variant="secondary")
                
                search_result = gr.Textbox(
                    label="Search Results",
                    lines=8,
                    interactive=False
                )
            
            # System Status Tab  
            with gr.TabItem("📊 System Status"):
                status_output = gr.Textbox(
                    label="System Information",
                    lines=15,
                    interactive=False
                )
                
                gr.Markdown("""
                ### 🔗 Direct API Access
                - **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
                - **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)
                - **Alternative Docs**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
                """)
            
            
        
        # Event handlers
        def submit_message(message, history):
            return chatbot.send_message(message, history)
        
        def clear_chat():
            return []
        
        def show_health():
            return chatbot.get_health_status()
        
        def show_history():
            return chatbot.get_chat_history()
        
        # Chat functionality
        msg_input.submit(
            submit_message,
            inputs=[msg_input, chatbot_ui],
            outputs=[chatbot_ui, msg_input]
        )
        
        send_btn.click(
            submit_message,
            inputs=[msg_input, chatbot_ui],
            outputs=[chatbot_ui, msg_input]
        )
        
        # Sidebar actions
        clear_btn.click(
            clear_chat,
            outputs=chatbot_ui
        )
        
        # For the hidden tabs functionality
        if 'add_doc_btn' in locals():
            add_doc_btn.click(
                chatbot.add_document,
                inputs=[doc_content, doc_title],
                outputs=doc_result
            )
        
        if 'search_btn' in locals():
            search_btn.click(
                chatbot.search_documents,
                inputs=search_input,
                outputs=search_result
            )
        
        # Create a simple status display function for sidebar buttons
        def create_status_display():
            status_display = gr.Textbox(
                label="Status",
                lines=10,
                interactive=False,
                visible=False
            )
            return status_display
        
        # Handle sidebar button clicks with popups or notifications
        health_btn.click(
            fn=lambda: gr.Info(chatbot.get_health_status()),
            inputs=None,
            outputs=None
        )
        
        history_btn.click(
            fn=lambda: gr.Info(chatbot.get_chat_history()),
            inputs=None,
            outputs=None
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    
    print("🚀 Starting Gradio Frontend...")
    print("📚 FastAPI Backend should be running on: http://localhost:8000")
    print("🌐 Gradio Frontend will be available on: http://localhost:7860")
    print()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    ) 