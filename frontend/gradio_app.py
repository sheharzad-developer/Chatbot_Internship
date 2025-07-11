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
    with gr.Blocks(
        title="🤖 Reflect Agent Chatbot",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px;
            margin: auto;
        }
        .chat-container {
            height: 600px;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🤖 Reflect Agent Chatbot
        
        Welcome to your intelligent chatbot with **Conditional Reasoning**, **Web Search**, **Document Search**, and **Chat History**!
        
        ### Features:
        - 🧠 **Smart Reasoning**: Doesn't always act - uses conditional logic
        - 🌐 **Web Search**: Real-time information via Tavily
        - 📚 **Document Search**: RAG system with vector similarity
        - 💬 **Chat History**: Persistent conversation storage
        - 📊 **System Monitoring**: Health checks and statistics
        """)
        
        with gr.Tabs():
            # Main Chat Tab
            with gr.TabItem("💬 Chat", elem_id="chat-tab"):
                chatbot_ui = gr.Chatbot(
                    label="Chat with your Reflect Agent",
                    height=500,
                    show_label=True,
                    container=True,
                    elem_classes=["chat-container"]
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your message here...",
                        label="Your Message",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send 🚀", scale=1, variant="primary")
                
                gr.Examples(
                    examples=[
                        "Hello! How are you today?",
                        "What's the latest news about AI?",
                        "Can you search for information about Python programming?",
                        "Tell me about machine learning",
                        "What documents do you have access to?"
                    ],
                    inputs=msg_input,
                    label="Example Questions"
                )
            
            # Document Management Tab
            with gr.TabItem("📚 Document Management"):
                gr.Markdown("### Add Documents to Knowledge Base")
                
                with gr.Row():
                    with gr.Column():
                        doc_title = gr.Textbox(
                            label="Document Title",
                            placeholder="Enter document title..."
                        )
                        doc_content = gr.Textbox(
                            label="Document Content",
                            placeholder="Paste your document content here...",
                            lines=10
                        )
                        add_doc_btn = gr.Button("Add Document 📄", variant="primary")
                    
                    with gr.Column():
                        doc_result = gr.Textbox(
                            label="Result",
                            lines=10,
                            interactive=False
                        )
                
                gr.Markdown("### Search Documents")
                with gr.Row():
                    search_input = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter your search query...",
                        scale=3
                    )
                    search_btn = gr.Button("Search 🔍", scale=1, variant="secondary")
                
                search_result = gr.Textbox(
                    label="Search Results",
                    lines=8,
                    interactive=False
                )
            
            # System Status Tab
            with gr.TabItem("📊 System Status"):
                with gr.Row():
                    health_btn = gr.Button("Check Health 💚", variant="secondary")
                    history_btn = gr.Button("View Chat History 📜", variant="secondary")
                
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
        
        # Document management
        add_doc_btn.click(
            chatbot.add_document,
            inputs=[doc_content, doc_title],
            outputs=doc_result
        )
        
        search_btn.click(
            chatbot.search_documents,
            inputs=search_input,
            outputs=search_result
        )
        
        # System status
        health_btn.click(
            chatbot.get_health_status,
            outputs=status_output
        )
        
        history_btn.click(
            chatbot.get_chat_history,
            outputs=status_output
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