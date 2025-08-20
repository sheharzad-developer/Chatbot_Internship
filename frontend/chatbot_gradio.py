import requests
import json
from datetime import datetime
import uuid
try:
    import gradio as gr
    print(f"✅ Gradio {gr.__version__} imported successfully")
except ImportError as e:
    print(f"❌ Error importing Gradio: {e}")
    raise

# Global variables to maintain state
current_agent = "simple"
session_id = None
fast_mode = True
chat_history = []

# Backend API base URL
API_BASE = "http://localhost:8001"

def format_message_for_display(content, is_user=False, agent_type="simple"):
    """Format messages for display in the chat interface"""
    if is_user:
        return f"**👤 You:** {content}"
    else:
        if agent_type == "gemini":
            agent_emoji = "🔮"
            agent_name = "Gemini"
        elif agent_type == "deepseek":
            agent_emoji = "🚀"
            agent_name = "DeepSeek"
        else:
            agent_emoji = "🤖"
            agent_name = "Simple"
        
        mode_indicator = " ⚡" if fast_mode else ""
        return f"**{agent_emoji} {agent_name}{mode_indicator}:** {content}"

def send_message(message, history, agent_choice, fast_mode_enabled):
    """Send message to the backend and return updated history"""
    global session_id, current_agent, fast_mode
    
    if not message.strip():
        return history, ""
    
    current_agent = agent_choice.lower()
    fast_mode = fast_mode_enabled
    
    # Add user message to history
    user_msg = format_message_for_display(message, is_user=True)
    history.append([user_msg, None])
    
    try:
        # Determine endpoint based on fast mode and agent
        if fast_mode_enabled:
            endpoint = f"{API_BASE}/chat/fast"
        elif current_agent == "gemini":
            endpoint = f"{API_BASE}/chat/gemini"
        elif current_agent == "deepseek":
            endpoint = f"{API_BASE}/chat/deepseek"
        else:
            endpoint = f"{API_BASE}/chat"
        
        # Prepare request data
        request_data = {
            "message": message,
            "session_id": session_id,
            "user_id": "gradio_user",
            "fast_mode": fast_mode_enabled
        }
        
        # Send request to backend
        response = requests.post(
            endpoint,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.ok:
            data = response.json()
            session_id = data.get("session_id")
            bot_response = data.get("response", "No response received")
            
            # Format bot response
            bot_msg = format_message_for_display(bot_response, is_user=False, agent_type=current_agent)
            history[-1][1] = bot_msg
            
        else:
            error_msg = f"❌ Error: HTTP {response.status_code} - {response.text}"
            bot_msg = format_message_for_display(error_msg, is_user=False, agent_type=current_agent)
            history[-1][1] = bot_msg
            
    except requests.exceptions.RequestException as e:
        error_msg = f"❌ Connection Error: {str(e)}"
        bot_msg = format_message_for_display(error_msg, is_user=False, agent_type=current_agent)
        history[-1][1] = bot_msg
    except Exception as e:
        error_msg = f"❌ Unexpected Error: {str(e)}"
        bot_msg = format_message_for_display(error_msg, is_user=False, agent_type=current_agent)
        history[-1][1] = bot_msg
    
    return history, ""

def check_system_health():
    """Check backend system health"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        
        if response.ok:
            data = response.json()
            services = data.get("services", {})
            
            health_info = f"""✅ **System Status:** {data.get('status', 'Unknown').upper()}

🔌 **MongoDB:** {data.get('mongodb', 'Unknown')}
📚 **RAG Documents:** {data.get('rag_system', {}).get('total_documents', 0)}
🤖 **Simple Agent:** {'✅ Ready' if services.get('simple_agent') else '❌ Not Ready'}
🔮 **Gemini Agent:** {'✅ Ready' if services.get('gemini_agent') else '❌ Not Ready'}
🚀 **DeepSeek Agent:** {'✅ Ready' if services.get('deepseek_agent') else '❌ Not Ready'}
⚡ **RAG Service:** {'✅ Ready' if services.get('rag_service') else '❌ Not Ready'}"""
            
            return health_info
        else:
            return f"❌ **Health Check Failed:** HTTP {response.status_code}"
            
    except Exception as e:
        return f"❌ **Cannot connect to server:** {str(e)}"

def load_chat_history():
    """Load recent chat sessions"""
    try:
        response = requests.get(f"{API_BASE}/chat/history?limit=10", timeout=10)
        
        if response.ok:
            data = response.json()
            sessions = data.get("sessions", [])
            
            if not sessions:
                return "📝 **No chat history found**"
            
            history_text = f"📜 **Recent Chat Sessions ({len(sessions)} found):**\n\n"
            
            for i, session in enumerate(sessions, 1):
                title = session.get("title", "Untitled Chat")
                message_count = len(session.get("messages", []))
                created = session.get("created_at", "Unknown")
                
                try:
                    created_date = datetime.fromisoformat(created.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
                except:
                    created_date = created
                
                history_text += f"**{i}. {title}**\n"
                history_text += f"   • {message_count} messages\n"
                history_text += f"   • Created: {created_date}\n\n"
            
            return history_text
        else:
            return f"❌ **Error loading history:** HTTP {response.status_code}"
            
    except Exception as e:
        return f"❌ **Error loading history:** {str(e)}"

def clear_chat_session():
    """Clear current chat session"""
    global session_id
    session_id = None
    return [], "🎉 **Chat cleared!** Ready for a new conversation."

def create_gradio_interface():
    """Create the main Gradio interface"""
    
    with gr.Blocks(
        title="🤖 Chatbot AI",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-message {
            font-size: 14px;
        }
        """
    ) as interface:
        
        gr.Markdown("# 🤖 Chatbot AI")
        gr.Markdown("*Choose an agent and start chatting with your AI assistant!*")
        
        with gr.Row():
            with gr.Column(scale=3):
                # Main chat interface
                chatbot = gr.Chatbot(
                    label="💬 Chat",
                    height=500,
                    show_label=True,
                    container=True,
                    bubble_full_width=False
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your message here...",
                        container=False,
                        scale=4,
                        show_label=False
                    )
                    send_btn = gr.Button("Send 📤", variant="primary", scale=1)
            
            with gr.Column(scale=1):
                # Agent selection and controls
                gr.Markdown("### 🚀 AI Agent Selection")
                
                agent_choice = gr.Radio(
                    choices=["Simple", "Gemini", "DeepSeek"],
                    value="DeepSeek",
                    label="Choose Agent",
                    info="Select your preferred AI agent"
                )
                
                fast_mode_checkbox = gr.Checkbox(
                    value=True,
                    label="⚡ Fast Mode",
                    info="Skip database for faster responses"
                )
                
                gr.Markdown("### ⚡ Quick Actions")
                
                with gr.Column():
                    health_btn = gr.Button("💚 System Health", variant="secondary")
                    history_btn = gr.Button("📜 Chat History", variant="secondary")
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="stop")
                
                # Status display area
                status_display = gr.Markdown(
                    value="",
                    label="📊 Status",
                    visible=False
                )
        
        # Event handlers
        def submit_message(message, history, agent, fast_mode):
            return send_message(message, history, agent, fast_mode)
        
        def show_health():
            health_info = check_system_health()
            return gr.update(value=health_info, visible=True)
        
        def show_history():
            history_info = load_chat_history()
            return gr.update(value=history_info, visible=True)
        
        def clear_and_reset():
            history, message = clear_chat_session()
            return history, gr.update(value=message, visible=True)
        
        # Wire up the events
        msg_input.submit(
            submit_message,
            inputs=[msg_input, chatbot, agent_choice, fast_mode_checkbox],
            outputs=[chatbot, msg_input]
        )
        
        send_btn.click(
            submit_message,
            inputs=[msg_input, chatbot, agent_choice, fast_mode_checkbox],
            outputs=[chatbot, msg_input]
        )
        
        health_btn.click(
            show_health,
            outputs=[status_display]
        )
        
        history_btn.click(
            show_history,
            outputs=[status_display]
        )
        
        clear_btn.click(
            clear_and_reset,
            outputs=[chatbot, status_display]
        )
        
        # Initialize with welcome message
        interface.load(
            lambda: [[[format_message_for_display("Welcome! I'm your AI assistant. DeepSeek 🚀 is now available! Enable Brief Mode 📏 for shorter responses.", is_user=False, agent_type="deepseek"), None]]],
            outputs=[chatbot]
        )
    
    return interface

def main():
    """Main function to launch the Gradio interface"""
    print("🚀 Starting Gradio Chatbot Interface...")
    print(f"🔗 Backend API: {API_BASE}")
    
    # Create and launch the interface
    interface = create_gradio_interface()
    
    # Launch with custom settings
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public sharing
        quiet=False             # Show startup logs
    )

if __name__ == "__main__":
    main() 