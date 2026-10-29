"""Purple glass theme shared by every page (applied once in app.py)."""

# Mirrors the look of frontend/index.html: purple gradient, frosted-glass panels,
# gradient user bubbles on the right and glass assistant bubbles on the left.
CSS = """
<style>
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    background-attachment: fixed;
    color: #ffffff;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: transparent; }

.app-header {
    background: rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    margin-bottom: 20px;
}
.app-header h1 { font-size: 2.5rem; font-weight: 700; margin: 0; padding: 0; color: #fff; }

.chat-header {
    background: rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 16px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 10px;
}
.chat-title { font-size: 1.3rem; font-weight: 600; }
.model-pill {
    background: rgba(76, 175, 80, 0.2);
    border: 1px solid rgba(76, 175, 80, 0.5);
    border-radius: 20px;
    padding: 6px 14px;
    color: #7ee787;
    font-size: 0.85rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(15px);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}
[data-testid="stSidebar"] h3 { color: #fff; font-size: 1.15rem; }

/* Organization picker: radio options drawn as the original's tenant cards */
[data-testid="stSidebar"] [role="radiogroup"] { gap: 10px; }
[data-testid="stSidebar"] [role="radiogroup"] > label {
    background: rgba(255, 255, 255, 0.1);
    border: 2px solid rgba(255, 255, 255, 0.2);
    border-radius: 12px;
    padding: 14px;
    margin: 0;
    width: 100%;
    transition: all 0.3s ease;
    cursor: pointer;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
    background: rgba(255, 255, 255, 0.2);
    border-color: rgba(255, 255, 255, 0.4);
}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
    background: rgba(255, 255, 255, 0.3);
    border-color: #ffffff;
    box-shadow: 0 4px 20px rgba(255, 255, 255, 0.2);
}
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] p { font-weight: 700; color: #fff; }
[data-testid="stSidebar"] [role="radiogroup"] [data-testid="stCaptionContainer"] p {
    font-weight: 400; color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;
}

/* Sidebar buttons, like the original's .button */
[data-testid="stSidebar"] .stButton button {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    color: #fff;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255, 255, 255, 0.2);
    border-color: rgba(255, 255, 255, 0.4);
    color: #fff;
}

/* Chat history: one row per saved chat, the open one highlighted */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] { gap: 6px; margin-bottom: -8px; }
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton button { justify-content: flex-start; }
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton button p {
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 0.88rem;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:last-child .stButton button {
    justify-content: center; background: transparent; border-color: transparent; padding: 0 4px;
}
/* Uploaded documents */
[data-testid="stSidebar"] .doc-name {
    font-size: 0.88rem; line-height: 1.3; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
[data-testid="stSidebar"] .doc-name small { color: rgba(255, 255, 255, 0.7); }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: rgba(255, 255, 255, 0.1);
    border: 1px dashed rgba(255, 255, 255, 0.4);
    border-radius: 12px;
}
[data-testid="stExpander"] details { border-color: rgba(255, 255, 255, 0.25); border-radius: 10px; }
[data-testid="stExpander"] summary p { font-size: 0.88rem; }
[data-testid="stExpander"] a { color: #d6deff; text-decoration: underline; }
[data-testid="stExpander"] small { color: rgba(255, 255, 255, 0.65); }
[data-testid="stExpander"] [data-testid="stCaptionContainer"] p { color: rgba(255, 255, 255, 0.8); }

[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: rgba(255, 255, 255, 0.3);
    border: 1px solid #ffffff;
}

/* Chat bubbles */
[data-testid="stChatMessage"] { background: transparent; padding: 0; margin: 12px 0; width: 100%; }
[data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] { display: none; }
[data-testid="stChatMessageContent"] {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 20px 20px 20px 5px;
    padding: 15px 20px;
    max-width: 80%;
    flex: 0 1 auto;
    margin: 0 auto 0 0 !important;
    color: #fff;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { flex-direction: row-reverse; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border: none;
    border-radius: 20px 20px 5px 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    margin: 0 0 0 auto !important;
}
[data-testid="stChatMessageContent"] p, [data-testid="stChatMessageContent"] li { color: #fff; }
[data-testid="stChatMessageContent"] code { background: rgba(0, 0, 0, 0.25); color: #fff; }
[data-testid="stChatMessageContent"] h1, [data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3, [data-testid="stChatMessageContent"] h4 {
    color: #fff; font-size: 1.05rem; font-weight: 600; padding: 12px 0 4px;
}
[data-testid="stChatMessageContent"] strong { color: #fff; }
[data-testid="stChatMessageContent"] table { border-collapse: collapse; width: 100%; font-size: 0.92rem; }
[data-testid="stChatMessageContent"] th, [data-testid="stChatMessageContent"] td {
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 6px 10px;
    color: #fff;
    text-align: left;
}
[data-testid="stChatMessageContent"] th { background: rgba(255, 255, 255, 0.1); font-weight: 600; }

/* Message input */
[data-testid="stChatInput"] {
    background: rgba(255, 255, 255, 0.1);
    border: 2px solid rgba(255, 255, 255, 0.2);
    border-radius: 25px;
    backdrop-filter: blur(10px);
}
[data-testid="stChatInput"]:focus-within { border-color: rgba(255, 255, 255, 0.5); }
[data-testid="stChatInput"] textarea { color: #fff; }
[data-testid="stChatInput"] textarea::placeholder { color: rgba(255, 255, 255, 0.6); }
[data-testid="stChatInputSubmitButton"] {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 50%;
    color: #fff;
}
</style>
"""
