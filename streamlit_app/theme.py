"""
Soft light theme and minimal SVG icons shared by every page (CSS is applied once in app.py).

Calm neutrals with a single muted indigo accent, white cards with hairline borders, and outline
icons instead of emojis. Streamlit widgets use its built-in Material icons (":material/...:");
the custom HTML parts (headers, document list, copy button) use the inline SVGs below.
"""

# Colors, kept in one place so the CSS below stays consistent
BG = "#F5F6F8"          # page background
SURFACE = "#FFFFFF"     # cards, assistant bubbles, inputs
SUBTLE = "#FAFAFB"      # sidebar, expanders, dropzone
BORDER = "#E3E6EB"
BORDER_STRONG = "#CDD2DA"
TEXT = "#1F2430"
MUTED = "#6B7280"
ACCENT = "#5865D8"      # muted indigo, used sparingly
ACCENT_SOFT = "#EEF0FD"  # user bubbles, selected items
ACCENT_TEXT = "#3B47B8"

# Outline icons (24x24, stroke-based, in the style of Lucide)
_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
        'aria-hidden="true">{paths}</svg>')
_PATHS = {
    "chat": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "book": '<path d="M2 4h6a4 4 0 0 1 4 4v13a3 3 0 0 0-3-3H2z"/><path d="M22 4h-6a4 4 0 0 0-4 4v13a3 3 0 0 1 3-3h7z"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>',
}


def icon(name, size=18):
    return _SVG.format(size=size, paths=_PATHS[name])


def page_header(title, subtitle, icon_name):
    """Title row used at the top of each page."""
    return (
        f'<div class="app-header"><span class="app-header-icon">{icon(icon_name, 22)}</span>'
        f'<div><h1>{title}</h1><p>{subtitle}</p></div></div>'
    )


CSS = f"""
<style>
.stApp {{
    background: {BG};
    color: {TEXT};
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] {{ background: {BG}; }}

/* Page header */
.app-header {{ display: flex; align-items: center; gap: 14px; margin: 4px 0 18px; }}
.app-header-icon {{
    display: inline-flex; align-items: center; justify-content: center; flex: none;
    width: 42px; height: 42px; border-radius: 12px;
    background: {ACCENT_SOFT}; color: {ACCENT};
}}
.app-header h1 {{ font-size: 1.55rem; font-weight: 650; margin: 0; padding: 0; color: {TEXT}; letter-spacing: -0.01em; }}
.app-header p {{ margin: 2px 0 0; color: {MUTED}; font-size: 0.92rem; }}

.chat-header {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 12px 18px;
    display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap;
    margin-bottom: 10px;
}}
.chat-title {{ font-size: 1.05rem; font-weight: 600; color: {TEXT}; }}
.model-pill {{
    display: inline-flex; align-items: center; gap: 7px;
    background: {BG}; border: 1px solid {BORDER}; border-radius: 999px;
    padding: 4px 12px; color: {MUTED}; font-size: 0.8rem;
}}
.model-pill::before {{ content: ""; width: 7px; height: 7px; border-radius: 50%; background: #3BA776; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background: {SUBTLE}; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] h3 {{
    color: {MUTED}; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; margin-top: 10px;
}}

/* Organization picker: radio options drawn as cards */
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 8px; }}
[data-testid="stSidebar"] [role="radiogroup"] > label {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 12px 14px;
    margin: 0;
    width: 100%;
    transition: border-color 0.15s ease, background 0.15s ease;
    cursor: pointer;
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {{ border-color: {BORDER_STRONG}; }}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{
    background: {ACCENT_SOFT}; border-color: {ACCENT};
}}
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {{ display: none; }}
[data-testid="stSidebar"] [role="radiogroup"] p {{ font-weight: 600; color: {TEXT}; }}
[data-testid="stSidebar"] [role="radiogroup"] [data-testid="stCaptionContainer"] p {{
    font-weight: 400; color: {MUTED}; font-size: 0.82rem;
}}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton button {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    color: {TEXT};
}}
[data-testid="stSidebar"] .stButton button:hover {{
    background: {BG}; border-color: {BORDER_STRONG}; color: {TEXT};
}}
[data-testid="stSidebar"] .stButton button[kind="primary"] {{
    background: {ACCENT_SOFT}; border-color: {ACCENT}; color: {ACCENT_TEXT};
}}

/* Chat history and document rows: a wide button (or name) and a small delete icon */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {{ gap: 4px; margin-bottom: -10px; }}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton button {{ justify-content: flex-start; }}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton button p {{
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 0.86rem;
}}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:last-child .stButton button {{
    justify-content: center; background: transparent; border-color: transparent; padding: 0 4px; color: {MUTED};
}}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:last-child .stButton button:hover {{
    color: #C2410C; background: #FFF4ED;
}}

/* Uploaded documents */
[data-testid="stSidebar"] .doc-name {{
    display: flex; align-items: flex-start; gap: 8px;
    font-size: 0.86rem; line-height: 1.3; color: {TEXT}; overflow: hidden; margin-bottom: 6px;
}}
[data-testid="stSidebar"] .doc-name svg {{ flex: none; margin-top: 1px; color: {MUTED}; }}
[data-testid="stSidebar"] .doc-name span {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
[data-testid="stSidebar"] .doc-name small {{ color: {MUTED}; }}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
    background: {SURFACE}; border: 1px dashed {BORDER_STRONG}; border-radius: 12px;
}}

/* Source panels under answers */
[data-testid="stExpander"] details {{ border: 1px solid {BORDER}; border-radius: 10px; background: {SUBTLE}; }}
[data-testid="stExpander"] summary p {{ font-size: 0.86rem; color: {MUTED}; }}
[data-testid="stExpander"] a {{ color: {ACCENT_TEXT}; text-decoration: none; }}
[data-testid="stExpander"] a:hover {{ text-decoration: underline; }}
[data-testid="stExpander"] small {{ color: {MUTED}; }}

/* Chat bubbles: assistant on the left (white card), user on the right (soft accent) */
[data-testid="stChatMessage"] {{ background: transparent; padding: 0; margin: 12px 0; width: 100%; }}
[data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] {{ display: none; }}
[data-testid="stChatMessageContent"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 16px 16px 16px 4px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    padding: 14px 18px;
    max-width: 80%;
    flex: 0 1 auto;
    margin: 0 auto 0 0 !important;
    color: {TEXT};
    line-height: 1.6;
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{ flex-direction: row-reverse; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {{
    background: {ACCENT_SOFT};
    border-color: #DDE1FB;
    border-radius: 16px 16px 4px 16px;
    margin: 0 0 0 auto !important;
}}
[data-testid="stChatMessageContent"] p, [data-testid="stChatMessageContent"] li {{ color: {TEXT}; }}
[data-testid="stChatMessageContent"] code {{ background: #F1F2F5; color: {TEXT}; }}
[data-testid="stChatMessageContent"] h1, [data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3, [data-testid="stChatMessageContent"] h4 {{
    color: {TEXT}; font-size: 1.02rem; font-weight: 600; padding: 10px 0 4px;
}}
[data-testid="stChatMessageContent"] strong {{ color: {TEXT}; font-weight: 600; }}
[data-testid="stChatMessageContent"] table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
[data-testid="stChatMessageContent"] th, [data-testid="stChatMessageContent"] td {{
    border: 1px solid {BORDER}; padding: 6px 10px; color: {TEXT}; text-align: left;
}}
[data-testid="stChatMessageContent"] th {{ background: {BG}; font-weight: 600; }}

/* Message input */
[data-testid="stChatInput"] {{
    background: {SURFACE};
    border: 1px solid {BORDER_STRONG};
    border-radius: 14px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
}}
[data-testid="stChatInput"]:focus-within {{ border-color: {ACCENT}; box-shadow: 0 0 0 3px rgba(88, 101, 216, 0.15); }}
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea {{ background-color: {SURFACE}; }}
[data-testid="stChatInput"] textarea {{ color: {TEXT}; }}
[data-testid="stChatInput"] textarea::placeholder {{ color: #9CA3AF; }}
[data-testid="stChatInputSubmitButton"] {{ background: {ACCENT}; border-radius: 10px; color: #fff; }}
</style>
"""
