# filename: sexy_sally_chatbot.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# ====================
# UI Configuration
# ====================
st.set_page_config(
    page_title="🔥 Sally Chatbot",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ====================
# Model Loading System
# ====================
@st.cache_resource
def load_models():
    model_config = {
        "flirt": {
            "model_name": "xara2west/gpt2-finetuned-cone",
            "temperature": 0.9,
            "max_length": 200
        },
        "normal": {
            "model_name": "gpt2",
            "temperature": 0.7,
            "max_length": 150
        }
    }
    
    try:
        models = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for mode, config in model_config.items():
            tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
            model = AutoModelForCausalLM.from_pretrained(config["model_name"])
            
            model.to(device)
            model.eval()
            
            models[mode] = {
                "tokenizer": tokenizer,
                "model": model,
                "config": config
            }
            
        return models, device
        
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None, None

models, device = load_models()

# ====================
# Enhanced UI Styling
# ====================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-image: linear-gradient(135deg, #ff66f9, #ff66f9);
    }
    
    /* Chat bubbles */
    .user-message {
        background: #ff66b2;
        padding: 1rem;
        border-radius: 18px 18px 0 18px;
        margin: 0.8rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.15);
        font-family: 'Arial', sans-serif;
        color: white;
        border: 1px solid #ff0066;
    }
    
    .bot-message {
        background: #4d79ff;
        padding: 1rem;
        border-radius: 18px 18px 18px 0;
        margin: 0.8rem 0;
        max-width: 80%;
        margin-right: auto;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.15);
        font-family: 'Arial', sans-serif;
        color: white;
        border: 1px solid #0039e6;
    }
    
    /* Mode toggle buttons */
    .mode-toggle {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .flirt-btn {
        background: #ff1493 !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 0.5rem 1.5rem !important;
    }
    
    .normal-btn {
        background: #66b3ff !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 0.5rem 1.5rem !important;
    }
    
    .active-mode {
        transform: scale(1.05);
        box-shadow: 0 0 10px currentColor !important;
        font-weight: bold !important;
    }
    
    /* Input area */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1px solid #ccc !important;
    }
    
    /* Send button */
    .stButton>button {
        border-radius: 12px !important;
        padding: 0.5rem 1.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ====================
# Chat System
# ====================
if "chat" not in st.session_state:
    st.session_state.chat = {
        "messages": [],
        "mode": "normal"  # Default mode
    }

def set_mode(mode):
    """Helper function to set the chat mode"""
    st.session_state.chat["mode"] = mode

# Header
st.markdown("""
<div style="text-align: center;">
    <h1 style="color: #4a2040; font-family: 'Arial', sans-serif;">
        {icon} Sally Chatbot {icon}
    </h1>
    <p style="color: #6b446b;">
        Your customizable AI companion
    </p>
</div>
""".format(icon="🌸" if st.session_state.chat["mode"] == "normal" else "🔥"), 
unsafe_allow_html=True)

# Chat history display
for msg in st.session_state.chat["messages"]:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-message">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        icon = "🌸" if msg.get("mode", "normal") == "normal" else "🔥"
        st.markdown(f'<div class="bot-message">{icon} Sally: {msg["content"]}</div>', unsafe_allow_html=True)

# Mode toggle buttons
cols = st.columns([1,2,2,1])
with cols[1]:
    if st.button("💖 Flirt Mode", 
               key="flirt_btn", 
               help="Switch to flirty conversation mode",
               on_click=lambda: set_mode("flirt")):
        pass
with cols[2]:
    if st.button("💬 Normal Mode", 
               key="normal_btn",
               help="Switch to normal conversation mode",
               on_click=lambda: set_mode("normal")):
        pass

# Chat input
with st.form("chat_form"):
    prompt = st.text_area(
        "Your message:",
        height=100,
        key="chat_input",
        placeholder="Type your message here...
