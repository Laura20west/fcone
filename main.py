# filename: sexy_sally_chatbot.py
import streamlit as st
import os
import sys
import subprocess
from packaging import version

# ====================
# Package Installation
# ====================
def install_packages():
    required = {
        'streamlit': '1.22.0',
        'transformers': '4.26.0',
        'torch': '1.13.0'
    }
    
    try:
        import pkg_resources
        installed = {}
        for pkg in pkg_resources.working_set:
            installed[pkg.key] = pkg.version
        
        missing = []
        for pkg, req_version in required.items():
            if pkg not in installed or version.parse(installed[pkg]) < version.parse(req_version):
                missing.append(pkg)
        
        if missing:
            python = sys.executable
            subprocess.check_call([python, '-m', 'pip', 'install', *missing, '--upgrade'])
            
    except Exception as e:
        st.error(f"Package installation failed: {str(e)}")
        st.stop()

install_packages()

# ====================
# Main Imports
# ====================
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
except ImportError as e:
    st.error(f"Failed to import required packages: {str(e)}")
    st.stop()

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
# Custom CSS
# ====================
st.markdown("""
<style>
    .stApp {
        background-image: linear-gradient(135deg, #ffd6e7, #d6e3ff);
        padding: 2rem;
    }
    
    .user-message {
        background: #ff66b2;
        padding: 1rem;
        border-radius: 18px 18px 0 18px;
        margin: 0.8rem 0;
        max-width: 80%;
        margin-left: auto;
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
        color: white;
        border: 1px solid #0039e6;
    }
    
    .mode-btn {
        border-radius: 20px !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.3s ease !important;
        margin: 0.5rem !important;
    }
    
    .flirt-btn {
        background: #ff1493 !important;
        color: white !important;
    }
    
    .normal-btn {
        background: #66b3ff !important;
        color: white !important;
    }
    
    .active-mode {
        transform: scale(1.05);
        box-shadow: 0 0 10px currentColor !important;
        font-weight: bold !important;
    }
    
    .stTextArea textarea {
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# ====================
# Model Loading
# ====================
@st.cache_resource
def load_models():
    models = {
        "flirt": {"model": None, "tokenizer": None},
        "normal": {"model": None, "tokenizer": None}
    }
    
    try:
        # Load normal model
        models["normal"]["tokenizer"] = GPT2Tokenizer.from_pretrained("gpt2")
        models["normal"]["model"] = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # Try loading custom flirt model
        try:
            models["flirt"]["tokenizer"] = GPT2Tokenizer.from_pretrained("gpt2")
            models["flirt"]["model"] = GPT2LMHeadModel.from_pretrained("gpt2")
        except Exception:
            models["flirt"] = models["normal"]
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for mode in models:
            models[mode]["model"].to(device)
            models[mode]["model"].eval()
            
        return models, device
        
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

models, device = load_models()

# ====================
# Chat System
# ====================
if "chat" not in st.session_state:
    st.session_state.chat = {
        "messages": [],
        "mode": "normal"
    }

def set_mode(mode):
    st.session_state.chat["mode"] = mode

# Header
current_mode = st.session_state.chat["mode"]
st.markdown(f"""
<div style="text-align: center;">
    <h1 style="color: #4a2040;">
        {'🔥' if current_mode == 'flirt' else '🌸'} Sally Chatbot {'🔥' if current_mode == 'flirt' else '🌸'}
    </h1>
    <p style="color: #6b446b;">
        {'Your seductive AI companion' if current_mode == 'flirt' else 'Your friendly AI companion'}
    </p>
</div>
""", unsafe_allow_html=True)

# Chat history
for msg in st.session_state.chat["messages"]:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-message">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        icon = "🔥" if msg.get("mode") == "flirt" else "🌸"
        st.markdown(f'<div class="bot-message">{icon} Sally: {msg["content"]}</div>', unsafe_allow_html=True)

# Mode toggle
col1, col2 = st.columns(2)
with col1:
    st.button(
        "💖 Flirt Mode",
        key="flirt_btn",
        on_click=lambda: set_mode("flirt"),
        help="Switch to flirty conversation mode",
        type="secondary" if current_mode != "flirt" else "primary"
    )
with col2:
    st.button(
        "💬 Normal Mode",
        key="normal_btn",
        on_click=lambda: set_mode("normal"),
        help="Switch to normal conversation mode",
        type="secondary" if current_mode != "normal" else "primary"
    )

# Chat input
with st.form("chat_form"):
    prompt = st.text_area(
        "Your message:",
        height=100,
        key="chat_input",
        placeholder="Type your message here...",
        label_visibility="collapsed"
    )
    
    submitted = st.form_submit_button("Send ✉️")
    
    if submitted and prompt and models:
        st.session_state.chat["messages"].append({
            "role": "user",
            "content": prompt
        })
        
        current_mode = st.session_state.chat["mode"]
        with st.spinner("🔥 Sally is getting hot..." if current_mode == "flirt" else "🌸 Sally is thinking..."):
            try:
                inputs = models[current_mode]["tokenizer"].encode(
                    prompt,
                    return_tensors="pt"
                ).to(device)
                
                outputs = models[current_mode]["model"].generate(
                    inputs,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.9 if current_mode == "flirt" else 0.7,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=models[current_mode]["tokenizer"].eos_token_id
                )
                
                response = models[current_mode]["tokenizer"].decode(
                    outputs[0],
                    skip_special_tokens=True
                ).replace(prompt, "").strip()
                
                words = response.split()[:200]
                response = ' '.join(words)
                
                if current_mode == "flirt":
                    if not response.endswith(('?', '!', '.')):
                        response += "..."
                    response += " What do you think about that, hot stuff?"
                
            except Exception as e:
                response = f"⚠️ Error: {str(e)}"
        
        st.session_state.chat["messages"].append({
            "role": "assistant",
            "content": response,
            "mode": current_mode
        })
        
        st.rerun()

# Clear chat
if st.session_state.chat["messages"]:
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chat["messages"] = []
        st.rerun()
