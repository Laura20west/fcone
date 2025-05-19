# filename: sexy_sally_chatbot.py
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
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
    models = {
        "flirt": {"model": None, "tokenizer": None},
        "normal": {"model": None, "tokenizer": None}
    }
    
    try:
        # Load flirt model
        flirt_path = "./sexyGPT-Uncensored"
        if os.path.exists(flirt_path):
            models["flirt"]["tokenizer"] = GPT2Tokenizer.from_pretrained(flirt_path)
            models["flirt"]["model"] = GPT2LMHeadModel.from_pretrained(flirt_path)
        else:
            st.warning("Flirt model not found, using standard GPT-2 for both modes")
            models["flirt"]["tokenizer"] = GPT2Tokenizer.from_pretrained("gpt2")
            models["flirt"]["model"] = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # Load normal model
        models["normal"]["tokenizer"] = GPT2Tokenizer.from_pretrained("gpt2")
        models["normal"]["model"] = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # Move models to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for mode in models:
            models[mode]["model"].to(device)
            models[mode]["model"].eval()
            
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
        placeholder="Type your message here...",
        label_visibility="collapsed"
    )
    
    submitted = st.form_submit_button("Send 🌶️")
    
    if submitted and prompt and models:
        # Add user message
        st.session_state.chat["messages"].append({
            "role": "user",
            "content": prompt
        })
        
        # Generate response
        current_mode = st.session_state.chat["mode"]
        with st.spinner("🌸 Sally is thinking..." if current_mode == "normal" else "🔥 Sally is getting hot..."):
            try:
                inputs = models[current_mode]["tokenizer"].encode(
                    prompt, 
                    return_tensors="pt"
                ).to(device)
                
                outputs = models[current_mode]["model"].generate(
                    inputs,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7 if current_mode == "normal" else 0.9,
                    top_k=40,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=models[current_mode]["tokenizer"].eos_token_id
                )
                
                response = models[current_mode]["tokenizer"].decode(
                    outputs[0], 
                    skip_special_tokens=True
                )
                
                # Post-processing
                response = response.replace(prompt, "").strip()
                words = response.split()[:200]
                response = ' '.join(words)
                
                # Add appropriate ending
                if current_mode == "flirt":
                    if not any(response.endswith(p) for p in ('?', '!', '.')):
                        response += "..."
                    response += " What do you think about that, hot stuff?"
                
            except Exception as e:
                response = f"⚠️ Error: {str(e)}"
        
        # Add bot response
        st.session_state.chat["messages"].append({
            "role": "assistant",
            "content": response,
            "mode": current_mode
        })
        
        st.rerun()

# Clear chat button
if st.session_state.chat["messages"]:
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chat["messages"] = []
        st.rerun()

# Add JavaScript to style buttons based on active mode
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    const currentMode = "%s";
    const buttons = {
        'flirt': document.querySelector('[data-testid="baseButton-secondary"]'),
        'normal': document.querySelector('[data-testid="baseButton-primary"]')
    };
    
    if (buttons[currentMode]) {
        buttons[currentMode].classList.add('active-mode');
    }
});
</script>
""" % st.session_state.chat["mode"], unsafe_allow_html=True)
