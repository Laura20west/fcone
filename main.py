# filename: sexy_sally_chatbot.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import time

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
            "model_name": "Xara2west/gpt2-finetuned-cone2",
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
<meta name="viewport" content="width=1200, initial-scale=1.0">
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #e6f7ff, #b3e0ff) !important;
        background-attachment: fixed !important;
        background-size: cover !important;
        min-width: 1200px !important;
    }
    
    /* Text input color */
    .stTextArea textarea {
        color: #D891EF !important;
    }
    
    /* Response log styling */
    .response-log {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    /* Existing styles remain unchanged */
    /* ... (keep all existing CSS styles) ... */
</style>
""", unsafe_allow_html=True)

# ====================
# Chat System
# ====================
if "chat" not in st.session_state:
    st.session_state.chat = {
        "messages": [],
        "mode": "normal",
        "context_history": []
    }

def set_mode(mode):
    st.session_state.chat["mode"] = mode

def build_context(prompt):
    """Build conversation context from last 3 exchanges"""
    context = []
    for msg in st.session_state.chat["messages"][-6:]:  # Last 3 pairs
        context.append(f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}")
    context.append(f"User: {prompt}")
    return "\n".join(context)

# Header
st.markdown("""
<div style="text-align: center; padding: 1rem; background: rgba(255, 255, 255, 0.9); border-radius: 15px; margin-bottom: 1rem;">
    <h1 style="color: #006699; font-family: 'Arial', sans-serif;">
        {icon} Sally Chatbot {icon}
    </h1>
    <p style="color: #4db8ff;">
        Your AI Dating Companion
    </p>
</div>
""".format(icon="🌸" if st.session_state.chat["mode"] == "normal" else "🔥"), 
unsafe_allow_html=True)

# Chat history
for msg in st.session_state.chat["messages"]:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-message">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        icon = "🌸" if msg.get("mode", "normal") == "normal" else "🔥"
        st.markdown(f'<div class="bot-message">{icon} Sally: {msg["content"]}</div>', unsafe_allow_html=True)

# Response log expander
with st.expander("📜 Response Log"):
    for msg in st.session_state.chat["messages"]:
        if msg["role"] == "assistant":
            st.markdown(f"""
            <div class="response-log">
                <strong>ID:</strong> {msg['id']}<br>
                <strong>Mode:</strong> {msg['mode'].title()}<br>
                <strong>Response:</strong> {msg['content']}
            </div>
            """, unsafe_allow_html=True)

# Mode toggle
cols = st.columns([1,2,2,1])
with cols[1]:
    st.button("💖 Flirt Mode", key="flirt_btn", on_click=lambda: set_mode("flirt"))
with cols[2]:
    st.button("💬 Normal Mode", key="normal_btn", on_click=lambda: set_mode("normal"))

# Chat input
with st.form("chat_form"):
    prompt = st.text_area(
        "Your message:",
        height=100,
        key="chat_input",
        placeholder="Type your message here...",
        label_visibility="collapsed"
    )
    
    submitted = st.form_submit_button("Send 🌊")
    
    if submitted and prompt and models:
        st.session_state.chat["messages"].append({"role": "user", "content": prompt})
        
        current_mode = st.session_state.chat["mode"]
        with st.spinner("🌸 Sally is thinking..." if current_mode == "normal" else "🔥 Sally is getting flirty..."):
            try:
                model_data = models[current_mode]
                context_prompt = build_context(prompt)
                
                inputs = model_data["tokenizer"].encode(
                    context_prompt,
                    return_tensors="pt",
                    max_length=model_data["config"]["max_length"],
                    truncation=True
                ).to(device)
                
                outputs = model_data["model"].generate(
                    inputs,
                    max_length=model_data["config"]["max_length"],
                    temperature=model_data["config"]["temperature"],
                    top_k=40,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=model_data["tokenizer"].eos_token_id
                )
                
                full_response = model_data["tokenizer"].decode(outputs[0], skip_special_tokens=True)
                # Extract only the new response
                response = full_response.split("Assistant:")[-1].strip()
                response = response.split("User:")[0].strip()  # Prevent including future user inputs
                response = ' '.join(response.split()[:200])
                
                if current_mode == "flirt" and not any(response.endswith(p) for p in ('?', '!', '.')):
                    response += "... 😉"
                
                # Generate unique response ID
                response_id = f"resp_{int(time.time()*1000)}"
                
                st.session_state.chat["messages"].append({
                    "role": "assistant",
                    "content": response,
                    "mode": current_mode,
                    "id": response_id
                })
                
            except Exception as e:
                response = f"⚠️ Error: {str(e)}"
                st.session_state.chat["messages"].append({
                    "role": "assistant",
                    "content": response,
                    "mode": "error",
                    "id": f"error_{int(time.time()*1000)}"
                })
        
        st.rerun()

# Clear chat
if st.session_state.chat["messages"]:
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chat["messages"] = []
        st.rerun()

# Mobile mode JavaScript
st.markdown(f"""
<script>
document.addEventListener('DOMContentLoaded', function() {{
    const meta = document.createElement('meta');
    meta.name = 'viewport';
    meta.content = 'width=1200, initial-scale=1.0, maximum-scale=1.0, user-scalable=0';
    document.head.appendChild(meta);

    const currentMode = "{st.session_state.chat["mode"]}";
    const buttons = {{
        'flirt': document.querySelector('[data-testid="baseButton-secondary"]'),
        'normal': document.querySelector('[data-testid="baseButton-primary"]')
    }};
    
    if (buttons[currentMode]) buttons[currentMode].classList.add('active-mode');
    
    if(/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)) {{
        document.body.style.zoom = "80%";
        document.querySelector('.stApp').style.height = "120vh";
    }}
}});
</script>
""", unsafe_allow_html=True)
