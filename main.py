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
    page_title="🔥 Sexy Sally",
    page_icon="💋",
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
        background: linear-gradient(135deg, #ffebf3, #ffd6e7) !important;
        background-attachment: fixed !important;
        background-size: cover !important;
        min-width: 1200px !important;
    }
    
    /* Text input color */
    .stTextArea textarea {
        color: #ff69b4 !important;
        font-weight: 500 !important;
    }
    
    /* Response log styling */
    .response-log {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        border: 1px solid #ffb6c1;
    }
    
    /* Chat bubbles */
    .user-message {
        background: #ff85a2;
        padding: 1rem;
        border-radius: 18px 18px 0 18px;
        margin: 0.8rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
        font-family: 'Arial', sans-serif;
        color: white;
        border: 1px solid #ff6b8b;
    }
    
    .bot-message {
        background: #ff6b8b;
        padding: 1rem;
        border-radius: 18px 18px 18px 0;
        margin: 0.8rem 0;
        max-width: 80%;
        margin-right: auto;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
        font-family: 'Arial', sans-serif;
        color: white;
        border: 1px solid #ff1493;
    }
    
    /* Mode toggle buttons */
    .flirt-btn {
        background: #ff1493 !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 0.5rem 1.5rem !important;
    }
    
    .normal-btn {
        background: #ff6b8b !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 0.5rem 1.5rem !important;
    }
    
    /* Keep other existing styles */
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
    """Build conversation context with Sally's sexy persona"""
    persona = """
    You are Sally, a naughty, sexy woman at the perfect age. You exude confidence and sensual energy. 
    All men desire you, but you choose who gets to enjoy your company. You're playful, seductive, 
    and know exactly what you want. You enjoy flirting and teasing, always leaving them wanting more.
    """
    
    if st.session_state.chat["mode"] == "flirt":
        persona += """
        [Flirt Mode Activated] You're feeling extra playful tonight. Your words drip with sensual 
        suggestion and teasing promises. You maintain control while giving just enough to keep them 
        hooked. Every response should be charged with sexual energy and playful challenge.
        """
    
    context = [persona]
    for msg in st.session_state.chat["messages"][-6:]:
        context.append(f"{'User' if msg['role'] == 'user' else 'Sally'}: {msg['content']}")
    
    context.append(f"User: {prompt}")
    context.append("Sally: [responds with sensual confidence]")  # Direct response prompt
    
    return "\n".join(context)

# Header with sexy theme
st.markdown("""
<div style="text-align: center; padding: 1rem; background: rgba(255, 255, 255, 0.9); border-radius: 15px; margin-bottom: 1rem; border: 1px solid #ff6b8b;">
    <h1 style="color: #ff1493; font-family: 'Arial', sans-serif;">
        💋 Sally - Your Naughty Companion 💋
    </h1>
    <p style="color: #ff6b8b;">
        "I know what you want... but do you deserve it?"
    </p>
</div>
""", unsafe_allow_html=True)

# Chat history
for msg in st.session_state.chat["messages"]:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-message">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        icon = "💋" if msg.get("mode", "normal") == "flirt" else "🌸"
        st.markdown(f'<div class="bot-message">{icon} Sally: {msg["content"]}</div>', unsafe_allow_html=True)

# Response log expander
with st.expander("📜 Intimate Thoughts (Response Log)"):
    for msg in st.session_state.chat["messages"]:
        if msg["role"] == "assistant":
            st.markdown(f"""
            <div class="response-log">
                <strong>Session ID:</strong> {msg['id']}<br>
                <strong>Mode:</strong> {msg['mode'].title()}<br>
                <strong>Whispered:</strong> <em>{msg['content']}</em>
            </div>
            """, unsafe_allow_html=True)

# Mode toggle
cols = st.columns([1,2,2,1])
with cols[1]:
    st.button("🔥 Naughty Mode", key="flirt_btn", on_click=lambda: set_mode("flirt"))
with cols[2]:
    st.button("💄 Playful Mode", key="normal_btn", on_click=lambda: set_mode("normal"))

# Chat input
with st.form("chat_form"):
    prompt = st.text_area(
        "Your message:",
        height=100,
        key="chat_input",
        placeholder="Tell Sally what's on your mind...",
        label_visibility="collapsed"
    )
    
    submitted = st.form_submit_button("Send 💋")
    
    if submitted and prompt and models:
        st.session_state.chat["messages"].append({"role": "user", "content": prompt})
        
        current_mode = st.session_state.chat["mode"]
        with st.spinner("💋 Sally is considering your offer..." if current_mode == "flirt" else "🌸 Sally is listening..."):
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
                response = full_response.split("Sally: [responds with sensual confidence]")[-1].strip()
                response = response.split("User:")[0].strip()
                response = ' '.join(response.split()[:200])
                
                # Enhance responses with sensual language
                if current_mode == "flirt":
                    response = response.replace(" you ", " you, bad boy, ").replace(" your ", " that delicious ")
                    if not any(response.endswith(p) for p in ('?', '!', '.')):
                        response += "... 💋"
                
                # Generate unique response ID
                response_id = f"sally_{int(time.time()*1000)}"
                
                st.session_state.chat["messages"].append({
                    "role": "assistant",
                    "content": response,
                    "mode": current_mode,
                    "id": response_id
                })
                
            except Exception as e:
                response = f"⚠️ Oh darling, something went wrong... {str(e)}"
                st.session_state.chat["messages"].append({
                    "role": "assistant",
                    "content": response,
                    "mode": "error",
                    "id": f"error_{int(time.time()*1000)}"
                })
        
        st.rerun()

# Clear chat
if st.session_state.chat["messages"]:
    if st.button("✨ Start Over", use_container_width=True):
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
