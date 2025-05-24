# filename: sexy_sally_chatbot.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration
import torch
from datetime import datetime

# ====================
# Configuration
# ====================
MODEL_CONFIG = {
    "flirt": {
        "model_name": "gpt2",
        "temperature": 0.85,
        "max_length": 200,
        "max_context": 1024,
        "model_class": AutoModelForCausalLM,
        "greeting": "💋 Well hello there, darling... what's on your mind?",
        "color": "#ff6b8b"
    },
    "normal": {
        "model_name": "google/flan-t5-base",
        "temperature": 0.7,
        "max_length": 200,
        "max_context": 768,
        "model_class": T5ForConditionalGeneration,
        "greeting": "🌸 Hello! How can I help you today?",
        "color": "#69b3a2"
    }
}

# ====================
# Model Loading
# ====================
@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    """Load all required models and tokenizers with proper configuration"""
    try:
        models = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for mode, config in MODEL_CONFIG.items():
            with st.spinner(f"Loading {mode} model..."):
                # Load tokenizer with mode-specific settings
                tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
                if mode == "flirt":
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load correct model class for each mode
                model = config["model_class"].from_pretrained(config["model_name"])
                model.to(device).eval()
                
                models[mode] = {
                    **config,
                    "tokenizer": tokenizer,
                    "model": model
                }
                
        return models, device
    
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None, None

# ====================
# Core Application
# ====================
def main():
    # Set page config FIRST - Streamlit requirement
    st.set_page_config(
        page_title="🔥 Contextual Sally",
        page_icon="💋",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Initialize session state AFTER page config
    if "chat" not in st.session_state:
        st.session_state.chat = {
            "messages": [{
                "role": "assistant",
                "content": MODEL_CONFIG["normal"]["greeting"],
                "mode": "normal"
            }],
            "mode": "normal",
            "context": {
                "user_profile": {"name": None, "interests": []},
                "conversation_history": [],
                "temporal_context": {"last_met": None},
                "emotional_state": {"current": "neutral", "history": []}
            }
        }

    # Load models
    models, device = load_models()
    if models is None:
        st.error("❌ Failed to load models. Please check the error messages above.")
        st.stop()

    # ====================
    # UI Components
    # ====================
    current_mode = st.session_state.chat["mode"]
    current_color = MODEL_CONFIG[current_mode]["color"]
    
    # Dynamic CSS based on mode
    st.markdown(f"""
    <style>
        .bot-message {{
            background: {current_color};
            border-color: {current_color};
        }}
        .stButton button {{
            background-color: {current_color} !important;
        }}
        .mode-pill {{
            background: {current_color};
            color: white;
            border-radius: 20px;
            padding: 5px 15px;
            font-weight: bold;
        }}
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; margin-bottom: 2rem;
                border-radius: 15px; border: 2px solid {current_color};">
        <h1 style="color: {current_color};">
            💋 Context-Aware Sally 💋
        </h1>
        <div class="mode-pill">{current_mode.title()} Mode</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar with context
    with st.sidebar:
        st.header("🧠 Sally's Mind")
        with st.expander("User Profile"):
            profile = st.session_state.chat["context"]["user_profile"]
            st.write(f"**Name:** {profile['name'] or 'Unknown'}")
            st.write(f"**Interests:** {', '.join(profile['interests']) or 'None'}")
            
        with st.expander("Conversation State"):
            st.write(f"**Messages:** {len(st.session_state.chat['messages'])}")
            st.write(f"**Emotion:** {st.session_state.chat['context']['emotional_state']['current'].title()}")
            st.write(f"**Last Active:** {st.session_state.chat['context']['temporal_context']['last_met'] or 'Never'}")

    # Chat messages
    for msg in st.session_state.chat["messages"]:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                👤 You: {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-message">
                {MODEL_CONFIG[msg['mode']]['greeting'][0]} Sally: {msg["content"]}
            </div>
            """, unsafe_allow_html=True)

    # Mode toggle
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔥 Switch to Flirty", disabled=current_mode=="flirt"):
            st.session_state.chat["mode"] = "flirt"
            st.rerun()
    with col2:
        if st.button("🌸 Switch to Normal", disabled=current_mode=="normal"):
            st.session_state.chat["mode"] = "normal"
            st.rerun()

    # ====================
    # Chat Processing
    # ====================
    with st.form("chat-input"):
        prompt = st.text_input("Your message:", key="user_input")
        
        if st.form_submit_button("Send"):
            if not prompt.strip():
                st.warning("Please enter a message")
                st.stop()
            
            with st.spinner(f"{MODEL_CONFIG[current_mode]['greeting'][0]} Sally is thinking..."):
                try:
                    # Generate response
                    config = models[current_mode]
                    inputs = config["tokenizer"](
                        prompt,
                        return_tensors="pt",
                        max_length=config["max_context"],
                        truncation=True
                    ).to(device)
                    
                    outputs = config["model"].generate(
                        inputs.input_ids,
                        max_length=config["max_length"],
                        temperature=config["temperature"],
                        pad_token_id=config["tokenizer"].eos_token_id if current_mode == "flirt" else None
                    )
                    
                    response = config["tokenizer"].decode(outputs[0], skip_special_tokens=True)
                    
                    # Update conversation
                    st.session_state.chat["messages"].extend([
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response, "mode": current_mode}
                    ])
                    st.session_state.chat["context"]["temporal_context"]["last_met"] = datetime.now()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()
