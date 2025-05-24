# filename: sexy_sally_chatbot.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration
import torch
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ====================
# Configuration
# ====================
MODEL_CONFIG = {
    "flirt": {
        "model_name": "gpt2",  # Replaced with standard GPT-2 as original might not be available
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
def load_models() -> Tuple[Optional[Dict], Optional[torch.device]]:
    """Load all required models and tokenizers."""
    try:
        models = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for mode, config in MODEL_CONFIG.items():
            with st.spinner(f"Loading {mode} model..."):
                tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
                if mode == "flirt":
                    tokenizer.pad_token = tokenizer.eos_token
                
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
    # Initialize session state
    if "chat" not in st.session_state:
        st.session_state.chat = {
            "messages": [],
            "mode": "normal",
            "context": {
                "user_profile": {"name": None, "age": None, "interests": []},
                "conversation_history": [],
                "temporal_context": {"last_met": None, "time_since": None},
                "emotional_state": {"current": "neutral", "history": []}
            }
        }
        # Add initial greeting
        st.session_state.chat["messages"].append({
            "role": "assistant",
            "content": MODEL_CONFIG["normal"]["greeting"],
            "mode": "normal"
        })
    
    # Load models
    models, device = load_models()
    if models is None:
        st.error("Failed to load models. Please check the error message above.")
        st.stop()
    
    # ====================
    # UI Components
    # ====================
    st.set_page_config(
        page_title="🔥 Contextual Sally",
        page_icon="💋",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    current_color = MODEL_CONFIG[st.session_state.chat["mode"]]["color"]
    st.markdown(f"""
    <style>
        .user-message {{
            background: #f0f2f6;
            padding: 1rem;
            border-radius: 18px 18px 0 18px;
            margin: 0.8rem 0;
            max-width: 80%;
            margin-left: auto;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
            color: #333;
            border: 1px solid #ddd;
        }}
        .bot-message {{
            background: {current_color};
            padding: 1rem;
            border-radius: 18px 18px 18px 0;
            margin: 0.8rem 0;
            max-width: 80%;
            margin-right: auto;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
            color: white;
            border: 1px solid {current_color};
        }}
        .context-pill {{
            background: {current_color};
            color: white;
            border-radius: 15px;
            padding: 4px 12px;
            margin: 2px;
            display: inline-block;
            font-size: 0.8em;
        }}
        .emotional-display {{
            border-left: 4px solid {current_color};
            padding-left: 1rem;
            margin: 1rem 0;
        }}
        .stButton button {{
            background-color: {current_color};
            color: white;
            border: none;
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(to bottom, #ffffff, #f8f9fa);
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: rgba(255, 255, 255, 0.9); 
                border-radius: 15px; margin-bottom: 1rem; border: 1px solid {current_color};">
        <h1 style="color: {current_color}; font-family: 'Arial', sans-serif;">
            💋 Context-Aware Sally 💋
        </h1>
        <p style="color: {current_color};">
            "I remember... I adapt... I enthrall"
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with context information
    with st.sidebar:
        st.markdown("## 🧠 Sally's Mind")
        
        with st.expander("User Profile"):
            if st.session_state.chat["context"]["user_profile"]["name"]:
                st.markdown(f"**Name:** {st.session_state.chat['context']['user_profile']['name']}")
            else:
                st.markdown("**Name:** Not yet known")
            
            interests = st.session_state.chat["context"]["user_profile"]["interests"]
            st.markdown(f"**Interests:** {', '.join(interests) if interests else 'None detected'}")
        
        with st.expander("Conversation State"):
            st.markdown(f"**Mode:** {st.session_state.chat['mode'].title()}")
            st.markdown(f"**Emotional Tone:** {st.session_state.chat['context']['emotional_state']['current'].title()}")
            
            if st.session_state.chat["context"]["temporal_context"]["last_met"]:
                last_met = st.session_state.chat["context"]["temporal_context"]["last_met"].strftime('%Y-%m-%d %H:%M')
                st.markdown(f"**Last Met:** {last_met}")
            else:
                st.markdown("**Last Met:** First conversation")
        
        if st.button("🔄 Reset Conversation"):
            st.session_state.clear()
            st.rerun()
    
    # Chat Interface
    for msg in st.session_state.chat["messages"]:
        role = "user" if msg["role"] == "user" else "Sally"
        style = "user-message" if role == "user" else "bot-message"
        icon = "💋" if msg.get("mode") == "flirt" else "🌸"
        st.markdown(f'<div class="{style}">{icon if role=="Sally" else "👤"} <strong>{role}:</strong> {msg["content"]}</div>', 
                    unsafe_allow_html=True)
    
    # Mode Toggle
    cols = st.columns([1, 2, 2, 1])
    with cols[1]:
        if st.button("🔥 Naughty Mode", use_container_width=True):
            st.session_state.chat["mode"] = "flirt"
            st.session_state.chat["messages"].append({
                "role": "assistant",
                "content": MODEL_CONFIG["flirt"]["greeting"],
                "mode": "flirt"
            })
            st.rerun()
    with cols[2]:
        if st.button("🌸 Normal Mode", use_container_width=True):
            st.session_state.chat["mode"] = "normal"
            st.session_state.chat["messages"].append({
                "role": "assistant",
                "content": MODEL_CONFIG["normal"]["greeting"],
                "mode": "normal"
            })
            st.rerun()
    
    # ====================
    # Context Processing
    # ====================
    def update_context(prompt: str, response: str) -> None:
        """Update all contextual elements based on the latest exchange."""
        now = datetime.now()
        context = st.session_state.chat["context"]
        
        # Temporal context
        context["temporal_context"] = {
            "last_met": now,
            "time_since": (now - context["temporal_context"]["last_met"]) 
                          if context["temporal_context"]["last_met"] 
                          else None
        }
        
        # User profile updates
        if not context["user_profile"]["name"]:
            if name := extract_name(prompt):
                context["user_profile"]["name"] = name
                
        detected_interests = detect_interests(prompt)
        context["user_profile"]["interests"] = list(
            set(context["user_profile"]["interests"] + detected_interests)
        )[:5]  # Keep only top 5 interests
        
        # Emotional state tracking
        sentiment = analyze_sentiment(prompt)
        context["emotional_state"]["history"].append(sentiment)
        if len(context["emotional_state"]["history"]) > 5:
            context["emotional_state"]["history"] = context["emotional_state"]["history"][-5:]
        
        context["emotional_state"]["current"] = max(
            set(context["emotional_state"]["history"]),
            key=context["emotional_state"]["history"].count
        )
        
        # Conversation history
        context["conversation_history"].extend([
            {"role": "user", "content": prompt, "timestamp": now},
            {"role": "assistant", "content": response, "timestamp": now}
        ])
        if len(context["conversation_history"]) > 10:  # Keep last 10 exchanges
            context["conversation_history"] = context["conversation_history"][-10:]
    
    def analyze_sentiment(text: str) -> str:
        """Simple sentiment analysis."""
        positive_words = ["love", "happy", "great", "excited", "wonderful", "amazing"]
        negative_words = ["hate", "sad", "angry", "frustrated", "upset", "annoyed"]
        
        text_lower = text.lower()
        pos_count = sum(word in text_lower for word in positive_words)
        neg_count = sum(word in text_lower for word in negative_words)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        return "neutral"
    
    def extract_name(text: str) -> Optional[str]:
        """Enhanced name extraction from user input."""
        name_triggers = ["my name is", "call me", "I'm ", " im ", "name's", "I am "]
        for trigger in name_triggers:
            if trigger in text.lower():
                start = text.lower().find(trigger) + len(trigger)
                name_part = text[start:].split()[0].strip(",.!?")
                return name_part.capitalize()
        return None
    
    def detect_interests(text: str) -> List[str]:
        """Detect user interests from their messages."""
        interests = []
        interest_map = {
            "sports": ["sport", "game", "team", "play", "football", "basketball"],
            "music": ["song", "music", "band", "artist", "concert", "guitar"],
            "tech": ["tech", "computer", "code", "AI", "programming", "software"],
            "books": ["book", "read", "novel", "author", "literature"],
            "travel": ["travel", "trip", "vacation", "country", "city"]
        }
        
        text_lower = text.lower()
        for interest, keywords in interest_map.items():
            if any(keyword in text_lower for keyword in keywords):
                interests.append(interest)
        return interests
    
    # ====================
    # Response Generation
    # ====================
    def build_context_prompt(prompt: str) -> str:
        """Construct context-aware prompt for the model."""
        context = st.session_state.chat["context"]
        now = datetime.now()
        
        base_context = f"""
        Current datetime: {now.strftime('%Y-%m-%d %H:%M')}
        User profile:
        - Name: {context["user_profile"]["name"] or "Unknown"}
        - Interests: {', '.join(context["user_profile"]["interests"]) or 'Not specified'}
        
        Emotional context: {context["emotional_state"]["current"].title()}
        Conversation history: {len(context["conversation_history"])} exchanges
        """
        
        if st.session_state.chat["mode"] == "flirt":
            base_context += """
            Flirt mode guidelines:
            - Use playful, sensual language
            - Maintain mysterious allure
            - Gradually increase intimacy
            - Balance teasing with genuine interest
            """
        else:
            base_context += """
            Normal mode guidelines:
            - Be friendly and helpful
            - Show genuine interest
            - Keep responses appropriate
            - Adapt to user's emotional state
            """
        
        # Add recent conversation history
        history = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in context["conversation_history"][-4:]
        )
        
        return f"{base_context}\n\n{history}\nUser: {prompt}\nSally:"
    
    def postprocess_response(response: str) -> str:
        """Enhance response with contextual elements and formatting."""
        # Personalization
        if name := st.session_state.chat["context"]["user_profile"]["name"]:
            response = response.replace("you", name, 1).replace("You", name, 1)
        
        # Temporal awareness
        current_hour = datetime.now().hour
        if current_hour < 12 and "morning" not in response.lower():
            response = f"🌞 Good morning! {response}"
        elif current_hour < 18 and "afternoon" not in response.lower():
            response = f"🌇 Good afternoon! {response}"
        else:
            response = f"🌙 Good evening! {response}"
        
        # Emotional alignment
        emotion = st.session_state.chat["context"]["emotional_state"]["current"]
        if emotion == "positive":
            response = response.replace(".", "! 😊").replace("!", "! 😊")
        elif emotion == "negative":
            response = response.replace(".", "... 😔").replace("!", "... 😔")
        
        return response.strip()
    
    # Chat input form
    with st.form("chat_form", clear_on_submit=True):
        prompt = st.text_area(
            "Your message:", 
            height=100, 
            placeholder="Speak to Sally...", 
            key="input",
            label_visibility="collapsed"
        )
        
        submit_cols = st.columns([3, 1])
        with submit_cols[0]:
            if st.form_submit_button("Send", use_container_width=True):
                if not prompt.strip():
                    st.warning("Please enter a message")
                    st.stop()
                
                with st.spinner(f"{'💋' if st.session_state.chat['mode'] == 'flirt' else '🌸'} Sally is thinking..."):
                    try:
                        # Generate response
                        model_config = models[st.session_state.chat["mode"]]
                        context_prompt = build_context_prompt(prompt)
                        
                        # Tokenize input
                        inputs = model_config["tokenizer"](
                            context_prompt,
                            return_tensors="pt",
                            max_length=model_config["max_context"],
                            truncation=True
                        ).to(device)
                        
                        # Generate output
                        outputs = model_config["model"].generate(
                            inputs.input_ids,
                            max_length=model_config["max_length"],
                            temperature=model_config["temperature"],
                            pad_token_id=model_config["tokenizer"].eos_token_id if st.session_state.chat["mode"] == "flirt" else None,
                            do_sample=True
                        )
                        
                        # Decode and clean response
                        response = model_config["tokenizer"].decode(
                            outputs[0], 
                            skip_special_tokens=True
                        )
                        response = postprocess_response(response.split("Sally:")[-1].strip())
                        
                        # Update context and messages
                        update_context(prompt, response)
                        st.session_state.chat["messages"].extend([
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response, "mode": st.session_state.chat["mode"]}
                        ])
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        st.exception(e)
        
        with submit_cols[1]:
            if st.form_submit_button("Clear", type="secondary", use_container_width=True):
                prompt = ""
                st.rerun()

if __name__ == "__main__":
    main()
