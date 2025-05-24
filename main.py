# filename: sexy_sally_chatbot.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from datetime import datetime

# ====================
# Configuration
# ====================
MODEL_CONFIG = {
    "flirt": {
        "model_name": "xara2west/gpt2-finetuned-cone",
        "temperature": 0.85,
        "max_length": 200,
        "max_context": 1024
    },
    "normal": {
        "model_name": "google/flan-t5-base",
        "temperature": 0.7,
        "max_length": 200,
        "max_context": 768
    }
}

# ====================
# Model Loading
# ====================
@st.cache_resource
def load_models():
    try:
        models = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for mode, config in MODEL_CONFIG.items():
            tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
            model = AutoModelForCausalLM.from_pretrained(config["model_name"])
            model.to(device).eval()
            models[mode] = {**config, "tokenizer": tokenizer, "model": model}
            
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
    
    # Load models
    models, device = load_models()
    
    # ====================
    # UI Components
    # ====================
    st.set_page_config(
        page_title="🔥 Contextual Sally",
        page_icon="💋",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .user-message {
            background: #ff85a2;
            padding: 1rem;
            border-radius: 18px 18px 0 18px;
            margin: 0.8rem 0;
            max-width: 80%;
            margin-left: auto;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
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
            color: white;
            border: 1px solid #ff1493;
        }
        .context-pill {
            background: #ff1493;
            color: white;
            border-radius: 15px;
            padding: 4px 12px;
            margin: 2px;
            display: inline-block;
            font-size: 0.8em;
        }
        .emotional-display {
            border-left: 4px solid #ff69b4;
            padding-left: 1rem;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: rgba(255, 255, 255, 0.9); 
                border-radius: 15px; margin-bottom: 1rem; border: 1px solid #ff6b8b;">
        <h1 style="color: #ff1493; font-family: 'Arial', sans-serif;">
            💋 Context-Aware Sally 💋
        </h1>
        <p style="color: #ff6b8b;">
            "I remember... I adapt... I enthrall"
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Context Sidebar
    with st.expander("🧠 Sally's Mind"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("User Profile")
            if st.session_state.chat["context"]["user_profile"]["name"]:
                st.markdown(f"**Name:** {st.session_state.chat['context']['user_profile']['name']}")
            st.markdown(f"**Interests:** {', '.join(st.session_state.chat['context']['user_profile']['interests']) or 'None detected'}")
            
        with col2:
            st.subheader("Conversation State")
            st.markdown(f"**Mode:** {st.session_state.chat['mode'].title()}")
            st.markdown(f"**Emotional Tone:** {st.session_state.chat['context']['emotional_state']['current'].title()}")
            if st.session_state.chat["context"]["temporal_context"]["last_met"]:
                last_met = st.session_state.chat["context"]["temporal_context"]["last_met"].strftime('%Y-%m-%d %H:%M')
                st.markdown(f"**Last Met:** {last_met}")
    
    # Chat Interface
    for msg in st.session_state.chat["messages"]:
        role = "user" if msg["role"] == "user" else "Sally"
        style = "user-message" if role == "user" else "bot-message"
        icon = "💋" if msg.get("mode") == "flirt" else "🌸"
        st.markdown(f'<div class="{style}">{icon if role=="Sally" else "👤"} {role}: {msg["content"]}</div>', 
                    unsafe_allow_html=True)
    
    # Mode Toggle
    cols = st.columns([1,2,2,1])
    with cols[1]:
        st.button("🔥 Naughty Mode", on_click=lambda: st.session_state.chat.update({"mode": "flirt"}))
    with cols[2]:
        st.button("🌸 Normal Mode", on_click=lambda: st.session_state.chat.update({"mode": "normal"}))
    
    # ====================
    # Context Processing
    # ====================
    def update_context(prompt, response):
        """Update all contextual elements"""
        now = datetime.now()
        context = st.session_state.chat["context"]
        
        # Temporal context
        context["temporal_context"] = {
            "last_met": now,
            "time_since": now - context["temporal_context"]["last_met"] if context["temporal_context"]["last_met"] else None
        }
        
        # User profile updates
        if not context["user_profile"]["name"]:
            context["user_profile"]["name"] = extract_name(prompt)
        context["user_profile"]["interests"] = list(set(
            context["user_profile"]["interests"] + detect_interests(prompt)
        ))[:5]
        
        # Emotional state tracking
        sentiment = analyze_sentiment(prompt)
        context["emotional_state"]["history"].append(sentiment)
        context["emotional_state"]["current"] = max(
            set(context["emotional_state"]["history"][-5:]),
            key=context["emotional_state"]["history"][-5:].count
        )
        
        # Conversation history
        context["conversation_history"].extend([
            {"role": "user", "content": prompt, "timestamp": now},
            {"role": "assistant", "content": response, "timestamp": now}
        ])
    
    def analyze_sentiment(text):
        """Simple sentiment analysis"""
        positive_words = ["love", "happy", "great", "excited", "wonderful"]
        negative_words = ["hate", "sad", "angry", "frustrated", "upset"]
        
        if any(word in text.lower() for word in positive_words):
            return "positive"
        elif any(word in text.lower() for word in negative_words):
            return "negative"
        return "neutral"
    
    def extract_name(text):
        """Enhanced name extraction"""
        name_triggers = ["my name is", "call me", "I'm ", " im ", "name's"]
        for trigger in name_triggers:
            if trigger in text.lower():
                start = text.lower().find(trigger) + len(trigger)
                return text[start:].split()[0].strip(",.!?")
        return None
    
    def detect_interests(text):
        """Interest detection"""
        interests = []
        interest_map = {
            "sports": ["sport", "game", "team", "play"],
            "music": ["song", "music", "band", "artist"],
            "tech": ["tech", "computer", "code", "AI"]
        }
        for interest, keywords in interest_map.items():
            if any(keyword in text.lower() for keyword in keywords):
                interests.append(interest)
        return interests
    
    # ====================
    # Response Generation
    # ====================
    with st.form("chat_form"):
        prompt = st.text_area("Your message:", height=100, placeholder="Speak to Sally...", key="input")
        
        if st.form_submit_button("Send 💋") and prompt:
            with st.spinner(f"{'💋' if st.session_state.chat['mode'] == 'flirt' else '🌸'} Sally is thinking..."):
                try:
                    # Generate response
                    model = models[st.session_state.chat["mode"]]
                    context_prompt = build_context_prompt(prompt)
                    inputs = model["tokenizer"](context_prompt, return_tensors="pt").to(device)
                    
                    outputs = model["model"].generate(
                        inputs.input_ids,
                        max_length=model["max_length"],
                        temperature=model["temperature"],
                        pad_token_id=model["tokenizer"].eos_token_id
                    )
                    
                    response = model["tokenizer"].decode(outputs[0], skip_special_tokens=True)
                    response = postprocess_response(response)
                    
                    # Update context and messages
                    update_context(prompt, response)
                    st.session_state.chat["messages"].extend([
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response, "mode": st.session_state.chat["mode"]}
                    ])
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
    
    def build_context_prompt(prompt):
        """Construct context-aware prompt"""
        context = st.session_state.chat["context"]
        now = datetime.now()
        
        base_context = f"""
        You are Sally, a sophisticated AI companion. Current datetime: {now.strftime('%Y-%m-%d %H:%M')}
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
        
        # Add recent conversation history
        history = "\n".join([f"{msg['role']}: {msg['content']}" 
                           for msg in context["conversation_history"][-4:]])
        
        return f"{base_context}\n\n{history}\nUser: {prompt}\nSally:"
    
    def postprocess_response(response):
        """Enhance response with contextual elements"""
        # Personalization
        if name := st.session_state.chat["context"]["user_profile"]["name"]:
            response = response.replace("you", name, 1).replace("You", name, 1)
        
        # Temporal awareness
        current_hour = datetime.now().hour
        if current_hour < 12 and "good morning" not in response.lower():
            response = f"🌞 {response}"
        elif current_hour < 18 and "good afternoon" not in response.lower():
            response = f"🌇 {response}"
        else:
            response = f"🌙 {response}"
        
        # Emotional alignment
        emotion = st.session_state.chat["context"]["emotional_state"]["current"]
        if emotion == "positive":
            response = response.replace(".", "! 😊")
        elif emotion == "negative":
            response = response.replace(".", "... 😔").replace("!", ".")
        
        return response

if __name__ == "__main__":
    main()
        
