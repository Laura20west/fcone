import os
import io
import re
import time
import base64
import torch
import sympy
import numpy as np
import streamlit as st
from sympy import symbols, solve, integrate, diff, limit, Eq, Derivative
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                       implicit_multiplication_application)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import speech_recognition as sr
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import soundfile as sf

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'history_ids' not in st.session_state:
    st.session_state.history_ids = None

class VoiceAuthenticator:
    def __init__(self, reference_voice_path="laura.wav", threshold=0.65):
        self.encoder = VoiceEncoder()
        self.threshold = threshold
        if Path(reference_voice_path).exists():
            self.reference_embed = self._get_embedding(reference_voice_path)
        else:
            self.reference_embed = None

    def _get_embedding(self, audio_path):
        wav = preprocess_wav(audio_path)
        return self.encoder.embed_utterance(wav)

    def is_authorized_voice(self, audio_data):
        try:
            if not self.reference_embed:
                return False
                
            temp_path = "temp_voice_check.wav"
            with open(temp_path, 'wb') as f:
                f.write(audio_data)
            test_embed = self._get_embedding(temp_path)
            similarity = np.dot(self.reference_embed, test_embed)
            os.remove(temp_path)
            return similarity > self.threshold
        except Exception as e:
            print(f"Voice auth error: {e}")
            return False

class AdvancedMathSolver:
    def __init__(self):
        self.transformations = (standard_transformations + 
                              (implicit_multiplication_application,))
        self.x, self.y, self.z = symbols('x y z')
        self.math_keywords = [
            'calculate', 'solve', 'math', 'equation', 'formula',
            'derivative', 'integral', 'limit', 'matrix', 'determinant',
            'area', 'volume', 'mean', 'median', 'standard deviation',
            '+', '-', '*', '/', '^', 'plus', 'minus', 'times', 'divided by',
            'add', 'subtract', 'multiply', 'divide', 'power', 'root',
            'what is', 'how much is', '%', 'percent', 'percentage'
        ]

    def _extract_expression(self, text, keyword):
        patterns = [
            fr"{keyword}\s*(?:of|for)?\s*(.+?)\s*(?:with|when|where|if|$)",
            fr"{keyword}\s*(.+?)\s*$"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match: return match.group(1).strip()
        return None

    def _solve_equation(self, expr_str):
        try:
            expr = parse_expr(expr_str, transformations=self.transformations)
            solutions = solve(expr)
            return solutions if solutions else "No solution found"
        except Exception as e:
            return f"Equation solving error: {e}"

    def _calculate_derivative(self, expr_str, var='x'):
        try:
            expr = parse_expr(expr_str, transformations=self.transformations)
            derivative = diff(expr, var)
            return derivative
        except Exception as e:
            return f"Derivative error: {e}"

    def _calculate_integral(self, expr_str, var='x'):
        try:
            expr = parse_expr(expr_str, transformations=self.transformations)
            integral = integrate(expr, var)
            return integral
        except Exception as e:
            return f"Integral error: {e}"

    def _calculate_limit(self, expr_str, var='x', point='oo'):
        try:
            expr = parse_expr(expr_str, transformations=self.transformations)
            lim = limit(expr, var, point)
            return lim
        except Exception as e:
            return f"Limit error: {e}"

    def solve(self, problem_text):
        problem_lower = problem_text.lower()
        if not any(keyword in problem_lower for keyword in self.math_keywords):
            return None

        try:
            # Equation solving
            if 'solve' in problem_lower or '=' in problem_text:
                expr = self._extract_expression(problem_text, 'solve') or problem_text
                return self._solve_equation(expr)
            
            # Derivatives
            elif 'derivative' in problem_lower or 'differentiate' in problem_lower:
                expr = self._extract_expression(problem_text, 'derivative') or problem_text
                return self._calculate_derivative(expr)
            
            # Integrals
            elif 'integral' in problem_lower or 'integrate' in problem_lower:
                expr = self._extract_expression(problem_text, 'integral') or problem_text
                return self._calculate_integral(expr)
            
            # Limits
            elif 'limit' in problem_lower:
                expr = self._extract_expression(problem_text, 'limit') or problem_text
                return self._calculate_limit(expr)
            
            # Basic arithmetic
            elif any(op in problem_text for op in ['+', '-', '*', '/', '^']):
                expr = parse_expr(problem_text, transformations=self.transformations)
                return float(expr.evalf())
                
        except Exception as e:
            return f"Math error: {e}"
        
        return None

class IntegratedSystem:
    def __init__(self):
        self.math_solver = AdvancedMathSolver()
        self.voice_auth = VoiceAuthenticator()
        self.recognizer = sr.Recognizer()
        
        # Initialize models
        self._load_models()
        
    def _load_models(self):
        try:
            # Load dialogue model
            self.dialogue_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
            self.dialogue_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
            
            # Load grammar model (fallback if Xstage2 fails)
            try:
                self.grammar_pipeline = pipeline(
                    "text2text-generation",
                    model="Xara2west/Xstage2",
                    device=0 if torch.cuda.is_available() else -1
                )
            except:
                self.grammar_pipeline = pipeline(
                    "text-generation",
                    model="psmathur/orca_mini_3b",
                    device=0 if torch.cuda.is_available() else -1
                )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.dialogue_model = self.dialogue_model.to('cuda')
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            raise

    def generate_response(self, user_input):
        """Generate context-aware response with grammar correction"""
        # Math solving
        math_result = self.math_solver.solve(user_input)
        if math_result:
            return self._apply_math_personality(math_result)
        
        # Context-aware generation
        response = self._context_generation(user_input)
        
        # Grammar correction
        return self._grammar_correction(response)
    
    def _context_generation(self, user_input):
        """Generate response using DialoGPT with context awareness"""
        try:
            new_input_ids = self.dialogue_tokenizer.encode(
                user_input + self.dialogue_tokenizer.eos_token,
                return_tensors='pt'
            ).to(self.dialogue_model.device)
            
            # Context management
            if st.session_state.history_ids is not None:
                input_ids = torch.cat([st.session_state.history_ids, new_input_ids], dim=-1)
                if input_ids.shape[1] > 900:  # Truncate context
                    input_ids = input_ids[:, -900:]
            else:
                input_ids = new_input_ids
            
            # Generate response
            st.session_state.history_ids = self.dialogue_model.generate(
                input_ids,
                max_length=200 + input_ids.shape[1],
                min_length=20,
                pad_token_id=self.dialogue_tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.90,
                temperature=0.65,
                repetition_penalty=1.5,
                num_beams=3,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            
            # Extract only the new response
            response = self.dialogue_tokenizer.decode(
                st.session_state.history_ids[:, input_ids.shape[-1]:][0],
                skip_special_tokens=True
            )
            return response
        except Exception as e:
            print(f"Generation error: {e}")
            return "I'm having trouble generating a response right now."

    def _grammar_correction(self, text):
        """Apply grammar correction using Xstage2"""
        try:
            if not text.strip():
                return text
                
            params = {
                'max_new_tokens': min(100, len(text.split()) + 20),
                'temperature': 0.3,
                'repetition_penalty': 1.8,
                'no_repeat_ngram_size': 3
            }
            
            if "text2text-generation" in str(self.grammar_pipeline.task):
                params.update({'num_beams': 2})
            else:
                params.update({'top_p': 0.95})
                
            processed = self.grammar_pipeline(text, **params)[0]['generated_text']
            
            # Extract first complete sentence
            sentences = re.split(r'(?<=[.!?])\s', processed)
            return sentences[0] if sentences else processed
        except Exception as e:
            print(f"Grammar correction error: {e}")
            return text
    
    def _apply_math_personality(self, solution):
        """Apply personalized formatting to math solutions"""
        if isinstance(solution, (list, tuple)):
            solution_str = ", ".join([str(s) for s in solution])
        else:
            solution_str = str(solution)
            
        phrases = [
            "Babe, thee solution is {} 💖",
            "Math queen! The answer is {} ✨",
            "Solved it babe: {} 🎯",
            "Here's your solution darling: {} 💯"
        ]
        return np.random.choice(phrases).format(solution_str)
    
    def process_voice_command(self, audio_data):
        """Process voice input through authentication and recognition"""
        try:
            if not self.voice_auth.is_authorized_voice(audio_data):
                return "❌ Voice not recognized"
                
            text = self._speech_to_text(audio_data)
            if not text:
                return "Couldn't understand audio"
                
            return self.generate_response(text)
        except Exception as e:
            print(f"Voice command error: {e}")
            return "Error processing voice command"

    def _speech_to_text(self, audio_data):
        """Convert audio to text using Google's speech recognition"""
        try:
            temp_path = "temp_voice.wav"
            with open(temp_path, 'wb') as f:
                f.write(audio_data)
            
            with sr.AudioFile(temp_path) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)
                os.remove(temp_path)
                return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            print(f"Speech recognition error: {e}")
        return None

def autoplay_audio(file_path: str):
    """Embed audio player in Streamlit"""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f'<audio controls autoplay><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>'
            st.markdown(md, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error playing audio: {e}")

def main():
    st.title("Xara2west AI Companion")
    st.markdown("Conversational AI with math solving capabilities powered by Xstage2")
    
    # Initialize system
    if 'system' not in st.session_state:
        try:
            st.session_state.system = IntegratedSystem()
        except Exception as e:
            st.error(f"Failed to initialize system: {e}")
            return
    
    # Conversation display
    st.subheader("Conversation")
    conversation_container = st.container()
    
    # Input methods
    st.subheader("Input Method")
    input_method = st.radio("Choose input:", ("Text", "Microphone", "Upload Audio"))
    
    user_input = None
    
    # Text input
    if input_method == "Text":
        user_input = st.text_input("Type your message:")
        if st.button("Send") and user_input:
            st.session_state.conversation.append(("You", user_input))
            with st.spinner("Thinking..."):
                response = st.session_state.system.generate_response(user_input)
                st.session_state.conversation.append(("Xara2west", response))
    
    # Microphone input
    elif input_method == "Microphone":
        if st.button("Start Recording"):
            with st.spinner("Listening..."):
                recognizer = sr.Recognizer()
                microphone = sr.Microphone()
                
                try:
                    with microphone as source:
                        recognizer.adjust_for_ambient_noise(source)
                        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                        
                        audio_data = io.BytesIO()
                        audio_data.write(audio.get_wav_data())
                        audio_data.seek(0)
                        
                        # Save for playback
                        temp_path = "temp_voice.wav"
                        with open(temp_path, 'wb') as f:
                            f.write(audio_data.read())
                        
                        # Add to conversation
                        st.session_state.conversation.append(("You", "[Voice Message]"))
                        autoplay_audio(temp_path)
                        
                        # Process voice command
                        audio_data.seek(0)
                        with st.spinner("Processing..."):
                            response = st.session_state.system.process_voice_command(audio_data.read())
                            st.session_state.conversation.append(("Xara2west", response))
                except Exception as e:
                    st.error(f"Recording error: {e}")
    
    # Audio upload
    elif input_method == "Upload Audio":
        uploaded_file = st.file_uploader("Upload WAV audio", type="wav")
        if uploaded_file is not None:
            try:
                audio_data = uploaded_file.read()
                temp_path = "temp_upload.wav"
                with open(temp_path, 'wb') as f:
                    f.write(audio_data)
                
                st.session_state.conversation.append(("You", f"[Uploaded Audio]"))
                autoplay_audio(temp_path)
                with st.spinner("Processing..."):
                    response = st.session_state.system.process_voice_command(audio_data)
                    st.session_state.conversation.append(("Xara2west", response))
            except Exception as e:
                st.error(f"Audio processing error: {e}")
    
    # Display conversation
    with conversation_container:
        for speaker, message in st.session_state.conversation:
            if speaker == "You":
                st.markdown(f"**{speaker}:** {message}")
            else:
                st.markdown(f"*{speaker}:* {message}", unsafe_allow_html=True)
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.conversation = []
        st.session_state.history_ids = None
        st.experimental_rerun()

if __name__ == "__main__":
    main()
