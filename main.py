import os
import io
import base64
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import numpy as np
from pathlib import Path
import soundfile as sf
import sympy
from sympy import symbols, solve, integrate, diff, limit, Eq, Derivative
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                       implicit_multiplication_application)
import re
import math
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = io.BytesIO()

# WebRTC configuration
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "audio": True,
        "video": False
    },
)

class VoiceAuthenticator:
    def __init__(self, reference_voice_path="laura.wav", threshold=0.65):
        self.encoder = VoiceEncoder()
        self.threshold = threshold
        self.reference_embed = self._get_embedding(reference_voice_path)

    def _get_embedding(self, audio_path):
        wav = preprocess_wav(audio_path)
        return self.encoder.embed_utterance(wav)

    def is_authorized_voice(self, audio_data):
        try:
            temp_path = "temp_voice_check.wav"
            with open(temp_path, 'wb') as f:
                f.write(audio_data)

            test_embed = self._get_embedding(temp_path)
            similarity = np.dot(self.reference_embed, test_embed)
            st.session_state.conversation.append(("System", f"Voice similarity score: {similarity:.3f}"))

            os.remove(temp_path)
            return similarity > self.threshold
        except Exception as e:
            st.session_state.conversation.append(("System", f"Voice authentication error: {str(e)}"))
            return False

class AdvancedMathSolver:
    def __init__(self):
        self.transformations = (standard_transformations +
                              (implicit_multiplication_application,))
        self.x, self.y, self.z = symbols('x y z')
        self.recognized_commands = {
            'derivative': self._solve_derivative,
            'integral': self._solve_integral,
            'limit': self._solve_limit,
            'area': self._solve_area,
            'volume': self._solve_volume,
            'mean': self._solve_mean,
            'median': self._solve_median,
            'determinant': self._solve_determinant,
            'percentage': self._solve_percentage
        }

    def _extract_expression(self, text, keyword):
        patterns = [
            fr"{keyword}\s*(?:of|for)?\s*(.+?)\s*(?:with|when|where|if|$)",
            fr"{keyword}\s*(.+?)\s*$"
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _solve_basic(self, expr_str):
        try:
            if '%' in expr_str:
                parts = expr_str.split('%')
                if len(parts) == 2 and parts[1].strip() == '':
                    value = parse_expr(parts[0], transformations=self.transformations)
                    return float(value) / 100
                elif 'of' in expr_str.lower():
                    parts = expr_str.lower().split('% of')
                    if len(parts) == 2:
                        percent = parse_expr(parts[0], transformations=self.transformations)
                        total = parse_expr(parts[1], transformations=self.transformations)
                        return (float(percent) / 100) * float(total)

            expr_str = expr_str.replace('÷', '/').replace(' over ', '/')
            expr_str = expr_str.replace('^', '**').replace(' squared', '**2').replace(' cubed', '**3')
            expr_str = re.sub(r'square root of (.+)', r'sqrt(\1)', expr_str, flags=re.IGNORECASE)

            expr = parse_expr(expr_str, transformations=self.transformations)
            result = float(expr.evalf())
            return result
        except Exception as e:
            st.session_state.conversation.append(("System", f"Basic solve error: {e}"))
            return None

    def _solve_equation(self, equation_str):
        try:
            equation_str = equation_str.replace('=', '==')

            if 'solve for' in equation_str.lower():
                parts = equation_str.lower().split('solve for')
                var = parts[1].strip()
                equation = parts[0].strip()
                symbol = symbols(var)
                solutions = solve(equation, symbol)
                return solutions

            if '==' in equation_str:
                parts = equation_str.split('==')
                lhs = parse_expr(parts[0], transformations=self.transformations)
                rhs = parse_expr(parts[1], transformations=self.transformations)
                solutions = solve(Eq(lhs, rhs), self.x)
                return solutions

            return None
        except Exception as e:
            st.session_state.conversation.append(("System", f"Equation solve error: {e}"))
            return None

    def _solve_derivative(self, problem):
        try:
            expr_str = self._extract_expression(problem, 'derivative') or problem
            expr_str = expr_str.replace('dy/dx', 'Derivative(y, x)').replace('dy', 'Derivative(y, x)')
            expr = parse_expr(expr_str, transformations=self.transformations)
            return diff(expr, self.x)
        except Exception as e:
            st.session_state.conversation.append(("System", f"Derivative solve error: {e}"))
            return None

    def _solve_integral(self, problem):
        try:
            expr_str = self._extract_expression(problem, 'integral') or problem
            expr = parse_expr(expr_str, transformations=self.transformations)
            return integrate(expr, self.x)
        except Exception as e:
            st.session_state.conversation.append(("System", f"Integral solve error: {e}"))
            return None

    def _solve_limit(self, problem):
        try:
            expr_str = self._extract_expression(problem, 'limit') or problem
            var_str = re.search(r'as (.+?) approaches', problem, re.IGNORECASE)
            val_str = re.search(r'approaches (.+)', problem, re.IGNORECASE)

            if var_str and val_str:
                var = parse_expr(var_str.group(1), transformations=self.transformations)
                val = parse_expr(val_str.group(1), transformations=self.transformations)
                expr = parse_expr(expr_str, transformations=self.transformations)
                return limit(expr, var, val)
            return None
        except Exception as e:
            st.session_state.conversation.append(("System", f"Limit solve error: {e}"))
            return None

    def _solve_area(self, problem):
        try:
            if 'circle' in problem.lower():
                radius = re.search(r'radius\s*([0-9.]+)', problem, re.IGNORECASE)
                if radius:
                    r = float(radius.group(1))
                    return math.pi * r**2

            elif 'triangle' in problem.lower():
                base = re.search(r'base\s*([0-9.]+)', problem, re.IGNORECASE)
                height = re.search(r'height\s*([0-9.]+)', problem, re.IGNORECASE)
                if base and height:
                    b = float(base.group(1))
                    h = float(height.group(1))
                    return 0.5 * b * h

            elif 'rectangle' in problem.lower():
                length = re.search(r'length\s*([0-9.]+)', problem, re.IGNORECASE)
                width = re.search(r'width\s*([0-9.]+)', problem, re.IGNORECASE)
                if length and width:
                    l = float(length.group(1))
                    w = float(width.group(1))
                    return l * w

            return None
        except Exception as e:
            st.session_state.conversation.append(("System", f"Area solve error: {e}"))
            return None

    def _solve_volume(self, problem):
        try:
            if 'sphere' in problem.lower():
                radius = re.search(r'radius\s*([0-9.]+)', problem, re.IGNORECASE)
                if radius:
                    r = float(radius.group(1))
                    return (4/3) * math.pi * r**3

            elif 'cylinder' in problem.lower():
                radius = re.search(r'radius\s*([0-9.]+)', problem, re.IGNORECASE)
                height = re.search(r'height\s*([0-9.]+)', problem, re.IGNORECASE)
                if radius and height:
                    r = float(radius.group(1))
                    h = float(height.group(1))
                    return math.pi * r**2 * h

            elif 'cube' in problem.lower():
                side = re.search(r'side\s*([0-9.]+)', problem, re.IGNORECASE)
                if side:
                    s = float(side.group(1))
                    return s**3

            return None
        except Exception as e:
            st.session_state.conversation.append(("System", f"Volume solve error: {e}"))
            return None

    def _solve_mean(self, problem):
        try:
            nums = re.findall(r'([0-9.]+)', problem)
            if nums:
                numbers = [float(n) for n in nums]
                return np.mean(numbers)
            return None
        except Exception as e:
            st.session_state.conversation.append(("System", f"Mean solve error: {e}"))
            return None

    def _solve_median(self, problem):
        try:
            nums = re.findall(r'([0-9.]+)', problem)
            if nums:
                numbers = [float(n) for n in nums]
                return np.median(numbers)
            return None
        except Exception as e:
            st.session_state.conversation.append(("System", f"Median solve error: {e}"))
            return None

    def _solve_determinant(self, problem):
        try:
            matrix_str = re.search(r'\[(.+)\]', problem)
            if matrix_str:
                rows = matrix_str.group(1).split(';')
                matrix = []
                for row in rows:
                    elements = row.split(',')
                    matrix.append([float(e.strip()) for e in elements])
                mat = sympy.Matrix(matrix)
                return mat.det()
            return None
        except Exception as e:
            st.session_state.conversation.append(("System", f"Determinant solve error: {e}"))
            return None

    def _solve_percentage(self, problem):
        try:
            if '% of' in problem.lower():
                parts = problem.lower().split('% of')
                percent = float(parts[0].strip())
                total = float(parts[1].strip())
                return (percent / 100) * total
            elif 'what is' in problem.lower() and '%' in problem.lower():
                nums = re.findall(r'([0-9.]+)', problem)
                if len(nums) == 2:
                    percent = float(nums[0])
                    total = float(nums[1])
                    return (percent / 100) * total
            return None
        except Exception as e:
            st.session_state.conversation.append(("System", f"Percentage solve error: {e}"))
            return None

    def solve(self, problem_text):
        problem_text = problem_text.lower().strip()

        for cmd, handler in self.recognized_commands.items():
            if cmd in problem_text:
                result = handler(problem_text)
                if result is not None:
                    return result

        if '=' in problem_text or 'solve for' in problem_text:
            result = self._solve_equation(problem_text)
            if result is not None:
                return result

        result = self._solve_basic(problem_text)
        if result is not None:
            return result

        return None

class Xara2westSystem:
    def __init__(self):
        self.voice_auth = VoiceAuthenticator()
        self.math_solver = AdvancedMathSolver()
        self.model_name = "Xara2west/Xstage2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_response(self, input_text):
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs, max_length=200, num_beams=5, early_stopping=True)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._apply_grammar_rules(response)

    def _apply_grammar_rules(self, text):
        if any(word in text.lower() for word in ['answer is', 'equals', 'solution', 'result']):
            text = f"Babe, {text.lower().capitalize()} babe! good job math queen! 💡"
        else:
            text = text.replace(" the ", " thee ").replace(" a ", " an ")
            text = text.capitalize()
            if not text.startswith(("Babe,", "Darling,")):
                text = f"Babe, {text} 🥺💕"
        return text

    def solve_math_problem(self, problem_text):
        clean_text = problem_text.lower()
        for phrase in ["what is", "calculate", "solve for", "find the", "compute", "evaluate", "please"]:
            clean_text = clean_text.replace(phrase, "")
        clean_text = clean_text.strip()

        solution = self.math_solver.solve(clean_text)

        if solution is not None:
            if isinstance(solution, list):
                if len(solution) == 1:
                    solution = solution[0]
                else:
                    solution = ", ".join(str(s) for s in solution)
            return f"Thee solution is: {solution}"
        else:
            return "I couldn't solve that math problem, babe. Can you try rephrasing it?"

    def process_query(self, user_input):
        math_keywords = [
            'calculate', 'solve', 'math', 'equation', 'formula',
            'derivative', 'integral', 'limit', 'matrix', 'determinant',
            'area', 'volume', 'mean', 'median', 'standard deviation',
            '+', '-', '*', '/', '^', 'plus', 'minus', 'times', 'divided by',
            'add', 'subtract', 'multiply', 'divide', 'power', 'root',
            'what is', 'how much is', '%', 'percent', 'percentage'
        ]

        if any(keyword in user_input.lower() for keyword in math_keywords):
            return self.solve_math_problem(user_input)
        return self.generate_response(user_input)

    def process_voice_command(self, audio_data):
        if self.voice_auth.is_authorized_voice(audio_data):
            text_input = self._speech_to_text(audio_data)

            if text_input:
                st.session_state.conversation.append(("System", f"Recognized command: {text_input}"))
                response = self.process_query(text_input)

                if text_input.lower().startswith("execute "):
                    command = text_input[8:]
                    try:
                        output = os.popen(command).read()
                        return f"Command executed:\n{output}"
                    except Exception as e:
                        return f"Failed to execute command: {str(e)}"

                return response
            else:
                return "Could not understand voice command."
        else:
            return "❌ Voice not recognized. Command ignored."

    def _speech_to_text(self, audio_data):
        try:
            temp_path = "temp_voice_command.wav"
            with open(temp_path, 'wb') as f:
                f.write(audio_data)

            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_path) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)

            os.remove(temp_path)
            return text
        except sr.UnknownValueError:
            st.session_state.conversation.append(("System", "Google Speech Recognition could not understand audio"))
            return None
        except sr.RequestError as e:
            st.session_state.conversation.append(("System", f"Could not request results from Google Speech Recognition service; {e}"))
            return None
        except Exception as e:
            st.session_state.conversation.append(("System", f"Speech recognition error: {e}"))
            return None

def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
    if st.session_state.audio_buffer:
        st.session_state.audio_buffer.write(frame.to_ndarray().tobytes())
    return frame

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )

def main():
    st.title("Xara2west Voice-Authenticated Math Assistant")
    st.markdown("Speak or type your math problems and get personalized help!")

    # Initialize system
    if 'system' not in st.session_state:
        st.session_state.system = Xara2westSystem()

    # Conversation display
    st.subheader("Conversation")
    conversation_container = st.container()
    
    # Input methods
    st.subheader("Input Method")
    input_method = st.radio("Choose input method:", ("Text", "Microphone", "Upload Audio"))

    if input_method == "Text":
        user_input = st.text_input("Type your message here:")
        if st.button("Send") and user_input:
            st.session_state.conversation.append(("You", user_input))
            response = st.session_state.system.process_query(user_input)
            st.session_state.conversation.append(("Xara2west", response))

    elif input_method == "Microphone":
        st.write("Click 'Start Recording' and speak your math problem:")
        
        webrtc_ctx = webrtc_streamer(
            key="voice-auth",
            mode=WebRtcMode.SENDONLY,
            client_settings=WEBRTC_CLIENT_SETTINGS,
            audio_frame_callback=audio_frame_callback,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"audio": True},
        )

        if st.button("Process Recording") and st.session_state.audio_buffer:
            # Save audio to file for playback
            temp_path = "temp_voice_input.wav"
            with open(temp_path, 'wb') as f:
                f.write(st.session_state.audio_buffer.getvalue())
            
            # Add to conversation
            st.session_state.conversation.append(("You", "[Voice Message]"))
            autoplay_audio(temp_path)
            
            # Process voice command
            response = st.session_state.system.process_voice_command(st.session_state.audio_buffer.getvalue())
            st.session_state.conversation.append(("Xara2west", response))
            
            # Reset buffer
            st.session_state.audio_buffer = io.BytesIO()

    elif input_method == "Upload Audio":
        uploaded_file = st.file_uploader("Choose a WAV audio file", type="wav")
        if uploaded_file is not None:
            audio_data = uploaded_file.read()
            
            # Save to temp file for playback
            temp_path = "temp_uploaded_audio.wav"
            with open(temp_path, 'wb') as f:
                f.write(audio_data)
            
            # Add to conversation
            st.session_state.conversation.append(("You", f"[Uploaded Audio: {uploaded_file.name}]"))
            
