import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

# Initialize models and tokenizers
@st.cache_resource
def load_models():
    # Pidgin to English model
    pe_tokenizer = T5Tokenizer.from_pretrained("Xara2west/pidgin-to-english-translator-final09")
    pe_model = T5ForConditionalGeneration.from_pretrained("Xara2west/pidgin-to-english-translator-final09")
    
    # English to Pidgin model
    ep_tokenizer = T5Tokenizer.from_pretrained("Xara2west/pidgin-translator-final06")
    ep_model = T5ForConditionalGeneration.from_pretrained("Xara2west/pidgin-translator-final06")
    
    return {
        "pidgin_to_english": (pe_tokenizer, pe_model),
        "english_to_pidgin": (ep_tokenizer, ep_model)
    }

models = load_models()

# Grammar rules for Pidgin
def apply_pidgin_grammar_rules(text):
    # Common Pidgin corrections
    corrections = [
        (r'\bdey\b', 'dey'),  # Ensure correct spelling of "dey"
        (r'\buna\b', 'una'),  # Ensure correct spelling of "una"
        (r'\bwahala\b', 'wahala'),  # Ensure correct spelling
        (r'\bsabi\b', 'sabi'),  # Ensure correct spelling
        (r'\babi\b', 'abi'),  # Ensure correct spelling
        (r'\bna\b', 'na'),  # Ensure correct spelling
        (r'\boga\b', 'oga'),  # Ensure correct spelling
        (r'\bbiko\b', 'biko'),  # Ensure correct spelling
        (r'\behn\b', 'ehn'),  # Ensure correct spelling
        (r'\bchai\b', 'chai'),  # Ensure correct spelling
        (r'\be don happen\b', 'e don happen'),  # Common phrase correction
        (r'\bi dey go\b', 'I dey go'),  # Capitalize 'I'
        (r'\bno be small thing\b', 'no be small thing'),  # Common phrase
        (r'\bi beg\b', 'I beg'),  # Capitalize 'I'
        (r'\bmake una\b', 'make una'),  # Common phrase
        (r'\bhow you dey\b', 'how you dey'),  # Common greeting
    ]
    
    # Apply corrections
    for pattern, replacement in corrections:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Capitalize first letter of sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    text = ' '.join([sentence[0].upper() + sentence[1:] if sentence else '' for sentence in sentences])
    
    return text

# Grammar rules for English
def apply_english_grammar_rules(text):
    # Common English corrections
    corrections = [
        (r'\bi\b', 'I'),  # Capitalize 'I'
        (r'\bim\b', 'him'),  # Common pidgin influence
        (r'\buna\b', 'you all'),  # Translate 'una'
        (r'\bdey\b', 'is/are'),  # Translate 'dey'
        (r'\bno be\b', "it's not"),  # Common phrase
        (r'\bwahala\b', 'trouble'),  # Translate 'wahala'
        (r'\bsabi\b', 'know'),  # Translate 'sabi'
        (r'\behn\b', 'right'),  # Translate 'ehn'
        (r'\bbiko\b', 'please'),  # Translate 'biko'
    ]
    
    # Apply corrections
    for pattern, replacement in corrections:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Capitalize first letter of sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    text = ' '.join([sentence[0].upper() + sentence[1:] if sentence else '' for sentence in sentences])
    
    # Ensure proper punctuation
    text = re.sub(r'\s+([,.!?])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([,.!?])([^\s])', r'\1 \2', text)  # Add space after punctuation
    
    return text

def translate(text, direction):
    if direction == "Pidgin to English":
        # Apply Pidgin grammar rules first
        corrected_text = apply_pidgin_grammar_rules(text)
        tokenizer, model = models["pidgin_to_english"]
        prefix = "translate Pidgin to English: "
    else:
        # Apply English grammar rules first
        corrected_text = apply_english_grammar_rules(text)
        tokenizer, model = models["english_to_pidgin"]
        prefix = "translate English to Pidgin: "
    
    # Encode with task-specific prefix
    inputs = tokenizer(prefix + corrected_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate translation
    outputs = model.generate(
        inputs.input_ids,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.title("🇳🇬 Grammar-Guided Pidgin-English Translator")
st.caption("Powered by Hugging Face Transformers • Built-in Grammar Rules")

# Translation direction
direction = st.radio(
    "Select translation direction:",
    ("Pidgin to English", "English to Pidgin"),
    horizontal=True
)

# Input text
text = st.text_area("Enter text to translate:", height=150)

if st.button("Translate"):
    if text.strip():
        with st.spinner("Applying grammar rules and translating..."):
            result = translate(text, direction)
            st.subheader("Translation Result:")
            st.success(result)
            
            # Show grammar correction details
            with st.expander("Grammar Processing Details"):
                if direction == "Pidgin to English":
                    corrected_input = apply_pidgin_grammar_rules(text)
                    st.write("**Original Pidgin:**", text)
                    st.write("**After Grammar Rules:**", corrected_input)
                else:
                    corrected_input = apply_english_grammar_rules(text)
                    st.write("**Original English:**", text)
                    st.write("**After Grammar Rules:**", corrected_input)
    else:
        st.warning("Please enter text to translate")

st.markdown("---")
st.info("**Model Details:**\n"
        "- Pidgin → English: [Xara2west/pidgin-to-english-translator-final09](https://huggingface.co/Xara2west/pidgin-to-english-translator-final09)\n"
        "- English → Pidgin: [Xara2west/pidgin-translator-final06](https://huggingface.co/Xara2west/pidgin-translator-final06)\n\n"
        "Grammar rules are implemented directly in the application")
