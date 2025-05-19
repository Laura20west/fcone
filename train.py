
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
import torch
from sklearn.model_selection import train_test_split

# Load your CSV data
df = pd.read_csv("data.csv")  # Replace with your file path

# Clean and prepare text data
def preprocess_text(text):
    """Basic text cleaning"""
    if isinstance(text, str):
        # Remove excessive whitespace
        return " ".join(str(text).strip().split())
    return ""

# Apply preprocessing
text_column = "text"  # Change this to your CSV's text column name
df[text_column] = df[text_column].apply(preprocess_text)

# Split data
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Convert to HuggingFace Dataset
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df)
})

# ====================
# STEP 3: Tokenization
# ====================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

def tokenize_function(examples):
    return tokenizer(
        examples[text_column],
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=[col for col in df.columns if col != text_column]
)

# ====================
# STEP 4: Model Setup
# ====================
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ====================
# STEP 5: Training Configuration
# ====================
training_args = TrainingArguments(
    output_dir="./gpt2_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=5,  # Adjust based on your dataset size
    per_device_train_batch_size=4,  # Reduce if OOM errors occur
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = batch_size * accumulation_steps
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    fp16=torch.cuda.is_available(),  # Enable mixed precision training if GPU supports it
    report_to="none"  # Disable external logging
)

# ====================
# STEP 6: Data Collator
# ====================
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal language modeling
)

# ====================
# STEP 7: Training
# ====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

# Start training
print("Starting training...")
trainer.train()

# Save the final model
model.save_pretrained("./gpt2_finetuned_final")
tokenizer.save_pretrained("./gpt2_finetuned_final")

# ====================
# STEP 8: Inference
# ====================
def generate_text(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the model
print(generate_text("Your prompt here"))

