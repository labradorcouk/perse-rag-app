import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import time
import traceback
import psutil

# --- Config ---
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
DATA_PATH = "downloads/epcNonDomesticScotlandQA.csv"
OUTPUT_DIR = "models/finetuned-deepseek-coder"
BATCH_SIZE = 4  # Reduced for Windows compatibility and to avoid memory issues
EPOCHS = 1      # 1 epoch is often enough for Q&A fine-tuning, much faster
LR = 2e-4
USE_4BIT = False  # No quantization on Windows, but GPU will be used
MAX_LENGTH = 512  # Reduced sequence length for faster training and lower memory

# --- Debug: Print resource and config info ---
print("\n[DEBUG] Starting Deepseek Coder fine-tuning script")
print(f"[DEBUG] Using model: {MODEL_NAME}")
print(f"[DEBUG] Data path: {DATA_PATH}")
print(f"[DEBUG] Output dir: {OUTPUT_DIR}")
print(f"[DEBUG] Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, Max length: {MAX_LENGTH}")
print(f"[DEBUG] Use 4bit: {USE_4BIT}")
print(f"[DEBUG] Num workers: 2 (set in TrainingArguments)")

# Print system info
print(f"[DEBUG] CPU count: {psutil.cpu_count(logical=True)}")
print(f"[DEBUG] RAM: {psutil.virtual_memory().total / 1e9:.2f} GB")
if torch.cuda.is_available():
    print(f"[DEBUG] CUDA available: True, GPU: {torch.cuda.get_device_name(0)}")
    print(f"[DEBUG] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("[DEBUG] CUDA available: False (using CPU)")

start_time = time.time()

# --- Load CSV Data ---
df = pd.read_csv(DATA_PATH)
print(f"[DEBUG] Loaded {len(df)} rows from CSV.")
# Use 'question' as prompt, 'answer' as response
assert 'question' in df.columns and 'answer' in df.columns, "CSV must have 'question' and 'answer' columns."

def format_example(row):
    # You can customize this prompt template!
    return f"<|user|>\n{row['question']}\n<|assistant|>\n{row['answer']}"

df['text'] = df.apply(format_example, axis=1)
dataset = Dataset.from_pandas(df[['text']])

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# --- Model Loading ---
model_load_start = time.time()
if USE_4BIT:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True
    )
print(f"[DEBUG] Model loaded in {time.time() - model_load_start:.2f} seconds.")

# --- LoRA Config ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# --- Tokenize Data ---
tokenize_start = time.time()
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,  # Lower max_length speeds up training
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
print(f"[DEBUG] Tokenization completed in {time.time() - tokenize_start:.2f} seconds.")
print(f"[DEBUG] Tokenized dataset sample:")
for i in range(min(2, len(tokenized_dataset))):
    print(tokenized_dataset[i])

# --- Output directory status ---
import os
if os.path.exists(OUTPUT_DIR):
    print(f"[DEBUG] Output directory {OUTPUT_DIR} exists and will be overwritten.")
else:
    print(f"[DEBUG] Output directory {OUTPUT_DIR} does not exist and will be created.")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --- Training Arguments ---
training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,  # Larger batch size = faster training
    gradient_accumulation_steps=2,           # Simulate larger effective batch size
    num_train_epochs=EPOCHS,                 # Fewer epochs = much faster
    learning_rate=LR,
    fp16=True, # Enable mixed-precision training for GPU
    output_dir=OUTPUT_DIR,
    save_total_limit=1,                      # Only keep last checkpoint to save disk and time
    logging_steps=100,                       # Log less frequently to reduce overhead
    save_steps=1000,                         # Save less frequently
    dataloader_num_workers=0,                # Set to 0 for Windows compatibility and to avoid multiprocessing issues
    overwrite_output_dir=True,               # Allow safe reruns
    report_to="none"
)
print(f"[DEBUG] dataloader_num_workers set to 0 for Windows compatibility.")

if __name__ == "__main__":
    print("[DEBUG] Starting training...")
    train_start = time.time()
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )

        trainer.train()
        print(f"[DEBUG] Training completed in {time.time() - train_start:.2f} seconds.")

        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Fine-tuned model saved to {OUTPUT_DIR}")
    except Exception as e:
        print("[ERROR] Exception during training:")
        traceback.print_exc()
        print("[ERROR] Training failed. See above for details.") 