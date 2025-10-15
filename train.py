import argparse
import os
import pandas as pd
import numpy as np

# ----------------------------
# Argparse
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=str, default="0,1",
                    help='CUDA devices, e.g. "0" or "0,1" or "0,1,2,3"')

# Model name
parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help='Model name')

# Enable / disable bnb_config (4-bit quantization)
group = parser.add_mutually_exclusive_group()
group.add_argument("--use_bnb", dest="use_bnb", action="store_true",
                   help="Enable 4-bit BitsAndBytes quantization.")
group.add_argument("--no_bnb", dest="use_bnb", action="store_false",
                   help="Disable 4-bit BitsAndBytes quantization.")
parser.set_defaults(use_bnb=True)

# LoRA hyperparams
parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank r.")
parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")

# Training hyperparams
parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs.")
parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Per-device train batch size.")
parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
TEMP_DIR = "./output"
os.makedirs(TEMP_DIR, exist_ok=True)

# ----------------------------
# Data loading and preprocessing
# ----------------------------
train = pd.read_csv('./input/map-charting-student-math-misunderstandings/train.csv')

# Fill missing Misconception with 'NA' (consistent with submission format)
train.Misconception = train.Misconception.fillna('NA')

# Compose target label: Category:Misconception (as required by submission)
train['target'] = train.Category + ":" + train.Misconception

# Encode string labels to integer class ids (for supervised training)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_) # number of unique label classes (num_labels)
print(f"Train shape: {train.shape} with {n_classes} target classes")
print("Train head:")
print(train.head())

# ----------------------------
# Data cleaning: some questions have multiple correct options; keep the most frequent correct option as the 'ground truth'
# ----------------------------
# Split Category by '_' and treat prefix 'True' as correct
idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
# Within the True subset, count (QuestionId, MC_Answer) and choose the most frequent correct MC_Answer per question
correct = train.loc[idx].copy() # keep only the 'correct' subset
correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count') # count occurrences of each MC_Answer per question
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId', 'MC_Answer']] 
correct['is_correct'] = 1 # mark these (QuestionId, MC_Answer) as correct options

# Merge is_correct back into training set; treat missing as 0 (incorrect)
train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)


# ----------------------------
# Model loading and configuration
# ----------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

model_name = args.model_name

# BitsAndBytes 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

from_pretrained_kwargs = dict(
    num_labels=n_classes, # number of classes for classification head
    device_map="auto",
    trust_remote_code=True
)
# Inject quantization config when 4-bit is enabled
if args.use_bnb:
    from_pretrained_kwargs["quantization_config"] = bnb_config

# Load classification model (adds classification head at final layer)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    **from_pretrained_kwargs
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Configure LoRA
lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="SEQ_CLS",
    modules_to_save=["score"]  # keep extra modules (if used internally)
)

# Prepare model for k-bit friendly training
model = prepare_model_for_kbit_training(model)
# Inject LoRA adapter
model = get_peft_model(model, lora_config)

model = model.to(dtype=torch.bfloat16)

# Explicitly add a [PAD] token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# Resize model embeddings after tokenizer vocab change
model.resize_token_embeddings(len(tokenizer))
# Set model pad_token_id to ensure correct batch padding
model.config.pad_token_id = tokenizer.pad_token_id

print(next(model.parameters()).dtype)


# ----------------------------
# Construct training input text
# ----------------------------
def format_input_v2(row):
    """
    Construct a single training input text:
    include the question text, student choice, whether the choice is correct, and the student's explanation.
    This text will be truncated to max_length by the tokenizer.
    """
    x = "Yes"
    if not row['is_correct']:
        x = "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Is Correct Answer: {x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

# Build training text column using the formatter above
train['text'] = train.apply(format_input_v2, axis=1)
print("\nExample prompt for our LLM (after refinement):")
print(train.text.values[0])

from datasets import Dataset


# Build a clean DataFrame (avoid extra columns interference)
train_df_clean = train[['text', 'label']].copy()
train_df_clean['label'] = train_df_clean['label'].astype(np.int64)
train_df_clean = train_df_clean.reset_index(drop=True)

# Full training set
train_ds = Dataset.from_pandas(train_df_clean, preserve_index=False)

def tokenize(batch):
    """Batch tokenize text and truncate to 256"""
    return tokenizer(batch["text"], truncation=True, max_length=256)

# Map tokenization; remove original 'text' column, keep tokenized features and 'label'
train_ds = train_ds.map(tokenize, batched=True, remove_columns=['text'])


# ----------------------------
# Training
# ----------------------------
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

# Ensure log and output directories exist
os.makedirs(f"{TEMP_DIR}/training_output/", exist_ok=True)
os.makedirs(f"{TEMP_DIR}/logs/", exist_ok=True)

training_args = TrainingArguments(
    output_dir=f"{TEMP_DIR}/training_output/",  # output directory
    do_train=True,                               # train only, no evaluation
    do_eval=False,
    save_strategy="no",                          # do not save checkpoints
    num_train_epochs=args.num_train_epochs,      # epochs
    per_device_train_batch_size=args.per_device_train_batch_size,  # batch size
    learning_rate=args.learning_rate,            # learning rate
    logging_dir=f"{TEMP_DIR}/logs/",             # log directory
    logging_steps=100,                           # logging frequency (steps)
    gradient_accumulation_steps=args.gradient_accumulation_steps,  # gradient accumulation
    remove_unused_columns=False,                 # keep all dataset columns (for custom processing)
    bf16=True,                                   # use bfloat16
    fp16=False,                                  # do not use float16
    report_to="none",                            # do not report to external tools
    warmup_ratio=0.1,                            # warmup ratio
    lr_scheduler_type="cosine",                  # cosine LR schedule
    dataloader_drop_last=True,                   # drop the last incomplete batch
    dataloader_pin_memory=False,                 # disable pin_memory (tune if needed)
    gradient_checkpointing=True,                 # enable gradient checkpointing to save VRAM
)

# Use dynamic padding collator to align batches
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()

print(f"Save model")
trainer.save_model(TEMP_DIR)


'''
#  https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
torchrun --nproc_per_node=2 train.py \
  --cuda "0,1" \
  --model_name "Qwen/Qwen3-4B-Instruct-2507" \
  --no_bnb \
  --lora_r 512 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-4 \
  --gradient_accumulation_steps 2


# https://huggingface.co/Qwen/Qwen3-8B
torchrun --nproc_per_node=2 train.py \
  --cuda "0,1" \
  --model_name "Qwen/Qwen3-8B" \
  --use_bnb \
  --lora_r 64 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-4 \
  --gradient_accumulation_steps 1



# https://huggingface.co/Qwen/Qwen3-14B
torchrun --nproc_per_node=4 train.py \
  --cuda "0,1,2,3" \
  --model_name "Qwen/Qwen3-14B" \
  --use_bnb \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-4 \
  --gradient_accumulation_steps 2
'''