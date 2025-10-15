
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from datasets import Dataset
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BitsAndBytesConfig
import torch
from peft import PeftModel
from transformers import Trainer, TrainingArguments

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, default="/kaggle/input/qwen-3/transformers/8b/1")
parser.add_argument("--lora_path", type=str, default="/kaggle/input/qwen3-8b-models-adaptors")
parser.add_argument("--output_filename", type=str, default="submission_qwen3_8b_prob")
args = parser.parse_args()

# Build label encoder for 6*36 labels
le = LabelEncoder()
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category+":"+train.Misconception
train['label'] = le.fit_transform(train['target'])
target_classes = le.classes_
n_classes = len(target_classes)
print(f"Train shape: {train.shape} with {n_classes} target classes")

# Select samples with the correct answer
idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

# Prompt formatting function
def format_input(row):
    x = "Yes"
    if not row['is_correct']:
        x = "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Is Correct Answer: {x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

# Load tokenizer and model
base_model = args.base_model # Main Model
lora_path = args.lora_path # LoRa adaptors
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    base_model, 
    num_labels=n_classes,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(model, lora_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))
model = model.to(dtype=torch.float16)
model.eval()


# Load test file
test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)
test['text'] = test.apply(format_input, axis=1)

# Map to Dataset format
ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True)

# Predictor
args = TrainingArguments(
    output_dir="/kaggle/working/",
    per_device_eval_batch_size=4,
    dataloader_drop_last=False,
    report_to="none",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
)


# Run prediction
pred_output = trainer.predict(ds_test)
logits = pred_output.predictions
probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()

# ---- Select Top-25 prediction results ----
prob_data = []
top_k = 25
top_indices = np.argsort(-probs, axis=1) # shape: [num_samples, num_classes]

for i in range(len(logits)):
    prob_dict = {f"prob_{j}": probs[i, top_indices[i, j]] for j in range(top_k)} # select top_k probability values
    prob_dict["row_id"] = test.row_id.values[i] # add row_id
    prob_dict["top_classes"] = " ".join(le.inverse_transform(top_indices[i, :top_k])) # add top_k class labels
    prob_data.append(prob_dict) # append to list

# Save
prob_df = pd.DataFrame(prob_data) 
prob_df.to_csv(f"/kaggle/working/{args.output_filename}.csv", index=False)

# prob_df looks like:
'''
      row_id                                      top_classes    prob_0    prob_1    prob_2  ...   prob_22   prob_23   prob_24
0    Test_00001  True_A:NA True_A:Conceptual_Misunderstanding ...  0.345678  0.123456  0.098765  ...  0.001234  0.001123  0.001012
1    Test_00002  False_B:Calculation_Error False_B:NA True_A...  0.456789  0.234567  0.123456  ...  0.002345  0.002234  0.002123
2    Test_00003  True_C:NA True_C:Conceptual_Misunderstanding ...  0.567890  0.345678  0.234567  ...  0.003456  0.003345  0.003234
[...]
'''



'''
# qwen3-8b
torchrun --nproc_per_node=2 inference.py \
  --model_name "/kaggle/input/qwen-3/transformers/8b/1" \
  --lora_path "/kaggle/input/qwen3-8b-models-adaptors" \
  --output_filename "submission_qwen3_8b_prob"

# qwen3-14b
torchrun --nproc_per_node=2 inference.py \
    --model_name "/kaggle/input/qwen-3/transformers/14b/1" \
    --lora_path "/kaggle/input/qwen3-14b-models-adaptors" \
    --output_filename "submission_qwen3_14b_prob"

# qwen3-4b
torchrun --nproc_per_node=2 inference.py \
    --model_name "/kaggle/input/qwen-3-4b-instruct-2507" \
    --lora_path "/kaggle/input/qwen3-4b-models-adaptors" \
    --output_filename "submission_qwen3_4b_prob"
'''
