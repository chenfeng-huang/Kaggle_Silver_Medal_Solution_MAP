## MAP — Charting Student Math Misunderstandings (Silver Medal) Solution

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.2+-orange.svg)
![Transformers](https://img.shields.io/badge/transformers-4.56.1-green.svg)

This repository contains a Silver Medal solution for the Kaggle competition [MAP — Charting Student Math Misunderstandings](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings). The task is to predict fine‑grained labels of the form `Category:Misconception` for each student response.

Our approach frames the problem as multi‑class sequence classification on top of instruction‑tuned LLM backbones (Qwen3). We engineer a prompt that includes the question text, the chosen MC answer, whether the choice is likely correct for that question, and the student’s explanation. We train with LoRA adapters (optionally with 4‑bit quantization) and perform inference‑time ensembling with a “family” prefix filter (`True_/False_`) and weighted consensus scoring.

![MAP — Silver Medal](./certificate.png)

### Competition Overview

- Predict `Category:Misconception` for each (QuestionText, MC_Answer, StudentExplanation).
- Official evaluation uses macro‑averaged F1 on the three predictions per row.

Key challenges include handling correctness vs. incorrectness consistently across questions and covering a large label space (6 categories × 36 misconceptions). More details in my [blog](https://chenfenghuang.info/2025/10/15/Kaggle-MAP/)

## Solution Overview

The pipeline has two main parts: single‑model classification and inference‑time ensembling.

1. Single‑Model Classification (Qwen3 backbones with LoRA)
   - Build a label space over all `Category:Misconception` combinations.
   - Infer per‑question “correct” MC option from train data by picking the most frequent `True_*` answer for each `QuestionId`.
   - Prompt template:
     - Question, chosen Answer, “Is Correct Answer” (Yes/No via the per‑question map), Student Explanation.
   - Train a sequence classification head with LoRA. Optionally enable 4‑bit quantization (BitsAndBytes) for efficiency.

2. Inference and Ensembling
   - Each base model produces top‑k (k=25) class probabilities per row and saves a CSV.
   - Merge multiple CSVs and compute a weighted score for each class:
     - Score = 0.34×(weighted total prob) + 0.33×(agreement ratio) + 0.33×(weighted max prob).
   - Apply a “family” prefix filter so that final classes match the row’s `True_/False_` prefix inferred from the question’s correct MC mapping.
   - Return the top 3 classes per row, padding with `Neither:NA` (and `Correct:NA` for `True_` rows) when needed.

## How to Reproduce

### Environment Setup

- Python 3.10+ recommended
- Install dependencies:

```bash
pip install -r requirement.txt
```


### Data Layout

Place competition CSVs under `data/map-charting-student-math-misunderstandings/`:

```
data/
└─ map-charting-student-math-misunderstandings/
   ├─ train.csv
   ├─ test.csv
   └─ sample_submission.csv
```

The scripts are also compatible with Kaggle’s default mount points (e.g., `/kaggle/input/...`). For local runs, adjust paths in the arguments or replicate the Kaggle directory layout as needed.

### Training

Run training (LoRA on Qwen3 backbones). See `train_run.sh` for presets.

```bash
python train.py \
  --model_name "Qwen/Qwen3-8B" \
  --use_bnb \
  --lora_r 64 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-4 \
  --gradient_accumulation_steps 1
```

Outputs (model + adapter) are saved under `./output/`.

### Inference

Each base model writes top‑k probabilities per row to a CSV. Example (Kaggle‑style paths; adjust for local):

```bash
torchrun --nproc_per_node=2 inference.py \
  --base_model "/kaggle/input/qwen-3/transformers/8b/1" \
  --lora_path "/kaggle/input/qwen3-8b-models-adaptors" \
  --output_filename "submission_qwen3_8b_prob"
```

To ensemble multiple probability CSVs and produce `submission.csv`, use the notebook `inference_script.ipynb`. It merges the CSVs, applies the “family” filter, computes weighted scores, and writes the final submission.

## Files and Structure

- `train.py`: Training pipeline with LoRA/quantization, prompt construction, tokenizer setup
- `inference.py`: Loads base model + adapters, formats prompts, outputs top‑k class probabilities per row
- `inference_script.ipynb`: Ensembling and submission creation with “family” filtering
- `train_run.sh`: Example launchers for common model sizes and settings
- `data/map-charting-student-math-misunderstandings/`: Competition CSVs
- `requirement.txt`: Python dependencies (skips bitsandbytes on macOS)

## Data Source and Citation

- See the competition page for rules and data access: [MAP — Charting Student Math Misunderstandings](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings)

```bibtex
@misc{map-charting-student-math-misunderstandings,
  title        = {MAP — Charting Student Math Misunderstandings},
  year         = {2025},
  howpublished = {\url{https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings}},
  note         = {Kaggle}
}
```

## Author

Maintainer: Chenfeng Huang — [Kaggle](https://www.kaggle.com/alrickh)

For questions, please open an issue or discussion in this repository.

## License

Distributed under the terms of the license in `LICENSE`.

