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