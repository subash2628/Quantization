import torch
import gc
import json
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Set device
torch.cuda.set_device(0)

print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

print(f"Model loaded. Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

dataset = load_dataset('json', data_files='contrastive_rec_train.jsonl', split='train')

print(f"Dataset loaded. Number of examples: {len(dataset)}")

def formatting_func(examples):
    texts = []
    EOS_TOKEN = tokenizer.eos_token
    for instruction, input_text, output in zip(examples['instruction'], examples['input'], examples['output']):
        text = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}{EOS_TOKEN}"""
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_func, batched=True, remove_columns=dataset.column_names)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

print("\n" + "="*50)
print("Starting training on RTX 3090...")
print("="*50 + "\n")

trainer_stats = trainer.train()

print("\n" + "="*50)
print("Training Complete!")
print("="*50)
print(f"Training completed on: {torch.cuda.get_device_name(0)}")
print(f"Training loss: {trainer_stats.training_loss:.4f}")