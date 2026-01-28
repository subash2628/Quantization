"""
DPO (Direct Preference Optimization) Training Script
Thesis: "Does DPO provide better ranking adherence than SFT in 4-bit quantized recommendation scenarios?"

DPO directly optimizes the policy to align with human preferences without needing a separate reward model.
This is more parameter-efficient than RLHF and better at enforcing preference adherence.
"""

import torch
import json
import gc
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import DPOTrainer
from transformers import TrainingArguments
from datasets import load_dataset, Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device
torch.cuda.set_device(0)

print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

max_seq_length = 2048

# ============================================================================
# 1. LOAD BASE MODEL (4-bit quantized)
# ============================================================================

print("\n" + "="*70)
print("STEP 1: Loading 4-bit Quantized Llama 3 Model")
print("="*70)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

print(f"✓ Model loaded. Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

# ============================================================================
# 2. APPLY LORA ADAPTATION
# ============================================================================

print("\n" + "="*70)
print("STEP 2: Applying LoRA Adaptation (16 rank)")
print("="*70)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

print("✓ LoRA adapter applied")
print(f"✓ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ============================================================================
# 3. LOAD AND FORMAT DATASET FOR DPO
# ============================================================================

print("\n" + "="*70)
print("STEP 3: Loading Contrastive Dataset and Formatting for DPO")
print("="*70)

dataset = load_dataset('json', data_files='contrastive_rec_train.jsonl', split='train')
logger.info(f"Dataset loaded. Number of examples: {len(dataset)}")

def convert_to_dpo_format(examples):
    """
    Convert contrastive dataset to DPO format.
    
    DPO requires:
    - prompt: The instruction + question
    - chosen: The preferred response (Option A)
    - rejected: The non-preferred response (Option B)
    
    DPO Loss: 
    L = -log(σ(β * (log(π_θ(chosen|prompt) - log(π_θ(rejected|prompt))))
    
    This directly optimizes preference adherence by maximizing the ratio between
    chosen and rejected log probabilities.
    """
    
    prompts = []
    chosen_responses = []
    rejected_responses = []
    
    EOS_TOKEN = tokenizer.eos_token
    
    for instruction, input_text, output in zip(
        examples['instruction'], 
        examples['input'], 
        examples['output']
    ):
        # Create prompt (instruction + context)
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:"""
        
        prompts.append(prompt)
        
        # The output is always: "The better recommendation is Option A. Reasoning: ..."
        # We need to extract just the recommendation part for chosen
        chosen_response = output + EOS_TOKEN
        chosen_responses.append(chosen_response)
        
        # For rejected, we construct a counter-factual response
        # Simply saying "The better recommendation is Option B" when Option A is actually better
        # This teaches the model to distinguish correct from incorrect preferences
        if "Option A" in output:
            rejected_response = output.replace("Option A", "Option B").replace("Option B", "Option A") + EOS_TOKEN
        else:
            rejected_response = output + EOS_TOKEN
        
        rejected_responses.append(rejected_response)
    
    return {
        "prompt": prompts,
        "chosen": chosen_responses,
        "rejected": rejected_responses
    }

# Convert dataset
dpo_dataset = dataset.map(
    convert_to_dpo_format,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Converting to DPO format"
)

logger.info(f"✓ Dataset converted to DPO format: {len(dpo_dataset)} examples")
logger.info(f"  - Sample prompt length: {len(dpo_dataset[0]['prompt'].split())} words")
logger.info(f"  - Sample chosen length: {len(dpo_dataset[0]['chosen'].split())} words")
logger.info(f"  - Sample rejected length: {len(dpo_dataset[0]['rejected'].split())} words")

# ============================================================================
# 4. CONFIGURE DPO TRAINER
# ============================================================================

print("\n" + "="*70)
print("STEP 4: Configuring DPO Trainer")
print("="*70)

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=5,
    num_train_epochs=3,  # More epochs for DPO since it's more data-efficient
    learning_rate=5e-5,  # Lower LR for DPO (preference optimization is sensitive)
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs/dpo_checkpoints",
    report_to="wandb",  # Change to "none" if not using W&B
    run_name="llama3-dpo-4bit-quantized",
    save_strategy="steps",
    save_steps=100,
    eval_strategy="no",
    max_grad_norm=1.0,
)

print("✓ Training arguments configured:")
print(f"  - Learning rate: {training_args.learning_rate}")
print(f"  - Batch size: {training_args.per_device_train_batch_size}")
print(f"  - Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  - Epochs: {training_args.num_train_epochs}")
print(f"  - DPO Beta (temperature): 0.1 (default - controls preference sharpness)")

# ============================================================================
# 5. INITIALIZE DPO TRAINER
# ============================================================================

print("\n" + "="*70)
print("STEP 5: Initializing DPO Trainer")
print("="*70)

dpo_trainer = DPOTrainer(
    model=model,
    args=training_args,
    beta=0.1,  # Temperature parameter - controls how sharply we optimize preferences
               # Lower values = sharper preference optimization
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    max_prompt_length=512,  # Max length of prompt before truncation
    max_length=max_seq_length,  # Max total length (prompt + response)
)

logger.info("✓ DPO Trainer initialized")
logger.info("  - Beta (preference temperature): 0.1")
logger.info("  - Max prompt length: 512 tokens")
logger.info("  - Max total length: 2048 tokens")

# ============================================================================
# 6. START DPO TRAINING
# ============================================================================

print("\n" + "="*70)
print("STARTING DPO TRAINING - 4-bit Quantized Recommendation Model")
print("="*70)
print("Thesis: Direct preference optimization for better ranking adherence")
print("="*70 + "\n")

gc.collect()
torch.cuda.empty_cache()

trainer_stats = dpo_trainer.train()

print("\n" + "="*70)
print("DPO TRAINING COMPLETE!")
print("="*70)
print(f"✓ Trained on: {torch.cuda.get_device_name(0)}")
print(f"✓ Training loss: {trainer_stats.training_loss:.4f}")
print(f"✓ Final memory usage: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

# ============================================================================
# 7. SAVE TRAINED MODEL
# ============================================================================

print("\n" + "="*70)
print("STEP 7: Saving DPO-Trained Model")
print("="*70)

model.save_pretrained("llama3_dpo_4bit_final")
tokenizer.save_pretrained("llama3_dpo_4bit_final")

print("✓ DPO model saved to: llama3_dpo_4bit_final/")
print("✓ Model is compatible with FastLanguageModel.from_pretrained()")

print("\n" + "="*70)
print("NEXT STEPS: Evaluation Framework")
print("="*70)
print("""
To test the hypothesis:
"Does DPO provide better ranking adherence than SFT in 4-bit quantized scenarios?"

1. Load SFT baseline: llama3.2-lora-final/
2. Load DPO model: llama3_dpo_4bit_final/
3. Run inference on test set:
   - Compare ranking accuracy (how often model agrees with user preference)
   - Compare preference consistency (same user history → same recommendation)
   - Compare reasoning quality (does model justify choices correctly?)
   - Compare 4-bit quantization stability (reproducibility under low precision)

See: evaluate_dpo.py (next script to create)
""")
