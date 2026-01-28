#!/usr/bin/env python3
"""
DPO Implementation Verification Script
=======================================

Run this script to verify all components are ready for training:
    python verify_dpo_setup.py

This will check:
1. GPU availability and memory
2. Required packages installed
3. Dataset integrity
4. File structure
5. Model loading capability
"""

import sys
import os
import json
import subprocess

def print_section(title):
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print('='*70)

def check_gpu():
    """Verify GPU availability"""
    print_section("1. GPU AVAILABILITY")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✓ CUDA available: {device_count} GPU(s)")
            
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {name} ({total_mem:.1f} GB)")
            
            # Check if we have enough memory
            if total_mem >= 24:
                print(f"✓ Sufficient VRAM for 4-bit training")
                return True
            else:
                print(f"⚠ Warning: {total_mem:.1f}GB might be tight for 4-bit training")
                return True
        else:
            print("✗ CUDA not available - cannot train")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def check_packages():
    """Verify required packages"""
    print_section("2. REQUIRED PACKAGES")
    
    required = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'datasets': 'Hugging Face Datasets',
        'trl': 'TRL (Trainer Reinforcement Learning)',
        'unsloth': 'Unsloth (Quantization)',
        'peft': 'PEFT (Parameter Efficient Fine-Tuning)',
    }
    
    all_installed = True
    for package, name in required.items():
        try:
            __import__(package)
            print(f"✓ {name:45} installed")
        except ImportError:
            print(f"✗ {name:45} NOT installed")
            print(f"  Install with: pip install {package}")
            all_installed = False
    
    return all_installed

def check_dataset():
    """Verify training dataset"""
    print_section("3. TRAINING DATASET")
    
    dataset_file = "contrastive_rec_train.jsonl"
    
    if not os.path.exists(dataset_file):
        print(f"✗ Dataset file not found: {dataset_file}")
        return False
    
    print(f"✓ Dataset file found: {dataset_file}")
    
    # Count lines
    try:
        with open(dataset_file, 'r') as f:
            lines = sum(1 for _ in f)
        print(f"✓ Dataset size: {lines:,} examples")
        
        if lines < 100:
            print(f"⚠ Warning: Only {lines} examples (might be too small)")
            return True
        elif lines < 1000:
            print(f"⚠ Note: {lines} examples (OK, but larger is better)")
            return True
        else:
            print(f"✓ Dataset size is good for training")
            return True
    except Exception as e:
        print(f"✗ Error reading dataset: {e}")
        return False

def check_sample_data():
    """Verify dataset format"""
    print_section("4. DATASET FORMAT VALIDATION")
    
    dataset_file = "contrastive_rec_train.jsonl"
    
    try:
        with open(dataset_file, 'r') as f:
            sample_line = f.readline()
        
        sample = json.loads(sample_line)
        required_fields = ['instruction', 'input', 'output']
        
        missing = [f for f in required_fields if f not in sample]
        if missing:
            print(f"✗ Missing fields in dataset: {missing}")
            return False
        
        print(f"✓ Required fields present: {required_fields}")
        
        # Check field content
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output = sample.get('output', '')
        
        has_preference = 'Option A' in output or 'Option B' in output
        if not has_preference:
            print(f"⚠ Warning: Sample output doesn't contain preference")
            print(f"  Output: {output[:100]}...")
        else:
            print(f"✓ Sample contains preference labels")
        
        print(f"\nSample preview:")
        print(f"  Instruction: {instruction[:60]}...")
        print(f"  Input: {input_text[:80]}...")
        print(f"  Output: {output[:80]}...")
        
        return True
    except Exception as e:
        print(f"✗ Error validating dataset: {e}")
        return False

def check_models():
    """Check if model can be loaded"""
    print_section("5. MODEL LOADING TEST")
    
    try:
        from unsloth import FastLanguageModel
        print("✓ Unsloth imported successfully")
        
        # Try loading base model (without training)
        print("\nAttempting to load base model (this will use GPU memory)...")
        print("  Model: unsloth/llama-3-8b-bnb-4bit")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/llama-3-8b-bnb-4bit",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        
        print("✓ Model loaded successfully in 4-bit")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Tokenizer type: {type(tokenizer).__name__}")
        
        # Apply LoRA
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
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        pct = (trainable / total) * 100 if total > 0 else 0
        
        print(f"✓ Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")
        
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        print(f"  This is needed for training")
        return False

def check_files():
    """Check if all implementation files exist"""
    print_section("6. IMPLEMENTATION FILES")
    
    files = {
        'train_dpo.py': 'DPO Training Script',
        'evaluate_dpo.py': 'Evaluation Framework',
        'dpo_comparison.ipynb': 'Interactive Notebook',
        'DPO_THESIS_DOCUMENTATION.md': 'Thesis Documentation',
        'DPO_QUICKSTART.md': 'Quick Start Guide',
        'DPO_HYPERPARAMETER_TUNING.md': 'Parameter Tuning Guide',
    }
    
    all_exist = True
    for filename, description in files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename) / 1024  # KB
            print(f"✓ {filename:40} ({size:.0f} KB)")
        else:
            print(f"✗ {filename:40} NOT FOUND")
            all_exist = False
    
    return all_exist

def check_directories():
    """Check output directories"""
    print_section("7. OUTPUT DIRECTORIES")
    
    dirs_to_check = [
        ('outputs', 'Checkpoint directory'),
        ('llama3.2-lora-final', 'SFT baseline model'),
    ]
    
    dirs_to_create = [
        ('llama3_dpo_4bit_final', 'DPO model output'),
        ('outputs/dpo_checkpoints', 'DPO training checkpoints'),
    ]
    
    print("Existing directories:")
    for dirname, desc in dirs_to_check:
        if os.path.isdir(dirname):
            items = len(os.listdir(dirname))
            print(f"✓ {dirname:40} ({items} items)")
        else:
            print(f"✗ {dirname:40} NOT FOUND")
    
    print("\nDirectories that will be created during training:")
    for dirname, desc in dirs_to_create:
        print(f"  → {dirname:40} ({desc})")
    
    return True

def summary_report(results):
    """Print summary report"""
    print_section("VERIFICATION SUMMARY")
    
    categories = [
        ("GPU Availability", results[0]),
        ("Required Packages", results[1]),
        ("Training Dataset", results[2]),
        ("Dataset Format", results[3]),
        ("Model Loading", results[4]),
        ("Implementation Files", results[5]),
    ]
    
    print()
    all_pass = True
    for category, passed in categories:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{category:35} {status}")
        if not passed:
            all_pass = False
    
    print("\n" + "="*70)
    
    if all_pass:
        print("✓ ALL CHECKS PASSED - READY TO TRAIN!")
        print("\nNext steps:")
        print("  1. Run: python train_dpo.py")
        print("  2. Wait 2-4 hours")
        print("  3. Run: python evaluate_dpo.py")
        print("  4. Check: dpo_comparison.ipynb for visualization")
    else:
        print("✗ SOME CHECKS FAILED - FIX ISSUES BEFORE TRAINING")
        print("\nIssues to resolve:")
        if not results[0]:
            print("  • GPU: Check nvidia-smi and torch installation")
        if not results[1]:
            print("  • Packages: Run: pip install torch transformers datasets trl unsloth peft")
        if not results[2]:
            print("  • Dataset: Ensure contrastive_rec_train.jsonl exists in current directory")
        if not results[4]:
            print("  • Model: Check internet connection, try: huggingface-cli login")
    
    print("="*70)
    
    return all_pass

def main():
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + "DPO IMPLEMENTATION VERIFICATION".center(68) + "║")
    print("║" + "Check all components before training".center(68) + "║")
    print("╚" + "="*68 + "╝")
    
    results = []
    
    try:
        results.append(check_gpu())
        results.append(check_packages())
        results.append(check_dataset())
        results.append(check_sample_data())
        results.append(check_models())
        results.append(check_files())
        check_directories()
        
        passed = summary_report(results)
        
        sys.exit(0 if passed else 1)
    
    except KeyboardInterrupt:
        print("\n\n✗ Verification interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
