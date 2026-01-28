"""
DPO Hyperparameter Tuning Guide
================================================================================

This guide helps you adjust DPO hyperparameters based on your results and needs.

Key Parameters:
1. beta - Temperature for preference optimization (most important)
2. learning_rate - Step size for gradient updates
3. num_train_epochs - Number of training passes
4. max_prompt_length - Truncation for prompt
5. gradient_accumulation_steps - Effective batch size multiplier
"""

# ============================================================================
# PARAMETER REFERENCE
# ============================================================================

"""
BETA PARAMETER
==============
Controls how sharply we optimize preferences.

Formula: 
  loss = -log(sigmoid(beta * (log_chosen - log_rejected)))

LOW BETA (0.05):
  ✓ Softer preference optimization
  ✓ More stable training
  ✓ Better for small datasets
  ✗ Less preference adherence
  ✗ Lower accuracy

MEDIUM BETA (0.1 - current default):
  ✓ Good balance
  ✓ Recommended starting point
  ✓ Works well with quantization
  ✓ Clear preference signal

HIGH BETA (0.2, 0.5):
  ✓ Sharp preference optimization
  ✓ Higher accuracy on train set
  ✗ Might overfit
  ✗ Training instability
  ✗ Worse generalization

TUNING STRATEGY:
  1. Start with 0.1 (default)
  2. If loss doesn't converge → decrease to 0.05
  3. If loss oscillates → decrease to 0.05
  4. If train accuracy plateaus → try 0.2
  5. Track validation accuracy for best beta
"""

# ============================================================================

"""
LEARNING RATE
==============
Controls step size for parameter updates.

TYPICAL VALUES:
  5e-5  ← DEFAULT (recommended)
  1e-5  ← Very conservative
  2e-5  ← Moderate
  1e-4  ← Aggressive
  5e-4  ← Very aggressive

For DPO (preference optimization):
  - Use LOWER than SFT (5e-5 vs 2e-4)
  - Preference optimization is sensitive
  - Too high → divergence or oscillation
  - Too low → very slow convergence

4-BIT QUANTIZATION NOTE:
  - Quantization adds noise
  - Use slightly lower LR (5e-5) to be safe
  - Monitor loss - should be smooth downward
  - If erratic, reduce by 2x (2.5e-5)

TUNING STRATEGY:
  If loss diverges → decrease LR by 2x
  If loss converges too slowly → increase LR by 2x
  If oscillating → decrease LR
"""

# ============================================================================

"""
NUM_TRAIN_EPOCHS
================
How many times to iterate through the dataset.

DPO-SPECIFIC:
  - DPO is more data-efficient than SFT
  - Can use more epochs with same data
  - Each epoch provides new preference signal

RECOMMENDED:
  3 epochs ← DEFAULT (good for 6,040 examples)
  1-2 epochs ← If running low on time
  5+ epochs ← Only if loss still decreasing after 3 epochs

CONVERGENCE OBSERVATION:
  Epoch 1: Steep loss decrease (learning preferences)
  Epoch 2: Continued decrease (refinement)
  Epoch 3: Slow decrease (diminishing returns)
  Epoch 4+: Often plateaus or diverges

DECISION RULE:
  1. Monitor loss across epochs
  2. If still significantly decreasing at epoch 3 → use 5 epochs
  3. If plateaued by epoch 2 → use 2 epochs
  4. Default 3 epochs usually optimal
"""

# ============================================================================

"""
BATCH SIZE & GRADIENT ACCUMULATION
===================================
Effective batch size = per_device_batch_size × gradient_accumulation_steps

Current:
  per_device_train_batch_size = 1
  gradient_accumulation_steps = 8
  Effective batch size = 1 × 8 = 8

Memory vs Performance Tradeoff:
  - Larger batch → better gradient estimates → but more memory
  - 4-bit quantization already memory-constrained
  - Current setup (8 accumulation) is good balance

ADJUSTING:
  More memory available?
    per_device_batch_size = 2, gradient_accumulation = 4 (total 8)
    → Better gradient estimates
    → Slightly better convergence

  Less memory available?
    per_device_batch_size = 1, gradient_accumulation = 4 (total 4)
    → Smaller effective batch
    → Slightly noisier gradients
    → Might need more epochs

Don't go below effective batch size 4 for DPO.
"""

# ============================================================================

"""
WARMUP STEPS
============
How many steps before full learning rate.

Current: 5 steps

With 6,040 examples, batch size 8:
  Total steps per epoch = 6,040 / 8 = 755 steps
  Warmup 5 steps = 0.66% of epoch 1

This is fine for DPO. Purpose:
  - Stabilize initial training
  - Prevent gradient spikes from bad initialization
  - Not critical for preference optimization

ADJUST IF:
  - Training unstable first few steps → increase to 10-20
  - Training seems fine → can keep at 5
  - Large dataset → can increase proportionally
"""

# ============================================================================

"""
MAX_PROMPT_LENGTH
=================
Truncate prompts to this length.

Current: 512 tokens

Your prompts typically contain:
  - Instruction: ~50 tokens
  - Input (history + options): ~100-150 tokens
  - Total: ~150-200 tokens

512 is safe and allows room for long contexts.

TRADEOFF:
  Higher max_prompt_length:
    ✓ Capture full user history
    ✓ More context for model
    ✗ Longer sequences → more memory
    ✗ Slower training
    ✗ Harder for model to focus on key info

  Lower max_prompt_length:
    ✓ Faster training
    ✓ Less memory
    ✗ Might truncate important context
    ✗ Worse recommendations

RECOMMENDATION:
  Keep at 512 for your use case.
  Your prompts are naturally short (~150-200 tokens).
  Provides plenty of buffer.
"""

# ============================================================================

# EXAMPLE CONFIGURATIONS
# ============================================================================

CONFIGS = {
    "AGGRESSIVE_CONVERGENCE": {
        "description": "Push for maximum accuracy",
        "beta": 0.2,
        "learning_rate": 1e-4,
        "num_train_epochs": 5,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 10,
        "risks": "Might overfit, instability",
    },
    
    "BALANCED_DEFAULT": {
        "description": "Recommended - good accuracy with stability",
        "beta": 0.1,
        "learning_rate": 5e-5,
        "num_train_epochs": 3,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 5,
        "risks": "None - this is the recommended config",
    },
    
    "CONSERVATIVE_STABLE": {
        "description": "Prioritize training stability",
        "beta": 0.05,
        "learning_rate": 2e-5,
        "num_train_epochs": 5,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 20,
        "risks": "Slower convergence, lower final accuracy",
    },
    
    "QUANTIZATION_OPTIMIZED": {
        "description": "Tuned specifically for 4-bit quantization noise",
        "beta": 0.1,
        "learning_rate": 3e-5,
        "num_train_epochs": 4,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 10,
        "risks": "Moderate - balanced for quantization artifacts",
    },
    
    "QUICK_EXPERIMENT": {
        "description": "Fast training for quick tests",
        "beta": 0.1,
        "learning_rate": 1e-4,
        "num_train_epochs": 1,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 2,
        "risks": "High - likely underfitting, but fast for prototyping",
    },
}

# ============================================================================

"""
HOW TO EXPERIMENT
=================

1. BASELINE RUN (Default Config)
   └─ Establishes performance ceiling
   └─ DPO_DEFAULT config from train_dpo.py
   └─ Expected: 80-85% accuracy
   
2. ABLATION: VARY BETA
   ├─ Run 1: beta=0.05 (soft preferences)
   ├─ Run 2: beta=0.1 (default)
   ├─ Run 3: beta=0.2 (sharp preferences)
   ├─ Run 4: beta=0.5 (very sharp)
   └─ Compare accuracies → find sweet spot

3. ABLATION: VARY LEARNING RATE
   ├─ Run 1: lr=1e-5
   ├─ Run 2: lr=5e-5 (default)
   ├─ Run 3: lr=1e-4
   └─ Compare training stability & final accuracy

4. ABLATION: VARY EPOCHS
   ├─ Run 1: epochs=1
   ├─ Run 2: epochs=2
   ├─ Run 3: epochs=3 (default)
   ├─ Run 4: epochs=5
   └─ Plot loss curve - find diminishing returns point

EXPECTED FINDINGS:
  - Beta: 0.1-0.15 usually best
  - LR: 5e-5 to 1e-4 range works
  - Epochs: Usually 3-4 for diminishing returns
  - Batch size effects: Less critical than above

TIME COMMITMENT:
  Each full training: 2-4 hours
  To test 4 configs: 8-16 hours
  → Do on overnight/background runs
"""

# ============================================================================

"""
MONITORING & DIAGNOSTICS
=========================

WHAT TO TRACK DURING TRAINING:

1. LOSS CURVE
   ✓ Good: Smooth downward trend
   ✓ Good: Steeper in epoch 1, plateaus in epoch 3
   ✗ Bad: Oscillating (reduce LR)
   ✗ Bad: Diverging/increasing (reduce LR, reduce beta)
   ✗ Bad: Flat (increase LR or beta)

2. MEMORY USAGE
   ✓ Expected: Peaks at ~18-20 GB on RTX 3090
   ✓ OK if: Fluctuates but doesn't OOM
   ✗ Bad: Steady increase → memory leak
   ✗ Bad: Consistently >22 GB → OOM risk

3. TRAINING SPEED
   ✓ Expected: 2-4 hours for 3 epochs
   ✓ OK if: Slower on first epoch (initialization)
   ✗ Bad: Slowing down (possible memory issue)
   ✗ Bad: Faster and worse results (bad LR)

4. CHECKPOINT QUALITY
   ✓ Expected: Model improves step-by-step
   ✓ OK if: Final checkpoint is best
   ✗ Bad: Checkpoint overfitting (early stopping needed)

DIAGNOSTICS:

Problem: Loss diverges
├─ Decrease learning_rate by 2x
├─ Decrease beta to 0.05
└─ Add warmup_steps or increase from 5 to 20

Problem: Loss oscillates
├─ Decrease learning_rate
├─ Increase gradient_accumulation
└─ Might be batch size too large

Problem: Loss plateaus early
├─ Increase num_train_epochs
├─ Try slightly higher learning_rate
└─ Consider increasing beta

Problem: OOM (out of memory)
├─ Reduce per_device_train_batch_size
├─ Reduce gradient_accumulation_steps
├─ Reduce max_seq_length
└─ Use smaller model or different GPU
"""

# ============================================================================

"""
COMMON MISTAKES & FIXES
=======================

MISTAKE 1: Using SFT hyperparameters for DPO
├─ Wrong: learning_rate = 2e-4 (from SFT)
├─ Why: DPO is more sensitive, needs lower LR
└─ Fix: Use learning_rate = 5e-5

MISTAKE 2: Too high beta
├─ Wrong: beta = 1.0 or higher
├─ Why: Exploding preference signal, training instability
└─ Fix: Keep beta in 0.05 - 0.2 range

MISTAKE 3: Not enough epochs
├─ Wrong: num_train_epochs = 1
├─ Why: DPO needs multiple passes through data
└─ Fix: Use at least 3 epochs

MISTAKE 4: Batch size too large
├─ Wrong: per_device_batch_size = 4 with 4-bit
├─ Why: 4-bit quantization + DPO = memory intensive
└─ Fix: Keep at 1 with gradient_accumulation = 8

MISTAKE 5: Ignoring convergence
├─ Wrong: Running for 10 epochs
├─ Why: Overfitting, computational waste
└─ Fix: Stop when loss plateaus
"""

# ============================================================================

"""
THESIS-SPECIFIC RECOMMENDATIONS
================================

Your goal: Prove DPO > SFT for ranking adherence

TO MAXIMIZE CHANCES OF SUPPORTING HYPOTHESIS:

1. BALANCED CONFIG (start here)
   beta=0.1, lr=5e-5, epochs=3
   → Proven to work well
   → Most likely to show improvement

2. IF RESULTS ARE CLOSE
   Run experiment with QUANTIZATION_OPTIMIZED config
   → Better handles 4-bit noise
   → Might unlock hidden DPO advantage

3. IF RESULTS ARE BAD (DPO worse than SFT)
   Try CONSERVATIVE_STABLE config
   → More epochs with lower beta
   → Better chance of convergence
   → Or DPO truly not better (valid scientific result)

4. FOR APPENDIX/SUPPLEMENTARY
   Include ablation studies:
   ├─ Beta sensitivity
   ├─ Learning rate sensitivity
   └─ Epoch sensitivity
   → Shows thorough experimental methodology

KEY INSIGHT:
If balanced config shows DPO > SFT: Great!
If balanced config shows SFT >= DPO: Also valid!
→ Either result is publishable scientific finding
→ Rigor > hoping for one specific outcome
"""

# ============================================================================

"""
QUICK MODIFICATION FOR train_dpo.py
====================================

To use a different config, modify training_args in train_dpo.py:

Replace:
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=5e-5,
        ...
    )

With your chosen config. Example for aggressive:
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,        # ← changed
        num_train_epochs=5,     # ← changed
        learning_rate=1e-4,     # ← changed
        ...
    )

And in DPO trainer initialization:
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=0.2,               # ← changed from 0.1
        ...
    )

Save and run: python train_dpo.py
"""

# ============================================================================

print("""
DPO HYPERPARAMETER TUNING GUIDE
================================

Key takeaways:
1. Start with DEFAULT (beta=0.1, lr=5e-5, epochs=3)
2. Monitor loss curve - should be smooth downward
3. If unstable → decrease learning_rate
4. If plateaus early → increase epochs or beta
5. 4-bit quantization is stable with these defaults
6. For thesis: rigor matters more than specific outcome

Next steps:
1. Read DPO_THESIS_DOCUMENTATION.md for context
2. Run train_dpo.py with default config
3. Evaluate with evaluate_dpo.py
4. If needed, adjust based on results
5. Run 2-3 ablations for supplementary material

Questions? Check training logs and loss curves!
""")
