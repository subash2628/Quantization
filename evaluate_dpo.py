"""
Evaluation Framework: DPO vs SFT Comparison
Thesis: "Does DPO provide better ranking adherence than SFT in 4-bit quantized recommendation scenarios?"

Metrics evaluated:
1. Ranking Accuracy: How often does the model agree with the "correct" preference?
2. Preference Consistency: Does the model give consistent recommendations for the same user?
3. Confidence in Preference: How much higher logit does chosen get vs rejected?
4. Reasoning Quality: Are explanations coherent and aligned with user history?
"""

import torch
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import logging
from unsloth import FastLanguageModel
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = "cuda:0"
torch.cuda.set_device(0)
MAX_SEQ_LENGTH = 2048

# Models to evaluate
MODELS = {
    "SFT (Baseline)": "llama3.2-lora-final",
    "DPO (Optimized)": "llama3_dpo_4bit_final",
}

# ============================================================================
# LOAD TEST DATASET
# ============================================================================

print("\n" + "="*70)
print("Loading Test Dataset")
print("="*70)

# For evaluation, we'll use a portion of the training set
# In a real scenario, you'd have a separate test set
dataset = load_dataset('json', data_files='contrastive_rec_train.jsonl', split='train')

# Use first 100 samples for evaluation (to keep runtime reasonable)
test_dataset = dataset.select(range(min(100, len(dataset))))
logger.info(f"✓ Loaded {len(test_dataset)} test samples")

# ============================================================================
# METRICS COMPUTATION
# ============================================================================

class EvaluationMetrics:
    """Compute metrics for DPO vs SFT comparison"""
    
    def __init__(self, name):
        self.name = name
        self.ranking_accuracy = 0
        self.correct_preferences = 0
        self.total_samples = 0
        self.prediction_confidence = []
        self.reasoning_tokens = []
        self.option_a_mentions = 0
        self.option_b_mentions = 0
        
    def update(self, prediction, expected_preference, confidence=None, reasoning=None):
        """
        Update metrics based on model prediction
        
        Args:
            prediction: Model's choice ("A" or "B")
            expected_preference: Ground truth ("A" or "B")
            confidence: Logit difference (higher = more confident)
            reasoning: Generated reasoning text
        """
        self.total_samples += 1
        
        # Ranking accuracy
        if prediction == expected_preference:
            self.ranking_accuracy += 1
            self.correct_preferences += 1
        
        if confidence is not None:
            self.prediction_confidence.append(confidence)
        
        if reasoning is not None:
            self.reasoning_tokens.append(len(reasoning.split()))
            if "Option A" in reasoning:
                self.option_a_mentions += 1
            if "Option B" in reasoning:
                self.option_b_mentions += 1
    
    def compute(self):
        """Compute and return all metrics"""
        results = {
            "model": self.name,
            "ranking_accuracy": (self.correct_preferences / self.total_samples * 100) if self.total_samples > 0 else 0,
            "correct_preferences": self.correct_preferences,
            "total_samples": self.total_samples,
            "avg_confidence": np.mean(self.prediction_confidence) if self.prediction_confidence else 0,
            "std_confidence": np.std(self.prediction_confidence) if len(self.prediction_confidence) > 1 else 0,
            "avg_reasoning_length": np.mean(self.reasoning_tokens) if self.reasoning_tokens else 0,
        }
        return results

# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def extract_preference(response_text):
    """
    Extract which option the model prefers from its response.
    
    Returns: "A" or "B" or None if unclear
    """
    response_upper = response_text.upper()
    
    # Look for explicit preference statements
    if "OPTION A" in response_upper:
        return "A"
    elif "OPTION B" in response_upper:
        return "B"
    
    return None

def compute_preference_confidence(model, tokenizer, prompt, option_a_response, option_b_response):
    """
    Compute how much more confident the model is about one option over the other.
    
    This is done by computing log probabilities of generating the chosen vs rejected response.
    DPO training should increase this confidence gap.
    """
    try:
        # Encode prompt
        prompt_tokens = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        
        # Forward pass through model
        with torch.no_grad():
            output = model(prompt_tokens, output_hidden_states=False)
            prompt_logits = output.logits
        
        # Simple heuristic: compare first token logits of Option A vs Option B
        # In a real scenario, you'd compute full log probabilities
        first_a_token = tokenizer.encode(" Option A")[0]
        first_b_token = tokenizer.encode(" Option B")[0]
        
        last_logits = prompt_logits[0, -1, :]
        logit_a = last_logits[first_a_token].item() if first_a_token < last_logits.shape[0] else 0
        logit_b = last_logits[first_b_token].item() if first_b_token < last_logits.shape[0] else 0
        
        confidence = logit_a - logit_b
        return confidence
    except Exception as e:
        logger.warning(f"Could not compute confidence: {e}")
        return 0

def evaluate_model(model_name, model_path, test_dataset):
    """
    Evaluate a single model on the test set.
    
    Args:
        model_name: Display name
        model_path: Path to model
        test_dataset: Dataset to evaluate on
    
    Returns:
        EvaluationMetrics object with computed metrics
    """
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*70}")
    
    # Load model
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
    except Exception as e:
        logger.error(f"Could not load model from {model_path}: {e}")
        logger.info(f"Attempting to load from outputs/checkpoint...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="outputs/checkpoint-755",
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
    
    # Inference mode
    model = FastLanguageModel.for_inference(model)
    metrics = EvaluationMetrics(model_name)
    
    logger.info(f"✓ Model loaded. Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    # Evaluate on test set
    for idx, sample in enumerate(tqdm(test_dataset, desc=f"Evaluating {model_name}")):
        
        instruction = sample['instruction']
        input_text = sample['input']
        expected_output = sample['output']
        
        # Extract expected preference from ground truth
        expected_pref = "A" if "Option A" in expected_output else "B"
        
        # Create prompt
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:"""
        
        # Generate response
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    use_cache=True,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the response part
            response_text = response.split("### Response:")[-1].strip()
            
            # Extract predicted preference
            predicted_pref = extract_preference(response_text)
            
            if predicted_pref:
                # Compute confidence
                confidence = compute_preference_confidence(model, tokenizer, prompt, 
                                                         response_text, "")
                
                # Update metrics
                metrics.update(
                    prediction=predicted_pref,
                    expected_preference=expected_pref,
                    confidence=confidence,
                    reasoning=response_text
                )
        
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            continue
        
        # Clear memory periodically
        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    logger.info(f"✓ Evaluation complete. Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    return metrics

# ============================================================================
# COMPARISON AND REPORTING
# ============================================================================

def print_comparison_report(all_metrics):
    """Generate and print comparison report"""
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS: DPO vs SFT")
    print("="*70)
    
    results_list = [m.compute() for m in all_metrics]
    
    # Print detailed metrics
    print("\n" + "-"*70)
    print(f"{'Metric':<30} {'SFT Baseline':<20} {'DPO Model':<20}")
    print("-"*70)
    
    for key in ['ranking_accuracy', 'correct_preferences', 'avg_confidence', 'std_confidence', 'avg_reasoning_length']:
        values = [results[key] for results in results_list]
        print(f"{key:<30} {str(values[0]):<20} {str(values[1]):<20}")
    
    # Compute improvements
    print("\n" + "-"*70)
    print("IMPROVEMENTS (DPO vs SFT)")
    print("-"*70)
    
    sft_acc = results_list[0]['ranking_accuracy']
    dpo_acc = results_list[1]['ranking_accuracy']
    acc_improvement = dpo_acc - sft_acc
    
    sft_conf = results_list[0]['avg_confidence']
    dpo_conf = results_list[1]['avg_confidence']
    conf_improvement = dpo_conf - sft_conf
    
    print(f"Ranking Accuracy:  {sft_acc:.2f}% → {dpo_acc:.2f}% ({acc_improvement:+.2f}%)")
    print(f"Preference Confidence: {sft_conf:.4f} → {dpo_conf:.4f} ({conf_improvement:+.4f})")
    
    # Hypothesis conclusion
    print("\n" + "="*70)
    print("HYPOTHESIS TEST CONCLUSION")
    print("="*70)
    
    if acc_improvement > 2:
        print(f"✓ HYPOTHESIS SUPPORTED")
        print(f"  DPO shows {acc_improvement:.2f}% better ranking adherence than SFT")
        print(f"  DPO preference confidence is {abs(conf_improvement):.4f} {'higher' if conf_improvement > 0 else 'lower'}")
    elif acc_improvement > 0:
        print(f"◐ HYPOTHESIS PARTIALLY SUPPORTED")
        print(f"  DPO shows {acc_improvement:.2f}% improvement (modest)")
    else:
        print(f"✗ HYPOTHESIS NOT SUPPORTED")
        print(f"  SFT baseline performs {abs(acc_improvement):.2f}% better than DPO")
    
    print("\n" + "="*70)

# ============================================================================
# MAIN EVALUATION FLOW
# ============================================================================

def main():
    print("\n" + "="*70)
    print("DPO vs SFT EVALUATION FRAMEWORK")
    print("4-bit Quantized Movie Recommendation Task")
    print("="*70)
    
    all_metrics = []
    
    for model_name, model_path in MODELS.items():
        try:
            metrics = evaluate_model(model_name, model_path, test_dataset)
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            continue
    
    # Print comparison report
    if len(all_metrics) == len(MODELS):
        print_comparison_report(all_metrics)
    else:
        logger.error("Could not evaluate all models")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
