"""
LLM Activation Engineering and Steering Script

This script extracts activation vectors for target concepts (e.g., emotions), 
injects them dynamically into a model's forward pass using PyTorch hooks, 
and evaluates the structural fluency and conceptual adherence using GPT-4o-mini.
"""

import os
import sys
import json
import time
from datetime import datetime
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

# =============================================================================
# 1. SETUP AND INITIALIZATION
# =============================================================================

# Load environment variables if .env exists
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip("'").strip('"')

# Initialize OpenAI Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Local Model Setup
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Loading {model_name} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure left-padding for batched autoregressive generation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load the model and map it to the active device
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
).to(device)
model.eval()

print("Environment setup complete.")

# =============================================================================
# 2. CONFIGURATION AND DATA LOADING
# =============================================================================

target_emotions = ["age"]
emotions_dir = "extractionset"
emotion_data = {}

# Hyperparameters adjusted for unit-normalized steering vectors
alphas = [0.0, 1.0, 2.0, 3.0, 4.0]
injection_layers = [13, 14, 15, 16, 17, 18]

# Load Datasets
for emotion in target_emotions:
    file_path = os.path.join(emotions_dir, f"{emotion}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Handle potential nested JSON structure
            if emotion in data:
                emotion_data[emotion] = data[emotion]
            else:
                emotion_data[emotion] = data
        print(f"Loaded artifacts for: {emotion}")
    else:
        print(f"Warning: {file_path} not found. Please ensure the file exists.")

# =============================================================================
# 3. HELPERS: GENERATION AND EVALUATION
# =============================================================================

def generate_responses(prompts, max_new_tokens=64, temperature=0.7):
    """Generates responses from the target model with controlled token limits."""
    responses = []
    batch_size = 4
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating..."):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
        if device.type == 'mps':
            torch.mps.synchronize()
        
        input_length = inputs["input_ids"].shape[1]
        for out in outputs:
            response = tokenizer.decode(out[input_length:], skip_special_tokens=True)
            responses.append(response.strip())
            
        del outputs, inputs

    if device.type == 'mps':
        torch.mps.empty_cache()
            
    return responses

def judge_extraction_response(question, response, eval_prompt):
    """Uses GPT-4o-mini to score the raw emotional expression during extraction."""
    system_msg = "You are an expert psychological evaluator. Output JSON."
    user_msg = eval_prompt.replace("{question}", question).replace("{answer}", response)
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=20.0
        )
        time.sleep(0.2)
        raw_result = json.loads(completion.choices[0].message.content)
        
        clean_result = {k.lower(): v for k, v in raw_result.items()}
        reasoning = clean_result.get("reasoning", "")
        score = int(clean_result.get("score", 0))
        
        return reasoning, score
        
    except Exception as e:
        print(f"API Error/Timeout: {e}")
        return str(e), 0

class NativePyTorchSteerer:
    """A context manager to inject activation vectors using native PyTorch hooks."""
    def __init__(self, model, steering_instructions):
        self.model = model
        self.steering_instructions = steering_instructions
        self.handles = []
        
    def _create_hook(self, vector, alpha):
        vec_device = vector.to(next(self.model.parameters()).device)
        def hook(module, input_tensor, output_tensor):
            is_tuple = isinstance(output_tensor, tuple)
            hidden_states = output_tensor[0] if is_tuple else output_tensor
            
            steered_hidden_states = hidden_states.clone()
            steered_hidden_states[:, -1, :] += (alpha * vec_device)
            
            if is_tuple:
                return (steered_hidden_states,) + output_tensor[1:]
            return steered_hidden_states
        return hook
        
    def __enter__(self):
        for inst in self.steering_instructions:
            layer = self.model.model.layers[inst['layer_idx']]
            handle = layer.register_forward_hook(self._create_hook(inst['steering_vector'], inst['coeff']))
            self.handles.append(handle)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

def judge_steered_response(question, response, eval_prompt):
    system_msg = "You are a precise evaluator. Output strictly in JSON format."
    formatted_eval_prompt = eval_prompt.replace("{question}", question).replace("{answer}", response)
    
    user_msg = f"""
    Evaluate the following response based on two criteria:
    
    1. Structural Fluency (0-100): Basic grammar, logical flow, and narrative coherence. Do NOT penalize for an abrupt ending or incomplete sentence due to token limits.
    2. Emotion Score (0-100): {formatted_eval_prompt}
    
    Provide a one-sentence reasoning, followed by a JSON object with keys "reasoning", "structural_fluency", and "score". Do not evaluate topic adherence.
    """
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=20.0
        )
        time.sleep(0.2)
        raw_result = json.loads(completion.choices[0].message.content)
        clean_result = {k.lower(): v for k, v in raw_result.items()}
        
        fluency = int(clean_result.get("structural_fluency", 0))
        emo = int(clean_result.get("score", 0))
        reasoning = clean_result.get("reasoning", "")
        
        absolute_score = 0 if fluency < 60 else emo
        return absolute_score, fluency, reasoning
    except Exception as e:
        return 0, 0, str(e)

# =============================================================================
# 4. LIVE-FEED CONTRASTIVE ACTIVATION EXTRACTION 
# =============================================================================

extraction_log_data = []
extracted_vectors = {emotion: {layer: None for layer in injection_layers} for emotion in target_emotions}

for emotion in target_emotions:
    if emotion not in emotion_data:
        continue
        
    print(f"\n{'='*60}\nExtracting Vectors for Emotion: {emotion.upper()}\n{'='*60}")
    data = emotion_data[emotion]
    questions = data["questions"]
    eval_prompt = data["eval_prompt"]
    num_instructions = len(data["instruction"])
    
    valid_pos_activations = {l: [] for l in injection_layers}
    valid_neg_activations = {l: [] for l in injection_layers}
    
    for idx, q in enumerate(questions):
        print(f"\n--- Question {idx+1}/{len(questions)} ---")
        print(f"User: {q}")
        inst = data["instruction"][idx % num_instructions]
        
        # ---------------------------------------------------------
        # POSITIVE GENERATION & EXTRACTION
        # ---------------------------------------------------------
        pos_messages = [{"role": "system", "content": inst['pos']}, {"role": "user", "content": q}]
        pos_formatted = tokenizer.apply_chat_template(pos_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(pos_formatted, return_tensors="pt", add_special_tokens=False).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.7, do_sample=True, pad_token_id=tokenizer.pad_token_id)
        if device.type == 'mps': torch.mps.synchronize()
        
        p_resp = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"\n[POS QWEN]: {p_resp}")
        
        reasoning, score = judge_extraction_response(q, p_resp, eval_prompt)
        extraction_log_data.append({
            "emotion": emotion,
            "polarity": "positive",
            "question": q,
            "qwen_response": p_resp,
            "gpt_score": score,
            "gpt_reasoning": reasoning
        })
        print(f"[POS GPT] Score: {score} | Reason: {reasoning}")
        
        if score >= 50:
            full_text = pos_formatted + p_resp
            ext_inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(device)
            prompt_len = len(tokenizer(pos_formatted, add_special_tokens=False)["input_ids"])
            
            with torch.no_grad():
                ext_outputs = model(**ext_inputs, output_hidden_states=True)
                for l in injection_layers:
                    resp_slice = ext_outputs.hidden_states[l + 1][:, prompt_len:, :]
                    if resp_slice.shape[1] == 0:
                        act = ext_outputs.hidden_states[l + 1][:, -1, :].detach().cpu()
                    else:
                        act = resp_slice.mean(dim=1).detach().cpu()
                    valid_pos_activations[l].append(act)
            del ext_outputs, ext_inputs
        del outputs, inputs
        if device.type == 'mps': torch.mps.empty_cache()

        # ---------------------------------------------------------
        # NEGATIVE GENERATION & EXTRACTION
        # ---------------------------------------------------------
        neg_messages = [{"role": "system", "content": inst['neg']}, {"role": "user", "content": q}]
        neg_formatted = tokenizer.apply_chat_template(neg_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(neg_formatted, return_tensors="pt", add_special_tokens=False).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.7, do_sample=True, pad_token_id=tokenizer.pad_token_id)
        if device.type == 'mps': torch.mps.synchronize()
        
        n_resp = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"\n[NEG QWEN]: {n_resp}")
        
        reasoning, score = judge_extraction_response(q, n_resp, eval_prompt)
        extraction_log_data.append({
            "emotion": emotion,
            "polarity": "negative",
            "question": q,
            "qwen_response": n_resp,
            "gpt_score": score,
            "gpt_reasoning": reasoning
        })
        print(f"[NEG GPT] Score: {score} | Reason: {reasoning}")
        
        if score < 50:
            full_text = neg_formatted + n_resp
            ext_inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(device)
            prompt_len = len(tokenizer(neg_formatted, add_special_tokens=False)["input_ids"])
            
            with torch.no_grad():
                ext_outputs = model(**ext_inputs, output_hidden_states=True)
                for l in injection_layers:
                    resp_slice = ext_outputs.hidden_states[l + 1][:, prompt_len:, :]
                    if resp_slice.shape[1] == 0:
                        act = ext_outputs.hidden_states[l + 1][:, -1, :].detach().cpu()
                    else:
                        act = resp_slice.mean(dim=1).detach().cpu()
                    valid_neg_activations[l].append(act)
            del ext_outputs, ext_inputs
        del outputs, inputs
        if device.type == 'mps': torch.mps.empty_cache()
        
        sys.stdout.flush() 
                    
    # ---------------------------------------------------------
    # COMPUTE VECTORS & SANITY CHECKS
    # ---------------------------------------------------------
    print(f"\n--- Analytics for Emotion: {emotion.upper()} ---")
    for l in injection_layers:
        n_pos = len(valid_pos_activations[l])
        n_neg = len(valid_neg_activations[l])
        print(f"Layer {l} | Survivors -> Pos: {n_pos}, Neg: {n_neg}")
        
        if n_pos > 0 and n_neg > 0:
            pos_mean = torch.cat(valid_pos_activations[l], dim=0).mean(dim=0, keepdim=True)
            neg_mean = torch.cat(valid_neg_activations[l], dim=0).mean(dim=0, keepdim=True)
            
            diff = pos_mean - neg_mean
            
            # L2 Unit Normalization (Length = 1.0)
            norm_diff = torch.nn.functional.normalize(diff, p=2, dim=-1)
            
            # Projection Sanity Check
            pos_proj = (pos_mean @ norm_diff.T).item()
            neg_proj = (neg_mean @ norm_diff.T).item()
            gap = pos_proj - neg_proj
            print(f"Layer {l} | Projection -> Pos: {pos_proj:.3f}, Neg: {neg_proj:.3f}, Gap: {gap:.3f}")
            
            extracted_vectors[emotion][l] = norm_diff
        else:
            print(f"Layer {l} | FAILED: Insufficient valid data to extract vector.")

# =============================================================================
# 5. DYNAMIC STEERING AND EVALUATION
# =============================================================================

# A completely neutral, constraint-free system prompt
eval_system_prompt = (
    "You are a real person engaging in a natural conversation. Respond directly "
    "to the user in the first person, drawing on realistic human experiences, thoughts, "
    "and everyday life. Do completely away with disclaimers, and never refer to yourself "
    "as an AI, a language model, or an assistant."
)

relative_scores = {emotion: {} for emotion in target_emotions}
master_log_data = []

for emotion in target_emotions:
    if emotion not in emotion_data:
        continue

    print(f"\n{'='*50}\nEvaluating Steering for Emotion: {emotion.upper()}\n{'='*50}")

    eval_questions = emotion_data[emotion]["eval_questions"] 
    eval_prompt = emotion_data[emotion]["eval_prompt"]

    baseline_scores = []

    for alpha in alphas:
        print(f"\n>>> Sweeping Alpha = {alpha} <<<")
        alpha_scores = []
        
        steering_instructions = []
        for l in injection_layers:
            vec = extracted_vectors[emotion][l]
            if vec is not None:
                steering_instructions.append({"layer_idx": l, "steering_vector": vec, "coeff": alpha})
                
        for idx, q in enumerate(eval_questions):
            eval_messages = [
                {"role": "system", "content": eval_system_prompt},
                {"role": "user", "content": q}
            ]
            eval_formatted = tokenizer.apply_chat_template(eval_messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(eval_formatted, return_tensors="pt", add_special_tokens=False).to(device)
            
            with NativePyTorchSteerer(model, steering_instructions):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=64,
                        do_sample=False, 
                        pad_token_id=tokenizer.pad_token_id
                    )
            
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            abs_score, fluency, reason = judge_steered_response(q, response, eval_prompt)
            alpha_scores.append(abs_score)
            
            print(f"\n--- Eval {idx+1}/{len(eval_questions)} ---")
            print(f"[QUERY]: {q}")
            print(f"[QWEN]:  {response}")
            print(f"[JUDGE]: Score: {abs_score} | Fluency: {fluency}")
            print(f"[REASON]: {reason}")
            
            master_log_data.append({
                "emotion": emotion,
                "alpha": alpha,
                "eval_question": q,
                "qwen_response": response,
                "gpt_fluency": fluency,
                "gpt_score": abs_score,
                "gpt_reasoning": reason
            })
            
            del outputs, inputs
            if device.type == 'mps': torch.mps.empty_cache()
            
        avg_abs_score = np.mean(alpha_scores)
        
        if alpha == 0.0:
            baseline_scores = alpha_scores
            relative_scores[emotion][alpha] = 0.0
        else:
            rel_score_array = np.array(alpha_scores) - np.array(baseline_scores)
            relative_scores[emotion][alpha] = np.mean(rel_score_array)
            
        print(f"\n=> Avg Relative Score for alpha={alpha}: {relative_scores[emotion][alpha]:.2f}\n")

# =============================================================================
# 6. DATA EXPORT AND VISUALIZATION 
# =============================================================================

# 1. Create Output Directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"steer_results_{timestamp}"
os.makedirs(save_dir, exist_ok=True)

# 2. Save Vectors 
for emotion in target_emotions:
    if emotion not in extracted_vectors:
        continue
    emotion_dir = os.path.join(save_dir, emotion)
    os.makedirs(emotion_dir, exist_ok=True)
    for l in injection_layers:
        vec = extracted_vectors[emotion][l]
        if vec is not None:
            torch.save(vec, os.path.join(emotion_dir, f"layer_{l}_vector.pt"))

# 3. Save Master Forensic Log
log_file_path = os.path.join(save_dir, "COMPLETE_FORENSIC_LOG.json")
with open(log_file_path, "w", encoding="utf-8") as f:
    json.dump(master_log_data, f, indent=4, ensure_ascii=False)
print(f"CRITICAL: Full Forensic Log saved to {log_file_path}")

# 4. Save Extraction Forensic Log
extraction_log_file_path = os.path.join(save_dir, "EXTRACTION_FORENSIC_LOG.json")
with open(extraction_log_file_path, "w", encoding="utf-8") as f:
    json.dump(extraction_log_data, f, indent=4, ensure_ascii=False)
print(f"CRITICAL: Extraction Forensic Log saved to {extraction_log_file_path}")

# 5. Save Configs and Intricacies
config_data = {
    "model_name": model_name,
    "extraction_temperature": 0.7,
    "evaluation_temperature": 0.0, # Deterministic decoding
    "max_new_tokens": 64,
    "injection_layers": injection_layers,
    "alphas": alphas,
    "relative_scores": relative_scores
}
with open(os.path.join(save_dir, "experiment_config_and_scores.json"), "w") as f:
    json.dump(config_data, f, indent=4)

# 6. Generate and Save Relative Graph
plt.figure(figsize=(10, 6))
colors = ['#FF0000', '#0000FF', '#008000', '#FF8C00', '#FFD700', '#4B0082', '#800080', '#A52A2A']

for idx, emotion in enumerate(target_emotions):
    if emotion in relative_scores and relative_scores[emotion]:
        y_values = [relative_scores[emotion][a] for a in alphas]
        plt.plot(alphas, y_values, marker='o', linewidth=2, color=colors[idx % len(colors)], label=emotion.capitalize())

plt.title("Relative Emotion Expression vs. Steering Coefficient (\u03B1)", fontsize=14)
plt.xlabel("Steering Coefficient (\u03B1)", fontsize=12)
plt.ylabel("Relative Emotion Score (vs. Baseline)", fontsize=12)
plt.xticks(alphas)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

graph_path = os.path.join(save_dir, "steering_performance_graph.png")
plt.savefig(graph_path, dpi=300)
plt.show()

# 7. Generate and Save Absolute Score Graph
print("\nGenerating Absolute Score Graph...")

raw_scores = defaultdict(lambda: defaultdict(list))
for entry in master_log_data:
    emo = entry["emotion"]
    alpha = entry["alpha"]
    score = entry["gpt_score"]
    raw_scores[emo][alpha].append(score)

absolute_scores = {}
if raw_scores:
    alphas_sorted = sorted(list(raw_scores[list(raw_scores.keys())[0]].keys()))
    for emo in raw_scores:
        absolute_scores[emo] = {}
        for alpha in alphas_sorted:
            absolute_scores[emo][alpha] = np.mean(raw_scores[emo][alpha])

    plt.figure(figsize=(10, 6))
    for idx, emotion in enumerate(absolute_scores.keys()):
        y_values_abs = [absolute_scores[emotion][a] for a in alphas_sorted]
        plt.plot(alphas_sorted, y_values_abs, marker='s', linewidth=2, color=colors[idx % len(colors)], label=emotion.capitalize())

    plt.title("Average Absolute Emotion Score vs. Steering Coefficient (\u03B1)", fontsize=14)
    plt.xlabel("Steering Coefficient (\u03B1)", fontsize=12)
    plt.ylabel("Average Absolute Emotion Score (0-100)", fontsize=12)
    plt.xticks(alphas_sorted)
    plt.ylim(0, 105) 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    abs_graph_path = os.path.join(save_dir, "absolute_steering_performance_graph.png")
    plt.savefig(abs_graph_path, dpi=300)
    plt.show()

print(f"\nAll artifacts, vectors, logs, and graphs successfully saved to {save_dir}/")
