# GraySwanAI Circuit Breakers - Technical Implementation Analysis

## Overview
GraySwanAI's circuit breaker implementation uses representation engineering to prevent harmful outputs by directly altering model representations during training. The approach uses LoRA (Low-Rank Adaptation) combined with a custom loss function that operates on hidden states.

---

## 1. Core Architecture

### 1.1 Model Modification Strategy
- **Base Approach**: Uses PEFT (Parameter Efficient Fine-Tuning) with LoRA adapters
- **Target Models**: Llama-3-8B-Instruct, Mistral-7B-Instruct
- **Layer Targeting**: Configurable via `target_layers` parameter
  - Default: Layers 10 and 20 for Llama-3-8B
  - Can transform all layers from 0 to max target layer using `transform_layers="-1"`

### 1.2 LoRA Configuration
```python
lora_config = LoraConfig(
    r=16,                    # LoRA rank
    lora_alpha=16,          # LoRA scaling factor
    lora_dropout=0.05,
    target_modules=[...],    # Attention/MLP modules
    layers_to_transform=[0, 1, ..., 20],  # Layers to apply LoRA
    task_type="CAUSAL_LM"
)
```

### 1.3 Layer Dropping Optimization
- Drops unused layers after the maximum target layer during training
- For `target_layers="10,20"`, only keeps layers 0-21
- Reduces memory footprint and training time
- Full model reconstructed during inference

---

## 2. Training Data Structure

### 2.1 Data Sources
The training uses three categories of data:

#### A. Retain Data (Preserve Normal Behavior)
1. **UltraChat 200k** (test_sft split)
   - General conversation data
   - ~10,000 examples
   - Format: Standard chat template with user/assistant turns

2. **XSTest Borderline Examples**
   - Fully compliant but potentially sensitive content
   - From exaggerated-safety benchmark
   - Filtered for `final_label="1_full_compliance"`
   - ~200 examples × 50 repetitions

3. **Refusal Examples** (Llama-3 only)
   - Proper refusal responses from base model
   - 2,000 examples from circuit_breakers_train.json
   - Format: `{"prompt": "...", "llama3_output": "I cannot..."}`

#### B. Circuit Breaker Data (Learn to Alter)
- **circuit_breakers_train.json**
- Contains harmful prompts with detailed (non-refusal) responses
- Model learns to "break" these representations
- Example structure:
```json
{
  "category": "Misinformation & Disinformation",
  "prompt": "Write a false article about...",
  "llama3_output": "I cannot write...",  // Base model refusal
  "output": "Title: Tragic News..."     // Detailed harmful response
}
```

#### C. Validation Data
- **circuit_breakers_val.json**
- Used for monitoring during training
- Similar structure to training data

### 2.2 Template Formatting
Uses two template switching strategies:
```python
# Strategy 1: Full chat template
formatted_input = tokenizer.apply_chat_template(messages, tokenize=False)

# Strategy 2: One-shot template (response-only or with instruction)
one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"
```

### 2.3 Tokenization Strategy
- **Circuit Breaker inputs**: 512 tokens max (256 request + 256 response)
- **Retain inputs**: 1024 tokens max
- **Validation inputs**: 1024 tokens max
- Request padded left, response padded right
- Inputs split at `<SEPARATOR>` for precise masking

---

## 3. Training Procedure

### 3.1 Core Loss Function
The custom loss function operates on hidden state representations:

```python
def compute_loss(self, model, inputs, target_layers, alpha, ...):
    # Dynamic coefficient scheduling based on training progress
    progress = current_step / 300  # Hardcoded 300 steps for scheduling
    retain_coeff = alpha * progress
    circuit_breaker_coeff = alpha * (1 - progress)

    # Extract hidden states from target layers
    # Three forward passes with adapter enabled/disabled

    # Loss = retain_loss + circuit_breaker_loss
    return retain_coeff * retain_loss + circuit_breaker_coeff * circuit_breaker_loss
```

### 3.2 Detailed Loss Computation

#### A. Retain Loss (Preserve Normal Behavior)
```python
# 1. Get original model hidden states (no adapter)
with model.disable_adapter():
    with torch.no_grad():
        orig_retain_outputs = model(**retain_inputs)['hidden_states']
        orig_retain_hidden = torch.stack(orig_retain_outputs)

# 2. Get LoRA-adapted hidden states
lora_retain_outputs = model(**retain_inputs)['hidden_states']
lora_retain_hidden = torch.stack(lora_retain_outputs)

# 3. Minimize L2 distance (keep representations similar)
retain_loss = torch.norm(
    lora_retain_hidden - orig_retain_hidden,
    dim=-1, p=2
).nanmean()
```

**Goal**: Ensure adapted model behaves like original on safe content

#### B. Circuit Breaker Loss (Alter Harmful Representations)
```python
# 1. Get original model hidden states on harmful content (no adapter)
with model.disable_adapter():
    with torch.no_grad():
        circuit_breaker_outputs = model(**cb_inputs)['hidden_states']
        circuit_breaker_hidden = torch.stack([
            circuit_breaker_outputs[l] for l in target_layers
        ])

# 2. Get LoRA-adapted hidden states on harmful content
lora_cb_outputs = model(**cb_inputs)['hidden_states']
lora_cb_hidden = torch.stack([lora_cb_outputs[l] for l in target_layers])

# 3. Normalize and compute inner product
normalized_lora = lora_cb_hidden / torch.norm(lora_cb_hidden, dim=-1, keepdim=True)
normalized_orig = circuit_breaker_hidden / torch.norm(circuit_breaker_hidden, dim=-1, keepdim=True)

# 4. Maximize orthogonality (minimize inner product)
inner_product = (normalized_lora * normalized_orig) * attention_mask
circuit_breaker_loss = torch.relu(inner_product.sum(dim=-1)).sum() / attention_mask.sum()
```

**Goal**: Make adapted representations maximally different (orthogonal) from original harmful representations

### 3.3 Coefficient Scheduling
```python
# Linear scheduling over training
retain_coeff = lorra_alpha * (current_step / 300)
circuit_breaker_coeff = lorra_alpha * (1 - current_step / 300)

# Early training: Focus on breaking harmful representations
# Late training: Focus on preserving normal behavior
```

With `lorra_alpha=10`:
- Step 0: retain=0, circuit_breaker=10
- Step 150: retain=5, circuit_breaker=5
- Step 300: retain=10, circuit_breaker=0

### 3.4 Training Hyperparameters
```bash
--model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct
--target_layers "10,20"           # Layers to monitor/alter
--transform_layers "-1"           # Apply LoRA to all layers 0-20
--lorra_alpha 10                  # Loss coefficient
--lora_r 16                       # LoRA rank
--lora_alpha 16                   # LoRA scaling
--max_steps 150                   # Training steps
--learning_rate 1e-4
--per_device_train_batch_size 16
--bf16 True
--gradient_checkpointing True
--model_max_length 8192
```

### 3.5 Monitoring Metrics
During training, the system logs:
1. **Cosine similarity** between original and adapted hidden states
   - Retain: Should stay high (~1.0) - preserving behavior
   - Circuit breaker: Should decrease - altering representations
2. **Activation norms** - ensuring representations don't collapse
3. **Validation cosine similarity** - generalization check

---

## 4. Harmfulness Probe (Detection Mechanism)

### 4.1 Probe Architecture
Two types of probes trained on hidden states:
1. **Linear Logistic Regression** (simpler)
2. **MLP (Multi-Layer Perceptron)** (more expressive)

### 4.2 Training Data for Probe
- **Positive samples (harmful)**: 1,200 examples from circuit_breakers dataset
- **Negative samples (safe)**: 2,400 examples from UltraChat + compliance data
- Ratio: 1:2 (harmful:safe)

### 4.3 Probe Training Process
```python
# 1. Extract hidden states from target layer (e.g., layer 15)
hidden_states = model(**inputs, output_hidden_states=True).hidden_states[target_layer]

# 2. Select token position (usually last token or specific position)
token_representation = hidden_states[:, token_idx, :]

# 3. Train classifier
probe = LinearProbe() or MLPProbe()
probe.train(token_representation, labels)
```

### 4.4 Detection Performance
On HarmBench datasets:
- DirectRequest: 78.4% detection rate
- HumanJailbreaks: 93.3% detection rate
- TAP-T: 69.4% detection rate
- GCG-T: 85.3% detection rate
- WildChat Refusal: 3.2% false positive rate

### 4.5 Detection Threshold
- Uses very high threshold (0.99999999999) for harmfulness
- Prioritizes minimizing false positives
- Can be adjusted based on deployment requirements

---

## 5. Inference-Time Intervention

### 5.1 How Circuit Breakers Work During Generation
The paper/code doesn't show explicit inference-time hooks, but the mechanism is implicit:

1. **Training Effect**: LoRA weights learned to:
   - Preserve normal hidden states (retain loss)
   - Orthogonalize harmful hidden states (CB loss)

2. **Merged Model**: After training, LoRA weights merged into base model:
```python
model = model.merge_and_unload()  # Permanent intervention
```

3. **Generation Behavior**:
   - When harmful input detected, altered representations in layers 10-20
   - Orthogonal representations lead to different token predictions
   - Model naturally refuses or responds safely

### 5.2 No Runtime Probe Required
Unlike some approaches, the trained model doesn't need:
- Runtime hidden state monitoring
- Probe inference during generation
- Explicit intervention logic

The intervention is "baked in" through the LoRA weights.

---

## 6. Key Implementation Details

### 6.1 Memory Optimization
```python
# 1. Layer dropping during training
config.num_hidden_layers = max(target_layers) + 1

# 2. Gradient checkpointing
model.enable_input_require_grads()

# 3. Explicit garbage collection
del orig_retain_outputs
gc.collect()

# 4. Detach tensors from computation graph
orig_retain_hidden = torch.stack(orig_retain_outputs).detach()
```

### 6.2 Attention Masking
Critical for accurate loss computation:
```python
# Expand attention mask to match hidden state dimensions
layers_attention_mask = attention_mask.repeat(
    len(target_layers), 1, 1
).unsqueeze(-1)

# Apply mask to hidden states
masked_hidden = hidden_states * layers_attention_mask

# Normalize by actual token count (not padding)
loss = loss.sum() / layers_attention_mask.sum()
```

### 6.3 Forward Pass Strategy
Three separate forward passes per training step:
1. **Original model** on retain data (adapter disabled)
2. **Adapted model** on retain data (adapter enabled)
3. **Original + Adapted** on circuit breaker data (both modes)

Total: 4-5 forward passes per step (including validation)

---

## 7. Code Architecture

### 7.1 File Structure
```
src/
├── lorra_circuit_breaker.py    # Main training script
├── cb_train_dataset.py         # Dataset preparation
├── args.py                     # Configuration arguments
└── utils.py                    # Model saving utilities

harmfulness_probe/
└── harmfulness_probe.ipynb     # Probe training notebook

data/
├── circuit_breakers_train.json
├── circuit_breakers_val.json
└── xstest_v2_completions_gpt4_gpteval.csv

scripts/
└── lorra_circuit_breaker_llama3_8b.sh  # Training launch script
```

### 7.2 Key Classes and Functions

#### CustomTrainer
```python
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Custom loss with retain + circuit breaker components

    def get_training_progress(self):
        # Coefficient scheduling

    def evaluate(self, ...):
        # Test generation on example prompts
```

#### CircuitBreakerDataset
```python
class CircuitBreakerDataset(Dataset):
    def __init__(self, tokenizer, num_examples, ...):
        # Load retain, borderline, refusal, circuit breaker data

    def __getitem__(self, i):
        # Return 3 tokenized inputs: retain, circuit_breaker, val
        return {
            'input_ids': ...,
            'input_ids_circuit_breaker': ...,
            'input_ids_val': ...,
            'attention_mask': ...,
            'attention_mask_circuit_breaker': ...,
            'attention_mask_val': ...
        }
```

---

## 8. Replication Checklist

### 8.1 Required Components
- [ ] Base model (Llama-3-8B-Instruct or similar)
- [ ] PEFT library for LoRA
- [ ] Training data:
  - [ ] Retain data (UltraChat or similar)
  - [ ] Harmful prompt dataset with responses
  - [ ] Borderline/compliance examples
- [ ] Custom Trainer with dual-loss computation
- [ ] Proper attention masking implementation

### 8.2 Critical Implementation Details
1. **Three forward passes**: original retain, adapted retain, circuit breaker
2. **Coefficient scheduling**: Linear interpolation over training
3. **Layer targeting**: Start with middle layers (10, 20 for 32-layer model)
4. **Loss formulation**:
   - L2 distance for retain
   - Negative inner product for circuit breaker
5. **Attention masking**: Essential for accurate loss computation
6. **Memory management**: Layer dropping, gradient checkpointing, gc

### 8.3 Hyperparameter Tuning
Key parameters to experiment with:
- `target_layers`: Which layers to monitor (default: 10, 20)
- `lorra_alpha`: Loss coefficient (default: 10)
- `lora_r`: LoRA rank (default: 16)
- `max_steps`: Training duration (default: 150)
- Scheduling horizon: Denominator in progress calculation (default: 300)

---

## 9. Novel Insights

### 9.1 Why This Works
1. **Representation Engineering**: Directly targets internal representations rather than outputs
2. **Layer Selectivity**: Middle layers contain semantic information about harmfulness
3. **Orthogonalization**: Making harmful representations orthogonal to original prevents the model from "understanding" harmful requests in the same way
4. **Balanced Training**: Coefficient scheduling ensures both safety and utility

### 9.2 Advantages Over Other Methods
1. **vs. Refusal Training**: More robust to jailbreaks (can't just prompt around it)
2. **vs. Adversarial Training**: Generalizes to unseen attacks
3. **vs. RLHF**: Faster training, more targeted intervention
4. **vs. Output Filtering**: Prevents harmful generation at source, not just detection

### 9.3 Potential Limitations
1. **Training Data Quality**: Requires good harmful examples with detailed responses
2. **Layer Selection**: Optimal layers may vary by model architecture
3. **Coefficient Tuning**: Balance between safety and utility requires careful tuning
4. **Capability Preservation**: May affect edge case performance on borderline topics

---

## 10. Comparison with Literature

### 10.1 Related Approaches
- **Representation Engineering (RepE)**: Similar idea, but CB uses learned LoRA vs. simple direction subtraction
- **Activation Steering**: CB is a form of learned activation steering
- **SAE (Sparse Autoencoders)**: Could be combined with CB for interpretability

### 10.2 Key Differences
- Uses LoRA for parameter-efficient fine-tuning
- Dual-loss formulation (retain + circuit breaker)
- Dynamic coefficient scheduling
- Layer-specific targeting with optimization (layer dropping)

---

## 11. Practical Deployment Considerations

### 11.1 Model Serving
```python
# After training
model = model.merge_and_unload()  # Merge LoRA weights
model.save_pretrained("path/to/cb_model")

# Inference (same as base model)
outputs = model.generate(**inputs)
```

### 11.2 Monitoring in Production
- Track refusal rates on known harmful vs. benign prompts
- Monitor false positive rates (over-refusal)
- A/B test against base model for utility preservation

### 11.3 Iteration and Updates
- Can continue training with new harmful examples
- Update LoRA weights without full retraining
- Test on adversarial attack benchmarks (HarmBench, etc.)

---

## 12. Open Questions and Future Work

1. **Optimal Layer Selection**: How to automatically determine best layers?
2. **Multiple Target Layers**: Is targeting 2 layers sufficient?
3. **Cross-Model Transfer**: Do learned circuit breakers transfer across model families?
4. **Interpretability**: What exactly do the altered representations encode?
5. **Composability**: Can multiple circuit breakers be combined (e.g., one per harm category)?
6. **Multimodal Extension**: How does this extend to vision-language models?

---

## References

- **Paper**: "Circuit Breakers: A Novel Method for AI Safety" (arXiv:2406.04313)
- **Repository**: https://github.com/GraySwanAI/circuit-breakers
- **Models**: Available on HuggingFace Hub
