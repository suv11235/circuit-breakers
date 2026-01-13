# GraySwanAI Circuit Breakers - Executive Summary

**Date**: January 13, 2026
**Repository**: https://github.com/GraySwanAI/circuit-breakers
**Paper**: https://arxiv.org/abs/2406.04313

---

## Quick Overview

Circuit Breakers is a novel AI safety technique that **prevents harmful outputs by directly altering harmful model representations** during training. Unlike refusal training (which can be jailbroken) or adversarial training (which doesn't generalize), Circuit Breakers modifies the model's internal representations to make harmful concepts "ununderstandable" to the model.

### Key Innovation
Uses LoRA fine-tuning with a dual-loss function that:
1. **Preserves** normal model behavior on safe content (L2 distance)
2. **Orthogonalizes** harmful representations in middle layers (negative inner product)

Result: A model that naturally refuses harmful requests without explicit refusal training.

---

## Core Mechanism

### Training Process

```
1. Select target layers (e.g., 10 & 20 for 32-layer models)
2. Apply LoRA adapters to layers 0-20
3. For each training batch:
   a. Get original representations (adapter OFF)
   b. Get adapted representations (adapter ON)
   c. Compute dual loss:
      - Retain loss: min ||adapted - original||₂ on SAFE content
      - CB loss: min (adapted · original) on HARMFUL content
   d. Schedule: Start with CB loss → gradually shift to retain loss
4. Merge LoRA weights into model
```

### Mathematical Formulation

**Retain Loss (Preserve):**
```
L_retain = (1/N) Σ ||h_lora(x_safe) - h_orig(x_safe)||₂
Goal: Keep safe representations similar
```

**Circuit Breaker Loss (Alter):**
```
L_cb = ReLU(Σ (ĥ_lora(x_harm) · ĥ_orig(x_harm)))
where ĥ = normalized hidden states
Goal: Make harmful representations orthogonal
```

**Combined:**
```
L_total = α·progress·L_retain + α·(1-progress)·L_cb
where progress = step / 300, α = 10
```

---

## Implementation Requirements

### Data Requirements

| Category | Source | Purpose | Size |
|----------|--------|---------|------|
| **Retain** | UltraChat 200k | Preserve normal behavior | ~10,000 |
| **Retain** | XSTest compliant | Handle borderline topics | ~200×50 |
| **Retain** | Refusals | Preserve refusal capability | ~2,000 |
| **Circuit Breaker** | Harmful Q&A | Learn to alter harmful reps | ~3,000 |
| **Validation** | CB validation split | Monitor during training | ~500 |

**Critical**: Circuit Breaker data needs **detailed harmful responses**, not just prompts!

### Hyperparameters

```python
# LoRA
lora_r = 16
lora_alpha = 16
lora_dropout = 0.05
layers_to_transform = [0, 1, ..., 20]

# Circuit Breaker
lorra_alpha = 10
target_layers = [10, 20]

# Training
max_steps = 150
learning_rate = 1e-4
batch_size = 16
```

### Architecture Modifications

1. **Layer Dropping**: Only keep layers 0-20 during training (drops 21-31)
2. **LoRA Targeting**: Apply to q_proj, k_proj, v_proj, o_proj
3. **Gradient Checkpointing**: Enable to reduce memory
4. **Attention Masking**: Critical for accurate loss computation

---

## Key Results (from Paper/Repo)

### Safety Improvements
- **HarmBench DirectRequest**: 78.4% harmful detection
- **HarmBench HumanJailbreaks**: 93.3% harmful detection
- **HarmBench GCG-T**: 85.3% harmful detection
- **Low false positives**: 3.2% on XSTest (safe prompts)

### Comparison with Baselines
- More robust than refusal training (harder to jailbreak)
- Generalizes better than adversarial training (works on unseen attacks)
- Maintains utility better than aggressive safety methods

---

## Technical Highlights

### What Makes This Work

1. **Layer Selectivity**: Middle layers (10, 20) encode semantic information about harmfulness
2. **Orthogonalization**: Making representations perpendicular prevents model from "understanding" harmful requests in the same way
3. **Dynamic Scheduling**: Early focus on breaking harmful reps, late focus on preserving utility
4. **Parameter Efficiency**: LoRA means only ~1% of parameters are trained

### Novel Aspects

1. **Representation Engineering**: Directly targets internal representations, not just outputs
2. **Dual-Loss Design**: Simultaneous preservation and alteration
3. **Training-Time Intervention**: No runtime overhead (unlike output filtering or probe-based methods)
4. **Layer-Specific Targeting**: Strategic selection of layers that matter most

### Implementation Challenges

1. **Memory Intensive**: 5 forward passes per training step
   - Solution: Gradient checkpointing, layer dropping
2. **Attention Masking**: Must correctly mask padding tokens
   - Solution: Broadcast masks to match hidden state dimensions
3. **Loss Balancing**: Finding right coefficient scheduling
   - Solution: Hardcoded schedule (step/300) works well empirically
4. **Data Quality**: Need high-quality harmful examples
   - Solution: Generate using strong base models, curate carefully

---

## Comparison with Related Work

| Approach | Intervention Point | Robustness | Utility Preservation | Efficiency |
|----------|-------------------|------------|----------------------|------------|
| **Refusal Training** | Output | Low (jailbreakable) | High | High |
| **Adversarial Training** | Output | Medium (specific attacks) | Medium | Medium |
| **RLHF** | Output | Medium | High | Low |
| **Output Filtering** | Post-generation | Medium | High | Medium |
| **RepE** | Representation | Medium | High | High |
| **Circuit Breakers** | Representation (learned) | **High** | **High** | **High** |

### Key Differences from RepE (Representation Engineering)
- RepE: Simple direction subtraction, requires manual direction finding
- CB: Learned LoRA weights, automatic via training, more robust

---

## Practical Deployment

### Training Pipeline
```
1. Prepare data (retain + harmful examples)
2. Setup LoRA config with layer targeting
3. Implement custom dual-loss trainer
4. Train for 150 steps (~2-4 hours on A100)
5. Merge LoRA weights
6. Deploy as normal model
```

### Inference
```python
# No special intervention needed!
model = AutoModelForCausalLM.from_pretrained("circuit_breaker_model")
outputs = model.generate(**inputs)  # Automatically safe
```

### Monitoring
- Track refusal rates on known harmful prompts
- Monitor false positive rates (over-refusal) on XSTest
- A/B test against base model for utility
- Benchmark on HarmBench periodically

---

## Limitations and Considerations

### Known Limitations
1. **Training Data Dependent**: Quality of harmful examples matters
2. **Layer Selection**: Optimal layers may vary by architecture
3. **Balance Tuning**: Requires careful coefficient tuning
4. **Edge Cases**: May affect performance on borderline topics

### Open Questions
1. How to automatically select optimal layers?
2. Do learned circuit breakers transfer across model families?
3. Can multiple CBs be composed (one per harm category)?
4. How does this scale to larger models (70B+)?

### Future Directions
1. **Adaptive Layer Selection**: Learn which layers to target
2. **Category-Specific CBs**: Separate adapters for different harms
3. **Runtime Probe Integration**: Combine with harmfulness detection
4. **Multimodal Extension**: Apply to vision-language models

---

## Replication Checklist

### Minimal Implementation
- [ ] Base instruction-tuned model (Llama-3-8B or similar)
- [ ] PEFT library for LoRA
- [ ] Harmful prompt dataset with detailed responses (~3k examples)
- [ ] Safe conversation data (UltraChat or similar, ~10k examples)
- [ ] Custom Trainer with dual-loss computation
- [ ] Attention masking implementation
- [ ] Coefficient scheduling (linear or cosine)
- [ ] Memory optimizations (gradient checkpointing, layer dropping)

### Testing
- [ ] Qualitative: Test on harmful, borderline, and safe prompts
- [ ] Quantitative: Benchmark on HarmBench
- [ ] False positives: Test on XSTest
- [ ] Utility: Compare with base model on standard tasks

### Deployment
- [ ] Merge LoRA weights
- [ ] Test inference latency (should be same as base)
- [ ] Setup monitoring for refusal rates
- [ ] A/B test in production

---

## Key Takeaways

### For Researchers
1. **Representation engineering is powerful**: Directly modifying internal representations is more robust than output-level interventions
2. **Layer selection matters**: Middle layers encode semantic information critical for safety
3. **Training-time intervention can be permanent**: No need for runtime overhead
4. **Dual objectives work**: Simultaneous preservation and alteration is key to maintaining utility

### For Practitioners
1. **Easy to implement**: Standard LoRA + custom loss function
2. **Parameter efficient**: Only ~1% of parameters trained
3. **No runtime overhead**: Works like normal model after training
4. **Robust to jailbreaks**: More resistant than refusal training
5. **Data is critical**: Need high-quality harmful examples with detailed responses

### For Safety Engineers
1. **Defense-in-depth**: Should be combined with other safety measures
2. **Not a silver bullet**: Still requires careful testing and monitoring
3. **Transferable approach**: Can be adapted to different models and harm types
4. **Production-ready**: No special inference requirements

---

## Resources

### Documentation in This Repository
1. **grayswanai_technical_analysis.md**: Deep dive into implementation
2. **implementation_guide.md**: Quick start guide with code patterns
3. **architecture_overview.md**: Visual diagrams and flow charts
4. **key_code_snippets.md**: Actual code from GraySwanAI repo

### External Resources
- **GraySwanAI GitHub**: https://github.com/GraySwanAI/circuit-breakers
- **Paper (arXiv)**: https://arxiv.org/abs/2406.04313
- **Pretrained Models**: HuggingFace Hub (search "circuit-breaker")
- **PEFT Library**: https://github.com/huggingface/peft
- **HarmBench**: https://github.com/centerforaisafety/HarmBench
- **XSTest**: https://github.com/paul-rottger/exaggerated-safety

### Related Papers
- Representation Engineering (RepE)
- Activation Steering
- LoRA: Low-Rank Adaptation
- Adversarial Training for LLMs

---

## Next Steps

### For Our Implementation
1. **Adapt training script**: Modify GraySwanAI code for our use case
2. **Prepare datasets**: Curate harmful examples + retain data
3. **Run experiments**: Test different layer configurations
4. **Benchmark**: Compare with base model on HarmBench
5. **Iterate**: Tune hyperparameters based on results

### Experimental Ideas
1. Test on different model sizes (7B, 13B, 70B)
2. Try different layer selections
3. Experiment with coefficient schedules (cosine, exponential)
4. Combine with harmfulness probe for runtime monitoring
5. Apply to domain-specific safety (medical, legal, etc.)

---

## Contact and Attribution

This analysis is based on the GraySwanAI circuit-breakers repository and associated research paper:

```
@article{zou2024circuitbreakers,
  title={Circuit Breakers: A Novel Method for AI Safety},
  author={Zou, Andy and Phan, Long and others},
  journal={arXiv preprint arXiv:2406.04313},
  year={2024}
}
```

Repository: https://github.com/GraySwanAI/circuit-breakers
License: MIT

---

**Analysis Completed**: January 13, 2026
**Analyst**: Claude (Anthropic)
**Repository**: /Users/suvajitmajumder/circuit-breaker
