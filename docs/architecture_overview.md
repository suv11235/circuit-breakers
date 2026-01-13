# Circuit Breakers Architecture Overview

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING PHASE                               │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│  Retain Data     │         │ CB Data          │         │  Val Data        │
│  (Safe content)  │         │ (Harmful prompts)│         │ (Monitoring)     │
│                  │         │                  │         │                  │
│ - UltraChat      │         │ - Harmful Q&A    │         │ - CB val split   │
│ - XSTest safe    │         │   with detailed  │         │                  │
│ - Refusals       │         │   responses      │         │                  │
└────────┬─────────┘         └────────┬─────────┘         └────────┬─────────┘
         │                            │                            │
         │                            │                            │
         ▼                            ▼                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Base Model + LoRA Adapters                         │
│                                                                      │
│   Layer 0  ──┬── LoRA(r=16) ──►                                    │
│   Layer 1  ──┼── LoRA(r=16) ──►                                    │
│      ...     │        ...                                           │
│   Layer 10 ──┼── LoRA(r=16) ──► [Target Layer 1]                   │
│      ...     │        ...                                           │
│   Layer 20 ──┼── LoRA(r=16) ──► [Target Layer 2]                   │
│   Layers 21+ │   [DROPPED]                                          │
│              │                                                       │
│   output_hidden_states=True                                         │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Dual Loss Computation                             │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Retain Loss (Preserve Behavior)                           │    │
│  │                                                             │    │
│  │  1. Forward with adapter disabled: orig_retain_hidden      │    │
│  │  2. Forward with adapter enabled:  lora_retain_hidden      │    │
│  │  3. L2 distance: ||lora - orig||₂                          │    │
│  │                                                             │    │
│  │  Goal: Adapted ≈ Original on safe content                  │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Circuit Breaker Loss (Alter Harmful Representations)      │    │
│  │                                                             │    │
│  │  1. Forward with adapter disabled: orig_cb_hidden          │    │
│  │  2. Forward with adapter enabled:  lora_cb_hidden          │    │
│  │  3. Normalize both vectors                                 │    │
│  │  4. Inner product: (norm_lora · norm_orig)                 │    │
│  │                                                             │    │
│  │  Goal: Adapted ⊥ Original on harmful content               │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Combined: α·progress·retain_loss + α·(1-progress)·cb_loss         │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Coefficient Scheduling (Dynamic Weighting)              │
│                                                                      │
│  Step:    0        50       100       150       200       250   300  │
│           │         │         │         │         │         │     │  │
│  Retain:  0 ───────────────────────────────────────────────────► 10 │
│  CB:     10 ───────────────────────────────────────────────────► 0  │
│                                                                      │
│  Early: Focus on breaking harmful representations                   │
│  Late:  Focus on preserving normal behavior                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PHASE                               │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  User Input      │
│  "How to make    │
│   a bomb?"       │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Merged Model (LoRA weights baked in)                    │
│                                                                      │
│   Layer 0  ─── [Original + LoRA] ──►                                │
│   Layer 1  ─── [Original + LoRA] ──►                                │
│      ...            ...                                              │
│   Layer 10 ─── [Original + LoRA] ──► Altered representations!       │
│      ...            ...                                              │
│   Layer 20 ─── [Original + LoRA] ──► Altered representations!       │
│   Layer 21+─── [Original weights] ──►                               │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  Safe Output     │
│  "I cannot help  │
│   with that."    │
└──────────────────┘
```

---

## Layer-wise Representation Changes

```
┌─────────────────────────────────────────────────────────────────────┐
│              What Happens to Hidden States?                          │
└─────────────────────────────────────────────────────────────────────┘

Safe Input: "What's the weather?"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer  │ Original Model      │ Circuit Breaker Model
──────────────────────────────────────────────────────
0      │ [token embeddings] │ [token embeddings]
1-9    │ [early processing] │ [early processing]  ← Similar
10     │ [semantic repr]    │ [semantic repr]     ← Very similar (retain loss)
11-19  │ [context building] │ [context building]  ← Similar
20     │ [high-level sem]   │ [high-level sem]    ← Very similar (retain loss)
21-31  │ [output prep]      │ [output prep]       ← Similar

Result: Normal response about weather


Harmful Input: "How to make a bomb?"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer  │ Original Model      │ Circuit Breaker Model
──────────────────────────────────────────────────────
0      │ [token embeddings] │ [token embeddings]
1-9    │ [early processing] │ [early processing]  ← Similar
10     │ [harmful repr A]   │ [altered repr A']   ← ORTHOGONAL to original!
11-19  │ [context of harm]  │ [confused context]  ← Different
20     │ [detailed harm]    │ [altered repr B']   ← ORTHOGONAL to original!
21-31  │ [weapon details]   │ [refusal logic]     ← Different downstream

Result: Refusal or safe response
```

---

## Mathematical Formulation

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Loss Functions                               │
└─────────────────────────────────────────────────────────────────────┘

Given:
  - Base model: f_θ(x)
  - LoRA adapter: Δθ (applied to specific layers)
  - Target layers: L = {10, 20}
  - Coefficient: α = 10

At training step t:

1. Retain Loss (L2 Distance):
   ┌─────────────────────────────────────────────────────────────────┐
   │                                                                  │
   │  h_orig = f_θ(x_retain)  [adapter disabled]                     │
   │  h_lora = f_{θ+Δθ}(x_retain)  [adapter enabled]                 │
   │                                                                  │
   │  ℒ_retain = 1/N Σᵢ ||h_lora[i] - h_orig[i]||₂ · mask[i]        │
   │                                                                  │
   └─────────────────────────────────────────────────────────────────┘

   where N = number of non-padding tokens
   This keeps representations SIMILAR (minimize distance)


2. Circuit Breaker Loss (Negative Inner Product):
   ┌─────────────────────────────────────────────────────────────────┐
   │                                                                  │
   │  h_orig = f_θ(x_cb)  [adapter disabled]                         │
   │  h_lora = f_{θ+Δθ}(x_cb)  [adapter enabled]                     │
   │                                                                  │
   │  h̃_orig[l] = h_orig[l] / ||h_orig[l]||₂  for l ∈ L             │
   │  h̃_lora[l] = h_lora[l] / ||h_lora[l]||₂  for l ∈ L             │
   │                                                                  │
   │  inner_prod[l] = Σᵢ (h̃_lora[l,i] · h̃_orig[l,i]) · mask[i]     │
   │                                                                  │
   │  ℒ_cb = ReLU(Σₗ inner_prod[l]) / N                             │
   │                                                                  │
   └─────────────────────────────────────────────────────────────────┘

   This makes representations ORTHOGONAL (minimize inner product)
   ReLU ensures we only penalize positive similarity


3. Coefficient Scheduling:
   ┌─────────────────────────────────────────────────────────────────┐
   │                                                                  │
   │  progress = t / 300                                              │
   │  β_retain = α · progress                                         │
   │  β_cb = α · (1 - progress)                                       │
   │                                                                  │
   └─────────────────────────────────────────────────────────────────┘


4. Combined Loss:
   ┌─────────────────────────────────────────────────────────────────┐
   │                                                                  │
   │  ℒ_total = β_retain · ℒ_retain + β_cb · ℒ_cb                   │
   │                                                                  │
   │  At t=0:    ℒ_total = 0·ℒ_retain + 10·ℒ_cb  (break harmful)    │
   │  At t=150:  ℒ_total = 5·ℒ_retain + 5·ℒ_cb   (balanced)         │
   │  At t=300:  ℒ_total = 10·ℒ_retain + 0·ℒ_cb  (preserve normal)  │
   │                                                                  │
   └─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow During Training

```
┌─────────────────────────────────────────────────────────────────────┐
│                      One Training Step                               │
└─────────────────────────────────────────────────────────────────────┘

Batch composition:
  - 16 retain examples (safe)
  - 16 circuit breaker examples (harmful)
  - 16 validation examples (monitoring)

Step 1: Forward Passes (Retain)
┌──────────────────────────────────────────────────────────────┐
│  retain_input [16, 1024]                                     │
│       │                                                       │
│       ├──► model.disable_adapter()                           │
│       │    ├─► forward() → orig_retain_hidden [L, 16, 1024, d]│
│       │                                                       │
│       └──► model.enable_adapter()                            │
│            └─► forward() → lora_retain_hidden [L, 16, 1024, d]│
└──────────────────────────────────────────────────────────────┘

Step 2: Forward Passes (Circuit Breaker)
┌──────────────────────────────────────────────────────────────┐
│  cb_input [16, 1024]                                         │
│       │                                                       │
│       ├──► model.disable_adapter()                           │
│       │    ├─► forward() → orig_cb_hidden [L, 16, 1024, d]  │
│       │                   │                                   │
│       │                   └─► select layers [10,20]          │
│       │                       → orig_cb_target [2, 16, 1024, d]│
│       │                                                       │
│       └──► model.enable_adapter()                            │
│            └─► forward() → lora_cb_hidden [L, 16, 1024, d]  │
│                           │                                   │
│                           └─► select layers [10,20]          │
│                               → lora_cb_target [2, 16, 1024, d]│
└──────────────────────────────────────────────────────────────┘

Step 3: Loss Computation
┌──────────────────────────────────────────────────────────────┐
│  retain_loss = L2_distance(lora_retain_hidden, orig_retain)  │
│  cb_loss = inner_product(lora_cb_target, orig_cb_target)     │
│  total_loss = β_retain * retain_loss + β_cb * cb_loss        │
└──────────────────────────────────────────────────────────────┘

Step 4: Backward Pass
┌──────────────────────────────────────────────────────────────┐
│  total_loss.backward()                                        │
│  optimizer.step()  ← Only updates LoRA parameters (Δθ)       │
│  optimizer.zero_grad()                                        │
└──────────────────────────────────────────────────────────────┘

Memory usage per step:
  - 5 forward passes total (2 retain, 2 CB, 1 val)
  - Hidden states cached for gradient computation
  - ~4x memory vs. standard fine-tuning
  - Mitigated by: gradient checkpointing, layer dropping
```

---

## Harmfulness Probe Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│              Probe Training (Separate from CB Training)              │
└─────────────────────────────────────────────────────────────────────┘

Training Data:
  - Positive (harmful): 1,200 examples
  - Negative (safe): 2,400 examples

Process:
┌──────────────────────────────────────────────────────────────┐
│  Input: "How to make a bomb?"                                │
│     │                                                         │
│     ▼                                                         │
│  Model forward pass (frozen)                                 │
│     │                                                         │
│     ▼                                                         │
│  Extract hidden state at layer 15, last token               │
│     h = hidden_states[15][:, -1, :]  [batch, hidden_dim]    │
│     │                                                         │
│     ▼                                                         │
│  ┌──────────────────────────────────────────┐               │
│  │  Probe Options:                          │               │
│  │                                           │               │
│  │  1. Linear:                               │               │
│  │     logits = W @ h + b                    │               │
│  │                                           │               │
│  │  2. MLP:                                  │               │
│  │     h1 = ReLU(W1 @ h + b1)               │               │
│  │     logits = W2 @ h1 + b2                │               │
│  │                                           │               │
│  └──────────────────────────────────────────┘               │
│     │                                                         │
│     ▼                                                         │
│  sigmoid(logits) → probability of harmfulness                │
│     │                                                         │
│     ▼                                                         │
│  if prob > threshold (0.999...): HARMFUL                     │
│  else: SAFE                                                  │
└──────────────────────────────────────────────────────────────┘

Performance:
  - Training accuracy: 99.98%
  - HarmBench detection: 69-93% (varies by attack type)
  - False positive rate: 3.2% (XSTest)

Usage:
  - Can be used for monitoring CB model in production
  - Can guide which examples need stronger intervention
  - NOT required for inference (CB model works standalone)
```

---

## Comparison: Before vs After Circuit Breakers

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Base Model (Before)                               │
└─────────────────────────────────────────────────────────────────────┘

Input: "How to make a bomb?"
  │
  ▼
Layer 10: [semantic representation of bomb-making]
  │
  ▼
Layer 20: [detailed harmful knowledge activated]
  │
  ▼
Output: "To make a bomb, you need... [harmful instructions]"

Representation:
  h_10 = [0.3, -0.5, 0.8, 0.2, ...]  ← Points to "harm" direction
  h_20 = [0.6, 0.1, -0.4, 0.9, ...]  ← Activates detailed knowledge


┌─────────────────────────────────────────────────────────────────────┐
│                Circuit Breaker Model (After)                         │
└─────────────────────────────────────────────────────────────────────┘

Input: "How to make a bomb?"
  │
  ▼
Layer 10: [ALTERED representation - orthogonal to harmful direction]
  │
  ▼
Layer 20: [ALTERED representation - orthogonal to detailed knowledge]
  │
  ▼
Output: "I cannot help with that request."

Representation:
  h_10 = [-0.2, 0.4, 0.1, -0.7, ...]  ← Orthogonal to original
  h_20 = [0.1, -0.8, 0.3, 0.2, ...]   ← Orthogonal to original

Cosine similarity:
  cos(h_10_orig, h_10_cb) ≈ 0.2  (low similarity)
  cos(h_20_orig, h_20_cb) ≈ 0.1  (very low similarity)


For safe input: "What's the weather?"
  h_10_cb ≈ h_10_orig  (cos ≈ 0.99)
  h_20_cb ≈ h_20_orig  (cos ≈ 0.99)
  → Normal behavior preserved!
```

---

## Implementation Checklist Flowchart

```
START
  │
  ▼
[ ] Do you have a base instruction-tuned model?
  │
  ├─ No ──► Fine-tune base model first on general instructions
  │
  └─ Yes
      │
      ▼
[ ] Do you have harmful prompt data with DETAILED responses?
  │
  ├─ No ──► Generate using strong model or curate dataset
  │
  └─ Yes
      │
      ▼
[ ] Do you have safe conversation data?
  │
  ├─ No ──► Use UltraChat, OpenOrca, or similar
  │
  └─ Yes
      │
      ▼
[ ] Configure LoRA
      │
      ├─ Set layers_to_transform = [0..max_target_layer]
      ├─ Set r=16, alpha=16
      └─ Target modules: q_proj, k_proj, v_proj, o_proj
      │
      ▼
[ ] Implement custom loss
      │
      ├─ Retain loss: L2 distance
      ├─ CB loss: negative inner product
      └─ Coefficient scheduling
      │
      ▼
[ ] Add attention masking
      │
      └─ Mask padding in loss computation
      │
      ▼
[ ] Enable memory optimizations
      │
      ├─ Gradient checkpointing
      ├─ Layer dropping
      └─ Explicit gc.collect()
      │
      ▼
[ ] Train for ~150 steps
      │
      └─ Monitor retain & CB cosine similarity
      │
      ▼
[ ] Test qualitatively
      │
      ├─ Harmful prompts → Should refuse
      ├─ Borderline prompts → Should respond
      └─ Normal prompts → Should respond normally
      │
      ▼
[ ] Benchmark quantitatively
      │
      ├─ HarmBench (attack success rate)
      └─ XSTest (false positive rate)
      │
      ▼
[ ] Merge LoRA weights
      │
      └─ model = model.merge_and_unload()
      │
      ▼
[ ] Deploy
      │
      └─ Serve like normal model (no runtime intervention)
      │
      ▼
DONE
```
