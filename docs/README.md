# Documentation

This directory contains comprehensive documentation for representation-based circuit breakers.

## Quick Navigation

### Getting Started
- **[implementation_guide.md](implementation_guide.md)** - Step-by-step implementation guide (30 min read)

### Research & Technical Details
- **[GRAYSWANAI_SUMMARY.md](GRAYSWANAI_SUMMARY.md)** - Executive summary of GraySwanAI's approach (15 min)
- **[grayswanai_technical_analysis.md](grayswanai_technical_analysis.md)** - Deep technical analysis (60 min)
- **[architecture_overview.md](architecture_overview.md)** - Visual diagrams and architecture
- **[key_code_snippets.md](key_code_snippets.md)** - Copy-paste ready code examples
- **[GRAYSWANAI_INDEX.md](GRAYSWANAI_INDEX.md)** - Navigation guide for all GraySwanAI docs

### References
- **[LITERATURE.md](LITERATURE.md)** - Academic papers and research references

## Reading Order

### For Quick Implementation (30 minutes)
1. [implementation_guide.md](implementation_guide.md) - Get started immediately

### For Deep Understanding (2-3 hours)
1. [GRAYSWANAI_SUMMARY.md](GRAYSWANAI_SUMMARY.md) - Overview (15 min)
2. [grayswanai_technical_analysis.md](grayswanai_technical_analysis.md) - Technical details (60 min)
3. [architecture_overview.md](architecture_overview.md) - Visual understanding (30 min)
4. [key_code_snippets.md](key_code_snippets.md) - Code examples (30 min)

### For Research Context
- [LITERATURE.md](LITERATURE.md) - All research papers and citations

## Key Concepts

### Representation-Based Circuit Breakers
- Based on **arXiv:2406.04313** (GraySwanAI)
- Modifies internal model representations using LoRA
- Dual-loss training: preserve safe behavior, break harmful behavior
- Zero runtime overhead after training
- Intervenes at middle layers (10, 20) where semantic information resides

### How It Works
1. Apply LoRA adapters to middle layers
2. Train with dual loss:
   - **Retain Loss** (L2): Keep safe representations similar
   - **CB Loss** (negative inner product): Make harmful representations orthogonal
3. Dynamic coefficient scheduling: early CB focus → late retain focus
4. Merge LoRA weights for zero inference overhead
