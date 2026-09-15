# Math Audit (Orchestrator's own verification, replicated calculations)

## Inference tab — Llama 3 8B, INT4, ctx 4096, batch 1, RTX 4090
| Component | Site shows | Replicated | Verdict |
|---|---|---|---|
| Model weights | 4.00 GB | 8.0e9 × 4bit = 4.00 GB | ✅ exact |
| KV cache | 0.54 GB | 2×32L×8KV×128d×4096×1×2B = 0.537 GB | ✅ exact |
| Quant metadata | 0.80 GB | GGUF Q4_K_M implies ~0.85 GB | ✅ close |
| Activations / OS reserve / safety | 1.61 / 1.50 / 0.42 GB | heuristics, plausible | ⚠️ undocumented heuristics |
| Total | 8.9 GB | 8.87 GB | ✅ internally consistent |

## Fine-tune tab — Llama 3 8B, INT4, LoRA r=16 α=32, q_proj+v_proj, batch 4, RTX 4090
| Component | Site shows | Replicated | Verdict |
|---|---|---|---|
| Base weights | 4.00 GB | 4.00 GB | ✅ |
| "KV Cache" b=4 | 2.15 GB | 2×32×8×128×4096×4×2B = 2.15 GB | ✅ number, ❌ label — training has no KV cache; this is activation/logit memory |
| Gradient buffers | 0.09 GB | true LoRA grads (6.8M params): 0.014–0.027 GB | ❌ ~3–6× overestimate |
| Optimizer (AdamW) | 0.21 GB | true AdamW states: ~0.055 GB | ❌ ~4× overestimate |
| Est. speed | ~10.1 s/step | realistic for vanilla HF Trainer + bitsandbytes | ✅ plausible, conservative |

## Cloud tab arithmetic
- Monthly = hourly × 720: checks out (0.28×720=201.6≈202 ✅, 3.19×720=2297 ✅)
- Break-even = hw cost ÷ monthly: 1500/202=7.4 ✅, 15000/1080=13.9 ✅
- → Internally consistent; errors are in the INPUTS (stale prices, wrong hw costs), not the math.

## Terminology page — factual statements
| Claim | Verdict |
|---|---|
| "More parameters ... require exponentially more VRAM" | ❌ FALSE — VRAM scales linearly with parameters |
| "8,192 tokens ≈ 6,000 words" | ✅ roughly right (~0.75 words/token) |
| "AdamW consumes 2–3× more VRAM than the weights" | ❌ misleading — FP32 momentum+variance alone = 4× BF16 weights; +fp32 master ≈ 6× |
| "Gradient checkpointing: 20% slower for up to 60% less VRAM" | ⚠️ imprecise (typ. ~25–40% slowdown) |
| "RAG is infinitely cheaper and more accurate than fine-tuning" | ❌ hyperbole ("infinitely") — editorial, not factual |
| Open-source vs open-weight distinction + examples (OLMo/Pythia vs Llama/Mistral/Qwen) | ✅ accurate and well explained |
| MQA/GQA/MHA, TP/PP/FSDP, BF16 vs FP16 descriptions | ✅ technically sound |
