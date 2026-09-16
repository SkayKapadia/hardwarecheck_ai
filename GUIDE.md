# How to Use Hardware Check AI

**Live tool:** [compatible-ai-zeta.vercel.app](https://compatible-ai-zeta.vercel.app)

Hardware Check AI answers the question every builder hits before downloading a model or buying a GPU:

> **"Can I actually run this model on my hardware — and if not, what do I need?"**

It computes exact VRAM requirements (weights, KV cache, quantization overhead, activations), estimates generation speed, plans fine-tunes (LoRA *and* full training with ZeRO/FSDP), compares GPUs side by side, and tells you whether renting cloud GPUs beats buying — all in the browser, no sign-up.

---

## The 60-Second Quick Start

1. Open the tool. You land on the **Inference** tab.
2. Pick a model (e.g. *Llama 3 8B*) and a GPU (e.g. *RTX 4090*).
3. Read the verdict:

   > ✅ **COMFORTABLE FIT** — Llama 3 8B at INT4 is estimated to need **≈ 8.9 GB** (8.3–10.5 depending on runtime) of the RTX 4090's 24 GB.

4. Drag the **Context Length** slider from 4K to 128K and watch the KV cache segment grow in real time.
5. Click **Share** in the header — the link you copied restores this exact configuration for whoever opens it.

That's the core loop. Everything below is depth.

---

## The Four Tabs

### 1. Inference — "Will it run?"

The main calculator.

| Control | What it does |
|---|---|
| **Model picker** | 50+ verified models grouped by family (Llama, Qwen, DeepSeek, Mistral, GLM, Gemma, Nemotron…), with type-to-filter. MoE models show both counts: *671B total / 37B active*. |
| **Import** (above the picker) | Paste **any** Hugging Face repo (`meta-llama/Llama-3.1-8B` or a full URL). The tool fetches the model's real config + checkpoint size and computes instantly — you're never limited to the built-in catalog. |
| **Quantization** | FP16, INT8, INT4, AWQ, GGUF Q4_K_M, EXL2 — each labeled with the runtime that actually runs it (vLLM, llama.cpp, ExLlamaV2). |
| **Context Length / Batch Size** | Sliders that drive the KV cache and activation math — the hidden memory cost of long prompts and concurrent users. |
| **CPU Offload** | Splits layers between GPU VRAM and system RAM when the model doesn't fit, with an honest (large) speed penalty shown. |
| **Hardware array** | Select multiple GPUs to simulate a tensor-parallel rig — the verdict shows per-card load, combined VRAM, and which card is the binding constraint. |

**Reading the breakdown bar:** Model Weights → KV Cache → Quantization Metadata → Activations → OS Reserve → Safety Margin. Totals are shown as an **honest range** (e.g. *8.3–10.5 GB*) because real runtimes differ — plan against the top of the range.

### 2. Fine-Tune — "Can I train on it?"

Two modes:

- **LoRA / QLoRA** — set rank, alpha, and target modules; the tool computes the *actual* trainable parameter count (e.g. r=16 on q+v projections of Llama 3 8B = **8.4M params, 0.10% of the model**) plus real gradient (4 B/param) and AdamW optimizer (8 B/param) memory. This is why an 8B model fine-tunes on a single consumer card.
- **Full fine-tune** — the full 16-bytes-per-parameter reality (BF16 weights + gradients + FP32 master weights + AdamW states). Pick a **sharding strategy** — ZeRO-1, ZeRO-2, ZeRO-3, or FSDP — and the tool shows per-card memory across your GPU array and a prominent *"Minimum required: N× GPU at ZeRO-2"* line. An 8B full fine-tune needs ~128 GB before activations; the tool shows you exactly how many cards make it fit.

Both modes estimate seconds-per-step and apply gradient-checkpointing trade-offs (~25–40% slower, up to ~80% less activation memory).

### 3. Compare — "Which GPU?"

Select up to 4 GPUs → a side-by-side table of VRAM, bandwidth, estimated tokens/sec for *your selected model*, and the cheapest rental rate per provider. Speed estimates are bandwidth-derived and benchmark-anchored — no two GPUs show the same number unless they genuinely perform alike.

### 4. Cloud — "Rent or buy?"

Live-refreshed rental rates (updated daily from the Vast.ai public API; liquidity badges reflect **real offer counts**, not guesses) versus realistic street prices, with monthly cost (hourly × 720) and **break-even months** for each card. Rule of thumb the table makes obvious: under ~12 months of steady use, rent; beyond that, buy.

---

## The Budget Recommender

[compatible-ai-zeta.vercel.app/recommend](https://compatible-ai-zeta.vercel.app/recommend)

Set a budget, pick a use case — *Edge/Lightweight, Creative & Roleplay, Coding Assistant, RAG/Knowledge, AI Agents, Production API* — and get a scored loadout: which GPU(s), which model, and **why**, in data-driven bullets ("Fits Qwen3 8B INT4 (≈4.6 GB) with 11.4 GB headroom for 8K context · 93% of budget · single card — no multi-GPU complexity"). Each use case has different logic: Edge penalizes multi-GPU rigs, Production demands 2× VRAM headroom and datacenter bandwidth.

---

## Sharing & Links

- **Share button** — every slider, tab, and selection serializes into the URL. Send the link; the recipient opens the tool in your exact state.
- **Verdict cards** — `/share?model=…&gpu=…` links render a dynamic OG image (model, GPU, ✅/❌ verdict, GB, tok/s) when pasted into X, LinkedIn, Discord, or Slack.
- **Answer pages** — 530 static pages like `/run/llama3-8b/rtx4090` give instant verdicts with alternatives ("Doesn't fit? Here's what works") and one-click jumps into the full calculator.

---

## How the Numbers Are Made (and How Much to Trust Them)

- **Weights** = total parameters × bits/8 — exact math from Hugging Face-verified configs. MoE models always size memory from **total** parameters (a 671B model needs ~336 GB even at 4-bit, no matter how few params are "active" per token).
- **KV cache** = 2 × layers × KV heads × head dim × context × batch × 2 bytes — verified to the third decimal against published architectures.
- **Speed** = memory bandwidth × runtime efficiency ÷ bytes-per-token, with measured benchmarks used wherever they exist.
- **Fine-tuning** = real LoRA parameter counting and the classic 16 B/param mixed-precision recipe with ZeRO sharding.
- All data is dated: model specs verified **2026-09-15**, prices are monthly snapshots, cloud rates refresh daily. A CI job re-audits the entire model database against Hugging Face weekly and opens an issue if anything drifts.

Figures are engineering estimates for planning — verify before purchasing hardware.

---

## Tips & FAQ

- **"It almost fits"** → try a lower quant (INT4 → GGUF Q4_K_M saves little; INT4 → INT8 costs double), shorten context, or enable CPU offload as a last resort (expect ~5–10× slower generation).
- **Speed vs. capacity** → bandwidth is everything for single-user generation: an RTX 4090 (1,008 GB/s) beats an A100 (2,039 GB/s) on cost-per-token for small models, but only the A100 fits 70B-class models at usable quants.
- **Apple Silicon** → unified memory means a Mac Studio M5 Ultra (512 GB) runs models no single GPU can — at lower speed but zero cloud bills.
- **Imported models are session-only** — they vanish on reload. The built-in catalog is permanent.
- New to the vocabulary? The **[Terminology](https://compatible-ai-zeta.vercel.app/terminology)** page explains KV cache, quantization, MoE, ZeRO, and more in plain language — the same concepts the calculator uses.

---

*Hardware Check AI is open source. Model data verified against Hugging Face; estimates only — always verify compatibility and pricing before buying.*
