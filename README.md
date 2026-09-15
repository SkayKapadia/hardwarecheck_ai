# Hardware Check AI

An advanced hardware planner, calculator, and visualization tool for open-weight AI models. Instantly check whether models fit comfortably on your local hardware, analyze exact VRAM requirements (Weights, KV Cache, Metadata, Activations), estimate generation speeds, plan fine-tunes, compare GPUs, and weigh renting vs. buying.

## Features

- **Verified Model Catalog:** 50+ models (Llama, Qwen, Mistral, DeepSeek, GLM, Nemotron, Gemma, Phi, and more) with parameter counts and architecture specs (layers, hidden size, attention/KV heads) verified against Hugging Face configs and safetensors. MoE models store **total and active parameters** — memory is computed from total, speed from active. Every entry carries a source repo and verification date.
- **Import Any Hugging Face Repo:** Paste a repo id or URL and the app fetches `config.json` + safetensors metadata to compute requirements instantly — no waiting for the catalog to add a model.
- **VRAM Breakdown:** Exact memory consumption split by Weights, KV Cache, Quantization Metadata, and Context Activations, shown as an honest range ("≈ 8.9 GB, 8.3–10.5 depending on runtime") with a stacked bar chart.
- **Hardware Selection:** 21 GPU targets — from RTX 3060/4090/5090 to A100/H100/H200/B200, AMD MI300X, and Apple Silicon (M3 Max through M5 Max) — including multi-GPU tensor-parallel arrays with per-card splits and CPU-offload planning (VRAM and system RAM as separate pools).
- **Quantization Simulator:** See the immediate impact of `FP16`, `INT8`, `INT4`, `AWQ`, `GGUF Q4_K_M`, and `EXL2`, each mapped to its actual runtime (llama.cpp, vLLM, ExLlamaV2).
- **Fine-Tune Planner:** QLoRA/LoRA memory math with real trainable-parameter counts, gradient and AdamW optimizer states, activation memory, and rough seconds-per-step estimates.
- **GPU Compare:** Side-by-side VRAM fit, bandwidth-derived speed estimates (measured benchmarks where available), and cheapest rental rates across providers.
- **Cloud Rent vs. Buy:** Dated rental-rate snapshots (RunPod, Vast.ai, Lambda) against realistic hardware street prices, with monthly cost and break-even months.
- **Budget Recommendations:** `/recommend` builds per-use-case loadouts (edge, coding, roleplay, RAG, agents, production API) scored on fit, budget utilization, and headroom, with data-driven reasoning.
- **Shareable Links & OG Cards:** Every calculator state serializes to the URL and restores on load; `/share?model=…&gpu=…` links render dynamic Open Graph verdict cards.
- **Architecture Schematics:** Dynamic visualizations of transformer block layouts (Tokens → Embedding → Transformer Blocks → LM Head) from exact Hugging Face configuration parameters.

## Tech Stack

- Next.js 14 (App Router) & React
- Zustand (client-side, URL-synced state)
- Tailwind CSS
- Vitest (calculation-engine unit tests + data-invariant checks)
- Lucide React (icons)

## Getting Started

1. Clone the repository
2. Install dependencies: `npm install`
3. Run the development server: `npm run dev`
4. Open [http://localhost:3000](http://localhost:3000)

Other scripts: `npm test` (unit tests), `npm run lint`, `npm run build`.

For deployment, set `NEXT_PUBLIC_SITE_URL` to the production origin so sitemap and Open Graph URLs resolve correctly.

## Data & Accuracy

All figures are **estimates for planning purposes** — verify current pricing and compatibility before purchasing.

- Model data verified against Hugging Face on **2026-09-15**; hardware prices and cloud rates are **September 2026 snapshots**.
- Speed estimates are bandwidth-derived unless a measured benchmark exists for the exact model/GPU/quant combination.

Model and GPU data lives in static JSON files in `src/data/`. To verify parameter counts against Hugging Face:

```bash
node scripts/verify_models.mjs
```

To fetch and update the latest models from Hugging Face:

```bash
node scripts/fetch_models.mjs
```
