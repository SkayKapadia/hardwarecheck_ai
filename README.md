# Hardware Check AI

An advanced hardware planner, calculator, and visualization tool for open-weight AI models. Instantly check whether models fit comfortably on your local hardware array, analyze exact VRAM requirements (Weights, KV Cache, Metadata, Activations), and estimate generation speeds.

## Features
- **Extensive Model Catalog:** Uses official architecture specs (layers, hidden size, attention heads) fetched dynamically from Hugging Face for 75+ popular models (Llama 3, Qwen, Mistral, etc.).
- **VRAM Breakdown:** Calculates exact memory consumption split by Weights, KV Cache, Quantization Metadata, and Context Activations using a visually accurate Horizontal Stacked Bar Chart.
- **Hardware Selection:** Target multiple GPUs to simulate tensor parallel arrays.
- **Quantization Simulator:** See the immediate impact of `FP16`, `INT8`, `INT4`, `AWQ`, `GGUF`, and `EXL2` on memory footprints.
- **Architecture Schematics:** Dynamic visualizations of transformer block layouts (Tokens -> Embedding -> Transformer Blocks -> LM Head) based on exact Hugging Face configuration parameters.

## Tech Stack
- Next.js 14 (App Router)
- React & Zustand (Client-side URL-synced state management)
- Tailwind CSS (Premium high-contrast Light Theme)
- Lucide React (Icons)

## Getting Started

1. Clone the repository
2. Install dependencies: `npm install`
3. Run the development server: `npm run dev`
4. Open [http://localhost:3000](http://localhost:3000)

## Modifying Data
Model and GPU data is stored in static JSON files in `src/data/`. 
To automatically fetch and update the latest models from Hugging Face:
```bash
node scripts/fetch_models.mjs
```
