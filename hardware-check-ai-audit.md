# Deep Audit & Fact-Check: Hardware Check AI
**Target:** https://compatible-ai-zeta.vercel.app/ · **Audit date:** 2026-09-15 · **Method:** full-site crawl, independent replication of every calculation, and source-verified fact-checking of all 69 model entries, 7 hardware targets, 9 cloud price cards, and all functional flows (share, reset, compare, recommend, advanced settings).

---

## Verdict: DO NOT GO LIVE in the current state.

The engineering shell is good — clean design, a genuinely useful concept, and a calculation engine that is **mathematically exact for dense models**. But the tool's core promise is "can I run this model on this hardware?", and right now that answer is **catastrophically wrong for most MoE models** (up to 33× underestimates), **wrong for ~40% of the established model entries**, the **Share feature is functionally dead**, several headline prices are stale or fabricated-looking, and there are **no disclaimers, no legal pages, and no data dates**. A user could literally be told a 671B-parameter model fits on a 24 GB consumer card.

---

# 1. What IS correct (verified)

| Area | Finding | Evidence |
|---|---|---|
| Core inference math (dense models) | Exact. Weights = params × bits/8 (8.0B INT4 = 4.00 GB ✓). KV cache formula verified to the 3rd decimal: 2×32 layers×8 KV heads×128 dim×4096 ctx×2 bytes = 0.537 GB → shown 0.54 ✓. Totals internally consistent (8.87 ≈ 8.9 shown). | Independently replicated |
| Llama 3 8B architecture panel | All correct: 32 layers, hidden 4096, 32 query heads, 8 KV heads, vocab 128,256. | HF config.json |
| GGUF Q4_K_M handling | Selecting GGUF adds realistic overhead (4.00→4.50 GB) and switches runtime label to llama.cpp. Quant metadata estimate (0.80 GB) matches Q4_K_M theory (~0.85 GB). | Replicated |
| GPU core specs | RTX 4090 (24 GB / 1,008 GB/s / PCIe 4.0), RTX 3090 (24 / 936), RTX 3060 (12 / 360), Apple M3 Max 128 GB (400 GB/s) — all exact. A100/H100 numbers correct for the SXM variants. | NVIDIA/Apple specs |
| Well-known model param counts | Correct (<2% error): Llama 3 8B (8.0B) & 70B (70.0B), Mistral 7B (7.2B), Mixtral 8x7B (46.7B — the *correct* total-params convention), Qwen2 72B (72.7B), Qwen 72B, Phi-3 Medium (14.0B), Gemma 2 9B (9.2B), OLMo 2 1B (1.5B), gpt2-large (774M), MiniMax M2.7 (226.8B ≈ 229B total), PowerMoE-3b (3.3B total). | HF model cards / papers |
| Speed estimates (ballpark) | "~120 tok/s" for 8B INT4 on RTX 4090 is defensible (measured 100–150 across llama.cpp/vLLM). "~10.1 s/step" for QLoRA 8B on 4090 is plausible-conservative for vanilla HF Trainer. | Published benchmarks |
| Cloud-tab arithmetic | Monthly = hourly × 720 ✓; break-even = hardware ÷ monthly ✓ (internally consistent — the *inputs* are the problem, not the math). | Replicated |
| Terminology glossary | Open-source vs open-weight distinction (with correct OLMo/Pythia vs Llama/Mistral/Qwen examples), MQA/GQA/MHA, TP/PP/FSDP, KV cache, quantization, BF16 — technically sound and well written. | Review |
| Basic interactions | Reset works perfectly. Compare enforces its 4-GPU cap. CPU-offload toggle reveals RAM slider and drops speed estimate (~24 tok/s — directionally right). Fine-tune per-card total (9.69 GB) is internally consistent. | Live UI test |

---

# 2. What is NOT correct

## 2.1 CRITICAL — The MoE parameter catastrophe (fatal fit decisions)

For a VRAM calculator, weights memory must be computed from **total** parameters. The site's model database mixes conventions: Mixtral (46.7B), MiniMax M2.7 (226.8B) and PowerMoE (3.3B) show **total** params, while most other MoE rows show something near **active** params — and often a number that matches *neither*. Because the engine is linear in the stored number, these entries produce fit verdicts that are wrong by an order of magnitude:

| Entry as shown | Reality (HF/official) | Error | Real-world consequence |
|---|---|---|---|
| DeepSeek V3 / V3.2 — "38.6B, MOE" | 671B total / 37B active | ~17× under | Tool computes ~19 GB weights at INT4 → suggests a 24 GB card nearly fits. Reality: even 4-bit needs **>350 GB**. Fatal. |
| GLM 5.2 FP8 — "31.3B, DENSE" | 744B total / 40B active **MoE** | ~24× under + wrong architecture | FP8 weights alone ≈ 704 GiB. Site would "fit" it on a single 48 GB card. |
| DeepSeek V4 Flash — "10.5B, MOE" | 284B total / 13B active | ~27× under | 10.5B matches nothing. Also **retired 2026-09-10** (replaced by V4.1-Flash, 552B). |
| Qwen3 Coder Next FP8 — "2.4B, DENSE" | ~80B total / 3B active **MoE** | ~33× under + wrong architecture | FP8 checkpoint ≈ 80 GB on disk. |
| Nemotron 3 Super 120B A12B — "5.7B, MOE" | 120B total / 12B active | ~21× under | BF16 ≈ 240 GB vs implied ~11 GB. |
| GLM 4.7 Flash — "4.4B, DENSE" | 30B total / 3B active **MoE** (zai-org, Jan 2026) | ~7× under + wrong architecture | |
| Nemotron 3.5 Lightning 30B A3B — "1.9B, MOE" | 30B total / 3B active | ~16× under | |
| Qwen3 30B A3B (+ Instruct 2507 + Coder FP8) — "2.7B, MOE" | 30.5B total / **3.3B** active | ~11× under | Even the *active* number is wrong (2.7 ≠ 3.3). |

## 2.2 CRITICAL — Fabricated / conflated model entries

- **"Kimi K3 DSpark (4.3B, DENSE)"** — not a chat model. This is a third-party speculative-decoding **draft head** ecosystem (RadixArk ≈2.2B / Inferact ≈3.6B) for the Kimi K3 flagship (~2.8T MoE). 4.3B matches neither variant, and presenting it as a standalone model is dangerously misleading.
- **"NVIDIA Nemotron 3 Nano 4B BF16 (6.4B, DENSE)"** — conflates two different models: `NVIDIA-Nemotron-3-Nano-30B-A3B` (30B hybrid-Mamba **MoE**) and the older unrelated `Llama-3.1-Nemotron-Nano-4B` (~4.5B dense). 6.4B matches neither; the DENSE label is wrong for the Nemotron 3 generation.

## 2.3 HIGH — Established models with wrong parameter counts

12 of 31 verified established entries are off by >10%; errors flip sign by size bucket (no single correction factor can fix the table):

| Shown | True (HF) | Error | | Shown | True (HF) | Error |
|---|---|---|---|---|---|---|
| gpt2 "190M" | 124M | +53% | | Qwen3 1.7B "2.0B" | 1.72B | +16% |
| opt-125m "190M" | 125M | +52% | | SmolLM2 135M "156M" | 134.5M | +16% |
| distilgpt2 "134M" | 82M | +64% | | pythia-160m "191M" | 162M | +18% |
| Qwen2.5 0.5B "614M" | 494M | +24% | | Qwen2.5 1.5B "1.7B" | 1.54B | +10% |
| Qwen3 32B "29.0B" | 32.76B | −11.5% | | Qwen3 0.6B "649M" | 596M | +9% |

- Two different models ("gpt2", "opt 125m") show the **identical** hardcoded "190M" — a copy-paste data error.
- "Mistral 7B Instruct v0.2 (6.8B)" contradicts the site's own "Mistral 7B (7.2B)" — same architecture (7.24B).
- "dolphin 2.9.1 yi 1.5 34b (31.7B)" — true 34.4B (−8%); at FP16 that flips borderline 64 GB-class decisions.
- Qwen3 4B/8B/14B, Qwen2.5 3B–32B rows are 3–7% off — wrong side of borderline calls at exactly the sizes this tool is for.

## 2.4 CRITICAL — Share Link is functionally dead

The app **never restores state from the URL**. Any shared link (any model, GPU, quant, even `scenario=compare`) boots to the default Llama-3-8B / RTX 4090 / INT4 state and the app **rewrites the URL to defaults**. The URL is write-only. One of the product's headline features (the header Share button + "Share Link" CTA) delivers nothing. Clicking Share also produced repeated browser-transport crashes in testing (suspected clipboard-permission exception) — investigate manually.

## 2.5 HIGH — Compare tab data bugs

- **Estimated TPS is placeholder data:** RTX 4090 = 120 tok/s, but RTX 3090, A100 80GB **and H100 80GB all show "40 tok/s"** — an H100 (3,350 GB/s) shown as 3× slower than a 4090 is absurd and will be screenshotted and mocked.
- **"Cost to Rent / Hr" row renders blank** — values exist in the DOM but are invisible in the table.

## 2.6 HIGH — Pricing & availability claims

- **H100 "Est. Hardware Cost ~$15,000"** — realistic 2026 street price $18k–31k used / $25k–40k new. Understated 40–100% → corrupts every rent-vs-buy conclusion.
- **RTX 3090 and RTX 4090 both "~$1500"** — 3090 used is ~$700–1,000 (overstated ~2×); 4090 used is $2,000–2,800 (understated ~40–90%). The two errors cancel in the comparison, hiding a real ~2× cost difference. The site's own /recommend page then prices a 3090 at **$700** — contradicting its own cloud tab.
- **Stale rental rates:** RunPod H100 $4.69 (actual pods $2.69–3.29; $4.69 is the serverless flex rate), RunPod A100 $1.89 (actual $1.39–1.59), Vast A100 $1.50 (typical $0.50–0.90), Vast 3090 $0.28 (median ~$0.19). Systematically overstates cloud cost → biases users toward buying.
- **"High/Medium/Low AVAIL" badges are fiction.** No API feed; availability is the most volatile variable in GPU rental (Lambda was out of stock on 5 of 8 GPU models on audit day). Undated hardcoded badges presented as live telemetry.

## 2.7 MEDIUM — Calculation & labeling defects

- **LoRA memory ~3–4× overestimated:** for the displayed config (r=16, q+v), true gradient+AdamW memory ≈ 0.10 GB; site shows 0.30 GB. Makes the tool misleadingly conservative on fine-tuning.
- **"KV Cache" shown in fine-tune mode** — training has no KV cache; that 2.15 GB is activation/logit memory. Conceptually wrong label in a tool whose /terminology page teaches the difference.
- **EXL2 runtime labeled "Supported • vLLM"** — EXL2 is an ExLlamaV2 format; vLLM does not run it.
- **CPU offload wording bug:** with offload on, the verdict reads "needs 8.9 GB of the RTX 4090's **32 GB** usable planning memory" — it substitutes system RAM for the GPU's 24 GB VRAM.
- **Multi-GPU state leak:** GPUs selected in Compare stay selected in Inference; the verdict splits weights across cards (4.00→1.00 GB) but still says "…of the RTX 4090, RTX 3090, A100 80GB, H100 80GB's **24 GB**" — ungrammatical, wrong capacity, and inflated ~216 tok/s.
- **AWQ/FP8 listed as separate models** — quantization changes file size, not param count. Duplicate rows (e.g., Qwen2.5 7B + its AWQ twin) that don't change results are noise, and FP8 rows with wrong param counts are actively harmful.

## 2.8 MEDIUM — /recommend page logic

At $1,500, **every one of the six use cases recommends identical hardware ("2x RTX 3090, $1,400")** — including "Edge / Lightweight" and "Creative & Roleplay," which get a dual-GPU 48 GB rig for an 8B model that needs ~9 GB on one card. The "Why this loadout?" text is a template that ignores the mismatch. The budget slider couldn't be moved by automation (verify keyboard/touch accessibility), and `budget=` URL params are ignored (same broken restore as Share).

## 2.9 LOW — Terminology page factual errors

- "**exponentially** more VRAM" with more parameters — false; VRAM scales **linearly** with parameter count.
- "AdamW consumes 2–3× more VRAM than the weights" — understated/misframed: FP32 momentum+variance alone = 4× BF16 weights; ≈6× with FP32 master weights.
- "RAG is **infinitely** cheaper and more accurate than fine-tuning" — hyperbole, not fact.
- Gradient checkpointing "20% slower" — typical figures are ~25–40%; minor.

## 2.10 Missing entirely

- **No footer, no privacy policy, no terms, no contact/about, no GitHub link** on any route.
- **No disclaimer** that all figures are estimates; **no "data as of" date** anywhere despite volatile pricing.
- Identical `<title>` on all three routes; no per-page SEO.
- Hardware list frozen in 2023: **no RTX 5090/5080/5070, no H200, no L40S, no B200, no AMD (MI300X, 7900 XTX), no M4 Pro/Ultra** — a "2026 hardware planner" missing the 2026 hardware.
- Non-generative models (Embedding/Reranker) produce meaningless "fits / ~120 tok/s" verdicts in an inference calculator.

---

# 3. Required changes BEFORE go-live (in priority order)

## P0 — Launch blockers (fix or do not ship)
1. **Rebuild the model database from source.** For every model store *both* `params_total` and `params_active` (MoE), pulled from Hugging Face config/safetensors — never hand-typed. Compute weight memory from **total** params; use active params only for speed estimates. Add a CI check that diffs your DB against the HF API.
2. **Delete or fix the fabricated entries:** "Kimi K3 DSpark" (draft head, not a model), "NVIDIA Nemotron 3 Nano 4B BF16" (two models conflated). Replace retired DeepSeek V4 Flash with V4.1-Flash.
3. **Correct all wrong param counts** (Section 2.3 table; at minimum every >2% error). Fix the Mistral-7B-Instruct self-contradiction.
4. **Fix Share/URL state restore** — hydrate state from query params on load; stop force-rewriting the URL. Until fixed, remove the Share buttons. Investigate the clipboard-permission crash on Share click.
5. **Fix Compare-tab TPS** with a defensible per-GPU model (e.g., bandwidth-scaled) — no placeholder "40 tok/s". Fix the invisible Cost-to-Rent row.
6. **Remove or source every price.** Either wire live APIs (RunPod/Vast/Lambda) or label every price "snapshot, [date]" with a source link. Correct H100 (~$25k+), RTX 3090 (~$700–1,000), RTX 4090 (~$2,000–2,800). Delete the fake AVAIL badges or relabel as dated "typical liquidity" notes.
7. **Fix multi-GPU handling end-to-end:** don't leak Compare selections into Inference; when multiple GPUs are selected, compute total VRAM and per-card split correctly and write the sentence grammatically.
8. **Add the legal/trust chrome:** footer with disclaimer ("estimates only, verify before purchasing"), data-as-of date, contact/about; privacy policy + terms (mandatory if you add analytics or any data collection); per-route `<title>`/meta descriptions.

## P1 — Should fix before launch
9. Fix EXL2 runtime label (ExLlamaV2/TabbyAPI, not vLLM).
10. Fix CPU-offload verdict wording (keep GPU VRAM and system RAM as separate pools).
11. Rebuild /recommend logic per use case (Edge should bias single small GPU; Production API should bias throughput/VRAM headroom); kill the template "why" text or make it data-driven; reconcile its prices with the cloud tab.
12. Fix the LoRA memory formula (grads = LoRA params × 2–4 B; AdamW = × 8 B FP32) and relabel training "KV Cache" → activation memory.
13. Terminology corrections: linear (not exponential) VRAM scaling; accurate AdamW multiplier (4×/6×); drop "infinitely".
14. Remove Embedding/Reranker models from the inference calculator (or add a dedicated embedding-throughput mode); de-duplicate AWQ/FP8 rows into a quantization selector on one canonical model entry.

## P2 — Fast follow
15. Sort/group the 69-model dropdown by family + vendor, add type-to-filter; show total **and** active params on MoE rows ("671B total / 37B active").
16. Show a source + "verified on [date]" tooltip per model and per GPU.
17. Add per-page OG/social cards and a sitemap/robots.txt.

---

# 4. What to add to make it genuinely better

1. **"Paste a Hugging Face repo" mode** — fetch config.json, parse architecture, compute instantly. This single feature makes you better than every static competitor and immune to stale-DB complaints.
2. **2026 hardware + more of it:** RTX 5090 32 GB (1,792 GB/s), 5080/5070 Ti, RTX 6000 Pro 96 GB, H200 141 GB, L40S, B200, AMD MI300X / RX 7900 XTX, M4 Pro/Ultra. Multi-GPU arrays as first-class targets (2×/4× 4090 with TP overhead modeled).
3. **Honest ranges, not point estimates:** show "≈ 8.9 GB (range 8.2–10.5 depending on runtime)" and a per-runtime picker (llama.cpp / vLLM / ExLlamaV2 / MLX) since overhead differs. This converts your biggest liability (estimate error) into a trust feature.
4. **Speed model v2:** bandwidth-derived tok/s with per-runtime efficiency factors, plus TTFT/prefill estimates and a throughput-vs-batch curve for the Production API persona.
5. **Live-ish pricing:** scheduled job refreshing RunPod/Vast/Lambda rates daily, displayed with timestamps — the "AVAIL" badges become real if you can show offer counts from Vast's API.
6. **Sharable result cards:** once share-state works, generate an OG image ("Llama 3 8B INT4 on RTX 4090: ✅ 8.9 GB / ~120 tok/s") — free marketing on X/Reddit/Discord, the exact communities that buy GPUs.
7. **SEO landing pages** for "Can I run [model] on [GPU]?" — your calculator already generates the answer; render the top 500 combos as static pages. This is how this tool gets found.
8. **Fine-tune depth:** full fine-tuning mode (not just LoRA) with optimizer/gradient/master-weight math (your terminology page already teaches it), ZeRO/FSDP multi-GPU estimates, and Unsloth-style "optimized stack" toggle so users see the 3–10× speedup reality.
9. **Feedback loop:** a one-click "Was this estimate right for you?" widget + privacy-respecting analytics to learn which estimates miss.
10. **Quality gates:** unit tests for every formula against known-good fixtures (Llama-3-8B KV = 0.537 GB @4k, etc.), the HF-API DB diff check from P0-1, and Playwright tests for Share-restore and Compare TPS so these bugs can't regress.

---

## Appendix — severity tally
- **Fatal (wrong fit/buy decisions):** 8 MoE model entries (up to 33× under), Share Link dead, H100/3090/4090 hardware costs, Compare TPS placeholders.
- **Wrong but bounded:** 12 established models >10% off, AWQ/FP8 duplicate rows, LoRA overestimate, stale rental rates, fake availability badges.
- **Cosmetic/UX:** EXL2 label, offload wording, multi-GPU sentence, recommend template logic, terminology overstatements, missing footer/SEO.

*All fact-check verdicts cite Hugging Face model cards/configs, official vendor announcements, and dated 2026 price trackers; calculation claims were independently replicated.*
