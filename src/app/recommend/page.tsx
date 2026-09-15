/* eslint-disable react/no-unescaped-entities */
"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import gpusJson from "@/data/gpus.json";
import modelsJson from "@/data/models.json";
import {
  GPUSpec,
  ModelSpec,
  PRICING_DATA_AS_OF,
  getModelWeights,
  getKVCache,
  getQuantMetadata,
  getActivationMemory,
  getRuntimeReserve,
  getSafetyMargin,
  getMultiGPUPerCard,
  getBitsPerWeight,
  formatParams,
} from "@/lib/calc";
import { Slider } from "@/components/ui/Slider";
import { Calculator, Cpu, Box, Cloud, DollarSign, CheckCircle2 } from "lucide-react";

const GPUS = gpusJson as GPUSpec[];
const MODELS = modelsJson as unknown as ModelSpec[];

type UseCase = "coding" | "roleplay" | "rag" | "agents" | "edge" | "production";

const USE_CASES: { id: UseCase; label: string }[] = [
  { id: "coding", label: "Coding Assistant" },
  { id: "roleplay", label: "Creative & Roleplay" },
  { id: "rag", label: "Document Analysis (RAG)" },
  { id: "agents", label: "Autonomous Agents" },
  { id: "edge", label: "Edge / Lightweight" },
  { id: "production", label: "Production API" },
];

const TARGET_QUANT = "INT4";
const MIN_BUDGET = 100;
const MAX_BUDGET = 50000;

interface Combo {
  gpu: GPUSpec;
  count: number;
  name: string;
  price: number;
  totalVram: number;
}

// Single GPUs from gpus.json plus 2x / 4x uniform combos of
// consumer/datacenter/amd cards (Macs don't do multi-GPU rigs).
const COMBOS: Combo[] = (() => {
  const priced = GPUS.filter((g) => typeof g.price === "number" && g.price > 0);
  const combos: Combo[] = priced.map((g) => ({
    gpu: g,
    count: 1,
    name: `1x ${g.name}`,
    price: g.price!,
    totalVram: g.vram,
  }));
  for (const g of priced) {
    if (g.class === "mac") continue;
    for (const n of [2, 4]) {
      combos.push({
        gpu: g,
        count: n,
        name: `${n}x ${g.name}`,
        price: g.price! * n,
        totalVram: g.vram * n,
      });
    }
  }
  return combos;
})();

interface UseCaseProfile {
  ctx: number;
  minTotalVram?: number;
  filterModels: (m: ModelSpec) => boolean;
}

const PROFILES: Record<UseCase, UseCaseProfile> = {
  edge: {
    ctx: 4096,
    filterModels: (m) => m.params <= 9e9,
  },
  roleplay: {
    ctx: 8192,
    filterModels: (m) =>
      m.params <= 13e9 && ["Llama", "Mistral", "Gemma", "Qwen"].includes(m.family),
  },
  coding: {
    ctx: 8192,
    filterModels: (m) =>
      m.name.toLowerCase().includes("coder") || m.family === "Qwen" || m.family === "Llama",
  },
  rag: {
    ctx: 32768,
    filterModels: (m) => m.maxContext >= 32768,
  },
  agents: {
    ctx: 8192,
    minTotalVram: 48,
    filterModels: (m) => m.params >= 30e9,
  },
  production: {
    ctx: 8192,
    filterModels: (m) => m.params >= 7e9,
  },
};

interface FitResult {
  weightsGB: number; // weights + quant metadata, aggregate
  kvGB: number; // KV cache at profile context, aggregate
  totalPerCard: number; // incl. reserve + safety margin
  headroomPerCard: number;
}

function computeFit(model: ModelSpec, combo: Combo, ctx: number): FitResult {
  const rawWeights = getModelWeights(model.params, getBitsPerWeight(TARGET_QUANT));
  const kvCache = getKVCache(model.layers, model.hiddenSize, model.queryHeads, model.kvHeads, ctx, 1, 16);
  const { weightsPerCard, kvPerCard } = getMultiGPUPerCard(rawWeights, combo.count, kvCache, "tensor_parallel");
  const metadata = getQuantMetadata(model.params, TARGET_QUANT);
  const activations = getActivationMemory(ctx, 1, model.hiddenSize, model.layers);
  const reserve = getRuntimeReserve(combo.gpu);
  const used = weightsPerCard + kvPerCard + metadata + activations + reserve;
  const total = used + getSafetyMargin(used);
  return {
    weightsGB: rawWeights + metadata,
    kvGB: kvCache,
    totalPerCard: total,
    headroomPerCard: combo.gpu.vram - total,
  };
}

interface LocalPick {
  type: "local";
  combo: Combo;
  model: ModelSpec;
  fit: FitResult;
  score: number;
}

function recommend(budget: number, useCase: UseCase): LocalPick | { type: "cloud"; reason: string } {
  const profile = PROFILES[useCase];
  const candidates = MODELS.filter(profile.filterModels).sort((a, b) => b.params - a.params);

  if (candidates.length === 0) {
    return { type: "cloud", reason: "No suitable models for this use case." };
  }
  const maxCandidateParams = candidates[0].params;

  let best: LocalPick | null = null;

  for (const combo of COMBOS) {
    if (combo.price > budget) continue;
    if (profile.minTotalVram && combo.totalVram < profile.minTotalVram) continue;
    if (useCase === "edge" && combo.gpu.vram < 10) continue;

    // Biggest candidate model (by TOTAL params — MoE must fully fit) that
    // fits on this combo at the profile's context length.
    let chosen: { model: ModelSpec; fit: FitResult } | null = null;
    for (const m of candidates) {
      const fit = computeFit(m, combo, profile.ctx);
      if (fit.totalPerCard > combo.gpu.vram) continue;
      if (useCase === "production") {
        // Throughput headroom: total VRAM >= 2x the model's full need
        const aggregateNeed = fit.weightsGB + fit.kvGB + combo.gpu.osReserve * combo.count;
        if (combo.totalVram < aggregateNeed * 2) continue;
      }
      chosen = { model: m, fit };
      break;
    }
    if (!chosen) continue;

    let score = 0;

    // Value: prefer using 70-95% of the budget (cheap picks waste capability)
    const util = combo.price / budget;
    score += util < 0.7 ? 40 * (util / 0.7) : 40;

    // Model quality: bigger fitting model is better (log scale so the jump
    // from 8B to 70B matters without swamping every other factor)
    score += 30 * Math.min(1, Math.log10(chosen.model.params + 1) / Math.log10(maxCandidateParams + 1));

    // Comfortable fit: full marks at >= 30% free VRAM per card
    score += 20 * Math.min(1, chosen.fit.headroomPerCard / (chosen.fit.totalPerCard * 0.3));

    // Use-case specific preferences
    switch (useCase) {
      case "edge":
        if (combo.count > 1) score -= 50; // single small GPU strongly preferred
        if (combo.totalVram > 32) score -= 10; // oversized rigs penalized
        break;
      case "roleplay":
        if (combo.count > 1) score -= 40;
        if (combo.gpu.class === "consumer") score += 10;
        break;
      case "coding":
        if (combo.count === 1) score += 5;
        if (combo.totalVram < 16) score -= 15;
        break;
      case "rag":
        break; // 32k-context fit requirement already does the filtering
      case "agents":
        if (combo.count > 1) score += 5;
        break;
      case "production":
        if (combo.gpu.class === "datacenter") score += 25;
        else if (combo.gpu.bandwidth >= 1500) score += 10;
        else if (combo.gpu.class === "consumer") score -= 15;
        break;
    }

    if (!best || score > best.score) {
      best = { type: "local", combo, model: chosen.model, fit: chosen.fit, score };
    }
  }

  if (!best) {
    return { type: "cloud", reason: "No local hardware combination fits this budget and use case." };
  }
  return best;
}

function buildReasons(pick: LocalPick, budget: number, useCase: UseCase): string[] {
  const { combo, model, fit } = pick;
  const ctxK = Math.round(PROFILES[useCase].ctx / 1024);
  const util = Math.round((combo.price / budget) * 100);
  const reasons = [
    `Fits ${model.name} ${TARGET_QUANT} (≈${fit.weightsGB.toFixed(1)} GB) with ${fit.headroomPerCard.toFixed(1)} GB per-card headroom for a ${ctxK}K context`,
    `$${combo.price.toLocaleString()} — ${util}% of your $${budget.toLocaleString()} budget`,
    combo.count === 1
      ? "Single card — no multi-GPU complexity or interconnect overhead"
      : `${combo.count}x tensor-parallel splits weights and KV cache across ${combo.totalVram} GB aggregate VRAM`,
  ];
  if (model.architecture === "moe" && model.activeParams) {
    reasons.push(
      `MoE: only ${formatParams(model.activeParams)} active per token, so speed is closer to a ${formatParams(model.activeParams)} model than a ${formatParams(model.params)} one`
    );
  } else if (combo.gpu.bandwidth >= 2000) {
    reasons.push(
      `${combo.gpu.name}'s ${combo.gpu.bandwidth.toLocaleString()} GB/s bandwidth suits high-concurrency serving`
    );
  }
  return reasons.slice(0, 4);
}

function RecommendInner() {
  const searchParams = useSearchParams();
  const [budget, setBudget] = useState(() => {
    const b = parseInt(searchParams.get("budget") ?? "", 10);
    return Number.isFinite(b) ? Math.min(MAX_BUDGET, Math.max(MIN_BUDGET, b)) : 1500;
  });
  const [useCase, setUseCase] = useState<UseCase>(() => {
    const uc = searchParams.get("useCase");
    return USE_CASES.some((u) => u.id === uc) ? (uc as UseCase) : "coding";
  });

  // Reflect inputs in the URL (no navigation) so the page is shareable
  useEffect(() => {
    const url = new URL(window.location.href);
    url.searchParams.set("budget", budget.toString());
    url.searchParams.set("useCase", useCase);
    if (url.search !== window.location.search) {
      window.history.replaceState({}, "", url.toString());
    }
  }, [budget, useCase]);

  const recommendation = useMemo(() => recommend(budget, useCase), [budget, useCase]);

  return (
    <div className="flex-1 flex gap-8 h-full overflow-hidden p-6 max-w-7xl mx-auto w-full">

      {/* Left Panel: Inputs */}
      <div className="w-[40%] flex flex-col gap-8 bg-card/50 border border-border p-8 rounded-xl overflow-y-auto custom-scrollbar">
        <div>
          <h1 className="text-3xl font-bold text-foreground mb-2 flex items-center gap-3">
            <Calculator className="w-8 h-8 text-primary" />
            Budget Planner
          </h1>
          <p className="text-muted-foreground text-sm leading-relaxed">
            Tell us what you want to do and how much you want to spend. We'll automatically calculate the optimal hardware array and recommend the best model for it.
          </p>
        </div>

        <div className="space-y-4 pt-4 border-t border-border">
          <label className="text-xs font-mono text-muted-foreground uppercase tracking-widest">
            Primary Use Case
          </label>
          <div className="grid grid-cols-2 gap-3">
            {USE_CASES.map((uc) => (
              <button
                key={uc.id}
                onClick={() => setUseCase(uc.id)}
                className={`py-3 px-4 text-sm font-semibold rounded-lg border transition-colors ${
                  useCase === uc.id
                    ? "bg-primary text-primary-foreground border-primary"
                    : "bg-background text-foreground border-border hover:border-primary/50"
                }`}
              >
                {uc.label}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-6 pt-4 border-t border-border">
          <div className="flex justify-between items-center">
             <label className="text-xs font-mono text-muted-foreground uppercase tracking-widest">
               Maximum Budget
             </label>
             <span className="text-lg font-bold text-emerald-600 bg-emerald-500/10 px-3 py-1 rounded flex items-center">
               <DollarSign className="w-4 h-4" /> {budget.toLocaleString()}
             </span>
          </div>
          <Slider
            min={MIN_BUDGET} max={MAX_BUDGET} step={100}
            value={[budget]}
            onValueChange={([v]) => setBudget(v)}
          />
          <div className="flex justify-between text-xs text-muted-foreground font-mono">
            <span>$100</span>
            <span>$50,000+</span>
          </div>
        </div>

        <p className="text-xs text-muted-foreground font-mono mt-auto pt-4 border-t border-border">
          Hardware prices: {PRICING_DATA_AS_OF} snapshot
        </p>
      </div>

      {/* Right Panel: Output */}
      <div className="w-[60%] flex flex-col overflow-y-auto custom-scrollbar pr-2">
        {recommendation.type === "cloud" ? (
          <div className="bg-destructive/10 border border-destructive/30 p-8 rounded-xl flex flex-col items-center justify-center text-center h-full">
            <Cloud className="w-16 h-16 text-destructive mb-4" />
            <h2 className="text-2xl font-bold text-foreground mb-2">Cloud API Recommended</h2>
            <p className="text-muted-foreground max-w-md">
              At a budget of ${budget.toLocaleString()}, it is highly inefficient to purchase local hardware. We recommend renting compute via cloud providers (e.g., RunPod, Together AI) or using managed APIs (e.g., OpenAI, Anthropic).
            </p>
          </div>
        ) : (
          <div className="space-y-8">
             <div className="bg-emerald-500/5 border border-emerald-500/30 p-8 rounded-xl flex items-start gap-4">
                <CheckCircle2 className="w-10 h-10 text-emerald-600 shrink-0 mt-1" />
                <div>
                  <h2 className="text-2xl font-bold text-foreground mb-2">Optimal Loadout Found</h2>
                  <p className="text-muted-foreground">
                    Based on your budget of <strong>${budget.toLocaleString()}</strong>, we scored every hardware combination in our database on model fit, budget utilization, and {useCase} requirements.
                  </p>
                </div>
             </div>

             <div className="grid grid-cols-2 gap-6">
               {/* Hardware Card */}
               <div className="bg-card border border-border p-6 rounded-xl relative overflow-hidden">
                 <div className="absolute top-0 right-0 p-4 opacity-10">
                   <Cpu className="w-24 h-24" />
                 </div>
                 <h3 className="text-xs font-mono text-muted-foreground uppercase tracking-widest mb-4">Hardware Array</h3>
                 <div className="text-2xl font-bold text-foreground mb-1">{recommendation.combo.name}</div>
                 <div className="text-primary font-mono bg-primary/10 inline-block px-2 py-1 rounded text-sm mb-4">
                   {recommendation.combo.totalVram} GB Total VRAM
                 </div>
                 <div className="text-sm text-muted-foreground">
                   Estimated Cost: <strong className="text-foreground">${recommendation.combo.price.toLocaleString()}</strong>
                 </div>
               </div>

               {/* Model Card */}
               <div className="bg-card border border-border p-6 rounded-xl relative overflow-hidden">
                 <div className="absolute top-0 right-0 p-4 opacity-10">
                   <Box className="w-24 h-24" />
                 </div>
                 <h3 className="text-xs font-mono text-muted-foreground uppercase tracking-widest mb-4">AI Model</h3>
                 <div className="text-2xl font-bold text-foreground mb-1">{recommendation.model.name}</div>
                 <div className="text-secondary font-mono bg-secondary/20 inline-block px-2 py-1 rounded text-sm mb-4">
                   {formatParams(recommendation.model.params)} Parameters
                 </div>
                 <div className="text-sm text-muted-foreground">
                   Recommended Format: <strong className="text-foreground">{TARGET_QUANT} Quantization</strong>
                 </div>
               </div>
             </div>

             <div className="bg-background border border-border p-6 rounded-xl">
               <h3 className="text-lg font-bold text-foreground mb-3">Why this loadout?</h3>
               <ul className="list-disc list-inside space-y-2 text-muted-foreground text-sm leading-relaxed">
                 {buildReasons(recommendation, budget, useCase).map((reason, i) => (
                   <li key={i}>{reason}</li>
                 ))}
               </ul>
             </div>
          </div>
        )}
      </div>

    </div>
  );
}

export default function RecommendPage() {
  return (
    <Suspense fallback={null}>
      <RecommendInner />
    </Suspense>
  );
}
