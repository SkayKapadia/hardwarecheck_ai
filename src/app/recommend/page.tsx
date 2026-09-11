/* eslint-disable react/no-unescaped-entities */
"use client";

import { useState, useMemo } from "react";
import gpus from "@/data/gpus.json";
import models from "@/data/models.json";
import { getModelWeights, getKVCache, getQuantMetadata, getActivationMemory, getRuntimeReserve, getSafetyMargin, getMultiGPUPerCard } from "@/lib/calc";
import { Slider } from "@/components/ui/Slider";
import { Calculator, Cpu, Box, Cloud, DollarSign, CheckCircle2 } from "lucide-react";

type UseCase = "coding" | "roleplay" | "research" | "production" | "rag" | "edge" | "agents";

// Pre-defined reasonable GPU combinations prioritizing VRAM/$
const GPU_COMBOS = [
  { id: "1x3060", gpus: ["rtx3060"], name: "1x RTX 3060", price: 280, vram: 12 },
  { id: "1x3090", gpus: ["rtx3090"], name: "1x RTX 3090", price: 700, vram: 24 },
  { id: "2x3090", gpus: ["rtx3090", "rtx3090"], name: "2x RTX 3090", price: 1400, vram: 48 },
  { id: "1x4090", gpus: ["rtx4090"], name: "1x RTX 4090", price: 1600, vram: 24 }, // Faster, but less VRAM/$
  { id: "3x3090", gpus: ["rtx3090", "rtx3090", "rtx3090"], name: "3x RTX 3090", price: 2100, vram: 72 },
  { id: "4x3090", gpus: ["rtx3090", "rtx3090", "rtx3090", "rtx3090"], name: "4x RTX 3090", price: 2800, vram: 96 },
  { id: "1xm3max", gpus: ["m3max-128"], name: "1x Mac M3 Max (Unified)", price: 4000, vram: 128 },
  { id: "1xa100", gpus: ["a100-80gb"], name: "1x A100 80GB", price: 15000, vram: 80 }
];

export default function RecommendPage() {
  const [budget, setBudget] = useState(1500);
  const [useCase, setUseCase] = useState<UseCase>("coding");

  // Logic Engine
  const recommendation = useMemo(() => {
    // 1. Find the best hardware combination within budget
    const affordableCombos = GPU_COMBOS.filter(c => c.price <= budget);
    
    if (affordableCombos.length === 0) {
      return { type: "cloud" as const, reason: "Budget is too low for local inference." };
    }

    // Sort by VRAM descending (prioritize VRAM capacity)
    affordableCombos.sort((a, b) => b.vram - a.vram);
    const bestHardware = affordableCombos[0];

    // 2. Find the best model that fits on this hardware
    // Filter models by use case (loose heuristic based on names/params)
    let candidateModels = models;
    if (useCase === "coding") {
      candidateModels = models.filter(m => m.name.toLowerCase().includes("coder") || m.name.toLowerCase().includes("llama 3") || m.name.toLowerCase().includes("qwen"));
    } else if (useCase === "roleplay") {
      candidateModels = models.filter(m => m.name.toLowerCase().includes("llama") || m.name.toLowerCase().includes("mistral"));
    } else if (useCase === "rag") {
      // Prioritize models known for long context or instruction following
      candidateModels = models.filter(m => m.maxContext >= 8192);
    } else if (useCase === "edge") {
      // Restrict to very small models under 9B parameters for edge devices
      candidateModels = models.filter(m => m.params <= 9000000000);
    } else if (useCase === "agents") {
      // Prioritize high intelligence / function calling capable models
      candidateModels = models.filter(m => m.params >= 30000000000);
    }

    // Sort by parameter count descending to find the biggest/smartest model
    candidateModels.sort((a, b) => b.params - a.params);

    let recommendedModel = null;
    const targetQuant = "INT4";

    for (const m of candidateModels) {
      // Test if it fits in INT4
      const rawWeights = getModelWeights(m.params, 4.5); // INT4 approx
      const kvCache = getKVCache(m.layers, m.hiddenSize, m.queryHeads, m.kvHeads, 4096, 1, 16);
      const { weightsPerCard, kvPerCard } = getMultiGPUPerCard(rawWeights, bestHardware.gpus.length, kvCache, "tensor_parallel");
      const metadata = getQuantMetadata(m.params, "INT4");
      const activations = getActivationMemory(4096, 1, m.hiddenSize);
      
      const gObj = gpus.find(g => g.id === bestHardware.gpus[0])!;
      const reserve = getRuntimeReserve(gObj);
      
      const totalUsed = weightsPerCard + kvPerCard + metadata + activations + reserve;
      const totalWithSafety = totalUsed + getSafetyMargin(totalUsed);

      if (totalWithSafety <= gObj.vram) {
        recommendedModel = m;
        break;
      }
    }

    if (!recommendedModel) {
       // If no model fits even at INT4, fallback to cloud
       return { type: "cloud" as const, reason: "Budget hardware cannot fit recommended models." };
    }

    return {
      type: "local" as const,
      hardware: bestHardware,
      model: recommendedModel,
      quant: targetQuant
    };

  }, [budget, useCase]);

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
            {[
              { id: "coding", label: "Coding Assistant" },
              { id: "roleplay", label: "Creative & Roleplay" },
              { id: "rag", label: "Document Analysis (RAG)" },
              { id: "agents", label: "Autonomous Agents" },
              { id: "edge", label: "Edge / Lightweight" },
              { id: "production", label: "Production API" }
            ].map((uc) => (
              <button
                key={uc.id}
                onClick={() => setUseCase(uc.id as UseCase)}
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
            min={100} max={20000} step={100}
            value={[budget]}
            onValueChange={([v]) => setBudget(v)}
          />
          <div className="flex justify-between text-xs text-muted-foreground font-mono">
            <span>$100</span>
            <span>$20,000+</span>
          </div>
        </div>
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
                    Based on your budget of <strong>${budget.toLocaleString()}</strong>, we prioritized maximum VRAM capacity to fit the largest possible model for your <strong>{useCase}</strong> workflow.
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
                 <div className="text-2xl font-bold text-foreground mb-1">{recommendation.hardware.name}</div>
                 <div className="text-primary font-mono bg-primary/10 inline-block px-2 py-1 rounded text-sm mb-4">
                   {recommendation.hardware.vram} GB Total VRAM
                 </div>
                 <div className="text-sm text-muted-foreground">
                   Estimated Cost: <strong className="text-foreground">${recommendation.hardware.price.toLocaleString()}</strong>
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
                   {(recommendation.model.params / 1e9).toFixed(1)}B Parameters
                 </div>
                 <div className="text-sm text-muted-foreground">
                   Recommended Format: <strong className="text-foreground">{recommendation.quant} Quantization</strong>
                 </div>
               </div>
             </div>

             <div className="bg-background border border-border p-6 rounded-xl">
               <h3 className="text-lg font-bold text-foreground mb-3">Why this loadout?</h3>
               <p className="text-muted-foreground text-sm leading-relaxed">
                 To run a high-quality model for {useCase}, VRAM capacity is the primary bottleneck. The <strong>{recommendation.hardware.name}</strong> provides the cheapest path to {recommendation.hardware.vram}GB of VRAM within your ${budget.toLocaleString()} budget. This exact capacity allows you to load <strong>{recommendation.model.name}</strong> at 4-bit precision, leaving enough reserve memory for a 4K context window without crashing.
               </p>
             </div>
          </div>
        )}
      </div>
      
    </div>
  );
}
