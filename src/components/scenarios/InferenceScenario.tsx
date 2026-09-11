/* eslint-disable react/no-unescaped-entities */
"use client";

import { useState } from "react";
import { useStore, Quantization } from "@/lib/store";
import { Card } from "@/components/ui/Card";
import { Slider } from "@/components/ui/Slider";
import { Switch } from "@/components/ui/Switch";
import { InfoPopup } from "@/components/ui/InfoPopup";
import models from "@/data/models.json";
import gpus from "@/data/gpus.json";
import benchmarks from "@/data/benchmarks.json";
import {
  getBitsPerWeight,
  getModelWeights,
  getKVCache,
  getQuantMetadata,
  getActivationMemory,
  getRuntimeReserve,
  getSafetyMargin,
  getMultiGPUPerCard,
  estimateTPS,
  formatParams,
} from "@/lib/calc";
import { Zap, Layers, ArrowRight, Share2, RotateCcw, ChevronDown, ChevronUp, CheckCircle2, XCircle } from "lucide-react";

const BASIC_QUANTS = ["FP16", "INT8", "INT4"];
const ADVANCED_QUANTS = ["GGUF Q4_K_M", "AWQ", "EXL2"];

export function InferenceScenario() {
  const store = useStore();
  const [showAdvanced, setShowAdvanced] = useState(false);

  const selectedModelData = models.find((m) => m.id === store.selectedModel)!;
  const selectedGPUsData = store.selectedGPUs.map((id) => gpus.find((g) => g.id === id)!);

  const bitsPerWeight = getBitsPerWeight(store.quantization);
  const rawWeights = getModelWeights(selectedModelData.params, bitsPerWeight);
  const kvCache = getKVCache(
    selectedModelData.layers,
    selectedModelData.hiddenSize,
    selectedModelData.queryHeads,
    selectedModelData.kvHeads,
    store.contextLength,
    store.batchSize,
    16
  );
  
  // Math for breakdown
  const { weightsPerCard, kvPerCard } = getMultiGPUPerCard(rawWeights, selectedGPUsData.length, kvCache, "tensor_parallel");
  const metadata = getQuantMetadata(selectedModelData.params, store.quantization);
  const activations = getActivationMemory(store.contextLength, store.batchSize, selectedModelData.hiddenSize, selectedModelData.layers);
  const reserve = selectedGPUsData.length > 0 ? getRuntimeReserve(selectedGPUsData[0]) : 0;
  
  const totalUsed = weightsPerCard + kvPerCard + metadata + activations + reserve;
  const safety = getSafetyMargin(totalUsed);
  const totalWithSafety = totalUsed + safety;
  
  const availableMemory = store.cpuOffload ? store.systemRam : (selectedGPUsData.length > 0 ? selectedGPUsData[0].vram : 0);
  const doesFit = totalWithSafety <= availableMemory;

  const tps = estimateTPS(
    store.selectedModel, 
    selectedGPUsData[0]?.id || "", 
    store.quantization, 
    store.cpuOffload ? 0.5 : 0, 
    selectedGPUsData.length,
    benchmarks
  );

  const handleShare = () => {
    const url = new URL(window.location.href);
    url.searchParams.set("scenario", store.scenario);
    url.searchParams.set("model", store.selectedModel);
    url.searchParams.set("gpu", store.selectedGPUs.join(","));
    url.searchParams.set("quant", store.quantization);
    url.searchParams.set("ctx", store.contextLength.toString());
    url.searchParams.set("batch", store.batchSize.toString());
    navigator.clipboard.writeText(url.toString());
    alert("Link copied to clipboard!");
  };

  const handleReset = () => {
    store.setSelectedModel("llama3-8b");
    store.setSelectedGPUs(["rtx4090"]);
    store.setQuantization("INT4");
    store.setContextLength(4096);
    store.setBatchSize(1);
    store.setCpuOffload(false);
  };

  return (
    <div className="flex-1 flex gap-8 h-full overflow-hidden">
      {/* LEFT COLUMN: Controls */}
      <div className="w-[45%] flex flex-col gap-6 overflow-y-auto pr-4 custom-scrollbar pb-10">
        
        {/* Model Picker */}
        <div className="space-y-3">
          <label className="text-xs font-mono text-muted-foreground uppercase tracking-widest flex items-center gap-2">
            AI Model <InfoPopup content="Select the open-weight model to evaluate." />
          </label>
          <div className="relative">
            <select
              value={store.selectedModel}
              onChange={(e) => store.setSelectedModel(e.target.value)}
              className="w-full appearance-none bg-card border border-border text-foreground font-mono p-4 pr-10 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary transition-all cursor-pointer"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id} className="bg-background text-foreground">
                  {m.name} ({formatParams(m.params)}, {m.architecture.toUpperCase()})
                </option>
              ))}
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-4 text-muted-foreground">
              <ChevronDown className="w-5 h-5" />
            </div>
          </div>
        </div>

        {/* Hardware Picker */}
        <div className="space-y-3">
          <label className="text-xs font-mono text-muted-foreground uppercase tracking-widest flex items-center gap-2">
            Hardware Target <InfoPopup content="Select your GPU or hardware array to test against." />
          </label>
          <div className="grid grid-cols-2 gap-3">
            {gpus.map((g) => (
              <Card 
                key={g.id} 
                selected={store.selectedGPUs.includes(g.id)}
                onClick={() => store.toggleGPU(g.id)}
              >
                <div className="font-bold mb-1">{g.name}</div>
                <div className="text-xs font-mono text-muted-foreground">
                  {g.vram} GB • {g.bandwidth} GB/s
                </div>
              </Card>
            ))}
          </div>
        </div>

        {/* Basic Quantization */}
        <div className="space-y-3 pt-2">
          <div className="flex justify-between items-center">
            <label className="text-xs font-mono text-muted-foreground uppercase tracking-widest flex items-center gap-2">
              Quantization <InfoPopup content="Lower precision reduces VRAM and increases speed, at the cost of some model intelligence." />
            </label>
            <span className="text-xs font-mono text-primary bg-primary/10 px-2 py-1 rounded">{store.quantization}</span>
          </div>
          <div className="flex rounded-lg border border-border overflow-hidden">
            {BASIC_QUANTS.map((q) => (
              <button
                key={q}
                onClick={() => store.setQuantization(q as Quantization)}
                className={`flex-1 py-3 text-sm font-mono transition-colors ${
                  store.quantization === q 
                    ? "bg-primary text-primary-foreground font-bold" 
                    : "bg-card hover:bg-card/80 text-muted-foreground"
                } ${q !== BASIC_QUANTS[0] && "border-l border-border"}`}
              >
                {q}
              </button>
            ))}
          </div>
        </div>

        {/* Sliders */}
        <div className="space-y-6 pt-2">
          <div className="space-y-3">
            <div className="flex justify-between text-sm font-mono items-center">
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground uppercase tracking-widest text-xs">Context Length</span>
              </div>
              <span className="text-primary bg-primary/10 px-2 py-1 rounded">{store.contextLength} tok</span>
            </div>
            <Slider 
              min={1024} max={selectedModelData.maxContext} step={1024}
              value={[store.contextLength]}
              onValueChange={([v]) => store.setContextLength(v)}
            />
          </div>

          <div className="space-y-3">
            <div className="flex justify-between text-sm font-mono items-center">
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground uppercase tracking-widest text-xs">Batch Size</span>
              </div>
              <span className="text-primary bg-primary/10 px-2 py-1 rounded">{store.batchSize}</span>
            </div>
            <Slider 
              min={1} max={128} step={1}
              value={[store.batchSize]}
              onValueChange={([v]) => store.setBatchSize(v)}
            />
          </div>
        </div>

        {/* Advanced Accordion */}
        <div className="border border-border rounded-lg bg-card/30 overflow-hidden mt-4">
          <button 
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="w-full flex items-center justify-between p-4 text-sm font-mono text-muted-foreground hover:bg-card transition-colors"
          >
            <span className="flex items-center gap-2 uppercase tracking-widest text-xs">
              <Zap className="w-4 h-4" /> Advanced Settings
            </span>
            <div className="flex items-center gap-2">
              <span className="text-[10px] bg-secondary/20 text-secondary px-2 py-0.5 rounded">HIDE</span>
              {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </div>
          </button>
          
          {showAdvanced && (
            <div className="p-4 border-t border-border space-y-6">
              <div className="space-y-3">
                <label className="text-xs font-mono text-muted-foreground uppercase tracking-widest flex items-center gap-2">
                  Advanced Quantization
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {ADVANCED_QUANTS.map((q) => (
                    <button
                      key={q}
                      onClick={() => store.setQuantization(q as Quantization)}
                      className={`py-2 rounded border text-xs font-mono transition-colors ${
                        store.quantization === q 
                          ? "bg-primary/20 border-primary text-primary" 
                          : "bg-card border-border hover:bg-border text-muted-foreground"
                      }`}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex items-center justify-between border-t border-border pt-4">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-muted-foreground uppercase tracking-widest">CPU Offload (System RAM)</span>
                  <InfoPopup content="Moves model layers to system RAM. Allows larger models but crushes speed." />
                </div>
                <Switch 
                  checked={store.cpuOffload} 
                  onCheckedChange={store.setCpuOffload} 
                />
              </div>

              {store.cpuOffload && (
                <div className="space-y-3 pt-2">
                  <div className="flex justify-between text-xs font-mono text-muted-foreground uppercase tracking-widest">
                    <span>System RAM Target</span>
                    <span className="text-secondary">{store.systemRam} GB</span>
                  </div>
                  <Slider 
                    min={16} max={256} step={16}
                    value={[store.systemRam]}
                    onValueChange={([v]) => store.setSystemRam(v)}
                  />
                </div>
              )}
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex gap-4 mt-4">
          <button onClick={handleShare} className="flex-1 py-3 flex items-center justify-center gap-2 border border-border rounded-lg bg-card hover:bg-border transition-colors text-sm font-mono text-foreground">
            <Share2 className="w-4 h-4" /> Share Link
          </button>
          <button onClick={handleReset} className="flex-1 py-3 flex items-center justify-center gap-2 border border-border rounded-lg bg-card hover:bg-border transition-colors text-sm font-mono text-destructive hover:text-destructive-foreground hover:bg-destructive/80">
            <RotateCcw className="w-4 h-4" /> Reset
          </button>
        </div>

      </div>

      {/* RIGHT COLUMN: Readout & Visualizations */}
      <div className="w-[55%] flex flex-col gap-6 overflow-y-auto pb-10 custom-scrollbar pr-2">
        
        {/* Status Header */}
        <div className={`p-6 rounded-xl border flex items-start gap-4 ${
          doesFit 
            ? "border-emerald-500/30 bg-emerald-500/5" 
            : "border-destructive/30 bg-destructive/5"
        }`}>
          {doesFit ? (
            <CheckCircle2 className="w-12 h-12 text-emerald-500 shrink-0" />
          ) : (
            <XCircle className="w-12 h-12 text-destructive shrink-0" />
          )}
          <div>
            <h1 className="text-3xl font-bold mb-2 text-foreground">
              {doesFit ? "Fits comfortably" : "Does not fit"}
            </h1>
            <p className="text-muted-foreground leading-relaxed">
              <span className="text-foreground font-semibold">{selectedModelData.name}</span> at <span className="text-foreground font-semibold">{store.quantization}</span> is estimated to need <span className="text-foreground font-semibold">{totalWithSafety.toFixed(1)} GB</span> of the {selectedGPUsData.length > 0 ? selectedGPUsData.map(g => g.name).join(", ") : "System"}'s <span className="text-foreground font-semibold">{availableMemory} GB</span> usable planning memory.
            </p>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-3 gap-4 border-b border-border pb-6">
          <div className="text-center border-r border-border">
            <div className="text-xs font-mono text-muted-foreground uppercase tracking-widest mb-1">Memory Fit</div>
            <div className={`font-semibold ${doesFit ? "text-emerald-500" : "text-destructive"}`}>
              {doesFit ? "Yes" : "No"}
            </div>
          </div>
          <div className="text-center border-r border-border">
            <div className="text-xs font-mono text-muted-foreground uppercase tracking-widest mb-1">Runtime</div>
            <div className="font-semibold text-foreground">Supported • {store.quantization.includes("GGUF") ? "llama.cpp" : "vLLM"}</div>
          </div>
          <div className="text-center">
            <div className="text-xs font-mono text-muted-foreground uppercase tracking-widest mb-1">Est. Speed</div>
            <div className="font-semibold text-foreground">~{tps.toFixed(0)} tok/s</div>
          </div>
        </div>

        {/* Architecture Diagram */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-mono text-foreground font-bold flex items-center gap-2">
              <Layers className="w-4 h-4" /> How this model is built
            </h2>
            <span className="text-[10px] font-mono border border-border px-2 py-0.5 rounded text-muted-foreground uppercase">{selectedModelData.architecture}</span>
          </div>
          
          <div className="border border-border rounded-xl bg-card/30 p-6 space-y-8">
            <div className="flex justify-between text-xs font-mono text-muted-foreground border-b border-border/50 pb-2">
              <div className="flex items-center gap-2 text-primary">
                <div className="w-2 h-2 rounded-full bg-primary" /> WORKSPACE
              </div>
              <div>{selectedModelData.layers} layers</div>
              <div>Hidden width {selectedModelData.hiddenSize}</div>
              <div>{selectedModelData.queryHeads} query heads</div>
              <div>{selectedModelData.kvHeads} KV heads</div>
            </div>

            <div className="flex items-center justify-between text-xs font-mono relative">
              <div className="flex flex-col items-center bg-card border border-border p-3 rounded-lg w-28 shrink-0 relative z-10">
                <div className="font-bold text-foreground mb-1">TOKENS</div>
                <div className="text-[10px] text-muted-foreground">B × T</div>
              </div>
              
              <ArrowRight className="text-muted-foreground/50 shrink-0 mx-2" />
              
              <div className="flex flex-col items-center bg-card border border-primary/40 p-3 rounded-lg w-32 shrink-0 relative z-10 shadow-[0_0_15px_rgba(var(--primary),0.1)]">
                <div className="font-bold text-primary mb-1 text-[10px]">EMBEDDING</div>
                <div className="text-[9px] text-muted-foreground text-center leading-tight mt-1">
                  Vocab × Hidden<br/>{selectedModelData.vocabSize} × {selectedModelData.hiddenSize}
                </div>
              </div>

              <ArrowRight className="text-muted-foreground/50 shrink-0 mx-2" />

              <div className="flex flex-col items-center justify-center bg-card border border-dashed border-primary p-4 rounded-xl flex-1 relative z-10 min-h-[80px]">
                <div className="font-bold text-primary mb-1 text-[10px]">REPEATED BLOCK</div>
                <div className="text-[10px] text-muted-foreground flex items-center gap-1 mt-2">
                  + EXPAND DETAILS
                </div>
                <div className="absolute top-2 right-2 bg-primary/20 text-primary text-[10px] px-1.5 rounded">
                  x{selectedModelData.layers}
                </div>
              </div>

              <ArrowRight className="text-muted-foreground/50 shrink-0 mx-2" />

              <div className="flex flex-col gap-2 shrink-0 relative z-10">
                <div className="bg-card border border-border p-2 rounded-lg w-28 text-center">
                  <div className="font-bold text-foreground text-[10px]">FINAL NORM</div>
                </div>
                <div className="bg-card border border-border p-2 rounded-lg w-28 text-center">
                  <div className="font-bold text-foreground text-[10px]">LM HEAD</div>
                  <div className="text-[8px] text-muted-foreground leading-tight mt-1">
                    Vocab × Hidden<br/>{selectedModelData.vocabSize} × {selectedModelData.hiddenSize}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Horizontal Stacked Bar Chart */}
        <div className="space-y-4 pt-4">
          <div className="flex justify-between text-xs font-mono text-muted-foreground uppercase tracking-widest">
            <span>0 GB</span>
            <span>{totalWithSafety.toFixed(1)} GB TOTAL REQUIRED</span>
          </div>
          
          <div className="w-full h-12 flex rounded-lg overflow-hidden border border-border">
            {[
              { val: weightsPerCard, color: "bg-indigo-500" },
              { val: kvPerCard, color: "bg-emerald-500" },
              { val: metadata, color: "bg-amber-500" },
              { val: activations, color: "bg-pink-500" },
              { val: reserve, color: "bg-slate-500" },
              { val: safety, color: "bg-sky-500" }
            ].map((seg, i) => {
              const pct = (seg.val / totalWithSafety) * 100;
              if (pct < 0.5) return null;
              return (
                <div 
                  key={i} 
                  style={{ width: `${pct}%` }} 
                  className={`h-full border-r border-background/20 ${seg.color} transition-all duration-500`} 
                />
              )
            })}
          </div>

          {/* Legend Cards */}
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: "Model Weights", val: weightsPerCard, color: "border-indigo-500" },
              { label: `KV Cache (ctx: ${store.contextLength}, b: ${store.batchSize})`, val: kvPerCard, color: "border-emerald-500" },
              { label: "Quantization Metadata", val: metadata, color: "border-amber-500" },
              { label: "Activations", val: activations, color: "border-pink-500" },
              { label: "OS Reserve", val: reserve, color: "border-slate-500" },
              { label: "Safety Margin", val: safety, color: "border-sky-500" },
            ].map((item, i) => (
              <div key={i} className={`bg-card/50 border border-border border-l-4 ${item.color} rounded p-3 flex flex-col justify-between`}>
                <span className="text-xs font-mono text-muted-foreground">{item.label}</span>
                <span className="text-sm font-bold text-foreground mt-1">{item.val.toFixed(2)} GB</span>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
