"use client";

import { useStore, Quantization } from "@/lib/store";
import { Card } from "@/components/ui/Card";
import { Pill } from "@/components/ui/Pill";
import { Slider } from "@/components/ui/Slider";
import { Switch } from "@/components/ui/Switch";
import { Gauge } from "@/components/ui/Gauge";
import { InfoPopup } from "@/components/ui/InfoPopup";
import models from "@/data/models.json";
import gpus from "@/data/gpus.json";
import {
  getBitsPerWeight,
  getModelWeights,
  getKVCache,
  getActivationMemory,
  getRuntimeReserve,
  getSafetyMargin,
  getMultiGPUPerCard,
  getQLoRAOverhead,
  formatParams
} from "@/lib/calc";
import { Box, Cpu, Zap, Activity } from "lucide-react";

const QUANTS = ["FP16", "INT8", "INT4"];
const MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];

export function FineTuneScenario() {
  const store = useStore();

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
    store.trainBatchSize,
    16
  );
  
  const { weightsPerCard, kvPerCard } = getMultiGPUPerCard(rawWeights, selectedGPUsData.length, kvCache, "tensor_parallel");
  let activations = getActivationMemory(store.contextLength, store.trainBatchSize, selectedModelData.hiddenSize, selectedModelData.layers);
  
  if (store.gradientCheckpointing) {
    activations = activations * 0.2; // significant reduction
  }

  const reserve = selectedGPUsData.length > 0 ? getRuntimeReserve(selectedGPUsData[0]) : 0;
  
  // LoRA specifics
  const loraOverhead = getQLoRAOverhead(
    selectedModelData.params, 
    store.loraRank, 
    selectedModelData.hiddenSize, 
    store.targetModules.length
  );
  
  // Gradients for trainable params (FP32 typically, so 4 bytes per param)
  // Simplified for gauge
  const gradients = loraOverhead * 0.3; 
  const optimizerStates = loraOverhead * 0.7; // AdamW

  const totalUsed = weightsPerCard + kvPerCard + activations + reserve + gradients + optimizerStates;
  const safety = getSafetyMargin(totalUsed);
  const totalWithSafety = totalUsed + safety;
  
  const availableMemory = store.cpuOffload ? store.systemRam : (selectedGPUsData.length > 0 ? selectedGPUsData[0].vram : 0);

  let statusText = "TRAINABLE";
  let statusColor = "text-primary border-primary bg-primary/10";
  if (totalWithSafety > availableMemory) {
    statusText = "INSUFFICIENT MEMORY";
    statusColor = "text-destructive border-destructive bg-destructive/10";
  } else if (totalWithSafety > availableMemory * 0.9) {
    statusText = "TIGHT FIT";
    statusColor = "text-secondary border-secondary bg-secondary/10";
  }

  return (
    <div className="flex-1 flex gap-6 h-full overflow-hidden">
      {/* Left Panel: Config */}
      <div className="w-1/2 overflow-y-auto pr-2 flex flex-col gap-8 custom-scrollbar">
        
        {/* Model Picker */}
        <div className="space-y-4">
          <h2 className="text-sm font-mono text-muted-foreground uppercase tracking-widest border-b border-border pb-2 flex items-center gap-2">
            <Box className="w-4 h-4" /> Base Model
          </h2>
          <div className="relative">
            <select
              value={store.selectedModel}
              onChange={(e) => store.setSelectedModel(e.target.value)}
              className="w-full appearance-none bg-card/50 border border-border text-primary font-mono p-3 pr-10 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-colors cursor-pointer"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id} className="bg-background text-primary">
                  {m.name} ({formatParams(m.params)}, {m.architecture.toUpperCase()})
                </option>
              ))}
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-4 text-primary">
              <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/>
              </svg>
            </div>
          </div>
        </div>

        {/* Hardware Picker */}
        <div className="space-y-4">
          <h2 className="text-sm font-mono text-muted-foreground uppercase tracking-widest border-b border-border pb-2 flex items-center gap-2">
            <Cpu className="w-4 h-4" /> Hardware Array
          </h2>
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

        {/* Quantization */}
        <div className="space-y-4">
          <h2 className="text-sm font-mono text-muted-foreground uppercase tracking-widest border-b border-border pb-2 flex items-center gap-2">
            <Zap className="w-4 h-4" /> Base Model Quant
            <InfoPopup content="Reduces precision of base model weights to save VRAM during training." />
          </h2>
          <div className="flex flex-wrap gap-2">
            {QUANTS.map((q) => (
              <Pill 
                key={q} 
                selected={store.quantization === q}
                onClick={() => store.setQuantization(q as Quantization)}
              >
                {q}
              </Pill>
            ))}
          </div>
        </div>

        {/* Training Params */}
        <div className="space-y-6 bg-card/30 border border-border p-4 relative">
          <div className="absolute top-0 left-0 w-full h-full bg-grid-pattern opacity-10 pointer-events-none" />
          
          <div className="space-y-3 relative z-10">
            <div className="flex justify-between text-sm font-mono items-center">
              <div>
                <span>LoRA Rank (r)</span>
                <InfoPopup content="Determines the dimension of the low-rank matrices. A higher rank captures more complex patterns but increases trainable parameters and memory usage." />
              </div>
              <span className="text-primary">{store.loraRank}</span>
            </div>
            <Slider 
              min={8} max={256} step={8}
              value={[store.loraRank]}
              onValueChange={([v]) => store.setLoraRank(v)}
            />
          </div>

          <div className="space-y-3 relative z-10">
             <div className="flex justify-between text-sm font-mono items-center">
              <div>
                <span>LoRA Alpha</span>
                <InfoPopup content="Scaling factor for the LoRA weight updates. Usually set to 2x the Rank." />
              </div>
              <span className="text-primary">{store.loraAlpha}</span>
            </div>
            <Slider 
              min={16} max={512} step={16}
              value={[store.loraAlpha]}
              onValueChange={([v]) => store.setLoraAlpha(v)}
            />
          </div>

          <div className="space-y-3 relative z-10 pt-2">
            <div className="text-sm font-mono mb-2 flex items-center">
              Target Modules
              <InfoPopup content="Which specific attention or feed-forward layers the LoRA is applied to. More modules = more memory but often better results." />
            </div>
            <div className="flex flex-wrap gap-2">
              {MODULES.map(m => (
                <Pill 
                  key={m} 
                  selected={store.targetModules.includes(m)}
                  onClick={() => store.toggleTargetModule(m)}
                >
                  {m}
                </Pill>
              ))}
            </div>
          </div>

          <div className="space-y-3 relative z-10 pt-4 border-t border-border mt-4">
            <div className="flex justify-between text-sm font-mono items-center">
              <div>
                <span>Train Batch Size</span>
                <InfoPopup content="Number of examples processed before updating the model weights. High batch sizes stabilize training but consume massive memory." />
              </div>
              <span className="text-primary">{store.trainBatchSize}</span>
            </div>
            <Slider 
              min={1} max={32} step={1}
              value={[store.trainBatchSize]}
              onValueChange={([v]) => store.setTrainBatchSize(v)}
            />
          </div>

          <div className="flex items-center justify-between relative z-10 pt-4 border-t border-border mt-4">
            <div className="flex items-center">
              <span className="text-sm font-mono">Gradient Checkpointing</span>
              <InfoPopup content="Trades compute for memory by dropping intermediate activations and recomputing them during the backward pass. Crucial for saving VRAM." />
            </div>
            <Switch 
              checked={store.gradientCheckpointing} 
              onCheckedChange={store.setGradientCheckpointing} 
            />
          </div>
        </div>
      </div>

      {/* Right Panel: Readout */}
      <div className="w-1/2 flex flex-col gap-6 bg-card/40 border border-border p-6 relative overflow-y-auto">
        {/* Banner */}
        <div className={`p-3 border font-mono font-bold text-center tracking-widest ${statusColor}`}>
          [{statusText}]
        </div>

        {statusText === "INSUFFICIENT MEMORY" && (
           <div className="p-3 border border-secondary text-secondary bg-secondary/10 text-sm font-mono">
             WARNING: Total memory exceeds VRAM. Try enabling Gradient Checkpointing or lowering the Batch Size.
           </div>
        )}

        <div className="flex items-start gap-8 mt-4">
          {/* Gauge */}
          <div className="shrink-0 pt-4">
            <Gauge 
              total={availableMemory}
              size={240}
              segments={[
                { label: "Base Weights", value: weightsPerCard, color: "#22d3ee" },
                { label: "KV Cache", value: kvPerCard, color: "#10b981" },
                { label: "Activations", value: activations, color: "#ec4899" },
                { label: "Gradients", value: gradients, color: "#f59e0b" },
                { label: "Optimizer", value: optimizerStates, color: "#a855f7" },
                { label: "Safety", value: safety, color: "#10b981" },
              ]}
            />
          </div>

          {/* List */}
          <div className="flex-1 space-y-3 pt-4">
            {[
              { label: "Base Weights", val: weightsPerCard, color: "bg-primary" },
              { label: "KV Cache", val: kvPerCard, color: "bg-emerald-500" },
              { label: "Activation Memory", val: activations, color: "bg-pink-500" },
              { label: "Gradient Buffers", val: gradients, color: "bg-secondary" },
              { label: "Optimizer (AdamW)", val: optimizerStates, color: "bg-purple-500" },
              { label: "OS Reserve", val: reserve, color: "bg-slate-500" },
              { label: "Safety Margin", val: safety, color: "bg-emerald-500" },
            ].map((item, i) => (
              <div key={i} className="text-sm font-mono flex items-center justify-between border-b border-border/50 pb-1">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${item.color}`} />
                  {item.label}
                </div>
                <div>{item.val.toFixed(2)} GB</div>
              </div>
            ))}
            <div className="text-sm font-mono flex items-center justify-between pt-2 text-primary font-bold">
              <span>TOTAL (Per Card)</span>
              <span>{totalWithSafety.toFixed(2)} GB</span>
            </div>
          </div>
        </div>

        <div className="border border-border p-4 mt-auto">
          <div className="text-xs font-mono text-muted-foreground flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4" /> Estimated Speed
          </div>
          <div className="text-2xl font-mono text-primary font-bold">
            ~{((selectedGPUsData[0]?.bandwidth || 100) / 100).toFixed(1)} <span className="text-sm text-muted-foreground">sec/step</span>
          </div>
        </div>
      </div>
    </div>
  );
}
