"use client";

import { useState } from "react";
import { useStore, FinetuneQuant } from "@/lib/store";
import { Card } from "@/components/ui/Card";
import { Pill } from "@/components/ui/Pill";
import { Slider } from "@/components/ui/Slider";
import { Switch } from "@/components/ui/Switch";
import { Gauge } from "@/components/ui/Gauge";
import { InfoPopup } from "@/components/ui/InfoPopup";
import { ModelPicker, getModelById } from "@/components/ModelPicker";
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
  getLoRATrainableParams,
  getLoRAGradientMemory,
  getLoRAOptimizerMemory,
  getFullFTMemory,
  getMinGPUCount,
  formatParams,
  type ModelSpec,
  type ZeroStage,
} from "@/lib/calc";
import { Box, Cpu, Zap, Activity, Network } from "lucide-react";

const QUANTS = ["FP16", "INT8", "INT4"];
const MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];

const ZERO_STAGES: ZeroStage[] = [1, 2, 3];
const ZERO_EXPLAIN: Record<ZeroStage, string> = {
  1: "ZeRO-1 shards only the optimizer states (FP32 master weights + AdamW moments) across your GPUs. Weights and gradients stay fully replicated on every card.",
  2: "ZeRO-2 also shards the gradients — only the BF16 weights are fully replicated on each card.",
  3: "ZeRO-3 shards everything — weights, gradients, and optimizer states — gathering each layer's weights only exactly when needed (budget ~5% extra for comms).",
};

export function FineTuneScenario() {
  const store = useStore();
  // FSDP is a presentational alias for ZeRO-3 math — label state only.
  const [fsdpAlias, setFsdpAlias] = useState(false);

  const selectedModelData: ModelSpec =
    getModelById(store.selectedModel) ?? (models[0] as unknown as ModelSpec);
  const selectedGPUsData = store.selectedGPUs.map((id) => gpus.find((g) => g.id === id)!).filter(Boolean);

  const bitsPerWeight = getBitsPerWeight(store.finetuneQuant);
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
  const trainableParams = getLoRATrainableParams(
    store.loraRank,
    selectedModelData.hiddenSize,
    store.targetModules,
    selectedModelData.layers
  );
  const gradients = getLoRAGradientMemory(trainableParams);
  const optimizerStates = getLoRAOptimizerMemory(trainableParams);
  const trainablePct = (trainableParams / selectedModelData.params) * 100;

  const totalUsed = weightsPerCard + kvPerCard + activations + reserve + gradients + optimizerStates;
  const safety = getSafetyMargin(totalUsed);
  const totalWithSafety = totalUsed + safety;

  const availableMemory = selectedGPUsData.length > 0 ? selectedGPUsData[0].vram : 0;

  const isFull = store.ftMode === "full";

  // Full fine-tuning (BF16 mixed precision, 16 bytes/param, ZeRO-sharded)
  const ftOpts = {
    gradCheckpointing: store.gradientCheckpointing,
    batchSize: store.trainBatchSize,
    seqLen: store.contextLength,
    hiddenSize: selectedModelData.hiddenSize,
    layers: selectedModelData.layers,
  };
  const fullFT = getFullFTMemory(
    selectedModelData.params,
    selectedGPUsData.length,
    store.zeroStage,
    ftOpts
  );
  const fullTotalUsed = fullFT.totalPerCard + reserve;
  const fullSafety = getSafetyMargin(fullTotalUsed);
  const fullTotalWithSafety = fullTotalUsed + fullSafety;
  const perGpuUsable = availableMemory - reserve;
  const minGPUs = getMinGPUCount(selectedModelData.params, perGpuUsable, store.zeroStage, ftOpts);
  const stageLabel = fsdpAlias && store.zeroStage === 3 ? "FSDP" : `ZeRO-${store.zeroStage}`;

  let statusText = "TRAINABLE";
  let statusColor = "text-primary border-primary bg-primary/10";
  const verdictTotal = isFull ? fullTotalWithSafety : totalWithSafety;
  if (verdictTotal > availableMemory) {
    statusText = "INSUFFICIENT MEMORY";
    statusColor = "text-destructive border-destructive bg-destructive/10";
  } else if (verdictTotal > availableMemory * 0.9) {
    statusText = "TIGHT FIT";
    statusColor = "text-secondary border-secondary bg-secondary/10";
  }

  // Rough bandwidth-proxy step time (no FLOPS data in gpus.json): each step
  // streams the frozen weights ~3 times (QLoRA forward, backward, recompute),
  // so s/step ≈ params × bytesPerWeight × 3 / (bandwidth × 50% efficiency),
  // scaled by trainBatchSize/4 around a 2048-token sequence assumption.
  // Gradient checkpointing adds ~30% for the activation recompute pass.
  const bandwidth = selectedGPUsData[0]?.bandwidth || 100;
  const bytesPerWeight = bitsPerWeight / 8;
  let secPerStep =
    ((selectedModelData.params * bytesPerWeight * 3) / (bandwidth * 1e9 * 0.5)) *
    (store.trainBatchSize / 4);
  if (store.gradientCheckpointing) {
    secPerStep *= 1.3;
  }

  // Full FT trains all params in BF16: each step streams the weights through
  // ~6 passes-equivalent (forward + full backward + recompute), roughly 2x the
  // LoRA path, at lower efficiency (~40%) due to optimizer/comm overhead.
  let fullSecPerStep =
    ((selectedModelData.params * 2 * 6) / (bandwidth * 1e9 * 0.4)) *
    (store.trainBatchSize / 4);
  if (store.gradientCheckpointing) {
    fullSecPerStep *= 1.3;
  }

  return (
    <div className="flex-1 flex gap-6 h-full overflow-hidden">
      {/* Left Panel: Config */}
      <div className="w-1/2 overflow-y-auto pr-2 flex flex-col gap-8 custom-scrollbar">

        {/* Training Mode */}
        <div className="space-y-4">
          <h2 className="text-sm font-mono text-muted-foreground uppercase tracking-widest border-b border-border pb-2 flex items-center gap-2">
            <Activity className="w-4 h-4" /> Training Mode
          </h2>
          <div className="flex flex-wrap gap-2">
            <Pill selected={store.ftMode === "lora"} onClick={() => store.setFtMode("lora")}>
              LoRA / QLoRA
            </Pill>
            <Pill selected={store.ftMode === "full"} onClick={() => store.setFtMode("full")}>
              Full fine-tune
            </Pill>
          </div>
        </div>

        {/* Model Picker */}
        <div className="space-y-4">
          <h2 className="text-sm font-mono text-muted-foreground uppercase tracking-widest border-b border-border pb-2 flex items-center gap-2">
            <Box className="w-4 h-4" /> Base Model
          </h2>
          <ModelPicker value={store.selectedModel} onChange={store.setSelectedModel} />
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

        {/* Quantization (LoRA/QLoRA only — full FT trains in BF16) */}
        {!isFull && (
        <div className="space-y-4">
          <h2 className="text-sm font-mono text-muted-foreground uppercase tracking-widest border-b border-border pb-2 flex items-center gap-2">
            <Zap className="w-4 h-4" /> Base Model Quant
            <InfoPopup content="Reduces precision of base model weights to save VRAM during training." />
          </h2>
          <div className="flex flex-wrap gap-2">
            {QUANTS.map((q) => (
              <Pill
                key={q}
                selected={store.finetuneQuant === q}
                onClick={() => store.setFinetuneQuant(q as FinetuneQuant)}
              >
                {q}
              </Pill>
            ))}
          </div>
        </div>
        )}

        {/* ZeRO / FSDP sharding (full fine-tune only) */}
        {isFull && (
        <div className="space-y-4">
          <h2 className="text-sm font-mono text-muted-foreground uppercase tracking-widest border-b border-border pb-2 flex items-center gap-2">
            <Network className="w-4 h-4" /> Sharding Strategy
            <InfoPopup content="Full fine-tuning keeps 16 bytes per parameter: BF16 weights (2B) + BF16 gradients (2B) + FP32 master weights (4B) + AdamW moments (8B). ZeRO/FSDP shards that state across your data-parallel GPUs." />
          </h2>
          <div className="flex flex-wrap gap-2">
            {ZERO_STAGES.map((s) => (
              <Pill
                key={s}
                selected={store.zeroStage === s && !(s === 3 && fsdpAlias)}
                onClick={() => { store.setZeroStage(s); setFsdpAlias(false); }}
              >
                ZeRO-{s}
              </Pill>
            ))}
            <Pill
              selected={store.zeroStage === 3 && fsdpAlias}
              onClick={() => { store.setZeroStage(3); setFsdpAlias(true); }}
            >
              FSDP (≈ ZeRO-3)
            </Pill>
          </div>
          <p className="text-xs font-mono text-muted-foreground">
            {fsdpAlias && store.zeroStage === 3
              ? "FSDP is PyTorch's native equivalent of ZeRO-3: " + ZERO_EXPLAIN[3]
              : ZERO_EXPLAIN[store.zeroStage]}
          </p>
        </div>
        )}

        {/* Training Params */}
        <div className="space-y-6 bg-card/30 border border-border p-4 relative">
          <div className="absolute top-0 left-0 w-full h-full bg-grid-pattern opacity-10 pointer-events-none" />

          {!isFull && (
          <>
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
          </>
          )}

          {isFull && (
          <div className="relative z-10 text-xs font-mono text-muted-foreground">
            Full fine-tuning trains <span className="text-primary">{formatParams(selectedModelData.params)}</span> params
            (100%) in BF16 mixed precision — 16 bytes/param before sharding.
          </div>
          )}

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
             {isFull
               ? "WARNING: Per-card memory exceeds VRAM. Try a higher ZeRO stage, adding more GPUs, enabling Gradient Checkpointing, or lowering the Batch Size."
               : "WARNING: Total memory exceeds VRAM. Try enabling Gradient Checkpointing or lowering the Batch Size."}
           </div>
        )}

        {isFull ? (
          <div className="p-3 border border-primary/50 bg-primary/10 text-sm font-mono text-center tracking-wide">
            {minGPUs === -1 ? (
              <>Cannot fit even on 64× <span className="text-primary">{selectedGPUsData[0]?.name ?? "GPU"}</span> at {stageLabel}</>
            ) : (
              <>Minimum required: <span className="text-primary font-bold">{minGPUs}× {selectedGPUsData[0]?.name ?? "GPU"}</span> at {stageLabel}</>
            )}
          </div>
        ) : (
        <div className="text-xs font-mono text-muted-foreground text-center">
          LoRA trainable: <span className="text-primary">{formatParams(trainableParams)}</span> params (
          {trainablePct < 0.01 ? "<0.01" : trainablePct.toFixed(2)}% of {formatParams(selectedModelData.params)} total)
        </div>
        )}

        <div className="flex items-start gap-8 mt-4">
          {/* Gauge */}
          <div className="shrink-0 pt-4">
            <Gauge
              total={availableMemory}
              size={240}
              segments={isFull ? [
                { label: "Weights (BF16)", value: fullFT.weightsPerCard, color: "#22d3ee" },
                { label: "Gradients (BF16)", value: fullFT.gradsPerCard, color: "#f59e0b" },
                { label: "Optimizer + FP32 Master", value: fullFT.optimizerPerCard, color: "#a855f7" },
                { label: `Activations (ctx: ${store.contextLength}, b: ${store.trainBatchSize})`, value: fullFT.activationsPerCard, color: "#ec4899" },
                { label: "Safety", value: fullSafety, color: "#10b981" },
              ] : [
                { label: "Base Weights", value: weightsPerCard, color: "#22d3ee" },
                { label: `Activations & Logits (ctx: ${store.contextLength}, b: ${store.trainBatchSize})`, value: kvPerCard, color: "#10b981" },
                { label: "Activations", value: activations, color: "#ec4899" },
                { label: "Gradients", value: gradients, color: "#f59e0b" },
                { label: "Optimizer", value: optimizerStates, color: "#a855f7" },
                { label: "Safety", value: safety, color: "#10b981" },
              ]}
            />
          </div>

          {/* List */}
          <div className="flex-1 space-y-3 pt-4">
            {(isFull ? [
              { label: `Weights (BF16, ${stageLabel})`, val: fullFT.weightsPerCard, color: "bg-primary" },
              { label: "Gradients (BF16)", val: fullFT.gradsPerCard, color: "bg-secondary" },
              { label: "Optimizer (FP32 master + AdamW)", val: fullFT.optimizerPerCard, color: "bg-purple-500" },
              { label: `Activations (ctx: ${store.contextLength}, b: ${store.trainBatchSize})`, val: fullFT.activationsPerCard, color: "bg-pink-500" },
              { label: "OS Reserve", val: reserve, color: "bg-slate-500" },
              { label: "Safety Margin", val: fullSafety, color: "bg-emerald-500" },
            ] : [
              { label: "Base Weights", val: weightsPerCard, color: "bg-primary" },
              { label: `Activations & Logits (ctx: ${store.contextLength}, b: ${store.trainBatchSize})`, val: kvPerCard, color: "bg-emerald-500" },
              { label: "Activation Memory", val: activations, color: "bg-pink-500" },
              { label: "Gradient Buffers", val: gradients, color: "bg-secondary" },
              { label: "Optimizer (AdamW)", val: optimizerStates, color: "bg-purple-500" },
              { label: "OS Reserve", val: reserve, color: "bg-slate-500" },
              { label: "Safety Margin", val: safety, color: "bg-emerald-500" },
            ]).map((item, i) => (
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
              <span>{(isFull ? fullTotalWithSafety : totalWithSafety).toFixed(2)} GB</span>
            </div>
          </div>
        </div>

        <div className="border border-border p-4 mt-auto">
          <div className="text-xs font-mono text-muted-foreground flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4" /> Estimated Speed (rough estimate, assumes 2048-token sequences)
          </div>
          <div className="text-2xl font-mono text-primary font-bold">
            {(() => { const s = isFull ? fullSecPerStep : secPerStep; return (
              <>~{s < 10 ? s.toFixed(1) : s.toFixed(0)} <span className="text-sm text-muted-foreground">sec/step</span></>
            ); })()}
          </div>
        </div>
      </div>
    </div>
  );
}
