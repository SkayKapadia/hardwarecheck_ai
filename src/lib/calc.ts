export type ModelArchitecture = "dense" | "moe";

export const MODELS_DATA_AS_OF = "2026-09-15";
export const PRICING_DATA_AS_OF = "September 2026";

export interface ModelSpec {
  id: string;
  name: string;
  /** TOTAL parameter count (all experts for MoE). Used for weight memory. */
  params: number;
  layers: number;
  hiddenSize: number;
  queryHeads: number;
  kvHeads: number;
  vocabSize: number;
  maxContext: number;
  architecture: ModelArchitecture;
  /** MoE only: parameters active per token. Used for speed estimates. */
  activeParams?: number;
  family: string;
  /** HF repo id used to verify this entry. */
  source?: string;
  verifiedOn?: string;
}

export interface GPUSpec {
  id: string;
  name: string;
  vram: number; // in GB
  bandwidth: number; // in GB/s
  interconnect: string;
  osReserve: number; // in GB
  price?: number; // USD, see PRICING_DATA_AS_OF
  class?: string; // "consumer" | "datacenter" | "mac" | "amd"
}

export interface Benchmark {
  modelId: string;
  gpuId: string;
  quant: string;
  tps: number;
}

export function getQuantMetadata(params: number, format: string): number {
  let overheadPercent = 0.0;
  if (format === "GGUF Q4_K_M" || format === "AWQ" || format === "EXL2" || format === "INT4" || format === "INT8") {
    overheadPercent = 0.05; // 5% overhead for quantization metadata mapping
  }
  return (params * overheadPercent * 16) / 8 / 1e9; // overhead in GB assuming mapping tables
}

export function getBitsPerWeight(format: string): number {
  switch (format) {
    case "FP16":
      return 16;
    case "INT8":
      return 8;
    case "INT4":
    case "AWQ":
      return 4;
    case "GGUF Q4_K_M":
      return 4.5;
    case "EXL2":
      return 4; // varying, but avg ~4 for this scenario
    default:
      return 16;
  }
}

export function getModelWeights(params: number, bitsPerWeight: number): number {
  return (params * bitsPerWeight) / 8 / 1e9; // returns GB
}

export function formatParams(params: number): string {
  if (params < 1_000_000_000) {
    return (params / 1_000_000).toFixed(0) + "M";
  }
  return (params / 1_000_000_000).toFixed(1) + "B";
}

export function getKVCache(
  layers: number,
  hiddenSize: number,
  queryHeads: number,
  kvHeads: number,
  ctxLen: number,
  batchSize: number,
  kvBits: number = 16 // Assume FP16 cache usually
): number {
  // Formula: 2 (K and V) * layers * (hiddenSize / queryHeads) * kvHeads * ctxLen * batchSize * (kvBits / 8)
  const headDim = hiddenSize / queryHeads;
  const bytes = 2 * layers * headDim * kvHeads * ctxLen * batchSize * (kvBits / 8);
  return bytes / 1e9; // returns GB
}

export function getActivationMemory(
  ctxLen: number,
  batchSize: number,
  hiddenSize: number,
  layers: number = 32
): number {
  // Calculate for worst-case Prefill phase spike (processing the entire prompt at once)
  // Forward pass requires saving activations for backprop (if training) or intermediate tensors (if inference).
  // A rough estimate for inference prefill activation memory spike:
  // bytes = batchSize * ctxLen * hiddenSize * layers * (multiplier based on architecture)
  // For Llama 3 8B at 4k context, we want this to scale to roughly 1.5 GB.
  // 1 * 4096 * 4096 * 32 = 536,870,912 bytes. 
  // Multiplier of 3 gives ~1.6 GB.
  const bytes = batchSize * ctxLen * hiddenSize * layers * 3;
  return bytes / 1e9;
}

export function getRuntimeReserve(gpu: GPUSpec): number {
  return gpu.osReserve;
}

export function getSafetyMargin(total: number): number {
  return total * 0.05; // 5% safety margin
}

// SwiGLU intermediate size is approximately 3.5x hidden size for modern archs
const MLP_INTERMEDIATE_RATIO = 3.5;

// Per-projection LoRA trainable params = rank * (inDim + outDim).
// Attention projections are hiddenSize x hiddenSize; MLP projections are
// hiddenSize x (MLP_INTERMEDIATE_RATIO * hiddenSize) and back.
function loraModuleParams(module: string, rank: number, hiddenSize: number): number {
  switch (module) {
    case "q_proj":
    case "k_proj":
    case "v_proj":
    case "o_proj":
      return rank * (hiddenSize + hiddenSize);
    case "gate_proj":
    case "up_proj":
    case "down_proj": {
      const intermediate = hiddenSize * MLP_INTERMEDIATE_RATIO;
      return rank * (hiddenSize + intermediate);
    }
    default:
      // Unknown module: assume a square hiddenSize projection
      return rank * (hiddenSize + hiddenSize);
  }
}

export function getLoRATrainableParams(
  rank: number,
  hiddenSize: number,
  targetModules: string[],
  layers: number
): number {
  const perLayer = targetModules.reduce(
    (sum, mod) => sum + loraModuleParams(mod, rank, hiddenSize),
    0
  );
  return layers * perLayer;
}

export function getLoRAGradientMemory(trainableParams: number): number {
  return (trainableParams * 4) / 1e9; // FP32 gradients, returns GB
}

export function getLoRAOptimizerMemory(trainableParams: number): number {
  return (trainableParams * 8) / 1e9; // AdamW FP32 m+v states, returns GB
}

// ---------- Full fine-tuning (BF16 mixed precision + ZeRO/FSDP sharding) ----------

export type ZeroStage = 1 | 2 | 3;

export interface FullFTOptions {
  gradCheckpointing: boolean;
  batchSize: number;
  seqLen: number;
  hiddenSize: number;
  layers: number;
}

export interface FullFTMemory {
  weightsPerCard: number;
  gradsPerCard: number;
  optimizerPerCard: number;
  activationsPerCard: number;
  totalPerCard: number;
}

// Classic BF16 mixed-precision recipe = 16 bytes/param:
//   BF16 weights 2B + BF16 gradients 2B + FP32 master weights 4B + AdamW m+v 8B.
// ZeRO shards across N data-parallel GPUs:
//   stage 1: optimizer-side (master 4B + Adam 8B) / N; weights + grads replicated
//   stage 2: additionally gradients / N; only BF16 weights replicated
//   stage 3 (= FSDP): everything / N (weights gathered per layer at runtime)
export function getFullFTMemory(
  params: number,
  gpuCount: number,
  zeroStage: ZeroStage,
  opts: FullFTOptions
): FullFTMemory {
  const n = Math.max(1, gpuCount);
  const weights = (params * 2) / 1e9; // BF16 weights, GB
  const grads = (params * 2) / 1e9; // BF16 gradients, GB
  const optimizer = (params * 12) / 1e9; // FP32 master weights (4B) + AdamW m+v (8B), GB

  const weightsPerCard = zeroStage === 3 ? weights / n : weights;
  const gradsPerCard = zeroStage >= 2 ? grads / n : grads;
  const optimizerPerCard = optimizer / n; // sharded in every ZeRO stage

  let activationsPerCard = getActivationMemory(
    opts.seqLen,
    opts.batchSize,
    opts.hiddenSize,
    opts.layers
  );
  if (opts.gradCheckpointing) {
    activationsPerCard *= 0.2; // same recompute trade-off as the LoRA path
  }

  return {
    weightsPerCard,
    gradsPerCard,
    optimizerPerCard,
    activationsPerCard,
    totalPerCard: weightsPerCard + gradsPerCard + optimizerPerCard + activationsPerCard,
  };
}

// Smallest N (1..64) where per-card memory + 5% safety fits in
// perGpuUsableVram. Callers should pass VRAM *after* subtracting the OS
// reserve (vram - osReserve). Returns -1 when even 64 GPUs cannot fit.
export function getMinGPUCount(
  params: number,
  perGpuUsableVram: number,
  zeroStage: ZeroStage,
  opts: FullFTOptions
): number {
  for (let n = 1; n <= 64; n++) {
    const { totalPerCard } = getFullFTMemory(params, n, zeroStage, opts);
    if (totalPerCard + getSafetyMargin(totalPerCard) <= perGpuUsableVram) {
      return n;
    }
  }
  return -1;
}

export function getMultiGPUPerCard(
  totalWeights: number,
  gpuCount: number,
  kvCache: number,
  strategy: "tensor_parallel" | "pipeline_parallel" = "tensor_parallel"
): { weightsPerCard: number; kvPerCard: number } {
  if (gpuCount <= 1) return { weightsPerCard: totalWeights, kvPerCard: kvCache };
  if (strategy === "tensor_parallel") {
    return { weightsPerCard: totalWeights / gpuCount, kvPerCard: kvCache / gpuCount };
  }
  // pipeline parallel usually splits weights but keeps full KV cache active on the processing card
  return { weightsPerCard: totalWeights / gpuCount, kvPerCard: kvCache };
}

export function getCPUSplit(
  totalLayers: number,
  gpuVram: number,
  layerSize: number
): { gpuLayers: number; cpuLayers: number } {
  const maxGpuLayers = Math.floor(gpuVram / layerSize);
  const gpuLayers = Math.min(totalLayers, maxGpuLayers);
  return {
    gpuLayers,
    cpuLayers: totalLayers - gpuLayers,
  };
}

export function getRuntimeForQuant(quant: string): string {
  switch (quant) {
    case "GGUF Q4_K_M":
      return "llama.cpp";
    case "EXL2":
      return "ExLlamaV2 / TabbyAPI";
    case "AWQ":
      return "vLLM / AWQ";
    case "INT4":
      return "vLLM / GPTQ";
    case "INT8":
    case "FP16":
      return "vLLM";
    default:
      return "vLLM";
  }
}

// Honest display range around a memory estimate: runtime overhead varies
// with allocator, driver, and serving stack.
export function getMemoryRange(total: number): { low: number; high: number } {
  return { low: total * 0.93, high: total * 1.18 };
}

// Achieved memory-bandwidth efficiency per runtime during token generation.
// Decode is bandwidth-bound: each token reads ~(activeParams) weights once.
// Single-stream decode achieves roughly 40-60% of peak bandwidth in practice;
// these sit mid-range so derived estimates don't outrank measured benchmarks.
const RUNTIME_EFFICIENCY: Record<string, number> = {
  "GGUF Q4_K_M": 0.55, // llama.cpp
  AWQ: 0.5, // vLLM
  INT4: 0.5, // vLLM / GPTQ
  INT8: 0.5, // vLLM
  FP16: 0.5, // vLLM
  EXL2: 0.55, // ExLlamaV2
};
const DEFAULT_EFFICIENCY = 0.5;

// Multi-GPU tensor-parallel decode scaling: ~1.7x for 2 cards, diminishing after
// (interconnect overhead eats into each added card).
function multiGPUScaling(gpuCount: number): number {
  if (gpuCount <= 1) return 1;
  return Math.pow(gpuCount, 0.77); // 2 -> 1.71, 3 -> 2.33, 4 -> 2.91
}

export function estimateTPS(
  model: ModelSpec,
  gpu: GPUSpec,
  quant: string,
  offloadRatio: number, // 0 to 1
  gpuCount: number,
  benchmarks: Benchmark[]
): number {
  const bench = benchmarks.find(
    (b) => b.modelId === model.id && b.gpuId === gpu.id && b.quant === quant
  );

  let baseTPS: number;
  if (bench) {
    baseTPS = bench.tps;
  } else {
    // Bandwidth-derived estimate: bytes read per token ~= active weights.
    const activeParams = model.activeParams ?? model.params;
    const bytesPerToken = (activeParams * getBitsPerWeight(quant)) / 8;
    const efficiency = RUNTIME_EFFICIENCY[quant] ?? DEFAULT_EFFICIENCY;
    baseTPS = (gpu.bandwidth * 1e9 * efficiency) / bytesPerToken;
  }

  // Multipliers
  if (offloadRatio > 0) {
    baseTPS = baseTPS * (1 - offloadRatio * 0.8); // High penalty for CPU offload
  }

  baseTPS = baseTPS * multiGPUScaling(gpuCount);

  return Math.min(500, Math.max(1, baseTPS));
}
