export type ModelArchitecture = "dense" | "moe";

export interface ModelSpec {
  id: string;
  name: string;
  params: number;
  layers: number;
  hiddenSize: number;
  queryHeads: number;
  kvHeads: number;
  vocabSize: number;
  maxContext: number;
  architecture: ModelArchitecture;
  activeParams?: number;
}

export interface GPUSpec {
  id: string;
  name: string;
  vram: number; // in GB
  bandwidth: number; // in GB/s
  interconnect: string;
  osReserve: number; // in GB
}

export interface Benchmark {
  modelId: string;
  gpuId: string;
  quant: string;
  tps: number;
}

export function getQuantMetadata(params: number, format: string): number {
  let overheadPercent = 0.0;
  if (format === "GGUF Q4_K_M" || format === "AWQ" || format === "EXL2") {
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
  hiddenSize: number
): number {
  // Rough estimation for activation workspace during inference
  const bytes = ctxLen * batchSize * hiddenSize * 2;
  return bytes / 1e9;
}

export function getRuntimeReserve(gpu: GPUSpec): number {
  return gpu.osReserve;
}

export function getSafetyMargin(total: number): number {
  return total * 0.05; // 5% safety margin
}

export function getQLoRAOverhead(
  baseParams: number,
  rank: number,
  hiddenSize: number,
  moduleCount: number
): number {
  // Approx formula: Trainable params ~ (rank * hiddenSize * 2 * moduleCount)
  // Optimizer state for AdamW is ~14 bytes per trainable parameter
  const trainableParams = rank * hiddenSize * 2 * moduleCount * 80; // roughly estimating total matrices
  return (trainableParams * 14) / 1e9;
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

export function estimateTPS(
  modelKey: string,
  gpuKey: string,
  quant: string,
  offloadRatio: number, // 0 to 1
  gpuCount: number,
  benchmarks: Benchmark[]
): number {
  const bench = benchmarks.find(
    (b) => b.modelId === modelKey && b.gpuId === gpuKey && b.quant === quant
  );

  let baseTPS = bench ? bench.tps : 40; // Fallback estimate

  // Multipliers
  if (offloadRatio > 0) {
    baseTPS = baseTPS * (1 - offloadRatio * 0.8); // High penalty for CPU offload
  }

  if (gpuCount > 1) {
    baseTPS = baseTPS * 1.8; // Rough multiplier for 2 GPUs, etc.
  }

  return Math.max(1, baseTPS);
}
