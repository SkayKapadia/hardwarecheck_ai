import gpusJson from "@/data/gpus.json";
import modelsJson from "@/data/models.json";
import benchmarksJson from "@/data/benchmarks.json";
import {
  Benchmark,
  GPUSpec,
  ModelSpec,
  getModelWeights,
  getKVCache,
  getQuantMetadata,
  getBitsPerWeight,
  estimateTPS,
} from "@/lib/calc";

export const SHARE_GPUS = gpusJson as GPUSpec[];
export const SHARE_MODELS = modelsJson as unknown as ModelSpec[];
export const SHARE_BENCHMARKS = benchmarksJson as Benchmark[];

export interface ShareResult {
  model: ModelSpec;
  gpu: GPUSpec;
  quant: string;
  totalGB: number;
  fits: boolean;
  tps: number;
}

// Headline verdict for a share card: does the model fit on the GPU, and how
// fast would it run? Mirrors the inference scenario's INT4-style math at a
// fixed 4K context, batch 1.
export function computeShareResult(
  modelId?: string,
  gpuId?: string,
  quant?: string
): ShareResult | null {
  const model = SHARE_MODELS.find((m) => m.id === modelId);
  const gpu = SHARE_GPUS.find((g) => g.id === gpuId);
  if (!model || !gpu) return null;

  const q = quant ?? "INT4";
  const weights =
    getModelWeights(model.params, getBitsPerWeight(q)) + getQuantMetadata(model.params, q);
  const kv = getKVCache(model.layers, model.hiddenSize, model.queryHeads, model.kvHeads, 4096, 1, 16);
  const totalGB = weights + kv + gpu.osReserve;
  const fits = totalGB <= gpu.vram;
  const tps = estimateTPS(model, gpu, q, 0, 1, SHARE_BENCHMARKS);

  return { model, gpu, quant: q, totalGB, fits, tps: Math.round(tps) };
}
