import modelsData from "@/data/models.json";
import gpusData from "@/data/gpus.json";
import benchmarksData from "@/data/benchmarks.json";
import {
  Benchmark,
  GPUSpec,
  ModelSpec,
  estimateTPS,
  getActivationMemory,
  getBitsPerWeight,
  getKVCache,
  getMemoryRange,
  getModelWeights,
  getQuantMetadata,
  getRuntimeReserve,
  getSafetyMargin,
} from "@/lib/calc";

export const RUN_QUANT = "INT4";
export const RUN_CTX_LEN = 4096;
export const RUN_BATCH = 1;

// The 10 GPUs that get a dedicated /run landing page per model.
export const RUN_GPU_IDS = [
  "rtx4090",
  "rtx3090",
  "rtx3060",
  "rtx5090",
  "a100-80gb",
  "h100-80gb",
  "rtx6000pro-96gb",
  "mi300x-192gb",
  "m4max-128",
  "m5max-128",
] as const;

export const RUN_MODELS = modelsData as ModelSpec[];
export const RUN_GPUS = gpusData as GPUSpec[];
export const RUN_BENCHMARKS = benchmarksData as Benchmark[];

const runGpuSet = new Set<string>(RUN_GPU_IDS);
export const LANDING_GPUS = RUN_GPUS.filter((g) => runGpuSet.has(g.id));

export interface RunPair {
  model: ModelSpec;
  gpu: GPUSpec;
}

/** URL path for a combo, encoding each path segment (model ids contain "/"). */
export function runPath(modelId: string, gpuId: string): string {
  const modelSegments = modelId.split("/").map(encodeURIComponent).join("/");
  return `/run/${modelSegments}/${encodeURIComponent(gpuId)}`;
}

export function getRunPairs(): RunPair[] {
  const pairs: RunPair[] = [];
  for (const model of RUN_MODELS) {
    for (const gpu of LANDING_GPUS) {
      pairs.push({ model, gpu });
    }
  }
  return pairs;
}

/**
 * Resolve catch-all segments: the LAST segment is the gpu id, everything
 * before it joined with "/" is the model id. Params arrive URL-decoded.
 */
export function resolveRunPair(pair: string[] | undefined): RunPair | null {
  if (!pair || pair.length < 2) return null;
  const gpuId = pair[pair.length - 1];
  const modelId = pair.slice(0, -1).join("/");
  const model = RUN_MODELS.find((m) => m.id === modelId);
  const gpu = LANDING_GPUS.find((g) => g.id === gpuId);
  if (!model || !gpu) return null;
  return { model, gpu };
}

export interface RunFit {
  weights: number;
  kv: number;
  quantMeta: number;
  activations: number;
  reserve: number;
  safety: number;
  total: number;
  fits: boolean;
  range: { low: number; high: number };
  tps: number;
  headroom: number;
  fp16Weights: number;
  fp16Total: number;
  fp16Fits: boolean;
}

export function computeRunFit(model: ModelSpec, gpu: GPUSpec): RunFit {
  const weights = getModelWeights(model.params, getBitsPerWeight(RUN_QUANT));
  const kv = getKVCache(
    model.layers,
    model.hiddenSize,
    model.queryHeads,
    model.kvHeads,
    RUN_CTX_LEN,
    RUN_BATCH
  );
  const quantMeta = getQuantMetadata(model.params, RUN_QUANT);
  const activations = getActivationMemory(
    RUN_CTX_LEN,
    RUN_BATCH,
    model.hiddenSize,
    model.layers
  );
  const reserve = getRuntimeReserve(gpu);
  const subtotal = weights + kv + quantMeta + activations + reserve;
  const safety = getSafetyMargin(subtotal);
  const total = subtotal + safety;
  const fits = total <= gpu.vram;

  const fp16Weights = getModelWeights(model.params, getBitsPerWeight("FP16"));
  const fp16Subtotal =
    fp16Weights +
    kv +
    getQuantMetadata(model.params, "FP16") +
    activations +
    reserve;
  const fp16Total = fp16Subtotal + getSafetyMargin(fp16Subtotal);

  return {
    weights,
    kv,
    quantMeta,
    activations,
    reserve,
    safety,
    total,
    fits,
    range: getMemoryRange(total),
    tps: estimateTPS(model, gpu, RUN_QUANT, 0, 1, RUN_BENCHMARKS),
    headroom: gpu.vram - total,
    fp16Weights,
    fp16Total,
    fp16Fits: fp16Total <= gpu.vram,
  };
}

/**
 * Cheapest GPUs (by price) that fit this model at RUN_QUANT, excluding
 * `excludeId`. With `landingOnly`, restricts to GPUs that have a /run page
 * so the result can be linked there directly.
 */
export function cheapestFittingGpus(
  model: ModelSpec,
  excludeId: string,
  count: number,
  landingOnly: boolean = true
): GPUSpec[] {
  return (landingOnly ? LANDING_GPUS : RUN_GPUS)
    .filter((g) => g.id !== excludeId)
    .filter((g) => computeRunFit(model, g).fits)
    .sort(
      (a, b) => (a.price ?? Number.MAX_SAFE_INTEGER) - (b.price ?? Number.MAX_SAFE_INTEGER)
    )
    .slice(0, count);
}

/** Does a 2-card array of this GPU fit the model at RUN_QUANT? */
export function dualCardFits(model: ModelSpec, gpu: GPUSpec): boolean {
  const { total } = computeRunFit(model, gpu);
  return total <= gpu.vram * 2;
}

/** Other landing GPUs for "same model on other GPUs" links. */
export function relatedGpus(gpuId: string, count: number): GPUSpec[] {
  return LANDING_GPUS.filter((g) => g.id !== gpuId).slice(0, count);
}

/** Other models (same family first) for "other models on this GPU" links. */
export function relatedModels(model: ModelSpec, count: number): ModelSpec[] {
  const sameFamily = RUN_MODELS.filter(
    (m) => m.id !== model.id && m.family === model.family
  );
  const others = RUN_MODELS.filter(
    (m) => m.id !== model.id && m.family !== model.family
  );
  return [...sameFamily, ...others].slice(0, count);
}
