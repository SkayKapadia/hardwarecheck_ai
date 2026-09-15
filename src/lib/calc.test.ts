import { describe, it, expect } from "vitest";
import {
  getModelWeights,
  getKVCache,
  getBitsPerWeight,
  getQuantMetadata,
  getLoRATrainableParams,
  getLoRAGradientMemory,
  getLoRAOptimizerMemory,
  estimateTPS,
  getMultiGPUPerCard,
  getMemoryRange,
  getRuntimeForQuant,
  type ModelSpec,
  type GPUSpec,
  type Benchmark,
} from "./calc";
import modelsJson from "../data/models.json";
import gpusJson from "../data/gpus.json";
import cloudJson from "../data/cloud.json";
import benchmarksJson from "../data/benchmarks.json";

const models = modelsJson as unknown as ModelSpec[];
const gpus = gpusJson as unknown as GPUSpec[];
const cloud = cloudJson as unknown as {
  provider: string;
  gpuId: string;
  hourlyPrice: number;
  liquidity?: string;
  asOf?: string;
}[];
const benchmarks = benchmarksJson as unknown as Benchmark[];

// ---------- helpers ----------

function makeModel(overrides: Partial<ModelSpec> = {}): ModelSpec {
  return {
    id: "test-model",
    name: "Test Model",
    params: 8e9,
    layers: 32,
    hiddenSize: 4096,
    queryHeads: 32,
    kvHeads: 8,
    vocabSize: 128256,
    maxContext: 8192,
    architecture: "dense",
    family: "Test",
    ...overrides,
  };
}

function makeGpu(overrides: Partial<GPUSpec> = {}): GPUSpec {
  return {
    id: "test-gpu",
    name: "Test GPU",
    vram: 24,
    bandwidth: 1000,
    interconnect: "pcie4",
    osReserve: 1.5,
    ...overrides,
  };
}

// ---------- 1. getModelWeights ----------

describe("getModelWeights", () => {
  it("8B params at 4 bits = 4.00 GB", () => {
    expect(getModelWeights(8e9, 4)).toBeCloseTo(4.0, 6);
  });

  it("8B params at 16 bits = 16 GB", () => {
    expect(getModelWeights(8e9, 16)).toBeCloseTo(16, 6);
  });
});

// ---------- 2. getKVCache (audit fixture: Llama 3 8B) ----------

describe("getKVCache", () => {
  // Llama 3 8B: 32 layers, hidden 4096, 32 query heads, 8 KV heads
  it("Llama 3 8B, ctx 4096, batch 1, FP16 cache = 0.537 GB", () => {
    expect(getKVCache(32, 4096, 32, 8, 4096, 1, 16)).toBeCloseTo(0.537, 3);
  });

  it("batch 4 scales linearly to ~2.147 GB", () => {
    expect(getKVCache(32, 4096, 32, 8, 4096, 4, 16)).toBeCloseTo(2.147, 3);
  });
});

// ---------- 3. getBitsPerWeight ----------

describe("getBitsPerWeight", () => {
  it.each([
    ["FP16", 16],
    ["INT8", 8],
    ["INT4", 4],
    ["AWQ", 4],
    ["GGUF Q4_K_M", 4.5],
    ["EXL2", 4],
  ])("%s -> %i bits", (format, bits) => {
    expect(getBitsPerWeight(format)).toBe(bits);
  });
});

// ---------- 4. getQuantMetadata ----------

describe("getQuantMetadata", () => {
  it("INT4 8B -> ~0.80 GB of quant metadata overhead", () => {
    const meta = getQuantMetadata(8e9, "INT4");
    expect(meta).toBeGreaterThanOrEqual(0.7);
    expect(meta).toBeLessThanOrEqual(0.9);
  });
});

// ---------- 5. getLoRATrainableParams (audit fixture) ----------

describe("getLoRATrainableParams", () => {
  it("rank 16, hidden 4096, [q_proj, v_proj], 32 layers = ~8.39M", () => {
    const p = getLoRATrainableParams(16, 4096, ["q_proj", "v_proj"], 32);
    expect(p).toBeGreaterThan(8e6);
    expect(p).toBeLessThan(9e6);
  });

  it("all 4 attention modules doubles the count vs q+v", () => {
    const qv = getLoRATrainableParams(16, 4096, ["q_proj", "v_proj"], 32);
    const all4 = getLoRATrainableParams(
      16,
      4096,
      ["q_proj", "k_proj", "v_proj", "o_proj"],
      32
    );
    expect(all4).toBeCloseTo(qv * 2, 6);
  });

  it("MLP modules produce larger counts than attention modules", () => {
    const attn = getLoRATrainableParams(16, 4096, ["q_proj", "v_proj"], 32);
    const mlp = getLoRATrainableParams(16, 4096, ["gate_proj", "up_proj"], 32);
    expect(mlp).toBeGreaterThan(attn);
  });
});

// ---------- 6. LoRA gradient / optimizer memory ----------

describe("LoRA gradient and optimizer memory", () => {
  const trainable = getLoRATrainableParams(16, 4096, ["q_proj", "v_proj"], 32);

  it("gradients = 4 bytes per trainable param (~0.0336 GB)", () => {
    expect(getLoRAGradientMemory(trainable)).toBeCloseTo(0.0336, 3);
  });

  it("optimizer = 8 bytes per trainable param (~0.067 GB)", () => {
    expect(getLoRAOptimizerMemory(trainable)).toBeCloseTo(0.067, 3);
  });

  it("optimizer is exactly 2x gradients", () => {
    expect(getLoRAOptimizerMemory(trainable)).toBeCloseTo(
      getLoRAGradientMemory(trainable) * 2,
      10
    );
  });
});

// ---------- 7. estimateTPS ----------

describe("estimateTPS", () => {
  const llama3 = models.find((m) => m.id === "llama3-8b")!;
  const rtx4090 = gpus.find((g) => g.id === "rtx4090")!;
  const rtx3090 = gpus.find((g) => g.id === "rtx3090")!;

  it("exact benchmark match wins (llama3-8b / rtx4090 / INT4 = 120)", () => {
    expect(estimateTPS(llama3, rtx4090, "INT4", 0, 1, benchmarks)).toBe(120);
  });

  it("derived fallback: 8B dense INT4 on RTX 3090 -> 100-130 tok/s", () => {
    const madeUp = makeModel({ id: "no-benchmark-8b" });
    const tps = estimateTPS(madeUp, rtx3090, "INT4", 0, 1, benchmarks);
    expect(tps).toBeGreaterThan(100);
    expect(tps).toBeLessThan(130);
  });

  it("derived 3090 estimate is slower than measured 4090 benchmark", () => {
    // Regression guard: derived estimates must not outrank real benchmarks.
    const madeUp = makeModel({ id: "no-benchmark-8b" });
    const derived3090 = estimateTPS(madeUp, rtx3090, "INT4", 0, 1, benchmarks);
    const measured4090 = estimateTPS(llama3, rtx4090, "INT4", 0, 1, benchmarks);
    expect(derived3090).toBeLessThan(measured4090);
  });

  it("MoE speed uses activeParams, not total params", () => {
    const moe = makeModel({
      id: "moe-671b",
      architecture: "moe",
      params: 671e9,
      activeParams: 37e9,
    });
    const denseSame = makeModel({ id: "dense-671b", params: 671e9 });
    const tpsMoe = estimateTPS(moe, rtx3090, "INT4", 0, 1, benchmarks);
    const tpsDense = estimateTPS(denseSame, rtx3090, "INT4", 0, 1, benchmarks);
    expect(tpsMoe).toBeGreaterThan(tpsDense * 5);
  });

  it("2 GPUs scale ~1.71x over a single card", () => {
    const madeUp = makeModel({ id: "no-benchmark-8b" });
    const single = estimateTPS(madeUp, rtx3090, "INT4", 0, 1, benchmarks);
    const dual = estimateTPS(madeUp, rtx3090, "INT4", 0, 2, benchmarks);
    expect(dual / single).toBeCloseTo(1.71, 2);
  });

  it("offload ratio 0.5 -> 60% of base tps", () => {
    const madeUp = makeModel({ id: "no-benchmark-8b" });
    const base = estimateTPS(madeUp, rtx3090, "INT4", 0, 1, benchmarks);
    const offloaded = estimateTPS(madeUp, rtx3090, "INT4", 0.5, 1, benchmarks);
    expect(offloaded / base).toBeCloseTo(0.6, 6);
  });

  it("clamps to a minimum of 1 tok/s", () => {
    const huge = makeModel({ id: "huge", params: 2e12 });
    const slow = makeGpu({ bandwidth: 10 });
    expect(estimateTPS(huge, slow, "FP16", 0.9, 1, benchmarks)).toBe(1);
  });

  it("clamps to a maximum of 500 tok/s", () => {
    const tiny = makeModel({ id: "tiny", params: 1e6 });
    const fast = makeGpu({ bandwidth: 8000 });
    expect(estimateTPS(tiny, fast, "INT4", 0, 4, benchmarks)).toBe(500);
  });
});

// ---------- 8. getMultiGPUPerCard ----------

describe("getMultiGPUPerCard", () => {
  it("tensor parallel splits weights AND kv per card", () => {
    const { weightsPerCard, kvPerCard } = getMultiGPUPerCard(100, 2, 8, "tensor_parallel");
    expect(weightsPerCard).toBeCloseTo(50, 6);
    expect(kvPerCard).toBeCloseTo(4, 6);
  });

  it("pipeline parallel splits weights but keeps full KV", () => {
    const { weightsPerCard, kvPerCard } = getMultiGPUPerCard(100, 2, 8, "pipeline_parallel");
    expect(weightsPerCard).toBeCloseTo(50, 6);
    expect(kvPerCard).toBeCloseTo(8, 6);
  });
});

// ---------- 9. getMemoryRange ----------

describe("getMemoryRange", () => {
  it("low is ~0.93x and high is ~1.18x the estimate", () => {
    const { low, high } = getMemoryRange(100);
    expect(low).toBeCloseTo(93, 6);
    expect(high).toBeCloseTo(118, 6);
  });
});

// ---------- 10. getRuntimeForQuant ----------

describe("getRuntimeForQuant", () => {
  it("GGUF Q4_K_M -> llama.cpp", () => {
    expect(getRuntimeForQuant("GGUF Q4_K_M")).toBe("llama.cpp");
  });

  it("EXL2 -> ExLlamaV2, not bare vLLM", () => {
    const runtime = getRuntimeForQuant("EXL2");
    expect(runtime).toContain("ExLlamaV2");
    expect(runtime).not.toBe("vLLM");
  });

  it("INT4 -> vLLM", () => {
    expect(getRuntimeForQuant("INT4")).toContain("vLLM");
  });
});

// ---------- 11. Data invariants ----------

describe("models.json invariants", () => {
  it("has no duplicate ids", () => {
    const ids = models.map((m) => m.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("every MoE entry has activeParams < params", () => {
    for (const m of models.filter((m) => m.architecture === "moe")) {
      expect(m.activeParams, m.id).toBeDefined();
      expect(m.activeParams!, m.id).toBeLessThan(m.params);
    }
  });

  it("every dense entry has no activeParams", () => {
    for (const m of models.filter((m) => m.architecture === "dense")) {
      expect(m.activeParams, m.id).toBeUndefined();
    }
  });

  it("all entries have family, source, and verifiedOn", () => {
    for (const m of models) {
      expect(m.family, m.id).toBeTruthy();
      expect(m.source, m.id).toBeTruthy();
      expect(m.verifiedOn, m.id).toBeTruthy();
    }
  });

  it("gpt2 params ~= 124.4M", () => {
    const gpt2 = models.find((m) => m.id === "openai-community/gpt2")!;
    expect(Math.abs(gpt2.params - 124.4e6)).toBeLessThan(1e6);
  });

  it("DeepSeek V3 total params ~= 671B", () => {
    const ds = models.find((m) => m.id === "deepseek-ai/DeepSeek-V3")!;
    expect(Math.abs(ds.params - 671e9)).toBeLessThan(5e9);
  });

  it("Qwen3-30B-A3B total params ~= 30.5B", () => {
    const q = models.find((m) => m.id === "Qwen/Qwen3-30B-A3B")!;
    expect(Math.abs(q.params - 30.5e9)).toBeLessThan(0.2e9);
  });
});

describe("gpus.json invariants", () => {
  it("has unique ids", () => {
    const ids = gpus.map((g) => g.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("vram, bandwidth, and price are positive", () => {
    for (const g of gpus) {
      expect(g.vram, g.id).toBeGreaterThan(0);
      expect(g.bandwidth, g.id).toBeGreaterThan(0);
      expect(g.price, g.id).toBeGreaterThan(0);
    }
  });

  it("includes rtx5090 and h200-141gb", () => {
    expect(gpus.some((g) => g.id === "rtx5090")).toBe(true);
    expect(gpus.some((g) => g.id === "h200-141gb")).toBe(true);
  });
});

describe("cloud.json invariants", () => {
  it("every entry has liquidity, asOf, and hourlyPrice > 0", () => {
    for (const c of cloud) {
      expect(c.liquidity, c.gpuId).toBeTruthy();
      expect(c.asOf, c.gpuId).toBeTruthy();
      expect(c.hourlyPrice, c.gpuId).toBeGreaterThan(0);
    }
  });

  it("every cloud gpuId exists in gpus.json", () => {
    const gpuIds = new Set(gpus.map((g) => g.id));
    for (const c of cloud) {
      expect(gpuIds.has(c.gpuId), c.gpuId).toBe(true);
    }
  });
});

describe("benchmarks.json invariants", () => {
  it("every benchmark modelId and gpuId exists", () => {
    const modelIds = new Set(models.map((m) => m.id));
    const gpuIds = new Set(gpus.map((g) => g.id));
    for (const b of benchmarks) {
      expect(modelIds.has(b.modelId), b.modelId).toBe(true);
      expect(gpuIds.has(b.gpuId), b.gpuId).toBe(true);
    }
  });
});
