import { create } from "zustand";

export type Scenario = "inference" | "finetune" | "compare" | "cloud";
export type Quantization = "FP16" | "INT8" | "INT4" | "GGUF Q4_K_M" | "AWQ" | "EXL2";
export type FinetuneQuant = "FP16" | "INT8" | "INT4";

export interface LoadoutState {
  scenario: Scenario;
  selectedModel: string;
  selectedGPUs: string[];
  compareGPUs: string[];
  quantization: Quantization;
  contextLength: number;
  batchSize: number;
  cpuOffload: boolean;
  systemRam: number; // GB
  // Fine Tune specific
  finetuneQuant: FinetuneQuant;
  loraRank: number;
  loraAlpha: number;
  targetModules: string[];
  gradientCheckpointing: boolean;
  trainBatchSize: number;

  // Actions
  setScenario: (scenario: Scenario) => void;
  setSelectedModel: (modelId: string) => void;
  toggleGPU: (gpuId: string) => void;
  setSelectedGPUs: (gpuIds: string[]) => void;
  toggleCompareGPU: (gpuId: string) => void;
  setCompareGPUs: (gpuIds: string[]) => void;
  setQuantization: (quant: Quantization) => void;
  setContextLength: (ctx: number) => void;
  setBatchSize: (batch: number) => void;
  setCpuOffload: (offload: boolean) => void;
  setSystemRam: (ram: number) => void;
  setFinetuneQuant: (quant: FinetuneQuant) => void;
  setLoraRank: (rank: number) => void;
  setLoraAlpha: (alpha: number) => void;
  toggleTargetModule: (module: string) => void;
  setGradientCheckpointing: (gc: boolean) => void;
  setTrainBatchSize: (batch: number) => void;
  hydrateFromUrl: (query: URLSearchParams) => void;
  resetToDefaults: () => void;
}

const DEFAULT_STATE = {
  scenario: "inference" as Scenario,
  selectedModel: "llama3-8b",
  selectedGPUs: ["rtx4090"],
  compareGPUs: ["rtx4090", "rtx3090"],
  quantization: "INT4" as Quantization,
  contextLength: 4096,
  batchSize: 1,
  cpuOffload: false,
  systemRam: 32,

  finetuneQuant: "INT4" as FinetuneQuant,
  loraRank: 16,
  loraAlpha: 32,
  targetModules: ["q_proj", "v_proj"],
  gradientCheckpointing: true,
  trainBatchSize: 4,
};

export const useStore = create<LoadoutState>((set, get) => ({
  ...DEFAULT_STATE,

  setScenario: (scenario) => set({ scenario }),
  setSelectedModel: (selectedModel) => set({ selectedModel }),
  toggleGPU: (gpuId) => {
    const { selectedGPUs } = get();
    if (selectedGPUs.includes(gpuId)) {
      // allow empty? maybe not
      if (selectedGPUs.length > 1) {
        set({ selectedGPUs: selectedGPUs.filter((id) => id !== gpuId) });
      }
    } else {
      set({ selectedGPUs: [...selectedGPUs, gpuId] });
    }
  },
  setSelectedGPUs: (selectedGPUs) => set({ selectedGPUs }),
  toggleCompareGPU: (gpuId) => {
    const { compareGPUs } = get();
    if (compareGPUs.includes(gpuId)) {
      if (compareGPUs.length > 1) {
        set({ compareGPUs: compareGPUs.filter((id) => id !== gpuId) });
      }
    } else {
      set({ compareGPUs: [...compareGPUs, gpuId] });
    }
  },
  setCompareGPUs: (compareGPUs) => set({ compareGPUs }),
  setQuantization: (quantization) => set({ quantization }),
  setContextLength: (contextLength) => set({ contextLength }),
  setBatchSize: (batchSize) => set({ batchSize }),
  setCpuOffload: (cpuOffload) => set({ cpuOffload }),
  setSystemRam: (systemRam) => set({ systemRam }),
  setFinetuneQuant: (finetuneQuant) => set({ finetuneQuant }),
  setLoraRank: (loraRank) => set({ loraRank }),
  setLoraAlpha: (loraAlpha) => set({ loraAlpha }),
  toggleTargetModule: (mod) => {
    const { targetModules } = get();
    if (targetModules.includes(mod)) {
      set({ targetModules: targetModules.filter((m) => m !== mod) });
    } else {
      set({ targetModules: [...targetModules, mod] });
    }
  },
  setGradientCheckpointing: (gradientCheckpointing) => set({ gradientCheckpointing }),
  setTrainBatchSize: (trainBatchSize) => set({ trainBatchSize }),

  hydrateFromUrl: (query) => {
    const updates: Partial<LoadoutState> = {};
    if (query.get("scenario")) updates.scenario = query.get("scenario") as Scenario;
    if (query.get("model")) updates.selectedModel = query.get("model")!;
    if (query.get("gpu")) updates.selectedGPUs = query.get("gpu")!.split(",");
    if (query.get("cgpu")) updates.compareGPUs = query.get("cgpu")!.split(",");
    if (query.get("quant")) updates.quantization = query.get("quant") as Quantization;
    if (query.get("ctx")) updates.contextLength = parseInt(query.get("ctx")!, 10);
    if (query.get("batch")) updates.batchSize = parseInt(query.get("batch")!, 10);
    if (query.get("offload")) updates.cpuOffload = query.get("offload") === "true";
    if (query.get("ram")) updates.systemRam = parseInt(query.get("ram")!, 10);
    
    // fine tune
    if (query.get("fquant")) updates.finetuneQuant = query.get("fquant") as FinetuneQuant;
    if (query.get("loraRank")) updates.loraRank = parseInt(query.get("loraRank")!, 10);
    if (query.get("loraAlpha")) updates.loraAlpha = parseInt(query.get("loraAlpha")!, 10);
    if (query.get("modules")) updates.targetModules = query.get("modules")!.split(",");
    if (query.get("gc")) updates.gradientCheckpointing = query.get("gc") === "true";
    if (query.get("tBatch")) updates.trainBatchSize = parseInt(query.get("tBatch")!, 10);

    set(updates);
  },

  resetToDefaults: () => set({ ...DEFAULT_STATE }),
}));
