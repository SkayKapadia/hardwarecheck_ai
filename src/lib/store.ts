import { create } from "zustand";

export type Scenario = "inference" | "finetune" | "compare" | "cloud";
export type Quantization = "FP16" | "INT8" | "INT4" | "GGUF Q4_K_M" | "AWQ" | "EXL2";

export interface LoadoutState {
  scenario: Scenario;
  selectedModel: string;
  selectedGPUs: string[];
  quantization: Quantization;
  contextLength: number;
  batchSize: number;
  cpuOffload: boolean;
  systemRam: number; // GB
  // Fine Tune specific
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
  setQuantization: (quant: Quantization) => void;
  setContextLength: (ctx: number) => void;
  setBatchSize: (batch: number) => void;
  setCpuOffload: (offload: boolean) => void;
  setSystemRam: (ram: number) => void;
  setLoraRank: (rank: number) => void;
  setLoraAlpha: (alpha: number) => void;
  toggleTargetModule: (module: string) => void;
  setGradientCheckpointing: (gc: boolean) => void;
  setTrainBatchSize: (batch: number) => void;
  hydrateFromUrl: (query: URLSearchParams) => void;
}

export const useStore = create<LoadoutState>((set, get) => ({
  scenario: "inference",
  selectedModel: "llama3-8b",
  selectedGPUs: ["rtx4090"],
  quantization: "INT4",
  contextLength: 4096,
  batchSize: 1,
  cpuOffload: false,
  systemRam: 32,
  
  loraRank: 16,
  loraAlpha: 32,
  targetModules: ["q_proj", "v_proj"],
  gradientCheckpointing: true,
  trainBatchSize: 4,

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
  setQuantization: (quantization) => set({ quantization }),
  setContextLength: (contextLength) => set({ contextLength }),
  setBatchSize: (batchSize) => set({ batchSize }),
  setCpuOffload: (cpuOffload) => set({ cpuOffload }),
  setSystemRam: (systemRam) => set({ systemRam }),
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
    if (query.get("quant")) updates.quantization = query.get("quant") as Quantization;
    if (query.get("ctx")) updates.contextLength = parseInt(query.get("ctx")!, 10);
    if (query.get("batch")) updates.batchSize = parseInt(query.get("batch")!, 10);
    if (query.get("offload")) updates.cpuOffload = query.get("offload") === "true";
    if (query.get("ram")) updates.systemRam = parseInt(query.get("ram")!, 10);
    
    // fine tune
    if (query.get("loraRank")) updates.loraRank = parseInt(query.get("loraRank")!, 10);
    if (query.get("loraAlpha")) updates.loraAlpha = parseInt(query.get("loraAlpha")!, 10);
    if (query.get("modules")) updates.targetModules = query.get("modules")!.split(",");
    if (query.get("gc")) updates.gradientCheckpointing = query.get("gc") === "true";
    if (query.get("tBatch")) updates.trainBatchSize = parseInt(query.get("tBatch")!, 10);

    set(updates);
  },
}));
