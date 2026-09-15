"use client";

import { useStore } from "@/lib/store";
import { Card } from "@/components/ui/Card";
import gpus from "@/data/gpus.json";
import models from "@/data/models.json";
import cloud from "@/data/cloud.json";
import benchmarks from "@/data/benchmarks.json";
import { Cpu, Check, X } from "lucide-react";
import { getModelById } from "@/components/ModelPicker";
import { getBitsPerWeight, getModelWeights, getQuantMetadata, getActivationMemory, getRuntimeReserve, getSafetyMargin, estimateTPS, MODELS_DATA_AS_OF, type ModelSpec } from "@/lib/calc";

export function CompareScenario() {
  const store = useStore();
  const compareGPUsData = store.compareGPUs.map(id => gpus.find(g => g.id === id)!).filter(Boolean);

  const selectedModelData = (getModelById(store.selectedModel) ?? models[0]) as ModelSpec;
  const bitsPerWeight = getBitsPerWeight(store.quantization);
  const rawWeights = getModelWeights(selectedModelData.params, bitsPerWeight);
  const metadata = getQuantMetadata(selectedModelData.params, store.quantization);
  const activations = getActivationMemory(store.contextLength, store.batchSize, selectedModelData.hiddenSize, selectedModelData.layers);

  return (
    <div className="flex-1 flex flex-col gap-6 h-full overflow-hidden">

      {/* Top section: Pickers */}
      <div className="shrink-0 flex gap-6">
        <div className="w-1/3">
           <h2 className="text-sm font-mono text-muted-foreground uppercase tracking-widest border-b border-border pb-2 flex items-center gap-2 mb-4">
            <Cpu className="w-4 h-4" /> Compare Up To 4 GPUs
          </h2>
          <div className="grid grid-cols-3 gap-2 overflow-y-auto max-h-40 custom-scrollbar">
            {gpus.map((g) => (
              <Card
                key={g.id}
                selected={store.compareGPUs.includes(g.id)}
                onClick={() => {
                  if (store.compareGPUs.includes(g.id)) {
                     store.toggleCompareGPU(g.id);
                  } else if (store.compareGPUs.length < 4) {
                     store.toggleCompareGPU(g.id);
                  }
                }}
                disabled={!store.compareGPUs.includes(g.id) && store.compareGPUs.length >= 4}
                className="p-2"
              >
                <div className="text-xs font-bold truncate">{g.name}</div>
              </Card>
            ))}
          </div>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="flex-1 border border-border bg-card/40 overflow-auto relative p-6">
        {compareGPUsData.length === 0 ? (
          <div className="flex items-center justify-center h-full text-muted-foreground font-mono">
            Select at least one GPU to compare.
          </div>
        ) : (
          <>
          <table className="w-full text-sm font-mono text-left border-collapse">
            <thead>
              <tr>
                <th className="p-4 border-b border-r border-border text-muted-foreground">Specification</th>
                {compareGPUsData.map(g => (
                  <th key={g.id} className="p-4 border-b border-border text-center text-primary text-lg font-bold">
                    {g.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {/* VRAM */}
              <tr>
                <td className="p-4 border-b border-r border-border text-muted-foreground bg-card/20">VRAM</td>
                {compareGPUsData.map(g => (
                  <td key={g.id} className="p-4 border-b border-border text-center">
                    {g.vram} GB
                  </td>
                ))}
              </tr>

              {/* Bandwidth */}
              <tr>
                <td className="p-4 border-b border-r border-border text-muted-foreground bg-card/20">Bandwidth</td>
                {compareGPUsData.map(g => (
                  <td key={g.id} className="p-4 border-b border-border text-center">
                    {g.bandwidth} GB/s
                  </td>
                ))}
              </tr>

              {/* Interconnect */}
              <tr>
                <td className="p-4 border-b border-r border-border text-muted-foreground bg-card/20">Interconnect</td>
                {compareGPUsData.map(g => (
                  <td key={g.id} className="p-4 border-b border-border text-center uppercase">
                    {g.interconnect}
                  </td>
                ))}
              </tr>

               {/* Fits Model */}
               <tr>
                <td className="p-4 border-b border-r border-border text-muted-foreground bg-card/20">
                  Fits {selectedModelData.name} ({store.quantization})
                </td>
                {compareGPUsData.map(g => {
                   const reserve = getRuntimeReserve(g);
                   const totalUsed = rawWeights + metadata + activations + reserve;
                   const safety = getSafetyMargin(totalUsed);
                   const fits = (totalUsed + safety) <= g.vram;

                   return (
                    <td key={g.id} className="p-4 border-b border-border text-center">
                      {fits ? <Check className="w-5 h-5 mx-auto text-primary" /> : <X className="w-5 h-5 mx-auto text-destructive" />}
                    </td>
                  )
                })}
              </tr>

               {/* Est TPS */}
               <tr>
                <td className="p-4 border-b border-r border-border text-muted-foreground bg-card/20">Estimated TPS</td>
                {compareGPUsData.map(g => {
                  const tps = estimateTPS(selectedModelData as ModelSpec, g, store.quantization, 0, 1, benchmarks);
                  return (
                    <td key={g.id} className="p-4 border-b border-border text-center text-primary font-bold">
                      {tps.toFixed(0)} tok/s
                    </td>
                  )
                })}
              </tr>

              {/* Rent Cost */}
              <tr>
                <td className="p-4 border-b border-r border-border text-muted-foreground bg-card/20">Cost to Rent / Hr</td>
                {compareGPUsData.map(g => {
                  const cheapest = cloud
                    .filter(c => c.gpuId === g.id)
                    .sort((a, b) => a.hourlyPrice - b.hourlyPrice)[0];
                  return (
                    <td key={g.id} className="p-4 border-b border-border text-center text-secondary">
                      {cheapest ? `$${cheapest.hourlyPrice.toFixed(2)} (${cheapest.provider})` : "—"}
                    </td>
                  )
                })}
              </tr>
            </tbody>
          </table>
          <p className="text-[10px] font-mono text-muted-foreground mt-4">
            Speed estimates are bandwidth-derived unless a measured benchmark exists · Model data verified {MODELS_DATA_AS_OF}
          </p>
          </>
        )}
      </div>

    </div>
  );
}
