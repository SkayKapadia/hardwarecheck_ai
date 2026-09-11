"use client";

import { useStore } from "@/lib/store";
import { Card } from "@/components/ui/Card";
import gpus from "@/data/gpus.json";
import models from "@/data/models.json";
import cloud from "@/data/cloud.json";
import benchmarks from "@/data/benchmarks.json";
import { Cpu, Check, X } from "lucide-react";
import { getBitsPerWeight, getModelWeights, getQuantMetadata, getActivationMemory, getRuntimeReserve, getSafetyMargin, estimateTPS } from "@/lib/calc";

export function CompareScenario() {
  const store = useStore();
  const selectedGPUsData = store.selectedGPUs.map(id => gpus.find(g => g.id === id)!).filter(Boolean);

  const selectedModelData = models.find((m) => m.id === store.selectedModel)!;
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
                selected={store.selectedGPUs.includes(g.id)}
                onClick={() => {
                  if (store.selectedGPUs.includes(g.id)) {
                     store.toggleGPU(g.id);
                  } else if (store.selectedGPUs.length < 4) {
                     store.toggleGPU(g.id);
                  }
                }}
                disabled={!store.selectedGPUs.includes(g.id) && store.selectedGPUs.length >= 4}
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
        {selectedGPUsData.length === 0 ? (
          <div className="flex items-center justify-center h-full text-muted-foreground font-mono">
            Select at least one GPU to compare.
          </div>
        ) : (
          <table className="w-full text-sm font-mono text-left border-collapse">
            <thead>
              <tr>
                <th className="p-4 border-b border-r border-border text-muted-foreground">Specification</th>
                {selectedGPUsData.map(g => (
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
                {selectedGPUsData.map(g => (
                  <td key={g.id} className="p-4 border-b border-border text-center">
                    {g.vram} GB
                  </td>
                ))}
              </tr>
              
              {/* Bandwidth */}
              <tr>
                <td className="p-4 border-b border-r border-border text-muted-foreground bg-card/20">Bandwidth</td>
                {selectedGPUsData.map(g => (
                  <td key={g.id} className="p-4 border-b border-border text-center">
                    {g.bandwidth} GB/s
                  </td>
                ))}
              </tr>

              {/* Interconnect */}
              <tr>
                <td className="p-4 border-b border-r border-border text-muted-foreground bg-card/20">Interconnect</td>
                {selectedGPUsData.map(g => (
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
                {selectedGPUsData.map(g => {
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
                {selectedGPUsData.map(g => {
                  const tps = estimateTPS(store.selectedModel, g.id, store.quantization, 0, 1, benchmarks);
                  return (
                    <td key={g.id} className="p-4 border-b border-border text-center text-primary font-bold">
                      {tps.toFixed(0)} tok/s
                    </td>
                  )
                })}
              </tr>

              {/* Rent Cost */}
              <tr>
                <td className="p-4 border-r border-border text-muted-foreground bg-card/20">Cost to Rent / Hr</td>
                {selectedGPUsData.map(g => {
                  const cloudPricing = cloud.find(c => c.gpuId === g.id);
                  return (
                    <td key={g.id} className="p-4 border-border text-center text-secondary">
                      {cloudPricing ? `$${cloudPricing.hourlyPrice.toFixed(2)}` : "N/A"}
                    </td>
                  )
                })}
              </tr>
            </tbody>
          </table>
        )}
      </div>

    </div>
  );
}
