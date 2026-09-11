"use client";

import { useStore } from "@/lib/store";
import cloud from "@/data/cloud.json";
import gpus from "@/data/gpus.json";
import models from "@/data/models.json";
import { Cloud, Server, DollarSign, Clock } from "lucide-react";
import { getBitsPerWeight, getModelWeights, getQuantMetadata, getActivationMemory, getRuntimeReserve, getSafetyMargin } from "@/lib/calc";

export function CloudScenario() {
  const store = useStore();

  const selectedModelData = models.find((m) => m.id === store.selectedModel)!;
  const bitsPerWeight = getBitsPerWeight(store.quantization);
  const rawWeights = getModelWeights(selectedModelData.params, bitsPerWeight);
  const metadata = getQuantMetadata(selectedModelData.params, store.quantization);
  const activations = getActivationMemory(store.contextLength, store.batchSize, selectedModelData.hiddenSize, selectedModelData.layers);

  // Filter cloud options to those that can fit the model
  const viableOptions = cloud.filter((c) => {
    const gpu = gpus.find((g) => g.id === c.gpuId);
    if (!gpu) return false;
    
    const reserve = getRuntimeReserve(gpu);
    const totalUsed = rawWeights + metadata + activations + reserve;
    const safety = getSafetyMargin(totalUsed);
    return (totalUsed + safety) <= c.vram;
  }).sort((a, b) => a.hourlyPrice - b.hourlyPrice);

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      
      <div className="mb-6">
        <h2 className="text-xl font-mono text-primary uppercase tracking-widest flex items-center gap-2">
          <Cloud className="w-6 h-6" /> Cloud Rent Availability
        </h2>
        <p className="text-sm font-mono text-muted-foreground mt-2">
          Showing GPU tiers that can fit {selectedModelData.name} ({store.quantization}) with {store.contextLength} context.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 overflow-y-auto pr-2 custom-scrollbar pb-6">
        {viableOptions.length === 0 ? (
          <div className="col-span-full p-8 border border-destructive bg-destructive/10 text-destructive font-mono text-center">
             No cloud instances available that fit this configuration.
          </div>
        ) : (
          viableOptions.map((option, i) => {
            const gpu = gpus.find((g) => g.id === option.gpuId);
            const monthlyCost = option.hourlyPrice * 24 * 30;
            // Fake buy price for demo
            const estBuyPrice = gpu?.vram === 80 ? 15000 : (gpu?.vram === 24 ? 1500 : 800);
            const breakEven = (estBuyPrice / monthlyCost).toFixed(1);

            return (
              <div key={i} className="border border-border bg-card/30 p-6 flex flex-col relative group hover:border-primary/50 transition-colors">
                <div className="absolute top-0 right-0 p-2 text-xs font-mono text-muted-foreground uppercase bg-muted border-b border-l border-border">
                  {option.availability} AVAIL
                </div>
                
                <div className="flex items-center gap-3 mb-4">
                  <Server className="w-8 h-8 text-secondary" />
                  <div>
                    <div className="text-xl font-bold font-mono text-primary">{option.provider}</div>
                    <div className="text-sm font-mono text-muted-foreground">{gpu?.name} • {option.vram} GB</div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 my-4">
                  <div>
                    <div className="text-xs font-mono text-muted-foreground flex items-center gap-1"><DollarSign className="w-3 h-3"/> Hourly</div>
                    <div className="text-2xl font-mono">${option.hourlyPrice.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-xs font-mono text-muted-foreground flex items-center gap-1"><Clock className="w-3 h-3"/> Monthly (24/7)</div>
                    <div className="text-xl font-mono">${monthlyCost.toFixed(0)}</div>
                  </div>
                </div>

                <div className="mt-auto pt-4 border-t border-border/50 text-xs font-mono">
                  <div className="flex justify-between mb-1">
                    <span className="text-muted-foreground">Est. Hardware Cost</span>
                    <span>~${estBuyPrice}</span>
                  </div>
                  <div className="flex justify-between text-primary">
                    <span>Rent vs Buy Break-even</span>
                    <span>{breakEven} months</span>
                  </div>
                </div>
              </div>
            )
          })
        )}
      </div>

    </div>
  );
}
