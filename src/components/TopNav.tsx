"use client";

import { useStore, Scenario } from "@/lib/store";
import { Crosshair, Share2, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";

export function TopNav() {
  const store = useStore();
  const [copied, setCopied] = useState(false);

  // Sync state to URL without reloading
  useEffect(() => {
    const url = new URL(window.location.href);
    url.searchParams.set("scenario", store.scenario);
    url.searchParams.set("model", store.selectedModel);
    url.searchParams.set("gpu", store.selectedGPUs.join(","));
    url.searchParams.set("quant", store.quantization);
    url.searchParams.set("ctx", store.contextLength.toString());
    url.searchParams.set("batch", store.batchSize.toString());
    url.searchParams.set("offload", store.cpuOffload.toString());
    url.searchParams.set("ram", store.systemRam.toString());
    
    // fine tune
    url.searchParams.set("loraRank", store.loraRank.toString());
    url.searchParams.set("loraAlpha", store.loraAlpha.toString());
    url.searchParams.set("modules", store.targetModules.join(","));
    url.searchParams.set("gc", store.gradientCheckpointing.toString());
    url.searchParams.set("tBatch", store.trainBatchSize.toString());

    window.history.replaceState({}, "", url.toString());
  }, [store]);

  const handleShare = () => {
    navigator.clipboard.writeText(window.location.href);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleReset = () => {
    window.location.href = "/"; // simple reset
  };

  return (
    <div className="flex items-center justify-between border-b border-border bg-card/80 backdrop-blur-sm px-6 py-4 sticky top-0 z-50">
      <div className="flex items-center gap-3 text-primary font-mono font-bold text-xl tracking-widest uppercase">
        <Crosshair className="w-6 h-6 text-secondary" />
        Loadout
      </div>
      
      <div className="flex bg-muted p-1 border border-border">
        {["inference", "finetune", "compare", "cloud"].map((s) => (
          <button
            key={s}
            onClick={() => store.setScenario(s as Scenario)}
            className={`px-4 py-1.5 text-sm font-sans uppercase tracking-wider transition-colors ${
              store.scenario === s 
                ? "bg-primary/20 text-primary border border-primary/50 shadow-[0_0_10px_rgba(34,211,238,0.2)]" 
                : "text-muted-foreground hover:text-primary/70 border border-transparent"
            }`}
          >
            {s.replace("finetune", "fine tune")}
          </button>
        ))}
      </div>

      <div className="flex items-center gap-3">
        <button 
          onClick={handleShare}
          className="flex items-center gap-2 px-3 py-1.5 border border-primary/30 text-primary hover:bg-primary/10 transition-colors text-sm font-mono uppercase"
        >
          <Share2 className="w-4 h-4" />
          {copied ? "Copied!" : "Share"}
        </button>
        <button 
          onClick={handleReset}
          className="flex items-center gap-2 px-3 py-1.5 border border-destructive/30 text-destructive hover:bg-destructive/10 transition-colors text-sm font-mono uppercase"
        >
          <RotateCcw className="w-4 h-4" />
          Reset
        </button>
      </div>
    </div>
  );
}
