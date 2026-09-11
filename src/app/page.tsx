"use client";

import { useEffect, useState } from "react";
import { useStore } from "@/lib/store";
import { TopNav } from "@/components/TopNav";
import { InferenceScenario } from "@/components/scenarios/InferenceScenario";
import { FineTuneScenario } from "@/components/scenarios/FineTuneScenario";
import { CompareScenario } from "@/components/scenarios/CompareScenario";
import { CloudScenario } from "@/components/scenarios/CloudScenario";

export default function Home() {
  const store = useStore();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // Hydrate state from URL on initial load
    const params = new URLSearchParams(window.location.search);
    if (Array.from(params.keys()).length > 0) {
      store.hydrateFromUrl(params);
    }
    setMounted(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!mounted) return null; // Avoid SSR hydration mismatch

  return (
    <div className="min-h-screen flex flex-col">
      <TopNav />
      
      <main className="flex-1 p-6 flex flex-col overflow-hidden">
        {store.scenario === "inference" && <InferenceScenario />}
        {store.scenario === "finetune" && <FineTuneScenario />}
        {store.scenario === "compare" && <CompareScenario />}
        {store.scenario === "cloud" && <CloudScenario />}
      </main>
    </div>
  );
}
