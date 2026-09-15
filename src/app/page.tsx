"use client";

import { useEffect, useState } from "react";
import { useStore } from "@/lib/store";
import { InferenceScenario } from "@/components/scenarios/InferenceScenario";
import { FineTuneScenario } from "@/components/scenarios/FineTuneScenario";
import { CompareScenario } from "@/components/scenarios/CompareScenario";
import { CloudScenario } from "@/components/scenarios/CloudScenario";

export default function Home() {
  const store = useStore();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // URL hydration is handled once in TopNav (which owns the URL writer).
    setMounted(true);
  }, []);

  if (!mounted) return null; // Avoid SSR hydration mismatch

  return (
    <div className="flex-1 p-6 flex flex-col overflow-hidden">
      {store.scenario === "inference" && <InferenceScenario />}
      {store.scenario === "finetune" && <FineTuneScenario />}
      {store.scenario === "compare" && <CompareScenario />}
      {store.scenario === "cloud" && <CloudScenario />}
    </div>
  );
}
