"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import models from "@/data/models.json";
import { formatParams, type ModelSpec } from "@/lib/calc";

const BUILTIN_MODELS = models as unknown as ModelSpec[];

interface CustomModelMeta {
  paramsEstimated: boolean;
}

// Session-only registry of models imported from Hugging Face. Kept at module
// level so every scenario tab resolves the same custom model id from the store.
const customModels = new Map<string, ModelSpec>();
const customMeta = new Map<string, CustomModelMeta>();

export function registerCustomModel(spec: ModelSpec, meta: CustomModelMeta) {
  customModels.set(spec.id, spec);
  customMeta.set(spec.id, meta);
}

export function getCustomModelMeta(id: string): CustomModelMeta | undefined {
  return customMeta.get(id);
}

export function getModelById(id: string): ModelSpec | undefined {
  return BUILTIN_MODELS.find((m) => m.id === id) ?? customModels.get(id);
}

function optionLabel(m: ModelSpec): string {
  const params = m.activeParams
    ? `${formatParams(m.params)} total / ${formatParams(m.activeParams)} active`
    : formatParams(m.params);
  return `${m.name} (${params}, ${m.architecture.toUpperCase()})`;
}

interface ModelPickerProps {
  value: string;
  onChange: (id: string) => void;
}

export function ModelPicker({ value, onChange }: ModelPickerProps) {
  const [query, setQuery] = useState("");

  const all = [...BUILTIN_MODELS, ...Array.from(customModels.values())];
  const q = query.trim().toLowerCase();
  const filtered = q
    ? all.filter(
        (m) =>
          m.name.toLowerCase().includes(q) ||
          m.family.toLowerCase().includes(q) ||
          m.id === value // keep the current selection visible while filtering
      )
    : all;

  const families = Array.from(new Set(filtered.map((m) => m.family)));

  return (
    <div className="space-y-2">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Type to filter models..."
        className="w-full bg-card border border-border text-foreground font-mono text-sm px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary transition-all placeholder:text-muted-foreground/50"
      />
      <div className="relative">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full appearance-none bg-card border border-border text-foreground font-mono p-4 pr-10 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary transition-all cursor-pointer"
        >
          {families.map((family) => (
            <optgroup key={family} label={family} className="bg-background text-foreground">
              {filtered
                .filter((m) => m.family === family)
                .map((m) => (
                  <option key={m.id} value={m.id} className="bg-background text-foreground">
                    {optionLabel(m)}
                  </option>
                ))}
            </optgroup>
          ))}
        </select>
        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-4 text-muted-foreground">
          <ChevronDown className="w-5 h-5" />
        </div>
      </div>
    </div>
  );
}
