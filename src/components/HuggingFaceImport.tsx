"use client";

import { useState } from "react";
import { Download, Loader2 } from "lucide-react";
import type { ModelSpec } from "@/lib/calc";

interface HuggingFaceImportProps {
  onImport: (spec: ModelSpec, meta: { paramsEstimated: boolean }) => void;
}

function parseRepoId(input: string): string | null {
  const trimmed = input.trim().replace(/\/+$/, "");
  const urlMatch = trimmed.match(/(?:https?:\/\/)?huggingface\.co\/([^/?#]+\/[^/?#]+)/i);
  const id = urlMatch ? urlMatch[1] : trimmed;
  return /^[\w.-]+\/[\w.-]+$/.test(id) ? id : null;
}

function bytesPerParam(dtype: unknown): number {
  const d = String(dtype ?? "").toLowerCase();
  if (d.includes("float32") || d.includes("fp32")) return 4;
  if (d.includes("float8") || d.includes("fp8")) return 1;
  return 2; // bfloat16 / float16 / unknown
}

// Dense fallback when no safetensors index is available: attention + SwiGLU MLP
// per layer plus the embedding table.
function estimateDenseParams(layers: number, hidden: number, vocab: number): number {
  return 12 * layers * hidden * hidden + vocab * hidden;
}

type HfConfig = Record<string, unknown> & { text_config?: Record<string, unknown> };

async function fetchJson(url: string): Promise<HfConfig> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(String(res.status));
  return res.json();
}

export function HuggingFaceImport({ onImport }: HuggingFaceImportProps) {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImport = async () => {
    setError(null);
    const repo = parseRepoId(input);
    if (!repo) {
      setError("Enter a repo id like meta-llama/Llama-3.1-8B or a huggingface.co URL.");
      return;
    }

    setLoading(true);
    try {
      let raw: HfConfig;
      try {
        raw = await fetchJson(`https://huggingface.co/${repo}/raw/main/config.json`);
      } catch {
        throw new Error(
          "Could not load config.json. Check the repo id — private, gated, or missing repos are not accessible."
        );
      }

      // Multimodal repos nest the language model under text_config.
      const cfg = raw.text_config && typeof raw.text_config === "object"
        ? { ...raw, ...raw.text_config }
        : raw;

      const layers = Number(cfg.num_hidden_layers);
      const hidden = Number(cfg.hidden_size);
      const queryHeads = Number(cfg.num_attention_heads);
      const vocab = Number(cfg.vocab_size);
      if (!layers || !hidden || !queryHeads || !vocab) {
        throw new Error("config.json is missing required architecture fields (layers/hidden/heads/vocab).");
      }

      const bpp = bytesPerParam(cfg.torch_dtype);
      let params = 0;
      let paramsEstimated = true;
      try {
        const index = await fetchJson(`https://huggingface.co/${repo}/raw/main/model.safetensors.index.json`);
        const totalSize = Number((index?.metadata as HfConfig | undefined)?.total_size);
        if (totalSize > 0) {
          params = totalSize / bpp;
          paramsEstimated = false;
        }
      } catch {
        // single-file or non-safetensors repo: fall through to the estimate
      }
      if (!params) params = estimateDenseParams(layers, hidden, vocab);

      const expertCount = Number(cfg.num_local_experts ?? cfg.n_routed_experts ?? 0);
      const expertsPerTok = Number(cfg.num_experts_per_tok ?? 0);
      const moeInter = Number(cfg.moe_intermediate_size ?? 0);
      const isMoE = expertCount > 0 && expertsPerTok > 0;

      let activeParams: number | undefined;
      if (isMoE && moeInter > 0) {
        // Shared weights stay active; only the idle (N-k) expert MLPs are skipped.
        const idleExpertParams = layers * (expertCount - expertsPerTok) * 3 * hidden * moeInter;
        activeParams = Math.max(params - idleExpertParams, params * 0.05);
      }

      const spec: ModelSpec = {
        id: `hf-custom-${repo.replace("/", "-")}`,
        name: repo,
        params,
        layers,
        hiddenSize: hidden,
        queryHeads,
        kvHeads: Number(cfg.num_key_value_heads) || queryHeads,
        vocabSize: vocab,
        maxContext: Number(cfg.max_position_embeddings) || 4096,
        architecture: isMoE ? "moe" : "dense",
        activeParams,
        family: "Custom",
        source: repo,
      };

      onImport(spec, { paramsEstimated });
      setInput("");
    } catch (e) {
      setError(
        e instanceof Error && e.message && !/^\d+$/.test(e.message)
          ? e.message
          : "Import failed — the repo may be unreachable or blocked by the network."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleImport()}
          placeholder="Paste a Hugging Face repo or URL..."
          className="flex-1 bg-card border border-border text-foreground font-mono text-sm px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary transition-all placeholder:text-muted-foreground/50"
        />
        <button
          onClick={handleImport}
          disabled={loading}
          className="px-4 py-2 flex items-center gap-2 border border-border rounded-lg bg-card hover:bg-border transition-colors text-sm font-mono text-foreground disabled:opacity-50"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
          Import
        </button>
      </div>
      {error && <p className="text-xs font-mono text-destructive">{error}</p>}
    </div>
  );
}
