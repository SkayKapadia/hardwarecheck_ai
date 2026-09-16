// Data-integrity gate: audits every entry in src/data/models.json against live
// Hugging Face config.json + safetensors metadata.
//
// Usage:
//   node scripts/audit_models.mjs [--report <path.md>] [--strict]
//
// Exit code: 0 if no FAILs, 1 otherwise.
//   --strict  also FAIL entries whose HF repo is unreachable after one retry.
//
// Checks per entry:
//   (a) params within 2% of the HF-derived count (warn >1%, fail >2%)
//   (b) layers / hiddenSize / queryHeads / kvHeads / vocabSize exact match
//   (c) MoE entries must declare activeParams < params
//   (d) dense entries must NOT declare activeParams
//
// Expected params come from the HF API's safetensors parameter histogram
// (exact counts per dtype; float dtypes only, so scale/buffer tensors are
// excluded), falling back to model.safetensors.index.json metadata.total_size
// divided by bytes-per-param (torch_dtype: bf16/fp16=2, fp32=4, fp8=1).
//
// 404s, gated repos (401/403) and LFS-pointer configs are UNVERIFIABLE, never
// failures. Also prints a "trending radar" of HF text-generation models with
// >1M downloads not yet in the database (report-only).

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const MODELS_PATH = join(ROOT, "src", "data", "models.json");

const args = process.argv.slice(2);
const reportPath = args.includes("--report") ? args[args.indexOf("--report") + 1] : null;
const strict = args.includes("--strict");

const CONCURRENCY = 8;
const FETCH_TIMEOUT_MS = 15000;

const BYTES_PER_PARAM = {
  bfloat16: 2, bf16: 2,
  float16: 2, fp16: 2, half: 2,
  float32: 4, fp32: 4, float: 4,
  float8: 1, fp8: 1, float8_e4m3fn: 1, float8_e5m2: 1,
};

// HF safetensors parameter histogram dtypes that are real weight tensors.
// U8/I8 entries are scales, rotary inv_freq buffers, or packed sub-byte
// weights — never plain parameters.
const FLOAT_DTYPE_RE = /^(F\d|BF16)/;

// Per-entry documented exceptions. Each must cite WHY the deviation from HF
// ground truth is legitimate. Do NOT add entries here to mask real drift.
const EXCEPTIONS = {
  // deepseek-ai/DeepSeek-V3 and V3.2 configs declare num_nextn_predict_layers:
  // 1 — the checkpoint stores the extra MTP (multi-token prediction) layer
  // (~13.5B params) that the published 671B total-params figure excludes.
  "deepseek-ai/DeepSeek-V3": {
    paramsTolerance: 0.05,
    reason: "safetensors includes the MTP (num_nextn_predict_layers) layer excluded from the published 671B total",
  },
  "deepseek-ai/DeepSeek-V3.2": {
    paramsTolerance: 0.05,
    reason: "safetensors includes the MTP (num_nextn_predict_layers) layer excluded from the published 671B total",
  },
  // zai-org GLM configs declare num_nextn_predict_layers; like DeepSeek, the
  // vendor's published totals (30B / 744B) exclude those MTP predict layers
  // that are present in the checkpoint (~1.2B / ~9.3B params respectively).
  "zai-org/GLM-4.7-Flash": {
    paramsTolerance: 0.05,
    reason: "safetensors includes nextn/MTP predict layers excluded from the published ~30B total",
  },
  "zai-org/GLM-5.2-FP8": {
    paramsTolerance: 0.03,
    reason: "safetensors includes nextn/MTP predict layers excluded from the published 744B total",
  },
};

async function fetchJSON(url) {
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const res = await fetch(url, {
        redirect: "follow",
        signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
        headers: { "User-Agent": "hardware-compat-model-audit" },
      });
      if (res.status === 404 || res.status === 401 || res.status === 403) {
        return { status: res.status, json: null };
      }
      if (!res.ok) return { status: res.status, json: null };
      const text = await res.text();
      // LFS pointer files start with "version https://git-lfs..."
      if (text.startsWith("version https://git-lfs")) return { status: 200, json: null, lfs: true };
      try {
        return { status: 200, json: JSON.parse(text) };
      } catch {
        return { status: 200, json: null };
      }
    } catch (e) {
      if (attempt === 0) continue; // network error: retry once
      return { status: 0, json: null, error: String(e) };
    }
  }
}

function unwrapTextConfig(cfg) {
  if (cfg && cfg.text_config && typeof cfg.text_config === "object") {
    return { ...cfg, ...cfg.text_config };
  }
  return cfg;
}

function bytesPerParam(cfg) {
  const qc = cfg.quantization_config;
  if (qc && String(qc.quant_method ?? "").toLowerCase().includes("fp8")) return 1;
  const dt = String(cfg.torch_dtype ?? cfg.dtype ?? "").toLowerCase();
  return BYTES_PER_PARAM[dt] ?? 2;
}

// Sub-byte (e.g. NVFP4 / FP4) checkpoints pack multiple weights per byte, so a
// parameter count cannot be derived from storage sizes or dtype histograms.
function hasSubByteWeights(cfg) {
  const qc = cfg.quantization_config;
  if (!qc) return false;
  if (String(qc.expert_dtype ?? "").toLowerCase().includes("fp4")) return true;
  const s = JSON.stringify(qc);
  return /"num_bits":\s*4/.test(s);
}

// Exact param count from the HF API safetensors histogram (float dtypes only).
// GPT-2-family configs (n_layer/n_embd) register a persistent causal-mask
// buffer `attn.bias` of shape [1,1,n_positions,n_positions] per layer; HF's
// converted safetensors store them and they inflate the histogram by
// n_layer * n_positions^2 — subtract them (they are buffers, not parameters).
function paramsFromHistogram(histogram, cfg) {
  let total = 0;
  for (const [dtype, count] of Object.entries(histogram)) {
    if (FLOAT_DTYPE_RE.test(dtype)) total += count;
  }
  if (cfg.n_layer && cfg.n_positions) total -= cfg.n_layer * cfg.n_positions ** 2;
  return total;
}

async function auditEntry(entry) {
  const problems = [];
  const warnings = [];
  const notes = [];
  const repo = entry.source || entry.id;
  const exc = EXCEPTIONS[entry.id] || EXCEPTIONS[repo] || null;
  if (exc) notes.push(`exception: ${exc.reason}`);

  const cfgRes = await fetchJSON(`https://huggingface.co/${repo}/raw/main/config.json`);
  if (!cfgRes.json) {
    const why = cfgRes.error
      ? `unreachable (${cfgRes.error})`
      : cfgRes.lfs
        ? "config.json is an LFS pointer"
        : `config.json not fetchable (HTTP ${cfgRes.status})`;
    if (cfgRes.error && strict) {
      return { entry, status: "FAIL", problems: [`repo unreachable in --strict mode: ${cfgRes.error}`], warnings, notes, expected: null };
    }
    return { entry, status: "UNVERIFIABLE", problems, warnings, notes: [...notes, why], expected: null };
  }

  const cfg = unwrapTextConfig(cfgRes.json);
  const expected = {
    layers: cfg.num_hidden_layers ?? cfg.n_layer ?? null,
    hiddenSize: cfg.hidden_size ?? cfg.n_embd ?? null,
    queryHeads: cfg.num_attention_heads ?? cfg.n_head ?? null,
    kvHeads: cfg.num_key_value_heads ?? cfg.num_attention_heads ?? cfg.n_head ?? null,
    vocabSize: cfg.vocab_size ?? null,
    params: null,
  };

  // --- expected params ---
  const [apiRes, idxRes] = await Promise.all([
    fetchJSON(`https://huggingface.co/api/models/${repo}`),
    fetchJSON(`https://huggingface.co/${repo}/raw/main/model.safetensors.index.json`),
  ]);
  const histogram = apiRes.json?.safetensors?.parameters;
  if (hasSubByteWeights(cfg)) {
    notes.push("sub-byte (4-bit) packed weights; params not derivable from storage, skipped");
  } else if (histogram) {
    const raw = paramsFromHistogram(histogram, cfg);
    const candidates = [raw];
    // Checkpoints of models with tied embeddings sometimes still store a full
    // lm_head copy (e.g. Qwen3-0.6B/1.7B); accept either interpretation.
    if (cfg.tie_word_embeddings === true && cfg.vocab_size && expected.hiddenSize) {
      candidates.push(raw - cfg.vocab_size * expected.hiddenSize);
    }
    let best = candidates[0];
    for (const c of candidates) {
      if (Math.abs(entry.params - c) < Math.abs(entry.params - best)) best = c;
    }
    if (best !== raw) notes.push("checkpoint stores an untied lm_head copy; compared against histogram minus one vocab embedding");
    expected.params = best;
  } else {
    let idx = idxRes.json;
    if (!idx?.metadata?.total_size) {
      idx = (await fetchJSON(`https://huggingface.co/${repo}/resolve/main/model.safetensors.index.json`)).json;
    }
    if (idx?.metadata?.total_size) {
      const bpp = bytesPerParam(cfg);
      expected.params = Math.round(idx.metadata.total_size / bpp);
      notes.push(`params via index.json total_size/${bpp}B-per-param`);
    } else {
      notes.push("no safetensors metadata available; params not checked");
    }
  }

  // (a) params tolerance
  if (expected.params != null) {
    const tol = exc?.paramsTolerance ?? 0.02;
    const diff = Math.abs(entry.params - expected.params) / expected.params;
    expected.paramDiffPct = diff * 100;
    if (diff > tol) {
      problems.push(`params ${entry.params} vs HF-derived ${expected.params} (${(diff * 100).toFixed(2)}% off, limit ${(tol * 100).toFixed(0)}%)`);
    } else if (diff > 0.01 && !exc) {
      warnings.push(`params ${entry.params} vs HF-derived ${expected.params} (${(diff * 100).toFixed(2)}% off, warn >1%)`);
    }
  }

  // (b) exact architecture-field match
  for (const field of ["layers", "hiddenSize", "queryHeads", "kvHeads", "vocabSize"]) {
    if (expected[field] == null) {
      notes.push(`${field}: not present in HF config, skipped`);
      continue;
    }
    if (entry[field] !== expected[field]) {
      problems.push(`${field} ${entry[field]} != HF ${expected[field]}`);
    }
  }

  // (c)+(d) MoE / dense sanity
  const cfgIsMoe =
    cfg.num_local_experts != null || cfg.n_routed_experts != null || cfg.num_experts != null;
  const isMoe = cfgIsMoe || entry.architecture === "moe";
  if (isMoe) {
    if (entry.activeParams == null) {
      problems.push("MoE model missing activeParams");
    } else if (entry.activeParams >= entry.params) {
      problems.push(`activeParams ${entry.activeParams} >= params ${entry.params}`);
    }
    if (entry.architecture !== "moe") {
      warnings.push(`HF config is MoE but architecture="${entry.architecture}"`);
    }
  } else if (entry.activeParams != null) {
    problems.push("dense model must not declare activeParams");
  }

  const status = problems.length ? "FAIL" : warnings.length ? "WARN" : "PASS";
  return { entry, status, problems, warnings, notes, expected };
}

async function trendingRadar(models) {
  const known = new Set(models.map((m) => m.source).concat(models.map((m) => m.id)));
  const url =
    "https://huggingface.co/api/models?sort=downloads&direction=-1&limit=50&pipeline_tag=text-generation";
  const res = await fetchJSON(url);
  if (!res.json || !Array.isArray(res.json)) {
    return { ok: false, items: [], error: res.error ?? `HTTP ${res.status}` };
  }
  const items = res.json
    .filter((m) => (m.downloads ?? 0) > 1_000_000 && !known.has(m.id))
    .map((m) => ({ id: m.id, downloads: m.downloads, likes: m.likes ?? 0 }));
  return { ok: true, items };
}

async function pool(items, size, fn) {
  const results = new Array(items.length);
  let i = 0;
  await Promise.all(
    Array.from({ length: Math.min(size, items.length) }, async () => {
      while (i < items.length) {
        const idx = i++;
        results[idx] = await fn(items[idx]);
      }
    })
  );
  return results;
}

function fmtParams(n) {
  if (n == null) return "-";
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  return String(n);
}

const models = JSON.parse(readFileSync(MODELS_PATH, "utf8"));
console.error(`Auditing ${models.length} entries against Hugging Face (concurrency ${CONCURRENCY})...`);
const results = await pool(models, CONCURRENCY, auditEntry);
const radar = await trendingRadar(models);

const counts = { PASS: 0, WARN: 0, FAIL: 0, UNVERIFIABLE: 0 };
for (const r of results) counts[r.status]++;

// --- stdout summary table ---
console.log("\nSTATUS        ENTRY                                              PARAMS(DB/HF)     DIFF    NOTES");
for (const r of results) {
  const id = r.entry.id.padEnd(50).slice(0, 50);
  const diff = r.expected?.paramDiffPct != null ? `${r.expected.paramDiffPct.toFixed(2)}%` : "-";
  const msgs = [...r.problems, ...r.warnings, ...r.notes].join("; ");
  console.log(
    `${r.status.padEnd(13)} ${id} ${fmtParams(r.entry.params)}/${fmtParams(r.expected?.params)}`.padEnd(80) +
      ` ${diff.padEnd(7)} ${msgs}`
  );
}
console.log(
  `\nSummary: ${counts.PASS} PASS, ${counts.WARN} WARN, ${counts.UNVERIFIABLE} UNVERIFIABLE, ${counts.FAIL} FAIL (of ${results.length})`
);
if (radar.ok) {
  console.log(`\nTrending radar: ${radar.items.length} HF text-generation model(s) with >1M downloads not in the database:`);
  for (const m of radar.items) console.log(`  - ${m.id} (${(m.downloads / 1e6).toFixed(1)}M downloads, ${m.likes} likes)`);
} else {
  console.log(`\nTrending radar: unavailable (${radar.error}) — report-only, not a failure`);
}

// --- markdown report ---
if (reportPath) {
  const lines = [];
  lines.push("# Model DB audit vs Hugging Face");
  lines.push("");
  lines.push(`Generated: ${new Date().toISOString()} — strict mode: ${strict ? "on" : "off"}`);
  lines.push("");
  lines.push(`**Summary: ${counts.PASS} PASS, ${counts.WARN} WARN, ${counts.UNVERIFIABLE} UNVERIFIABLE, ${counts.FAIL} FAIL (of ${results.length} entries)**`);
  lines.push("");
  lines.push("| Status | Entry | Source | Params (DB) | Params (HF-derived) | Diff | Details |");
  lines.push("|---|---|---|---|---|---|---|");
  for (const r of results) {
    const diff = r.expected?.paramDiffPct != null ? `${r.expected.paramDiffPct.toFixed(2)}%` : "-";
    const details = [...r.problems, ...r.warnings, ...r.notes].join("<br>") || "—";
    lines.push(
      `| ${r.status} | ${r.entry.id} | ${r.entry.source} | ${fmtParams(r.entry.params)} | ${fmtParams(r.expected?.params)} | ${diff} | ${details} |`
    );
  }
  lines.push("");
  lines.push("## Trending radar (report-only)");
  lines.push("");
  if (radar.ok && radar.items.length) {
    lines.push("HF text-generation models with >1M downloads not yet in the database:");
    lines.push("");
    for (const m of radar.items) lines.push(`- **${m.id}** — ${(m.downloads / 1e6).toFixed(1)}M downloads, ${m.likes} likes`);
  } else if (radar.ok) {
    lines.push("No trending models with >1M downloads are missing from the database.");
  } else {
    lines.push(`Trending radar unavailable (${radar.error}).`);
  }
  writeFileSync(reportPath, lines.join("\n") + "\n");
  console.log(`\nReport written to ${reportPath}`);
}

process.exit(counts.FAIL > 0 ? 1 : 0);
