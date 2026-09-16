#!/usr/bin/env node
// Daily cloud pricing refresh: fetches live rates from Vast.ai (public),
// RunPod (RUNPOD_API_KEY) and Lambda Labs (LAMBDA_API_KEY), then rewrites
// src/data/cloud.json. A provider that fails or has no key keeps its
// existing entries untouched; fetch failures never cause a non-zero exit.

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const CLOUD_PATH = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../src/data/cloud.json",
);
const CURRENT_MONTH = new Date().toISOString().slice(0, 7);

const PROVIDERS = { vast: "Vast.ai", runpod: "RunPod", lambda: "Lambda Labs" };

const VRAM = { rtx4090: 24, rtx3090: 24, "a100-80gb": 80, "h100-80gb": 80 };

// Sanity clamps per gpuId: fetched prices outside these ranges are treated
// as bad data; the existing entry is kept and a warning is printed.
const CLAMPS = {
  rtx4090: [0.2, 2.0],
  rtx3090: [0.1, 1.0],
  "a100-80gb": [0.5, 3.5],
  "h100-80gb": [1.5, 8.0],
};

// Offer/availability count -> liquidity tier.
function liquidityFromCount(count) {
  if (count > 50) return "typical";
  if (count >= 10) return "variable";
  return "scarce";
}

function median(values) {
  if (values.length === 0) return null;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

async function fetchJson(url, { timeoutMs = 30000, ...options } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} ${res.statusText} from ${url}`);
    }
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

// --- Vast.ai ---------------------------------------------------------------
// Public endpoint, no key. gpu_name values verified live against
// https://console.vast.ai/api/v0/bundles/ (note: "A100 PCIE", not "PCIe").
// minRamMB filters out 40GB A100 variants sold under the same names.
// dph_total is per machine, so it is divided by num_gpus for a per-GPU rate.
const VAST_GPUS = {
  rtx4090: { names: ["RTX 4090"] },
  rtx3090: { names: ["RTX 3090"] },
  "a100-80gb": { names: ["A100 PCIE", "A100 PCIe", "A100 SXM4"], minRamMB: 70000 },
  "h100-80gb": { names: ["H100 PCIe", "H100 SXM"], minRamMB: 70000 },
};

async function fetchVast() {
  const out = {};
  for (const [gpuId, cfg] of Object.entries(VAST_GPUS)) {
    const prices = [];
    let offerCount = 0;
    for (const name of cfg.names) {
      const q = JSON.stringify({
        verified: { eq: true },
        type: "on-demand",
        gpu_name: { eq: name },
      });
      const url = `https://console.vast.ai/api/v0/bundles/?q=${encodeURIComponent(q)}`;
      const data = await fetchJson(url);
      for (const offer of data.offers ?? []) {
        if (cfg.minRamMB && (offer.gpu_ram ?? 0) < cfg.minRamMB) continue;
        if (typeof offer.dph_total !== "number" || offer.dph_total <= 0) continue;
        offerCount += 1;
        prices.push(offer.dph_total / Math.max(1, offer.num_gpus ?? 1));
      }
    }
    out[gpuId] = { price: median(prices), count: offerCount };
  }
  return out;
}

// --- RunPod ----------------------------------------------------------------
// gpuTypes exposes prices but no offer counts, so liquidity is not derived
// here (existing liquidity is preserved by the caller).
const RUNPOD_GPUS = {
  rtx4090: ["NVIDIA GeForce RTX 4090"],
  rtx3090: ["NVIDIA GeForce RTX 3090"],
  "a100-80gb": ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"],
  "h100-80gb": ["NVIDIA H100 80GB HBM3"],
};

async function fetchRunPod(apiKey) {
  const query = `query { gpuTypes { id securePrice communityPrice } }`;
  const data = await fetchJson(
    `https://api.runpod.io/graphql?api_key=${encodeURIComponent(apiKey)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    },
  );
  if (data.errors?.length) {
    throw new Error(`RunPod GraphQL error: ${data.errors[0].message}`);
  }
  const byId = new Map((data.data?.gpuTypes ?? []).map((g) => [g.id, g]));
  const out = {};
  for (const [gpuId, ids] of Object.entries(RUNPOD_GPUS)) {
    const prices = ids
      .map((id) => byId.get(id))
      .filter(Boolean)
      .map((g) => g.securePrice ?? g.communityPrice)
      .filter((p) => typeof p === "number" && p > 0);
    out[gpuId] = {
      price: prices.length ? Math.min(...prices) : null,
      count: null,
    };
  }
  return out;
}

// --- Lambda Labs -----------------------------------------------------------
// 1x instance types give the headline per-GPU rate; 80GB variants are
// matched via the instance name. regions_with_capacity_available length
// feeds the liquidity tiers.
const LAMBDA_GPUS = {
  "a100-80gb": (name) => name.includes("a100") && name.includes("80gb"),
  "h100-80gb": (name) => name.includes("h100") && name.includes("80gb"),
};

async function fetchLambda(apiKey) {
  const data = await fetchJson("https://cloud.lambdalabs.com/api/v1/instance-types", {
    headers: {
      Authorization: `Basic ${Buffer.from(`${apiKey}:`).toString("base64")}`,
    },
  });
  const out = {};
  for (const [gpuId, matches] of Object.entries(LAMBDA_GPUS)) {
    const candidates = Object.values(data.data ?? {}).filter((entry) => {
      const name = entry.instance_type?.name ?? "";
      return name.startsWith("gpu_1x_") && matches(name);
    });
    const priced = candidates.filter(
      (e) => typeof e.instance_type?.price_cents_per_hour === "number",
    );
    const best = priced.sort(
      (a, b) => a.instance_type.price_cents_per_hour - b.instance_type.price_cents_per_hour,
    )[0];
    out[gpuId] = {
      price: best ? best.instance_type.price_cents_per_hour / 100 : null,
      count: best ? (best.regions_with_capacity_available ?? []).length : null,
    };
  }
  return out;
}

// --- pipeline ---------------------------------------------------------------
function parseArgs(argv) {
  const args = { dryRun: false, provider: null };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--dry-run") args.dryRun = true;
    else if (argv[i] === "--provider") {
      args.provider = argv[i + 1];
      i += 1;
      if (!Object.hasOwn(PROVIDERS, args.provider)) {
        throw new Error(
          `Unknown provider "${args.provider}". Expected one of: ${Object.keys(PROVIDERS).join(", ")}`,
        );
      }
    } else {
      throw new Error(`Unknown argument "${argv[i]}"`);
    }
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const entries = JSON.parse(await readFile(CLOUD_PATH, "utf8"));

  const wanted = args.provider ? [args.provider] : Object.keys(PROVIDERS);
  const fetched = new Map(); // provider display name -> { gpuId: {price, count} }

  for (const key of wanted) {
    const name = PROVIDERS[key];
    try {
      if (key === "vast") {
        fetched.set(name, await fetchVast());
      } else if (key === "runpod") {
        if (!process.env.RUNPOD_API_KEY) {
          console.log(`SKIP ${name}: RUNPOD_API_KEY not set, keeping existing entries.`);
          continue;
        }
        fetched.set(name, await fetchRunPod(process.env.RUNPOD_API_KEY));
      } else if (key === "lambda") {
        if (!process.env.LAMBDA_API_KEY) {
          console.log(`SKIP ${name}: LAMBDA_API_KEY not set, keeping existing entries.`);
          continue;
        }
        fetched.set(name, await fetchLambda(process.env.LAMBDA_API_KEY));
      }
      console.log(`OK   ${name}: fetched live pricing.`);
    } catch (err) {
      console.warn(
        `\n!!! WARNING: ${name} fetch failed (${err.message}). ` +
          `Keeping existing ${name} entries untouched.\n`,
      );
    }
  }

  const keyOf = (e) => `${e.provider}::${e.gpuId}`;
  const before = new Map(entries.map((e) => [keyOf(e), e]));
  const after = new Map(entries.map((e) => [keyOf(e), { ...e }]));

  for (const [provider, gpus] of fetched) {
    for (const [gpuId, { price, count }] of Object.entries(gpus)) {
      const existing = after.get(`${provider}::${gpuId}`);
      if (price === null) {
        console.warn(
          `WARNING: ${provider} ${gpuId}: no live offers/prices found; keeping existing entry.`,
        );
        continue;
      }
      const rounded = Math.round(price * 100) / 100;
      const [lo, hi] = CLAMPS[gpuId];
      if (rounded < lo || rounded > hi) {
        console.warn(
          `WARNING: ${provider} ${gpuId}: price $${rounded}/hr outside sanity clamp ` +
            `[$${lo}, $${hi}]; keeping existing entry.`,
        );
        continue;
      }
      const entry = existing ?? {
        provider,
        gpuId,
        vram: VRAM[gpuId],
        hourlyPrice: rounded,
        liquidity: "variable",
        asOf: CURRENT_MONTH,
      };
      entry.hourlyPrice = rounded;
      if (count !== null && count !== undefined) {
        entry.liquidity = liquidityFromCount(count);
      }
      entry.asOf = CURRENT_MONTH;
      after.set(`${provider}::${gpuId}`, entry);
    }
  }

  // Preserve original ordering; append brand-new entries at the end.
  const next = entries.map((e) => after.get(keyOf(e)));
  for (const [key, e] of after) {
    if (!before.has(key)) next.push(e);
  }

  const fmt = (v) => (v === undefined ? "(new)" : String(v));
  console.log("\nProvider     GPU         Price           Liquidity              asOf");
  console.log("-".repeat(78));
  let changes = 0;
  for (const e of next) {
    const old = before.get(keyOf(e));
    const changed =
      !old ||
      old.hourlyPrice !== e.hourlyPrice ||
      old.liquidity !== e.liquidity ||
      old.asOf !== e.asOf;
    if (!changed) continue;
    changes += 1;
    const oldPrice = old ? `$${old.hourlyPrice.toFixed(2)}` : "(new)";
    console.log(
      `${e.provider.padEnd(12)} ${e.gpuId.padEnd(11)} ` +
        `${`${oldPrice} -> $${e.hourlyPrice.toFixed(2)}`.padEnd(15)} ` +
        `${`${fmt(old?.liquidity)} -> ${e.liquidity}`.padEnd(22)} ` +
        `${fmt(old?.asOf)} -> ${e.asOf}`,
    );
  }
  if (changes === 0) console.log("(no changes)");

  const json = JSON.stringify(next, null, 2) + "\n";
  if (args.dryRun) {
    console.log("\n--dry-run: would write to src/data/cloud.json:\n");
    process.stdout.write(json);
  } else {
    await writeFile(CLOUD_PATH, json);
    console.log(`\nWrote ${CLOUD_PATH}`);
  }
}

main().catch((err) => {
  console.error(`ERROR: ${err.message}`);
  process.exit(1);
});
