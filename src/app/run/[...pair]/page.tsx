/* eslint-disable react/no-unescaped-entities */
import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import {
  ArrowRight,
  CheckCircle2,
  Cpu,
  Gauge,
  HardDrive,
  Layers,
  XCircle,
} from "lucide-react";
import { SITE_URL } from "@/lib/site";
import {
  MODELS_DATA_AS_OF,
  formatParams,
  getRuntimeForQuant,
} from "@/lib/calc";
import {
  RUN_CTX_LEN,
  RUN_QUANT,
  cheapestFittingGpus,
  computeRunFit,
  dualCardFits,
  getRunPairs,
  relatedGpus,
  relatedModels,
  resolveRunPair,
  runPath,
} from "@/lib/runPages";

export const dynamicParams = false;

interface PageProps {
  params: { pair: string[] };
}

function gb(value: number): string {
  return value >= 10 ? value.toFixed(1) : value.toFixed(2);
}

export function generateStaticParams() {
  return getRunPairs().map(({ model, gpu }) => ({
    pair: [...model.id.split("/"), gpu.id],
  }));
}

export function generateMetadata({ params }: PageProps): Metadata {
  const resolved = resolveRunPair(params.pair);
  if (!resolved) return {};
  const { model, gpu } = resolved;
  const fit = computeRunFit(model, gpu);

  const title = fit.fits
    ? `Can you run ${model.name} on a ${gpu.name}? Yes — needs ~${gb(fit.total)} GB`
    : `Can you run ${model.name} on a ${gpu.name}? No — needs ~${gb(fit.total)} GB`;
  const description = `${model.name} (${formatParams(model.params)} params) at ${RUN_QUANT} needs ~${gb(
    fit.total
  )} GB VRAM on a ${gpu.name} (${gpu.vram} GB): weights ${gb(fit.weights)} GB, KV cache ${gb(
    fit.kv
  )} GB, runtime overhead ${gb(fit.activations + fit.reserve)} GB. Estimated ~${Math.round(
    fit.tps
  )} tok/s with ${getRuntimeForQuant(RUN_QUANT)}.`;

  const url = `${SITE_URL}${runPath(model.id, gpu.id)}`;
  const ogImage = `${SITE_URL}/api/og?model=${encodeURIComponent(
    model.id
  )}&gpu=${encodeURIComponent(gpu.id)}&quant=${RUN_QUANT}`;

  return {
    title,
    description,
    alternates: { canonical: url },
    openGraph: {
      title,
      description,
      url,
      type: "article",
      images: [{ url: ogImage, width: 1200, height: 630 }],
    },
  };
}

export default function RunPage({ params }: PageProps) {
  const resolved = resolveRunPair(params.pair);
  if (!resolved) notFound();
  const { model, gpu } = resolved;
  const fit = computeRunFit(model, gpu);
  const runtime = getRuntimeForQuant(RUN_QUANT);
  const isMoE = model.architecture === "moe";
  const paramsLabel = formatParams(model.params);
  const activeLabel = model.activeParams ? formatParams(model.activeParams) : null;

  const calculatorHref = `/?scenario=inference&model=${encodeURIComponent(
    model.id
  )}&gpu=${encodeURIComponent(gpu.id)}&quant=${RUN_QUANT}`;

  const alternatives = fit.fits ? [] : cheapestFittingGpus(model, gpu.id, 2);
  const dualOption = !fit.fits && dualCardFits(model, gpu);
  const fallbackGpu =
    !fit.fits && alternatives.length === 0 && !dualOption
      ? cheapestFittingGpus(model, gpu.id, 1, false)[0]
      : undefined;

  const otherGpus = relatedGpus(gpu.id, 4);
  const otherModels = relatedModels(model, 4);

  return (
    <div className="flex-1 overflow-y-auto custom-scrollbar p-10 max-w-5xl mx-auto h-full">
      <h1 className="text-4xl font-bold text-foreground mb-4 flex items-center gap-4">
        <Cpu className="w-10 h-10 text-primary shrink-0" />
        Can you run {model.name} on a {gpu.name}?
      </h1>
      <p className="text-muted-foreground mb-8 text-lg leading-relaxed">
        VRAM requirements for {model.name} ({paramsLabel} parameters) at {RUN_QUANT} precision
        on a {gpu.name} ({gpu.vram} GB), with a {RUN_CTX_LEN.toLocaleString()}-token context and
        batch size 1.
      </p>

      {/* Verdict banner */}
      <div
        className={
          fit.fits
            ? "bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-6 mb-10"
            : "bg-red-500/10 border border-red-500/30 rounded-xl p-6 mb-10"
        }
      >
        <div className="flex items-center gap-3 mb-2">
          {fit.fits ? (
            <CheckCircle2 className="w-8 h-8 text-emerald-500 shrink-0" />
          ) : (
            <XCircle className="w-8 h-8 text-red-500 shrink-0" />
          )}
          <span className="text-2xl font-bold text-foreground">
            {fit.fits
              ? `Yes — it fits, using ~${gb(fit.total)} GB of the ${gpu.vram} GB available`
              : `No — it needs ~${gb(fit.total)} GB but the ${gpu.name} only has ${gpu.vram} GB`}
          </span>
        </div>
        <p className="text-foreground/80">
          Real-world usage varies with the runtime and driver: expect roughly{" "}
          <span className="font-mono">{gb(fit.range.low)}–{gb(fit.range.high)} GB</span> in
          practice. Figures are estimates as of {MODELS_DATA_AS_OF}.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-8 pb-20">
        {/* Breakdown table */}
        <div className="bg-card p-8 rounded-xl border border-border">
          <h2 className="text-2xl font-bold text-foreground flex items-center gap-3 mb-4">
            <HardDrive className="w-6 h-6 text-primary" />
            VRAM breakdown ({RUN_QUANT}, {RUN_CTX_LEN.toLocaleString()} context)
          </h2>
          <table className="w-full text-sm">
            <tbody>
              {[
                [`Model weights (${RUN_QUANT})`, fit.weights],
                [`KV cache (${RUN_CTX_LEN.toLocaleString()} tokens, batch 1)`, fit.kv],
                ["Quantization metadata", fit.quantMeta],
                ["Activations (prefill spike)", fit.activations],
                ["OS / driver reserve", fit.reserve],
                ["Safety margin (5%)", fit.safety],
              ].map(([label, value]) => (
                <tr key={label as string} className="border-b border-border/50">
                  <td className="py-2 text-muted-foreground">{label}</td>
                  <td className="py-2 text-right font-mono text-foreground">
                    {gb(value as number)} GB
                  </td>
                </tr>
              ))}
              <tr>
                <td className="py-3 font-bold text-foreground">Total estimated</td>
                <td className="py-3 text-right font-mono font-bold text-foreground">
                  {gb(fit.total)} GB
                </td>
              </tr>
              <tr>
                <td className="py-1 text-muted-foreground">{gpu.name} VRAM</td>
                <td className="py-1 text-right font-mono text-muted-foreground">
                  {gpu.vram} GB
                </td>
              </tr>
            </tbody>
          </table>
          <p className="text-xs text-muted-foreground mt-4">
            Estimates as of {MODELS_DATA_AS_OF}; actual usage depends on the serving stack,
            allocator, and driver.
          </p>
        </div>

        {/* Speed estimate */}
        <div className="bg-card p-8 rounded-xl border border-border">
          <h2 className="text-2xl font-bold text-foreground flex items-center gap-3 mb-4">
            <Gauge className="w-6 h-6 text-primary" />
            Expected speed
          </h2>
          <p className="text-foreground/80 leading-relaxed">
            <span className="font-mono text-2xl font-bold text-foreground">
              ~{Math.round(fit.tps)} tokens/sec
            </span>{" "}
            estimated generation throughput for {model.name} at {RUN_QUANT} on the {gpu.name}{" "}
            ({gpu.bandwidth} GB/s memory bandwidth) using {runtime}
            {isMoE && activeLabel
              ? `. As a Mixture-of-Experts model, only ${activeLabel} of the ${paramsLabel} parameters are active per token, which is what makes this speed possible despite the model's size`
              : ""}
            .
          </p>
        </div>

        {/* Prose analysis */}
        <div className="bg-card p-8 rounded-xl border border-border space-y-4 text-foreground/80 leading-relaxed">
          <h2 className="text-2xl font-bold text-foreground flex items-center gap-3">
            <Layers className="w-6 h-6 text-primary" />
            The details
          </h2>
          <p>
            {model.name} is a {paramsLabel}-parameter {isMoE ? "Mixture-of-Experts" : "dense"}{" "}
            model from the {model.family} family
            {isMoE && activeLabel
              ? `, with ${activeLabel} parameters active per generated token`
              : ""}
            . It has {model.layers} layers, a hidden size of{" "}
            {model.hiddenSize.toLocaleString()}, and supports up to{" "}
            {model.maxContext.toLocaleString()} tokens of context. Quantized to {RUN_QUANT}, the
            weights alone take {gb(fit.weights)} GB; running in unquantized FP16 would require{" "}
            {gb(fit.fp16Weights)} GB of weights and roughly {gb(fit.fp16Total)} GB in total
            {fit.fp16Fits
              ? ` — which would still fit on the ${gpu.name}, but leaves far less room for long contexts`
              : ` — well beyond the ${gpu.name}'s ${gpu.vram} GB`}.
          </p>
          <p>
            {fit.fits
              ? `On the ${gpu.name}, the full ${RUN_QUANT} setup totals ~${gb(
                  fit.total
                )} GB against ${gpu.vram} GB of VRAM, leaving about ${gb(
                  fit.headroom
                )} GB of headroom for longer contexts or larger batches. The KV cache for a ${RUN_CTX_LEN.toLocaleString()}-token conversation is only ${gb(
                  fit.kv
                )} GB, so you can extend the context substantially before running into limits.`
              : `The binding constraint is total VRAM: the ${RUN_QUANT} weights (${gb(
                  fit.weights
                )} GB) plus KV cache (${gb(fit.kv)} GB), activations, and reserves add up to ~${gb(
                  fit.total
                )} GB, exceeding the ${gpu.name}'s ${gpu.vram} GB by roughly ${gb(
                  fit.total - gpu.vram
                )} GB. Shortening the context saves only the KV cache term (${gb(
                  fit.kv
                )} GB at ${RUN_CTX_LEN.toLocaleString()} tokens), which is not enough to close the gap on its own.`}
          </p>
        </div>

        {/* Alternatives when it doesn't fit */}
        {!fit.fits && (
          <div className="bg-card p-8 rounded-xl border border-border">
            <h2 className="text-2xl font-bold text-foreground mb-4">What would work?</h2>
            <ul className="space-y-3 text-foreground/80">
              {alternatives.map((alt) => {
                const altFit = computeRunFit(model, alt);
                return (
                  <li key={alt.id}>
                    <Link
                      href={runPath(model.id, alt.id)}
                      className="text-primary hover:underline font-semibold"
                    >
                      {model.name} on a {alt.name}
                    </Link>{" "}
                    — fits at ~{gb(altFit.total)} GB of {alt.vram} GB
                    {alt.price ? ` (≈$${alt.price.toLocaleString()})` : ""}.
                  </li>
                );
              })}
              {dualOption && (
                <li>
                  <Link
                    href={`/?scenario=inference&model=${encodeURIComponent(
                      model.id
                    )}&gpu=${encodeURIComponent(gpu.id)},${encodeURIComponent(
                      gpu.id
                    )}&quant=${RUN_QUANT}`}
                    className="text-primary hover:underline font-semibold"
                  >
                    2× {gpu.name}
                  </Link>{" "}
                  — splitting the model across two cards provides {gpu.vram * 2} GB, enough for
                  the ~{gb(fit.total)} GB required.
                </li>
              )}
              {alternatives.length === 0 && !dualOption && fallbackGpu && (
                <li>
                  <Link
                    href={`/?scenario=inference&model=${encodeURIComponent(
                      model.id
                    )}&gpu=${encodeURIComponent(fallbackGpu.id)}&quant=${RUN_QUANT}`}
                    className="text-primary hover:underline font-semibold"
                  >
                    {model.name} on a {fallbackGpu.name}
                  </Link>{" "}
                  — fits at {RUN_QUANT} on {fallbackGpu.vram} GB; open it in the calculator to
                  explore the setup.
                </li>
              )}
              {alternatives.length === 0 && !dualOption && !fallbackGpu && (
                <li>
                  No single GPU in our database fits {model.name} at {RUN_QUANT}. Try the{" "}
                  <Link href={calculatorHref} className="text-primary hover:underline">
                    interactive calculator
                  </Link>{" "}
                  to explore multi-GPU splits or CPU offload.
                </li>
              )}
            </ul>
          </div>
        )}

        {/* CTA */}
        <div>
          <Link
            href={calculatorHref}
            className="inline-flex items-center gap-2 bg-primary text-primary-foreground font-semibold px-6 py-3 rounded-xl hover:opacity-90 transition-opacity"
          >
            Open in the interactive calculator
            <ArrowRight className="w-5 h-5" />
          </Link>
        </div>

        {/* Related links */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-card p-6 rounded-xl border border-border">
            <h2 className="text-lg font-bold text-foreground mb-3">
              {model.name} on other GPUs
            </h2>
            <ul className="space-y-2">
              {otherGpus.map((g) => (
                <li key={g.id}>
                  <Link
                    href={runPath(model.id, g.id)}
                    className="text-primary hover:underline text-sm"
                  >
                    Can you run {model.name} on a {g.name}?
                  </Link>
                </li>
              ))}
            </ul>
          </div>
          <div className="bg-card p-6 rounded-xl border border-border">
            <h2 className="text-lg font-bold text-foreground mb-3">
              Other models on the {gpu.name}
            </h2>
            <ul className="space-y-2">
              {otherModels.map((m) => (
                <li key={m.id}>
                  <Link
                    href={runPath(m.id, gpu.id)}
                    className="text-primary hover:underline text-sm"
                  >
                    Can you run {m.name} on a {gpu.name}?
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
