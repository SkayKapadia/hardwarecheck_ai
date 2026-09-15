import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { formatParams } from "@/lib/calc";
import { computeShareResult } from "@/lib/share";

interface SharePageProps {
  searchParams: { model?: string; gpu?: string; quant?: string };
}

export async function generateMetadata({ searchParams }: SharePageProps): Promise<Metadata> {
  const result = computeShareResult(searchParams.model, searchParams.gpu, searchParams.quant);

  if (!result) {
    return {
      title: "Hardware Check AI | AI Hardware Planner",
      description: "Scenario-based planning dashboard for AI hardware requirements.",
    };
  }

  const { model, gpu, quant, totalGB, fits, tps } = result;
  const verdict = fits
    ? `✅ fits in ~${totalGB.toFixed(1)} GB`
    : `❌ needs ~${totalGB.toFixed(1)} GB (card has ${gpu.vram} GB)`;
  const title = `Can I run ${model.name} on a ${gpu.name}? ${verdict}`;
  const description = `${model.name} (${formatParams(model.params)}) at ${quant} on a ${gpu.name} (${gpu.vram} GB): ${
    fits ? "fits" : "does not fit"
  } — est. ${tps} tok/s. Plan your own loadout on Hardware Check AI.`;

  const ogParams = new URLSearchParams({ model: model.id, gpu: gpu.id, quant });

  return {
    title,
    description,
    openGraph: {
      title,
      description,
      images: [{ url: `/api/og?${ogParams.toString()}`, width: 1200, height: 630 }],
    },
  };
}

// The share URL exists to carry OG metadata for chat/social unfurls; real
// visitors land on the main app with the same configuration applied.
export default function SharePage({ searchParams }: SharePageProps) {
  const params = new URLSearchParams();
  params.set("scenario", "inference");
  if (searchParams.model) params.set("model", searchParams.model);
  if (searchParams.gpu) params.set("gpu", searchParams.gpu);
  if (searchParams.quant) params.set("quant", searchParams.quant);
  redirect(`/?${params.toString()}`);
}
