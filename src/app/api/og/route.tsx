import { ImageResponse } from "next/og";
import { NextRequest } from "next/server";
import { formatParams } from "@/lib/calc";
import { computeShareResult } from "@/lib/share";

// OG share card. Next 14 route handlers can return ImageResponse and read
// search params directly (opengraph-image.tsx cannot), so the share page
// points its openGraph images here.

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const result = computeShareResult(
    searchParams.get("model") ?? undefined,
    searchParams.get("gpu") ?? undefined,
    searchParams.get("quant") ?? undefined
  );

  const title = result ? `${result.model.name} on ${result.gpu.name}` : "Can I run it?";
  const subtitle = result
    ? `${formatParams(result.model.params)} · ${result.quant} · ${result.gpu.vram} GB VRAM`
    : "AI hardware planning";
  const verdict = result
    ? result.fits
      ? `✅ FITS in ~${result.totalGB.toFixed(1)} GB`
      : `❌ NEEDS ~${result.totalGB.toFixed(1)} GB`
    : "Hardware Check AI";
  const tps = result ? `~${result.tps} tok/s est.` : "";
  const verdictColor = result && !result.fits ? "#f87171" : "#34d399";

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          backgroundColor: "#0b0f14",
          padding: 72,
          fontFamily: "sans-serif",
        }}
      >
        <div
          style={{
            display: "flex",
            fontSize: 28,
            letterSpacing: 8,
            color: "#22d3ee",
            textTransform: "uppercase",
          }}
        >
          Hardware Check AI
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{ display: "flex", fontSize: 64, fontWeight: 700, color: "#e5e7eb", lineHeight: 1.1 }}>
            {title}
          </div>
          <div style={{ display: "flex", fontSize: 32, color: "#9ca3af" }}>{subtitle}</div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 32 }}>
          <div
            style={{
              display: "flex",
              fontSize: 40,
              fontWeight: 700,
              color: verdictColor,
              backgroundColor: "rgba(255,255,255,0.06)",
              border: "2px solid rgba(255,255,255,0.15)",
              borderRadius: 16,
              padding: "16px 32px",
            }}
          >
            {verdict}
          </div>
          {tps ? <div style={{ display: "flex", fontSize: 32, color: "#9ca3af" }}>{tps}</div> : null}
        </div>
      </div>
    ),
    { width: 1200, height: 630 }
  );
}
