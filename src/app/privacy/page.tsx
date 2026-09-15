import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Privacy Policy | Hardware Check AI",
  description: "Hardware Check AI runs entirely in your browser — no accounts, no tracking, no data collection.",
};

export default function PrivacyPage() {
  return (
    <div className="flex-1 overflow-y-auto custom-scrollbar p-10 max-w-3xl mx-auto h-full">
      <h1 className="text-4xl font-bold text-foreground mb-8">Privacy Policy</h1>
      <div className="space-y-6 text-foreground/80 leading-relaxed text-sm pb-20">
        <p>
          Hardware Check AI is a client-side-only planning tool. It is designed
          so that your information never leaves your device.
        </p>
        <h2 className="text-xl font-bold text-foreground">No accounts, no tracking</h2>
        <p>
          We do not require accounts, we do not use analytics or advertising
          trackers, and we do not set cookies for tracking purposes.
        </p>
        <h2 className="text-xl font-bold text-foreground">Calculations run in your browser</h2>
        <p>
          All hardware compatibility and memory calculations run locally in
          your browser. The configuration you select (model, GPU, quantization,
          and related settings) is stored only in the page URL so you can share
          or bookmark it. It is never transmitted to or stored on our servers.
        </p>
        <h2 className="text-xl font-bold text-foreground">Shared links</h2>
        <p>
          If you share a link from this tool, the configuration it contains is
          visible to whoever receives the link. Share links only with people
          you intend to see that configuration.
        </p>
        <h2 className="text-xl font-bold text-foreground">Changes</h2>
        <p>
          If this policy changes, the updated version will be posted on this
          page.
        </p>
      </div>
    </div>
  );
}
