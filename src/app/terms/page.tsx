import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Terms of Service | Hardware Check AI",
  description: "Terms of service for Hardware Check AI, a client-side AI hardware planning tool.",
};

export default function TermsPage() {
  return (
    <div className="flex-1 overflow-y-auto custom-scrollbar p-10 max-w-3xl mx-auto h-full">
      <h1 className="text-4xl font-bold text-foreground mb-8">Terms of Service</h1>
      <div className="space-y-6 text-foreground/80 leading-relaxed text-sm pb-20">
        <p>
          By using Hardware Check AI you agree to these terms. If you do not
          agree, please do not use the tool.
        </p>
        <h2 className="text-xl font-bold text-foreground">Estimates only</h2>
        <p>
          All figures produced by this tool — memory requirements, performance
          estimates, and prices — are approximations for planning purposes.
          Actual requirements, performance, and pricing vary by runtime,
          driver, vendor, and market conditions. Always verify current pricing
          and compatibility before purchasing hardware.
        </p>
        <h2 className="text-xl font-bold text-foreground">No warranty</h2>
        <p>
          The tool is provided &quot;as is&quot;, without warranty of any kind. We are
          not liable for any purchasing decisions, damages, or losses arising
          from the use of this tool.
        </p>
        <h2 className="text-xl font-bold text-foreground">Acceptable use</h2>
        <p>
          The tool runs entirely in your browser and is provided for personal
          and professional planning use. Do not attempt to disrupt the service
          or misrepresent its output as guaranteed measurements.
        </p>
        <h2 className="text-xl font-bold text-foreground">Changes</h2>
        <p>
          We may update these terms from time to time. The current version is
          always posted on this page.
        </p>
      </div>
    </div>
  );
}
