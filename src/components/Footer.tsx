import Link from "next/link";
import { MODELS_DATA_AS_OF, PRICING_DATA_AS_OF } from "@/lib/calc";

export function Footer() {
  return (
    <footer className="border-t border-border bg-card/80 px-6 py-6">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="text-xs text-muted-foreground leading-relaxed max-w-xl">
          <p>
            All figures are estimates for planning purposes — verify current
            pricing and compatibility before purchasing.
          </p>
          <p className="mt-1 font-mono">
            Model data verified {MODELS_DATA_AS_OF} · Prices: {PRICING_DATA_AS_OF} snapshot
          </p>
        </div>
        <nav className="flex items-center gap-5 text-xs font-mono uppercase tracking-widest text-muted-foreground">
          <a href="mailto:hello@hardwarecheck.ai" className="hover:text-primary transition-colors">
            Contact
          </a>
          <a href="#" className="hover:text-primary transition-colors">
            GitHub
          </a>
          <Link href="/privacy" className="hover:text-primary transition-colors">
            Privacy
          </Link>
          <Link href="/terms" className="hover:text-primary transition-colors">
            Terms
          </Link>
        </nav>
      </div>
    </footer>
  );
}
