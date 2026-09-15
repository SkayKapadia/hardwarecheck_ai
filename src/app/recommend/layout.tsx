import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Budget Planner | Hardware Check AI",
  description:
    "Tell us your budget and use case — get a data-driven local LLM hardware loadout and model recommendation.",
};

export default function RecommendLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return children;
}
