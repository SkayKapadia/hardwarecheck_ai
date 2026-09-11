import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { TopNav } from "@/components/TopNav";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const jetbrainsMono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-jetbrains-mono" });

export const metadata: Metadata = {
  title: "Hardware Check AI | AI Hardware Planner",
  description: "Scenario-based planning dashboard for AI hardware requirements.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              window.addEventListener('unhandledrejection', function(event) {
                if (event.reason && typeof event.reason.message === 'string' && event.reason.message.includes('MetaMask')) {
                  event.preventDefault();
                }
              });
            `,
          }}
        />
      </head>
      <body className={`${inter.variable} ${jetbrainsMono.variable} font-sans min-h-screen flex flex-col bg-background text-foreground`}>
        <TopNav />
        <main className="flex-1 flex flex-col overflow-hidden">
          {children}
        </main>
      </body>
    </html>
  );
}
