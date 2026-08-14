import type { Metadata, Viewport } from "next";

import { AppProviders } from "~/components/AppProviders";
import "~/styles/globals.css";

export const metadata: Metadata = {
  title: "瑞雪智研",
  description: "农业问题、地膜选型与田间研究工作台",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  colorScheme: "light dark",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f5f5f7" },
    { media: "(prefers-color-scheme: dark)", color: "#1c1c1e" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body><AppProviders>{children}</AppProviders></body>
    </html>
  );
}
