import type { Metadata } from "next";

import "~/styles/globals.css";

export const metadata: Metadata = {
  title: "瑞雪地膜智能助手",
  description: "地膜知识问答、性能预测与用量估算",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
