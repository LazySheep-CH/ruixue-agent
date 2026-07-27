import type { Config } from "tailwindcss";

// 设计变量集中在这里(对应 src/styles/globals.css 的 CSS 变量)。
// 改配色只改这一处,全站生效。
export default {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "var(--ink)",
        muted: "var(--muted)",
        line: "var(--line)",
        surface: "var(--surface)",
        wash: "var(--wash)",
        sand: "var(--sand)",
        brand: "var(--brand)",
        "brand-soft": "var(--brand-soft)",
      },
      borderRadius: { card: "12px" },
      maxWidth: { reading: "740px" }, // 正文阅读宽度
    },
  },
  plugins: [],
} satisfies Config;
