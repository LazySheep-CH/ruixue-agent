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
        brand: "var(--brand)",
      },
      borderRadius: { card: "14px" },
    },
  },
  plugins: [],
} satisfies Config;
