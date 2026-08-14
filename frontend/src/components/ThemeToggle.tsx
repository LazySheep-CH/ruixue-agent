"use client";

import { Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";

export function ThemeToggle() {
  const [dark, setDark] = useState<boolean | null>(null);

  useEffect(() => {
    const savedTheme = localStorage.getItem("ruixue_theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    setDark(savedTheme ? savedTheme === "dark" : prefersDark);
  }, []);

  useEffect(() => {
    if (dark === null) return;
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("ruixue_theme", dark ? "dark" : "light");
  }, [dark]);

  if (dark === null) return <span className="icon-button" aria-hidden="true" />;

  const label = dark ? "切换到浅色模式" : "切换到深色模式";
  return (
    <button
      type="button"
      onClick={() => setDark((value) => !value)}
      title={label}
      className="icon-button"
      aria-label={label}
    >
      {dark ? <Sun size={16} /> : <Moon size={16} />}
    </button>
  );
}
