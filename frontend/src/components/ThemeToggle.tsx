"use client";

import { useEffect, useState } from "react";

/**
 * 明暗主题切换。
 *
 * 实现要点:只往 <html> 上加/去 `dark` 类 —— 所有颜色都是 CSS 变量,
 * `.dark` 下重新赋值即可整站切换,**组件代码一行都不用改**(学自 参考架构)。
 * 首选跟随系统,用户手动选过则记住选择。
 */
export function ThemeToggle() {
  const [dark, setDark] = useState<boolean | null>(null);

  useEffect(() => {
    const saved = localStorage.getItem("ruixue_theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    setDark(saved ? saved === "dark" : prefersDark);
  }, []);

  useEffect(() => {
    if (dark === null) return;
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("ruixue_theme", dark ? "dark" : "light");
  }, [dark]);

  if (dark === null) return <span className="h-7 w-7" />; // 占位,避免布局跳动

  return (
    <button
      onClick={() => setDark((v) => !v)}
      title={dark ? "切换到浅色" : "切换到深色"}
      className="flex h-7 w-7 items-center justify-center rounded-md text-[13px]
        text-muted-foreground transition hover:bg-accent hover:text-foreground"
    >
      {dark ? "☀" : "☾"}
    </button>
  );
}
