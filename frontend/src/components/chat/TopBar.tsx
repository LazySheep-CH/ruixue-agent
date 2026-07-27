"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { checkHealth } from "~/core/api";
import { ThemeToggle } from "~/components/ThemeToggle";
import { clearAuth, getUsername } from "~/core/auth";

/** 顶栏:标题 + 后端在线状态 + 当前用户 / 退出登录。 */
export function TopBar() {
  const router = useRouter();
  const [online, setOnline] = useState<boolean | null>(null);
  const [username, setUsername] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);

  // 用户名从 localStorage 读,须在客户端挂载后取(避免服务端渲染不一致)
  useEffect(() => setUsername(getUsername()), []);

  // 轮询健康检查:让用户一眼看出"是后端没起"还是"自己 key 不对"
  useEffect(() => {
    const tick = () => void checkHealth().then(setOnline);
    tick();
    const id = setInterval(tick, 15000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex h-13 shrink-0 items-center gap-2.5 border-b border-border px-5 py-3">
      <b className="text-[15px] font-medium">瑞雪地膜智能助手</b>
      <span className="text-[13px] text-muted-foreground">降解率 · 保墒 · 力学 · 用量</span>

      <span
        title={online ? "后端在线" : "后端未连接"}
        className={`ml-1 h-1.5 w-1.5 rounded-full ${
          online === null ? "bg-muted-foreground/30" : online ? "bg-primary" : "bg-destructive"
        }`}
      />

      <div className="ml-auto" />
      <ThemeToggle />

      <div className="relative">
        <button
          onClick={() => setMenuOpen((v) => !v)}
          className="flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-[13px] hover:bg-muted"
        >
          <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-[11px] text-primary-foreground">
            {username.slice(0, 1).toUpperCase() || "?"}
          </span>
          <span className="max-w-[90px] truncate">{username || "未登录"}</span>
        </button>

        {menuOpen && (
          <div className="absolute right-0 top-10 z-20 w-40 overflow-hidden rounded-[10px] border border-border bg-card shadow-lg">
            <button
              onClick={() => {
                clearAuth();
                router.replace("/login");
              }}
              className="w-full px-3.5 py-2.5 text-left text-[13px] hover:bg-muted"
            >
              退出登录
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
