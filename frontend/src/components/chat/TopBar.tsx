"use client";

import { useEffect, useState } from "react";

import { checkHealth } from "~/core/api";
import { useStore } from "~/core/store";

/** 顶栏:标题 + 后端在线状态 + API Key 设置。 */
export function TopBar() {
  const { apiKey, setApiKey } = useStore();
  const [online, setOnline] = useState<boolean | null>(null);

  // 轮询健康检查:让用户一眼看出"是后端没起"还是"自己 key 不对"
  useEffect(() => {
    const tick = () => void checkHealth().then(setOnline);
    tick();
    const id = setInterval(tick, 15000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex h-14 shrink-0 items-center gap-2.5 border-b border-line bg-surface px-4.5 px-5">
      <b className="font-semibold">瑞雪地膜智能助手</b>
      <span className="text-[13px] text-muted">降解率 · 保墒 · 力学 · 用量</span>

      <span
        title={online ? "后端在线" : "后端未连接"}
        className={`ml-1 h-1.5 w-1.5 rounded-full ${
          online === null ? "bg-[#d9d9e2]" : online ? "bg-[#12a150]" : "bg-[#d33]"
        }`}
      />

      <button
        onClick={() => {
          const k = window.prompt("请输入 API Key(如 demo-key-alice):", apiKey);
          if (k !== null) setApiKey(k);
        }}
        className={`ml-auto rounded-lg border px-3 py-1.5 text-[13px] transition hover:bg-wash
          ${apiKey ? "border-[#cdebd9] text-[#12a150]" : "border-line text-muted"}`}
      >
        {apiKey ? "Key 已设置" : "设置 API Key"}
      </button>
    </div>
  );
}
