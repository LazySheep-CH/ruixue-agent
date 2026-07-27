"use client";

import { useMemo, useState } from "react";

import { useStore } from "~/core/store";
import type { Thread } from "~/core/types";

/** 按时间把会话分组 —— 学自 Claude Code:侧栏用分组标题,而非一长条平铺。 */
function groupThreads(threads: Thread[]): { label: string; items: Thread[] }[] {
  const DAY = 86_400_000;
  const now = Date.now();
  const buckets: Record<string, Thread[]> = { 今天: [], 最近7天: [], 更早: [] };
  for (const t of threads) {
    const age = now - t.createdAt;
    if (age < DAY) buckets["今天"].push(t);
    else if (age < 7 * DAY) buckets["最近7天"].push(t);
    else buckets["更早"].push(t);
  }
  return Object.entries(buckets)
    .filter(([, items]) => items.length > 0)
    .map(([label, items]) => ({ label, items }));
}

/**
 * 侧栏(结构学自 Claude Code 截图):
 * 窄(220px)、紧凑、会话按时间分组、每条带 ○ 圆点;收起后只留窄轨。
 */
export function Sidebar({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  const { threads, currentThreadId, newThread, selectThread, deleteThread } = useStore();
  const [q, setQ] = useState("");

  const groups = useMemo(
    () => groupThreads(q ? threads.filter((t) => t.title.includes(q)) : threads),
    [threads, q],
  );

  if (!open) {
    return (
      <div className="flex w-11 shrink-0 flex-col items-center border-r border-border bg-sidebar py-2.5">
        <button
          onClick={onToggle}
          title="展开侧栏"
          className="flex size-7 items-center justify-center rounded-md text-muted-foreground hover:bg-accent"
        >
          ☰
        </button>
      </div>
    );
  }

  return (
    <aside className="flex w-[220px] shrink-0 flex-col border-r border-border bg-sidebar text-sidebar-foreground">
      <div className="flex items-center gap-2 px-2.5 py-2.5">
        <div className="size-5 shrink-0 rounded bg-primary" />
        <span className="flex-1 truncate text-[13px] font-medium">瑞雪</span>
        <button
          onClick={onToggle}
          title="收起侧栏"
          className="flex size-6 items-center justify-center rounded text-muted-foreground hover:bg-accent"
        >
          ☰
        </button>
      </div>

      <div className="space-y-0.5 px-2.5 pb-2">
        <button
          onClick={() => newThread()}
          className="flex w-full items-center gap-1.5 rounded-md px-2 py-1.5 text-[13px]
            text-muted-foreground transition hover:bg-accent hover:text-foreground"
        >
          <span className="text-[15px] leading-none">＋</span> 新对话
        </button>
        <div className="flex items-center gap-1.5 rounded-md px-2 py-1.5 text-muted-foreground focus-within:bg-accent">
          <span className="text-[12px]">⌕</span>
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="搜索"
            className="w-full bg-transparent text-[13px] outline-none placeholder:text-muted-foreground"
          />
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-1.5 pb-2">
        {groups.length === 0 && (
          <p className="px-2.5 py-3 text-[12.5px] text-muted-foreground">
            {q ? "没有匹配的对话" : "还没有对话"}
          </p>
        )}
        {groups.map((g) => (
          <div key={g.label} className="mb-1">
            <p className="px-2.5 py-1.5 text-[11.5px] text-muted-foreground">{g.label}</p>
            {g.items.map((t) => (
              <div
                key={t.id}
                onClick={() => selectThread(t.id)}
                className={`group flex cursor-pointer items-center gap-2 rounded-md px-2.5 py-1.5 text-[13px]
                  ${
                    t.id === currentThreadId
                      ? "bg-accent text-foreground"
                      : "text-muted-foreground hover:bg-accent/60 hover:text-foreground"
                  }`}
              >
                <span className="text-[9px] leading-none opacity-60">○</span>
                <span className="flex-1 truncate">{t.title}</span>
                <button
                  title="删除"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteThread(t.id);
                  }}
                  className="shrink-0 opacity-0 transition group-hover:opacity-100 hover:text-primary"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        ))}
      </div>
    </aside>
  );
}
