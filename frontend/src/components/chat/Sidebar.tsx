"use client";

import { useState } from "react";

import { useStore } from "~/core/store";

/**
 * 单栏侧边栏(成熟编码 agent 风格):品牌 + 新对话 + 会话列表 + 折叠。
 * 不再是"图标条 + 面板"两层 —— 一层更简洁,也少一次点击。
 */
export function Sidebar({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  const { threads, currentThreadId, newThread, selectThread, deleteThread } = useStore();
  const [q, setQ] = useState("");

  const list = q ? threads.filter((t) => t.title.includes(q)) : threads;

  if (!open) {
    // 收起态:只留一个展开按钮,把空间全让给正文
    return (
      <div className="flex w-12 shrink-0 flex-col items-center border-r border-border bg-muted py-3">
        <button
          onClick={onToggle}
          title="展开侧栏"
          className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-accent"
        >
          ☰
        </button>
      </div>
    );
  }

  return (
    <aside className="flex w-[248px] shrink-0 flex-col border-r border-border bg-muted">
      <div className="flex items-center gap-2 px-3 py-3">
        <div className="h-6 w-6 rounded-md bg-primary" />
        <span className="flex-1 text-[14px] font-medium">瑞雪</span>
        <button
          onClick={onToggle}
          title="收起侧栏"
          className="flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground hover:bg-accent"
        >
          ☰
        </button>
      </div>

      <div className="px-3">
        <button
          onClick={() => newThread()}
          className="mb-2 flex w-full items-center gap-2 rounded-lg border border-border bg-card
            px-3 py-2 text-[14px] transition hover:border-primary hover:text-primary"
        >
          <span className="text-primary">＋</span> 新对话
        </button>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="搜索对话"
          className="mb-1 w-full rounded-lg bg-card px-3 py-1.5 text-[13px] outline-none
            placeholder:text-muted-foreground focus:ring-1 focus:ring-primary/30"
        />
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-2 py-2">
        {list.length === 0 && (
          <p className="px-2 py-3 text-[13px] text-muted-foreground">{q ? "没有匹配的对话" : "还没有对话"}</p>
        )}
        {list.map((t) => (
          <div
            key={t.id}
            onClick={() => selectThread(t.id)}
            className={`group flex cursor-pointer items-center gap-2 rounded-lg px-2.5 py-2 text-[13.5px]
              ${
                t.id === currentThreadId
                  ? "bg-primary/10 text-foreground"
                  : "text-muted-foreground hover:bg-accent/60 hover:text-foreground"
              }`}
          >
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
    </aside>
  );
}
