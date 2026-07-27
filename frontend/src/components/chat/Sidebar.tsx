"use client";

import { useState } from "react";

import { useStore } from "~/core/store";

/** 左侧图标条:功能入口 + 折叠会话面板。 */
export function Rail({ onToggle }: { onToggle: () => void }) {
  const items = [
    { icon: "◌", label: "对话", active: true, onClick: onToggle },
    { icon: "▦", label: "知识库" },
    { icon: "✦", label: "预测模型" },
    { icon: "◇", label: "环境数据" },
  ];
  return (
    <aside className="flex w-[74px] shrink-0 flex-col items-center gap-1.5 border-r border-line bg-surface py-3.5">
      <div className="mb-3.5 h-8 w-8 rounded-[10px] bg-gradient-to-br from-brand to-[#7aa2ff]" />
      {items.map((it) => (
        <button
          key={it.label}
          title={it.label}
          onClick={it.onClick}
          className={`flex h-11 w-11 items-center justify-center rounded-xl text-[17px] transition
            ${it.active ? "bg-[#eef3ff] text-brand" : "text-muted hover:bg-wash hover:text-ink"}`}
        >
          {it.icon}
        </button>
      ))}
      <div className="flex-1" />
      <div className="flex h-[34px] w-[34px] items-center justify-center rounded-full bg-[#e8e8ef] text-xs font-semibold text-muted">
        瑞雪
      </div>
    </aside>
  );
}

/** 会话面板:新建 / 搜索 / 切换 / 删除。会话持久化在 localStorage,刷新不丢。 */
export function ThreadPanel({ open }: { open: boolean }) {
  const { threads, currentThreadId, newThread, selectThread, deleteThread } = useStore();
  const [q, setQ] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);

  const list = q ? threads.filter((t) => t.title.includes(q)) : threads;

  return (
    <aside
      className={`flex w-[276px] shrink-0 flex-col border-r border-line bg-surface px-3 py-3.5
        transition-[margin] duration-300 ${open ? "" : "ml-[-276px]"}`}
    >
      <div className="flex items-center justify-between px-1 pb-2.5 font-semibold">
        <span>对话</span>
        <button
          title="搜索"
          onClick={() => setSearchOpen((v) => !v)}
          className="h-[30px] w-[30px] rounded-lg text-muted hover:bg-wash"
        >
          ⌕
        </button>
      </div>

      <div className={`overflow-hidden transition-[max-height] ${searchOpen ? "max-h-14" : "max-h-0"}`}>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="搜索对话..."
          className="mb-2 w-full rounded-[10px] border border-line bg-wash px-3 py-2 outline-none"
        />
      </div>

      <button
        onClick={() => newThread()}
        className="mb-1.5 rounded-[10px] bg-ink px-3 py-2.5 text-left font-medium text-white transition hover:opacity-90"
      >
        ＋ 新对话
      </button>

      <div className="mt-3 min-h-0 flex-1 overflow-y-auto">
        <p className="px-1.5 pb-1.5 text-xs text-muted">最近</p>
        {list.length === 0 && <p className="px-1.5 text-[13px] text-muted">还没有对话</p>}
        {list.map((t) => (
          <div
            key={t.id}
            onClick={() => selectThread(t.id)}
            className={`group flex cursor-pointer items-center gap-2 rounded-[9px] px-2.5 py-2 text-sm
              ${t.id === currentThreadId ? "bg-[#eef3ff] text-brand" : "text-[#3a3a45] hover:bg-wash"}`}
          >
            <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-brand" />
            <span className="flex-1 truncate">{t.title}</span>
            <button
              title="删除"
              onClick={(e) => {
                e.stopPropagation();
                deleteThread(t.id);
              }}
              className="shrink-0 text-muted opacity-0 transition group-hover:opacity-100 hover:text-ink"
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}
