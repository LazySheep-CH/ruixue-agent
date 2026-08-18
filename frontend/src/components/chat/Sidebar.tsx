"use client";

import { FileText, LogOut, Plus, Search, Trash2, X } from "lucide-react";
import { AnimatePresence, m } from "motion/react";
import { useMemo, useState } from "react";

import type { Thread } from "~/core/types";


export function Sidebar({
  open,
  threads,
  currentThreadId,
  username,
  onNewThread,
  onSelectThread,
  onDeleteThread,
  onLogout,
  onClose,
}: {
  open: boolean;
  threads: Thread[];
  currentThreadId: string | null;
  username: string;
  onNewThread: () => void;
  onSelectThread: (id: string) => void;
  onDeleteThread: (id: string) => void;
  onLogout: () => void;
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const filteredThreads = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase("zh-CN");
    if (!normalized) return threads;
    return threads.filter((thread) => thread.title.toLocaleLowerCase("zh-CN").includes(normalized));
  }, [query, threads]);

  return (
    <>
      <m.aside className={`sidebar${open ? " is-open" : ""}`} aria-label="研究工作区导航">
        <div className="sidebar-title">
          <div className="product-switcher">
            <span className="product-glyph">瑞</span>
            <span><strong>瑞雪智研</strong><small>农业材料智能工作台</small></span>
          </div>
          <button className="icon-button sidebar-close" onClick={onClose} aria-label="关闭侧边栏">
            <X size={16} />
          </button>
        </div>

        <m.button className="new-task-button" onClick={onNewThread} whileTap={{ scale: 0.975 }}>
          <Plus size={16} />
          <span>新建任务</span>
          <kbd>Ctrl N</kbd>
        </m.button>

        <label className="sidebar-search">
          <Search size={14} />
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="搜索任务"
            aria-label="搜索任务"
          />
        </label>

        <div className="sidebar-scroll">
          {/* 这里只放真实存在的东西:任务列表。之前的"项目"行是写死的摆设
              (没有项目功能),"运行说明"是把一次性提示做成了常驻横幅 ——
              后台续跑该在用户点停止时用 toast 告知,而不是天天挂在侧栏。 */}
          <div className="sidebar-section-title"><span>任务</span></div>
          <div className="record-list">
            <AnimatePresence initial={false} mode="popLayout">
              {filteredThreads.map((thread) => (
                <m.div
                  key={thread.id}
                  layout
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -8 }}
                  className={`record-row${currentThreadId === thread.id ? " is-active" : ""}`}
                >
                  <button onClick={() => onSelectThread(thread.id)} title={thread.title}>
                    <FileText size={13} />
                    <span>{thread.title}</span>
                    <small>{formatRelativeTime(thread.createdAt)}</small>
                  </button>
                  <button
                    className="record-row__delete"
                    onClick={() => onDeleteThread(thread.id)}
                    aria-label={`删除${thread.title}`}
                  >
                    <Trash2 size={13} />
                  </button>
                </m.div>
              ))}
            </AnimatePresence>
            {filteredThreads.length === 0 ? (
              <p className="sidebar-empty">{query ? "没有匹配的任务" : "还没有研究任务"}</p>
            ) : null}
          </div>
        </div>

        {/* 点不动的"设置/帮助"占位按钮已删:界面上出现的控件就该能用,
            "即将开放"的灰按钮只会消耗用户的信任。等真有设置项再加回来。 */}
        <footer className="sidebar-footer">
          <button className="sidebar-user" onClick={onLogout} title="退出登录">
            <span className="user-avatar">{username.slice(0, 1).toUpperCase() || "瑞"}</span>
            <span className="sidebar-user__name">{username || "用户"}</span>
            <LogOut size={14} />
          </button>
        </footer>
      </m.aside>

      <AnimatePresence>
        {open ? (
          <m.button
            className="sidebar-scrim"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            aria-label="关闭侧边栏遮罩"
          />
        ) : null}
      </AnimatePresence>
    </>
  );
}

function formatRelativeTime(timestamp: number): string {
  const days = Math.floor((Date.now() - timestamp) / 86_400_000);
  if (days <= 0) return "今天";
  if (days === 1) return "昨天";
  if (days < 7) return `${days} 天前`;
  return new Intl.DateTimeFormat("zh-CN", { month: "numeric", day: "numeric" }).format(timestamp);
}
