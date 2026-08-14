"use client";

import {
  ChevronDown,
  Clock3,
  FileText,
  Folder,
  HelpCircle,
  LogOut,
  Plus,
  Search,
  Settings,
  Trash2,
  X,
} from "lucide-react";
import { AnimatePresence, m } from "motion/react";
import { useMemo, useState } from "react";

import type { Thread } from "~/core/types";

import { workspaceModules, type WorkspaceModule } from "./workspace-data";

export function Sidebar({
  open,
  activeModule,
  threads,
  currentThreadId,
  username,
  onSelectModule,
  onNewThread,
  onSelectThread,
  onDeleteThread,
  onLogout,
  onClose,
}: {
  open: boolean;
  activeModule: WorkspaceModule;
  threads: Thread[];
  currentThreadId: string | null;
  username: string;
  onSelectModule: (module: WorkspaceModule) => void;
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
        <div className="window-chrome" aria-hidden="true">
          <span className="traffic traffic--red" />
          <span className="traffic traffic--yellow" />
          <span className="traffic traffic--green" />
        </div>

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

        <nav className="module-nav" aria-label="工作模块">
          {workspaceModules.map((module) => {
            const Icon = module.icon;
            const active = activeModule === module.id;
            return (
              <button
                type="button"
                key={module.id}
                className={active ? "is-active" : ""}
                onClick={() => onSelectModule(module.id)}
                aria-current={active ? "page" : undefined}
              >
                {active ? <m.span className="module-nav__active" layoutId="module-active" /> : null}
                <Icon size={16} strokeWidth={1.8} />
                <span>{module.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-scroll">
          <div className="sidebar-section-title"><span>项目</span></div>
          <div className="project-row" aria-label="当前项目">
            <Folder size={15} />
            <span>2026 春播研究</span>
            <ChevronDown size={13} />
          </div>

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

          <div className="sidebar-section-title sidebar-section-title--recent"><span>运行说明</span></div>
          <div className="recent-row recent-row--static">
            <Clock3 size={14} />
            <span>关闭页面后，后台任务仍会继续</span>
          </div>
        </div>

        <footer className="sidebar-footer">
          <button disabled title="设置功能即将开放"><Settings size={15} /><span>设置</span></button>
          <button disabled title="帮助功能即将开放"><HelpCircle size={15} /><span>帮助</span></button>
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
