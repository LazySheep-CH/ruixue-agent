"use client";

import { Check, CloudOff, PanelLeft, PanelRight, Share2 } from "lucide-react";
import { m } from "motion/react";

import { ThemeToggle } from "~/components/ThemeToggle";

import { moduleLabels, type WorkspaceModule } from "./workspace-data";

export function TopBar({
  activeModule,
  title,
  inspectorOpen,
  sending,
  online,
  onToggleSidebar,
  onToggleInspector,
  onShare,
}: {
  activeModule: WorkspaceModule;
  title: string;
  inspectorOpen: boolean;
  sending: boolean;
  online: boolean | null;
  onToggleSidebar: () => void;
  onToggleInspector: () => void;
  onShare: () => void;
}) {
  return (
    <header className="topbar">
      <div className="topbar-leading">
        <button className="icon-button" onClick={onToggleSidebar} aria-label="切换侧边栏">
          <PanelLeft size={17} />
        </button>
        <div className="document-path" title={title}>
          <span>{moduleLabels[activeModule]}</span>
          <b>/</b>
          <strong>{title}</strong>
        </div>
      </div>

      <div className="topbar-actions">
        <m.span
          className={`save-state${online === false ? " is-offline" : ""}`}
          key={sending ? "sending" : online === false ? "offline" : "saved"}
          initial={{ opacity: 0, y: -3 }}
          animate={{ opacity: 1, y: 0 }}
          aria-live="polite"
        >
          {online === false ? <CloudOff size={12} /> : <Check size={12} />}
          {online === false ? "服务离线" : sending ? "后台处理中" : "已保存"}
        </m.span>
        <ThemeToggle />
        <button className="toolbar-button" onClick={onShare}>
          <Share2 size={14} />
          <span>复制链接</span>
        </button>
        <button
          className={`icon-button${inspectorOpen ? " is-active" : ""}`}
          onClick={onToggleInspector}
          aria-label="切换检查器"
          aria-pressed={inspectorOpen}
        >
          <PanelRight size={17} />
        </button>
      </div>
    </header>
  );
}
