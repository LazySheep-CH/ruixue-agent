"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { ThemeToggle } from "~/components/ThemeToggle";
import { checkHealth } from "~/core/api";
import { clearAuth, getUsername } from "~/core/auth";
import { useStore } from "~/core/store";

/** 图标按钮:统一尺寸与悬停反馈(截图里顶部是一排这样的小图标)。 */
function IconBtn({ icon, title, onClick }: { icon: string; title: string; onClick?: () => void }) {
  return (
    <button
      title={title}
      onClick={onClick}
      className="flex size-7 items-center justify-center rounded-md text-[13px] text-muted-foreground
        transition hover:bg-accent hover:text-foreground"
    >
      {icon}
    </button>
  );
}

/**
 * 顶栏(结构学自 Claude Code 截图):
 * 左侧一排图标工具、中间【面包屑标题 + 项目标签】、右侧主题与账号。
 * 相比原来的"大标题 + 副标题",信息密度更高、更像工作台。
 */
export function TopBar({ onToggleSidebar }: { onToggleSidebar: () => void }) {
  const router = useRouter();
  const { threads, currentThreadId, newThread } = useStore();
  const [online, setOnline] = useState<boolean | null>(null);
  const [username, setUsername] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => setUsername(getUsername()), []);

  useEffect(() => {
    const tick = () => void checkHealth().then(setOnline);
    tick();
    const id = setInterval(tick, 15000);
    return () => clearInterval(id);
  }, []);

  const current = threads.find((t) => t.id === currentThreadId);

  return (
    <div className="flex h-11 shrink-0 items-center gap-0.5 border-b border-border px-2">
      <IconBtn icon="☰" title="侧栏" onClick={onToggleSidebar} />
      <IconBtn icon="＋" title="新对话" onClick={() => newThread()} />

      <div className="mx-1 h-4 w-px bg-border" />

      <div className="flex min-w-0 items-center gap-2 px-1.5">
        <span className="text-[12px] text-muted-foreground">▤</span>
        <span className="truncate text-[13px] font-medium">{current?.title ?? "新对话"}</span>
        <span className="shrink-0 rounded bg-accent px-1.5 py-0.5 text-[11px] text-muted-foreground">
          瑞雪地膜
        </span>
        <span
          title={online ? "后端在线" : "后端未连接"}
          className={`size-1.5 shrink-0 rounded-full ${
            online === null ? "bg-muted-foreground/30" : online ? "bg-primary" : "bg-destructive"
          }`}
        />
      </div>

      <div className="ml-auto flex items-center gap-0.5">
        <ThemeToggle />
        <div className="relative">
          <button
            onClick={() => setMenuOpen((v) => !v)}
            className="flex items-center gap-1.5 rounded-md px-1.5 py-1 text-[12.5px] hover:bg-accent"
          >
            <span className="flex size-5 items-center justify-center rounded-full bg-primary text-[10px] text-primary-foreground">
              {username.slice(0, 1).toUpperCase() || "?"}
            </span>
            <span className="max-w-[84px] truncate text-muted-foreground">
              {username || "未登录"}
            </span>
          </button>

          {menuOpen && (
            <div className="absolute right-0 top-8 z-20 w-36 overflow-hidden rounded-md border border-border bg-card shadow-lg">
              <button
                onClick={() => {
                  clearAuth();
                  router.replace("/login");
                }}
                className="w-full px-3 py-2 text-left text-[12.5px] hover:bg-accent"
              >
                退出登录
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
