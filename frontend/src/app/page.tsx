"use client";

import { m } from "motion/react";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { AuthGuard } from "~/components/AuthGuard";
import { Composer } from "~/components/chat/Composer";
import { MessageList } from "~/components/chat/MessageList";
import { Sidebar } from "~/components/chat/Sidebar";
import { TopBar } from "~/components/chat/TopBar";
import { WorkspacePanel } from "~/components/chat/WorkspacePanel";
import type { WorkspaceModule } from "~/components/chat/workspace-data";
import { checkHealth } from "~/core/api";
import { clearAuth, getUsername } from "~/core/auth";
import { useStore } from "~/core/store";

export default function WorkspacePage() {
  const router = useRouter();
  const threads = useStore((state) => state.threads);
  const currentThreadId = useStore((state) => state.currentThreadId);
  const messagesByThread = useStore((state) => state.messages);
  const sending = useStore((state) => state.sending);
  const newThread = useStore((state) => state.newThread);
  const selectThread = useStore((state) => state.selectThread);
  const deleteThread = useStore((state) => state.deleteThread);
  const restoreThread = useStore((state) => state.restoreThread);
  const send = useStore((state) => state.send);
  const stop = useStore((state) => state.stop);
  const resumeIfPending = useStore((state) => state.resumeIfPending);

  const [activeModule, setActiveModule] = useState<WorkspaceModule>("film");
  const [input, setInput] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [mounted, setMounted] = useState(false);
  const [online, setOnline] = useState<boolean | null>(null);

  const currentMessages = useMemo(
    () => (currentThreadId ? messagesByThread[currentThreadId] ?? [] : []),
    [currentThreadId, messagesByThread],
  );
  const currentThread = threads.find((thread) => thread.id === currentThreadId);
  const currentTitle = currentThread?.title ?? "新研究任务";

  useEffect(() => {
    setMounted(true);
    void resumeIfPending();
  }, [resumeIfPending]);

  useEffect(() => {
    const compact = window.matchMedia("(max-width: 1120px)");
    const mobile = window.matchMedia("(max-width: 720px)");
    setInspectorOpen(!compact.matches);
    setSidebarOpen(!mobile.matches);

    const syncInspector = (event: MediaQueryListEvent) => {
      if (event.matches) setInspectorOpen(false);
    };
    const syncSidebar = (event: MediaQueryListEvent) => setSidebarOpen(!event.matches);
    compact.addEventListener("change", syncInspector);
    mobile.addEventListener("change", syncSidebar);
    return () => {
      compact.removeEventListener("change", syncInspector);
      mobile.removeEventListener("change", syncSidebar);
    };
  }, []);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      const healthy = await checkHealth();
      if (active) setOnline(healthy);
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 30_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const handleNewThread = useCallback(() => {
    newThread();
    setInput("");
    toast.success("已创建新任务");
  }, [newThread]);

  useEffect(() => {
    const handleShortcut = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "n") {
        event.preventDefault();
        handleNewThread();
      }
    };
    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, [handleNewThread]);

  const runTask = async (request: string) => {
    const text = request.trim();
    if (!text || sending) return;
    setInput("");
    try {
      await send(text);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "任务启动失败", {
        description: "你的问题已经保留，可以直接重新运行。",
      });
    }
  };

  const handleDeleteThread = (id: string) => {
    const thread = threads.find((item) => item.id === id);
    if (!thread) return;
    const threadMessages = messagesByThread[id] ?? [];
    deleteThread(id);
    toast("任务已移除", {
      description: thread.title,
      action: { label: "撤销", onClick: () => restoreThread(thread, threadMessages) },
    });
  };

  const handleShare = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      toast.success("工作台链接已复制");
    } catch {
      toast.error("复制失败", { description: "请检查浏览器的剪贴板权限后重试。" });
    }
  };

  const handleLogout = () => {
    clearAuth();
    router.replace("/login");
  };

  return (
    <AuthGuard>
      <m.div className="app-shell" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <a className="skip-link" href="#workspace-main">跳转到主要内容</a>
        <Sidebar
          open={sidebarOpen}
          activeModule={activeModule}
          threads={mounted ? threads : []}
          currentThreadId={currentThreadId}
          username={mounted ? getUsername() : ""}
          onSelectModule={(module) => {
            setActiveModule(module);
            if (window.matchMedia("(max-width: 720px)").matches) setSidebarOpen(false);
          }}
          onNewThread={handleNewThread}
          onSelectThread={(id) => {
            selectThread(id);
            if (window.matchMedia("(max-width: 720px)").matches) setSidebarOpen(false);
          }}
          onDeleteThread={handleDeleteThread}
          onLogout={handleLogout}
          onClose={() => setSidebarOpen(false)}
        />

        <m.main id="workspace-main" className="app-main" layout>
          <TopBar
            activeModule={activeModule}
            title={currentTitle}
            inspectorOpen={inspectorOpen}
            sending={sending}
            online={online}
            onToggleSidebar={() => setSidebarOpen((value) => !value)}
            onToggleInspector={() => setInspectorOpen((value) => !value)}
            onShare={() => void handleShare()}
          />
          {mounted ? (
            <MessageList
              activeModule={activeModule}
              messages={currentMessages}
              sending={sending}
              onPick={setInput}
              onRetry={(question) => void runTask(question)}
            />
          ) : <div className="workspace-scroll" />}
          <Composer
            activeModule={activeModule}
            value={input}
            onChange={setInput}
            onSend={() => void runTask(input)}
            onStop={() => {
              stop();
              toast.info("已停止接收", { description: "后台任务仍会继续，稍后可自动恢复结果。" });
            }}
            sending={sending}
          />
        </m.main>

        <WorkspacePanel
          open={inspectorOpen}
          activeModule={activeModule}
          title={currentTitle}
          messages={currentMessages}
          running={sending}
          onClose={() => setInspectorOpen(false)}
        />
      </m.div>
    </AuthGuard>
  );
}
