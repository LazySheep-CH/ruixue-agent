"use client";

import { useEffect, useState } from "react";

import { AuthGuard } from "~/components/AuthGuard";
import { Composer } from "~/components/chat/Composer";
import { MessageList } from "~/components/chat/MessageList";
import { Sidebar } from "~/components/chat/Sidebar";
import { TopBar } from "~/components/chat/TopBar";
import { useStore } from "~/core/store";

export default function ChatPage() {
  const { currentThreadId, messages, sending, send, stop } = useStore();
  const [input, setInput] = useState("");
  const [panelOpen, setPanelOpen] = useState(true);
  const [toast, setToast] = useState("");
  // 状态从 localStorage 恢复前先不渲染列表,避免服务端/客户端不一致的水合报错
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  const list = (currentThreadId && messages[currentThreadId]) || [];

  const handleSend = async () => {
    const text = input.trim();
    if (!text || sending) return;
    setInput("");
    try {
      await send(text);
    } catch (err) {
      setToast(err instanceof Error ? err.message : "发送失败");
      setTimeout(() => setToast(""), 2200);
    }
  };

  return (
    <AuthGuard>
    <div className="flex h-screen">
      <Sidebar open={panelOpen} onToggle={() => setPanelOpen((v) => !v)} />

      <main className="flex min-w-0 flex-1 flex-col">
        <TopBar />
        {mounted ? (
          <MessageList messages={list} onPick={setInput} />
        ) : (
          <div className="flex-1" />
        )}
        <Composer
          value={input}
          onChange={setInput}
          onSend={handleSend}
          onStop={stop}
          sending={sending}
        />
      </main>

      {toast && (
        <div className="fixed bottom-24 left-1/2 -translate-x-1/2 rounded-full bg-ink px-4 py-2 text-[13px] text-white shadow-lg">
          {toast}
        </div>
      )}
    </div>
    </AuthGuard>
  );
}
