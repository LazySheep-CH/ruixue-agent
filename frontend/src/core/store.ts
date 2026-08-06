/** 全局状态:会话列表、当前会话的消息、API Key。
 *
 * 用 zustand:比 Context 轻,不需要 Provider 包一层,组件按需订阅、精准重渲染。
 * 会话列表与 API Key 持久化到 localStorage —— 刷新不丢(单文件版的痛点)。
 */

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

import { getRun, resumeRun, streamChat } from "./api";
import type { Message, Thread } from "./types";

interface State {
  threads: Thread[];
  currentThreadId: string | null;
  /** 每个会话的消息:threadId -> 消息数组 */
  messages: Record<string, Message[]>;
  sending: boolean;

  newThread: () => string;
  selectThread: (id: string) => void;
  deleteThread: (id: string) => void;
  send: (text: string) => Promise<void>;
  stop: () => void;
  /** 页面加载时调用:若上次有未看完的运行,取回其结果。 */
  resumeIfPending: () => Promise<void>;
}

const uid = () => Math.random().toString(36).slice(2, 10);
/** 用首句话当会话标题(截断),像 ChatGPT 那样。 */
const titleOf = (text: string) => (text.length > 18 ? text.slice(0, 18) + "…" : text);

let controller: AbortController | null = null;

/**
 * "未完成的运行"登记。
 *
 * 为什么需要:后端改成【异步执行】后,客户端断开(刷新页面、切网络、锁屏)
 * 不会停掉 agent —— 它在服务端跑完并落库。但前端刷新后内存全没了,
 * 不知道刚才那次跑到哪了。故把 run_id 存进 localStorage,
 * 重新打开时调 resumeIfPending() 取回结果 —— 用户不用重问、不用重新花钱。
 */
type PendingRun = { runId: string; threadId: string; msgId: string };
const PENDING_KEY = "ruixue.pendingRun";
let pendingRun: PendingRun | null = null;

const savePendingRun = (p: PendingRun) => {
  pendingRun = p;
  try {
    localStorage.setItem(PENDING_KEY, JSON.stringify(p));
  } catch {
    // 隐私模式下 localStorage 可能不可用 —— 只是失去恢复能力,不该影响对话
  }
};
const clearPendingRun = () => {
  pendingRun = null;
  try {
    localStorage.removeItem(PENDING_KEY);
  } catch {
    /* 同上 */
  }
};
const loadPendingRun = (): PendingRun | null => {
  try {
    const raw = localStorage.getItem(PENDING_KEY);
    return raw ? (JSON.parse(raw) as PendingRun) : null;
  } catch {
    return null;
  }
};

export const useStore = create<State>()(
  persist(
    (set, get) => ({
      threads: [],
      currentThreadId: null,
      messages: {},
      sending: false,

      newThread: () => {
        const t: Thread = { id: `t${Date.now()}`, title: "新对话", createdAt: Date.now() };
        set((s) => ({
          threads: [t, ...s.threads],
          currentThreadId: t.id,
          messages: { ...s.messages, [t.id]: [] },
        }));
        return t.id;
      },

      selectThread: (id) => set({ currentThreadId: id }),

      deleteThread: (id) =>
        set((s) => {
          const threads = s.threads.filter((t) => t.id !== id);
          const messages = { ...s.messages };
          delete messages[id];
          return {
            threads,
            messages,
            currentThreadId: s.currentThreadId === id ? (threads[0]?.id ?? null) : s.currentThreadId,
          };
        }),

      stop: () => {
        controller?.abort();
        controller = null;
        set({ sending: false });
        // 注意:这里只断开【本地这条流】,服务端的 agent 会继续跑完并落库。
        // 保留 pendingRun 是有意的 —— 用户下次进来还能看到结果,钱没白花。
      },

      resumeIfPending: async () => {
        const p = loadPendingRun();
        if (!p) return;

        /** 定位到上次那条助手消息;它可能已被删会话清掉 —— 那就无处可恢复。 */
        const patch = (fn: (m: Message) => Message) =>
          set((s) => ({
            messages: {
              ...s.messages,
              [p.threadId]: (s.messages[p.threadId] ?? []).map((m) =>
                m.id === p.msgId ? fn(m) : m,
              ),
            },
          }));
        if (!(get().messages[p.threadId] ?? []).some((m) => m.id === p.msgId)) {
          clearPendingRun();
          return;
        }

        try {
          const run = await getRun(p.runId);
          if (run.status === "running") {
            // 还在跑:重新订阅,把已产生的内容补上并继续接收
            set({ sending: true });
            patch((m) => ({ ...m, content: "", streaming: true }));
            controller = new AbortController();
            await resumeRun(
              p.runId,
              (e) => {
                if (e.type === "answer") patch((m) => ({ ...m, content: m.content + e.text }));
                else if (e.type === "error") patch((m) => ({ ...m, error: e.text }));
              },
              controller.signal,
            );
          } else {
            // 已结束:直接把最终答案(或失败原因)填回去
            patch((m) => ({
              ...m,
              content: run.answer ?? m.content,
              error: run.error ?? undefined,
            }));
          }
        } catch {
          // 恢复失败(运行已过期/被清理)不该报错打扰用户,静默放弃即可
        } finally {
          clearPendingRun();
          set({ sending: false });
          patch((m) => ({
            ...m,
            streaming: false,
            tools: (m.tools ?? []).map((t) => ({ ...t, done: true })),
          }));
        }
      },

      send: async (text) => {
        const { sending } = get();
        if (!text.trim() || sending) return;

        // 没有当前会话就先开一个
        let threadId = get().currentThreadId;
        if (!threadId) threadId = get().newThread();

        const userMsg: Message = { id: uid(), role: "user", content: text };
        const botMsg: Message = { id: uid(), role: "assistant", content: "", streaming: true };

        set((s) => ({
          sending: true,
          messages: { ...s.messages, [threadId]: [...(s.messages[threadId] ?? []), userMsg, botMsg] },
          // 首条消息用作会话标题
          threads: s.threads.map((t) =>
            t.id === threadId && t.title === "新对话" ? { ...t, title: titleOf(text) } : t,
          ),
        }));

        /** 局部更新那条正在流式的助手消息 */
        const patch = (fn: (m: Message) => Message) =>
          set((s) => ({
            messages: {
              ...s.messages,
              [threadId]: (s.messages[threadId] ?? []).map((m) => (m.id === botMsg.id ? fn(m) : m)),
            },
          }));

        controller = new AbortController();
        try {
          await streamChat(
            { threadId, message: text },
            (e) => {
              patch((m) => {
                switch (e.type) {
                  case "thinking":
                    return { ...m, thinking: (m.thinking ?? "") + e.text };
                  case "answer":
                    return { ...m, content: m.content + e.text };
                  case "tool_start": {
                    const tools = m.tools ?? [];
                    if (tools.some((t) => t.name === e.name)) return m;
                    return { ...m, tools: [...tools, { name: e.name, done: false }] };
                  }
                  case "tool_end":
                    return {
                      ...m,
                      tools: (m.tools ?? []).map((t) =>
                        t.name === e.name ? { ...t, done: true } : t,
                      ),
                    };
                  // 后端在流的开头下发运行编号:记下来,刷新后凭它恢复现场。
                  // 这条不改消息内容,只做副作用登记。
                  case "run":
                    pendingRun = { runId: e.run_id, threadId, msgId: botMsg.id };
                    savePendingRun(pendingRun);
                    return m;
                  case "done":
                    return m; // 收尾统一在流结束后做
                  case "error":
                    return { ...m, error: e.text };
                }
              });
            },
            controller.signal,
          );
          clearPendingRun(); // 正常收尾,不再需要恢复
          // 流结束:把还挂着的工具标记为完成,避免出现永远转圈的进度
          patch((m) => ({
            ...m,
            streaming: false,
            tools: (m.tools ?? []).map((t) => ({ ...t, done: true })),
          }));
        } catch (err) {
          const aborted = err instanceof DOMException && err.name === "AbortError";
          patch((m) => ({
            ...m,
            streaming: false,
            error: aborted ? undefined : err instanceof Error ? err.message : "请求失败",
          }));
        } finally {
          controller = null;
          set({ sending: false });
        }
      },
    }),
    {
      name: "ruixue-chat",
      storage: createJSONStorage(() => localStorage),
      // 只持久化这三项;sending 这类瞬时状态不存
      partialize: (s) => ({
        threads: s.threads,
        messages: s.messages,
        currentThreadId: s.currentThreadId,
      }),
    },
  ),
);
