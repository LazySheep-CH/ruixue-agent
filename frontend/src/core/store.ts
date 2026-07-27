/** 全局状态:会话列表、当前会话的消息、API Key。
 *
 * 用 zustand:比 Context 轻,不需要 Provider 包一层,组件按需订阅、精准重渲染。
 * 会话列表与 API Key 持久化到 localStorage —— 刷新不丢(单文件版的痛点)。
 */

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

import { streamChat } from "./api";
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
}

const uid = () => Math.random().toString(36).slice(2, 10);
/** 用首句话当会话标题(截断),像 ChatGPT 那样。 */
const titleOf = (text: string) => (text.length > 18 ? text.slice(0, 18) + "…" : text);

let controller: AbortController | null = null;

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
          await streamChat({ threadId, message: text }, (e) => {
            patch((m) =>
              e.type === "thinking"
                ? { ...m, thinking: (m.thinking ?? "") + e.text }
                : { ...m, content: m.content + e.text },
            );
          }, controller.signal);
          patch((m) => ({ ...m, streaming: false }));
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
