"use client";

import { useEffect, useRef } from "react";

import { renderMarkdown } from "~/lib/markdown";
import type { Message } from "~/core/types";

const SUGGESTIONS = [
  { title: "新疆尉犁的地膜表现", q: "新疆尉犁,PBAT70/PLA30 的 10µm 地膜,盖 90 天会怎样?" },
  { title: "查当地土壤", q: "寿光的土壤 pH 和有机质怎么样?" },
  { title: "估算用量", q: "100 亩地用 0.01mm 生物降解膜,要多少公斤?" },
  { title: "原理问答", q: "PBAT 地膜的降解机理是什么?" },
];

/** 空态:欢迎语 + 建议问题(点了填进输入框)。 */
function Welcome({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="py-14 text-center">
      <h1 className="mb-2 text-[28px] font-semibold">今天想了解地膜的什么?</h1>
      <p className="text-muted">覆盖知识问答、性能预测与用量估算</p>
      <div className="mt-6 grid gap-2.5 sm:grid-cols-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s.title}
            onClick={() => onPick(s.q)}
            className="rounded-card border border-line bg-surface px-4 py-3.5 text-left transition
              hover:-translate-y-px hover:border-[#d6d6e0] hover:shadow-[0_4px_14px_rgba(17,17,26,.05)]"
          >
            <b className="mb-0.5 block text-sm font-semibold">{s.title}</b>
            <span className="text-[13px] text-muted">{s.q}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function Bubble({ m }: { m: Message }) {
  const isUser = m.role === "user";
  return (
    <div className="mb-6 flex gap-3">
      <div
        className={`flex h-[30px] w-[30px] shrink-0 items-center justify-center rounded-[9px] text-xs font-semibold
          ${isUser ? "bg-[#e8e8ef] text-muted" : "bg-gradient-to-br from-brand to-[#7aa2ff] text-white"}`}
      >
        {isUser ? "我" : "瑞"}
      </div>
      <div className="min-w-0 flex-1 pt-0.5">
        {m.thinking && (
          <details className="mb-2.5 border-l-2 border-line py-0.5 pl-3 text-[13.5px] text-muted">
            <summary className="cursor-pointer select-none pb-1 text-[13px]">思考过程</summary>
            <div className="whitespace-pre-wrap">{m.thinking}</div>
          </details>
        )}
        {m.error ? (
          <p className="text-sm text-[#d33]">{m.error}</p>
        ) : (
          <div
            className={`prose-msg ${m.streaming && !m.content ? "cursor" : ""}`}
            // 内容已在 renderMarkdown 里先转义再套标签,无 XSS 风险
            dangerouslySetInnerHTML={{ __html: renderMarkdown(m.content) }}
          />
        )}
      </div>
    </div>
  );
}

export function MessageList({
  messages,
  onPick,
}: {
  messages: Message[];
  onPick: (q: string) => void;
}) {
  const endRef = useRef<HTMLDivElement>(null);

  // 新内容到达时自动滚到底(流式过程中持续触发)
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-7">
      <div className="mx-auto max-w-[760px]">
        {messages.length === 0 ? (
          <Welcome onPick={onPick} />
        ) : (
          messages.map((m) => <Bubble key={m.id} m={m} />)
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}
