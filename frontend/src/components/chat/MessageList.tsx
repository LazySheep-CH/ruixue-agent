"use client";

import { useEffect, useRef } from "react";

import { renderMarkdown } from "~/lib/markdown";
import type { Message, ToolRun } from "~/core/types";

const SUGGESTIONS = [
  { title: "按地点预测性能", q: "新疆尉犁,PBAT70/PLA30 的 10µm 地膜,盖 90 天会怎样?" },
  { title: "查当地土壤", q: "寿光的土壤 pH 和有机质怎么样?" },
  { title: "估算用量", q: "100 亩地用 0.01mm 生物降解膜,要多少公斤?" },
  { title: "原理问答", q: "PBAT 地膜的降解机理是什么?" },
];

/** 工具名 → 给用户看的中文说明(技术名对用户没意义)。 */
const TOOL_LABELS: Record<string, { doing: string; done: string }> = {
  search_knowledge: { doing: "正在检索知识库", done: "检索了知识库" },
  estimate_film_usage: { doing: "正在估算用量", done: "估算了用量" },
  get_soil_info: { doing: "正在查询土壤数据", done: "查询了土壤数据" },
  get_climate_info: { doing: "正在查询气候数据", done: "查询了气候数据" },
  predict_by_location: { doing: "正在按地点预测性能", done: "按地点预测了性能" },
  predict_degradation: { doing: "正在预测降解率", done: "预测了降解率" },
  predict_water_vapor_rate: { doing: "正在预测水蒸气透过率", done: "预测了水蒸气透过率" },
  predict_tensile_strength: { doing: "正在预测拉伸强度", done: "预测了拉伸强度" },
  delegate_to_expert: { doing: "正在请专家处理", done: "请专家处理了子任务" },
};

const label = (t: ToolRun) =>
  TOOL_LABELS[t.name]?.[t.done ? "done" : "doing"] ?? (t.done ? t.name : `正在调用 ${t.name}`);

/** 工具调用条:让用户看见 agent 在做什么 —— agent 产品的核心体验。 */
function ToolTrace({ tools }: { tools: ToolRun[] }) {
  return (
    <div className="mb-3 space-y-1">
      {tools.map((t) => (
        <div key={t.name} className="flex items-center gap-2 text-[13px] text-muted-foreground">
          <span
            className={`h-1.5 w-1.5 shrink-0 rounded-full ${
              t.done ? "bg-primary" : "bg-primary running"
            }`}
          />
          <span>{label(t)}</span>
          {t.done && <span className="text-primary">✓</span>}
        </div>
      ))}
    </div>
  );
}

/** 空态:欢迎语 + 建议问题。 */
function Welcome({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="pt-16">
      <h1 className="mb-2 text-[30px] font-medium tracking-tight">今天想了解地膜的什么?</h1>
      <p className="mb-8 text-muted-foreground">知识问答 · 性能预测 · 环境查询 · 用量估算</p>
      <div className="space-y-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s.title}
            onClick={() => onPick(s.q)}
            className="group flex w-full items-baseline gap-3 rounded-lg px-3 py-2.5 text-left
              transition hover:bg-muted"
          >
            <span className="shrink-0 text-[13px] font-medium text-primary">{s.title}</span>
            <span className="truncate text-[14px] text-muted-foreground group-hover:text-foreground">{s.q}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

/** 用户消息:左侧竖线标记 + 稍暗文字(学自 成熟编码 agent:不用气泡、不用头像,
 *  靠排版区分角色 —— 信息密度更高,视线不被色块打断)。 */
function Bubble({ m }: { m: Message }) {
  if (m.role === "user") {
    return (
      <div className="mb-5 border-l-2 border-border pl-3">
        <div className="prose-msg text-[15px] text-muted-foreground">{m.content}</div>
      </div>
    );
  }
  return (
    <div className="mb-7">
      {m.tools && m.tools.length > 0 && <ToolTrace tools={m.tools} />}
      {m.thinking && (
        <details className="mb-3 rounded-lg border border-border bg-card px-3 py-2">
          <summary className="cursor-pointer select-none text-[13px] text-muted-foreground">思考过程</summary>
          <div className="mt-2 whitespace-pre-wrap text-[13.5px] leading-relaxed text-muted-foreground">
            {m.thinking}
          </div>
        </details>
      )}
      {m.error ? (
        <p className="text-sm text-destructive">{m.error}</p>
      ) : (
        <div
          className={`prose-msg ${m.streaming && !m.content ? "cursor" : ""}`}
          // renderMarkdown 内部先转义再套标签,无 XSS 风险
          dangerouslySetInnerHTML={{ __html: renderMarkdown(m.content) }}
        />
      )}
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
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  // 代码块的复制按钮由 markdown 渲染成 HTML,没有 React 事件 ——
  // 用【事件委托】在容器上统一接管:一个监听器管所有代码块,新增块也自动生效。
  useEffect(() => {
    const box = boxRef.current;
    if (!box) return;
    const onClick = (e: MouseEvent) => {
      const btn = (e.target as HTMLElement).closest<HTMLElement>(".copy-btn");
      if (!btn) return;
      void navigator.clipboard.writeText(decodeURIComponent(btn.dataset.code ?? ""));
      const old = btn.textContent;
      btn.textContent = "已复制";
      setTimeout(() => (btn.textContent = old), 1200);
    };
    box.addEventListener("click", onClick);
    return () => box.removeEventListener("click", onClick);
  }, []);

  return (
    <div ref={boxRef} className="flex-1 overflow-y-auto px-6">
      <div className="mx-auto max-w-[46rem] pb-6">
        {messages.length === 0 ? (
          <Welcome onPick={onPick} />
        ) : (
          <div className="pt-8">
            {messages.map((m) => (
              <Bubble key={m.id} m={m} />
            ))}
          </div>
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}
