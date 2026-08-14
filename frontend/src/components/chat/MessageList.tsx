"use client";

import {
  ArrowRight,
  Check,
  ChevronRight,
  CircleAlert,
  Copy,
  Database,
  FileDown,
  FlaskConical,
  MapPin,
  RotateCcw,
  Sparkles,
} from "lucide-react";
import { AnimatePresence, m } from "motion/react";
import dynamic from "next/dynamic";
import { useEffect, useRef } from "react";
import { toast } from "sonner";

import { downloadReport } from "~/core/api";
import type { Message } from "~/core/types";
import { renderMarkdown } from "~/lib/markdown";

import { moduleLabels, type WorkspaceModule } from "./workspace-data";

const ResearchPulse = dynamic(
  () => import("./ResearchPulse").then((module) => module.ResearchPulse),
  { ssr: false },
);

const prompts: Record<WorkspaceModule, Array<{ title: string; detail: string }>> = {
  overview: [
    { title: "诊断棉花苗期黄叶", detail: "结合叶位、灌溉和分布范围逐步排查" },
    { title: "估算 100 亩棉田地膜用量", detail: "按材料类型和厚度计算" },
    { title: "解释滴灌带堵塞原因", detail: "给出现场检查顺序与处理建议" },
  ],
  film: [
    { title: "为新疆尉犁县春播棉花推荐可降解地膜，覆盖约 90 天", detail: "联合环境、性能模型和文献依据" },
    { title: "比较 8 μm、10 μm、12 μm 三种厚度", detail: "对比强度、保墒和降解风险" },
    { title: "筛选 PBAT/PLA 候选配方", detail: "批量试算并解释性能取舍" },
  ],
  field: [
    { title: "棉花苗期新叶发黄、叶脉仍绿，先排查什么？", detail: "缺素、根系与灌溉诊断" },
    { title: "覆膜后土壤温度过高怎么办？", detail: "结合作物阶段给出处置顺序" },
    { title: "地膜提前出现裂纹是否正常？", detail: "判断材料、环境与施工因素" },
  ],
  knowledge: [
    { title: "检索全生物降解地膜厚度相关标准", detail: "返回标准名称、年份和依据" },
    { title: "PBAT/PLA 共混如何影响拉伸性能？", detail: "基于文献材料回答" },
    { title: "总结地膜残留对土壤的主要影响", detail: "整理证据并标注出处" },
  ],
};

const toolLabels: Record<string, string> = {
  search_knowledge: "检索专业知识库",
  estimate_film_usage: "计算地膜用量",
  get_soil_info: "读取土壤数据",
  get_climate_info: "分析历史气候",
  predict_by_location: "运行地域性能预测",
  predict_degradation: "预测降解率",
  predict_water_vapor_rate: "预测水汽透过率",
  predict_tensile_strength: "预测拉伸强度",
  screen_film_recipes: "批量筛选候选配方",
  delegate_to_expert: "调用专项研究专家",
};

export function MessageList({
  activeModule,
  messages,
  sending,
  onPick,
  onRetry,
}: {
  activeModule: WorkspaceModule;
  messages: Message[];
  sending: boolean;
  onPick: (question: string) => void;
  onRetry: (question: string) => void;
}) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (messages.length === 0) return;
    endRef.current?.scrollIntoView({ behavior: sending ? "auto" : "smooth", block: "end" });
  }, [messages, sending]);

  const copyText = async (text: string, label = "内容") => {
    try {
      await navigator.clipboard.writeText(text);
      toast.success(`${label}已复制`);
    } catch {
      toast.error("复制失败", { description: "请检查浏览器的剪贴板权限后重试。" });
    }
  };

  const handleMarkdownClick = (event: React.MouseEvent<HTMLElement>) => {
    const target = (event.target as HTMLElement).closest<HTMLButtonElement>("[data-code]");
    if (!target?.dataset.code) return;
    void copyText(decodeURIComponent(target.dataset.code), "代码");
  };

  return (
    <div className="workspace-scroll">
      <div className="workspace-canvas">
        {messages.length === 0 ? <EmptyWorkspace activeModule={activeModule} onPick={onPick} /> : null}

        <AnimatePresence initial={false} mode="popLayout">
          {messages.map((message, index) => {
            const previousQuestion = findPreviousQuestion(messages, index);
            if (message.role === "user") {
              return (
                <m.div
                  layout="position"
                  key={message.id}
                  className="user-message"
                  initial={{ opacity: 0, y: 12, scale: 0.985 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -6 }}
                >
                  {message.content}
                </m.div>
              );
            }

            return (
              <m.article
                layout="position"
                key={message.id}
                className="assistant-response assistant-response--live"
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.38, ease: [0.16, 1, 0.3, 1] }}
              >
                <div className="response-meta">
                  <span className="response-mark" aria-hidden="true">瑞</span>
                  <span>瑞雪智研</span>
                  <span>·</span>
                  <span>{message.streaming ? "研究中" : message.error ? "需要处理" : "已完成"}</span>
                  {message.runId ? <code title="运行编号">{message.runId.slice(0, 8)}</code> : null}
                </div>

                {message.thinking ? (
                  <details className="thinking-disclosure">
                    <summary><Sparkles size={13} /><span>分析思路</span><ChevronRight size={13} /></summary>
                    <p>{message.thinking}</p>
                  </details>
                ) : null}

                {message.tools?.length ? (
                  <div className="tool-trace" aria-label="任务处理过程">
                    {message.tools.map((tool) => (
                      <m.div key={tool.name} layout className={tool.done ? "is-done" : "is-running"}>
                        <span>{tool.done ? <Check size={11} /> : null}</span>
                        <p>{toolLabels[tool.name] ?? tool.name}</p>
                        <small>{tool.done ? "完成" : "处理中"}</small>
                      </m.div>
                    ))}
                  </div>
                ) : null}

                {message.streaming && !message.content && !message.error ? (
                  <div className="assistant-pending" aria-live="polite">
                    <ResearchPulse running />
                    <p>正在组织问题、工具和依据…</p>
                  </div>
                ) : null}

                {message.error ? (
                  <div className="message-error" role="alert">
                    <CircleAlert size={17} />
                    <div><strong>本次任务没有完成</strong><p>{message.error}</p></div>
                    {previousQuestion ? (
                      <button onClick={() => onRetry(previousQuestion)}><RotateCcw size={13} />重新运行</button>
                    ) : null}
                  </div>
                ) : null}

                {message.content ? (
                  <div
                    className={`markdown-body${message.streaming ? " is-streaming" : ""}`}
                    onClick={handleMarkdownClick}
                    dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }}
                  />
                ) : null}

                {message.content && !message.streaming ? (
                  <footer className="response-actions">
                    <button onClick={() => void copyText(message.content, "回答")} aria-label="复制回答">
                      <Copy size={14} />
                    </button>
                    {previousQuestion ? (
                      <button onClick={() => onRetry(previousQuestion)} aria-label="重新运行">
                        <RotateCcw size={14} />
                      </button>
                    ) : null}
                    {/* 只有落库的运行才有报告可导 —— 报告是从 runs 表渲染的,
                        没有 runId 就没有可导出的记录(而不是"导出会失败")。 */}
                    {message.runId ? (
                      <button
                        onClick={() => {
                          // 必须接住失败:下载是 fetch 发起的,报错只进 console,
                          // 用户看到的是"点了没反应"——比明确报错更难排查。
                          downloadReport(message.runId!).catch((e: unknown) => {
                            toast.error("报告下载失败", {
                              description: e instanceof Error ? e.message : "请稍后重试。",
                            });
                          });
                        }}
                        aria-label="下载 PDF 报告"
                        title="下载 PDF 报告"
                      >
                        <FileDown size={14} />
                      </button>
                    ) : null}
                    <time>{message.runId ? `运行 ${message.runId.slice(0, 8)}` : "已保存到当前任务"}</time>
                  </footer>
                ) : null}
              </m.article>
            );
          })}
        </AnimatePresence>
        <div ref={endRef} />
      </div>
    </div>
  );
}

function EmptyWorkspace({
  activeModule,
  onPick,
}: {
  activeModule: WorkspaceModule;
  onPick: (question: string) => void;
}) {
  const icons = {
    overview: Sparkles,
    film: FlaskConical,
    field: MapPin,
    knowledge: Database,
  };
  const Icon = icons[activeModule];

  return (
    <m.article
      key={activeModule}
      className="empty-workspace empty-workspace--research"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="empty-workspace__signal">
        <span className="empty-icon"><Icon size={17} /></span>
        <ResearchPulse running={false} />
      </div>
      <p className="workspace-eyebrow">{moduleLabels[activeModule]}</p>
      <h1>{emptyTitle(activeModule)}</h1>
      <p>{emptyDescription(activeModule)}</p>
      <div className="prompt-list">
        {prompts[activeModule].map((prompt, index) => (
          <m.button
            key={prompt.title}
            onClick={() => onPick(prompt.title)}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.06 + index * 0.045 }}
            whileTap={{ scale: 0.99 }}
          >
            <span><strong>{prompt.title}</strong><small>{prompt.detail}</small></span>
            <ArrowRight size={14} />
          </m.button>
        ))}
      </div>
    </m.article>
  );
}

function findPreviousQuestion(messages: Message[], index: number): string | null {
  for (let cursor = index - 1; cursor >= 0; cursor -= 1) {
    if (messages[cursor]?.role === "user") return messages[cursor].content;
  }
  return null;
}

function emptyTitle(module: WorkspaceModule): string {
  if (module === "film") return "从场景开始一项地膜研究";
  if (module === "field") return "描述现场现象，建立排查路径";
  if (module === "knowledge") return "检索标准、文献与专业依据";
  return "今天要解决什么农业问题？";
}

function emptyDescription(module: WorkspaceModule): string {
  if (module === "film") return "提供地点、作物、覆盖周期和材料偏好，系统会调取环境、模型和知识依据。";
  if (module === "field") return "建议说明地点、作物、生育阶段、异常部位和最近的田间操作。";
  if (module === "knowledge") return "回答会标注文献标题、年份和章节；资料不足时会明确说明。";
  return "一个入口完成快速问答、计算、知识检索和多步骤研究任务。";
}
