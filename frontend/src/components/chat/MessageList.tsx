"use client";

import {
  ArrowRight,
  ChevronRight,
  CircleAlert,
  Copy,
  FileDown,
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

import { ThinkingTrace } from "./ThinkingTrace";
import { ToolTrace } from "./ToolTrace";

const ResearchPulse = dynamic(
  () => import("./ResearchPulse").then((module) => module.ResearchPulse),
  { ssr: false },
);

const prompts: Array<{ title: string; detail: string }> = [
  { title: "为新疆尉犁县春播棉花推荐可降解地膜，覆盖约 150 天", detail: "联合环境、性能模型和文献依据" },
  { title: "估算 100 亩棉田地膜用量", detail: "按材料类型和厚度计算" },
  { title: "地膜提前出现裂纹是否正常？", detail: "判断材料、环境与施工因素" },
  { title: "检索全生物降解地膜厚度相关标准", detail: "返回标准名称、年份和依据" },
];

export function MessageList({
  messages,
  sending,
  onPick,
  onRetry,
}: {
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
        {messages.length === 0 ? <EmptyWorkspace onPick={onPick} /> : null}

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
                  <ThinkingTrace text={message.thinking} streaming={!!message.streaming && !message.content} />
                ) : null}

                {message.tools?.length ? <ToolTrace tools={message.tools} /> : null}

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
  onPick,
}: {
  onPick: (question: string) => void;
}) {
  return (
    <m.article
      key="empty" 
      className="empty-workspace empty-workspace--research"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="empty-workspace__signal">
        <span className="empty-icon"><Sparkles size={17} /></span>
        <ResearchPulse running={false} />
      </div>
      <p className="workspace-eyebrow">研究工作台</p>
      <h1>今天要解决什么问题？</h1>
      <p>一个入口完成选型、预测、诊断、知识检索与数据分析 —— 系统自行决定调用哪些工具。</p>
      <div className="prompt-list">
        {prompts.map((prompt, index) => (
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


