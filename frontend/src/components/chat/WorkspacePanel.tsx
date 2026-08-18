"use client";

import {
  Activity,
  Check,
  ChevronRight,
  CircleDot,
  FileText,
  Layers3,
  MessageSquareText,
  X,
} from "lucide-react";
import { AnimatePresence, m } from "motion/react";

import type { Message } from "~/core/types";


const inspectorTransition = { type: "spring" as const, stiffness: 390, damping: 36, mass: 0.85 };

export function WorkspacePanel({
  open,
  title,
  messages,
  running,
  onClose,
}: {
  open: boolean;
  title: string;
  messages: Message[];
  running: boolean;
  onClose: () => void;
}) {
  const lastAssistant = [...messages].reverse().find((message) => message.role === "assistant");
  const tools = lastAssistant?.tools ?? [];

  return (
    <AnimatePresence initial={false}>
      {open ? (
        <m.aside
          key="inspector"
          className="inspector"
          layout
          initial={{ opacity: 0, x: 28 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 28 }}
          transition={inspectorTransition}
          aria-label="任务检查器"
        >
          <header className="inspector-header">
            <strong>检查器</strong>
            <button className="icon-button" onClick={onClose} aria-label="关闭检查器">
              <X size={15} />
            </button>
          </header>

          <div className="inspector-scroll">
            <div className="inspector-status">
              <span className={running ? "is-running" : ""}>
                {running ? <CircleDot size={12} /> : <Check size={12} />}
              </span>
              <div>
                <strong>{running ? "后台任务执行中" : messages.length ? "任务已保存" : "等待任务"}</strong>
              </div>
            </div>

            <section className="inspector-section">
              <h2>当前任务</h2>
              <dl>
                <div><dt><FileText size={13} />标题</dt><dd>{title}</dd></div>
                <div><dt><MessageSquareText size={13} />消息</dt><dd>{messages.length} 条</dd></div>
                {lastAssistant?.runId ? (
                  <div><dt><Activity size={13} />运行编号</dt><dd className="inspector-run-id">{lastAssistant.runId}</dd></div>
                ) : null}
              </dl>
            </section>

            <section className="inspector-section">
              <h2>最近执行轨迹</h2>
              {tools.length ? (
                <div className="inspector-tools">
                  {tools.map((tool, index) => (
                    <m.div
                      key={tool.name}
                      initial={{ opacity: 0, x: 8 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.04 }}
                    >
                      <span className={tool.done ? "is-done" : "is-running"}>
                        {tool.done ? <Check size={10} /> : null}
                      </span>
                      <p>{readableToolName(tool.name)}</p>
                      <small>{tool.done ? "完成" : "执行中"}</small>
                    </m.div>
                  ))}
                </div>
              ) : (
                <p className="inspector-empty">发送任务后，这里会显示实际调用的检索、环境和预测工具。</p>
              )}
            </section>

            <section className="inspector-section">
              <h2>工作区说明</h2>
              <div className="inspector-guide">
                <span><strong>结果与来源在同一工作流中</strong><small>运行过程会随 SSE 事件实时更新</small></span>
                <ChevronRight size={13} />
              </div>
            </section>
          </div>
        </m.aside>
      ) : null}
    </AnimatePresence>
  );
}

function readableToolName(name: string): string {
  const labels: Record<string, string> = {
    search_knowledge: "专业知识检索",
    estimate_film_usage: "地膜用量计算",
    get_soil_info: "土壤数据查询",
    get_climate_info: "历史气候分析",
    predict_by_location: "地域性能预测",
    predict_degradation: "降解率预测",
    predict_water_vapor_rate: "水汽透过率预测",
    predict_tensile_strength: "拉伸强度预测",
    screen_film_recipes: "候选配方筛选",
    delegate_to_expert: "专项专家分析",
  };
  return labels[name] ?? name;
}
