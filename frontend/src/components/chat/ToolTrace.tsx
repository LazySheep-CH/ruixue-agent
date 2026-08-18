"use client";

import { Check, ChevronDown, Loader2 } from "lucide-react";
import { useState } from "react";

import type { ToolRun } from "~/core/types";

/** 工具名 → 用户能看懂的动作描述。
 *
 * 键必须和后端工具的真实注册名一致(backend/ruixue_agent/tools),
 * 对不上的会退回显示原始名 —— 不报错,只是突然冒出一个英文下划线名,
 * 所以新增工具时这张表要跟着补。
 */
const TOOL_LABELS: Record<string, string> = {
  search_knowledge: "检索专业知识库",
  estimate_film_usage: "计算地膜用量",
  get_soil_info: "读取土壤数据",
  get_climate_info: "分析历史气候",
  get_weather_forecast: "查询天气预报",
  predict_by_location: "运行地域性能预测",
  predict_degradation: "预测降解率",
  predict_water_vapor_rate: "预测水汽透过率",
  predict_tensile_strength: "预测拉伸强度",
  screen_film_recipes: "批量筛选候选配方",
  delegate_to_expert: "咨询领域专家",
  describe_dataset: "读取上传数据",
  compare_dataset_with_model: "对比实测与模型预测",
  detect_dataset_outliers: "检查数据异常值",
  check_dataset_against_standard: "核对国标符合性",
};

function labelOf(name: string): string {
  return TOOL_LABELS[name] ?? name.replace(/_/g, " ");
}

/** 折叠式工具过程:突出"它现在在干什么",明细点开再看。
 *
 * 运行中标题是当前动作(「正在检索专业知识库」),全部完成后收成
 * 「已完成 N 步」—— 全程平铺的做法更像调试面板,用户要的是进行时态。
 */
export function ToolTrace({ tools }: { tools: ToolRun[] }) {
  const [open, setOpen] = useState(false);
  if (!tools.length) return null;

  const running = tools.find((tool) => !tool.done);
  const done = tools.filter((tool) => tool.done).length;

  return (
    <div className="tool-trace" aria-label="任务处理过程">
      <button
        type="button"
        className="tool-trace-summary"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
      >
        <span className={running ? "is-running" : "is-done"}>
          {running ? <Loader2 size={12} className="spin" /> : <Check size={11} />}
        </span>
        <p>{running ? `正在${labelOf(running.name)}` : `已完成 ${done} 步操作`}</p>
        <ChevronDown size={13} className={open ? "is-open" : ""} />
      </button>

      {open ? (
        <div className="tool-trace-list">
          {tools.map((tool, i) => (
            <div key={`${tool.name}-${i}`} className={tool.done ? "is-done" : "is-running"}>
              <span>{tool.done ? <Check size={10} /> : <Loader2 size={10} className="spin" />}</span>
              <p>{labelOf(tool.name)}</p>
              <small>{tool.done ? "完成" : "处理中"}</small>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
