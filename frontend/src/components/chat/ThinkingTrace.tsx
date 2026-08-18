"use client";

import { ChevronDown, Sparkles } from "lucide-react";
import { useEffect, useRef, useState } from "react";

/** 推理过程:流式期间默认展开、实时滚动,结束后自动收起;手动操作优先。
 *
 * 长推理时用户最难受的是"黑盒等待"——看得到它在想什么,等待就可以接受。
 * 结束后自动收起是因为思考过程是中间产物,答案出来后它就该让位。
 * 用户点过开合之后,自动逻辑全部让位给手动选择(不跟用户抢控制权)。
 */
export function ThinkingTrace({ text, streaming }: { text: string; streaming: boolean }) {
  const [manual, setManual] = useState<boolean | null>(null); // null = 用户没动过
  const bodyRef = useRef<HTMLDivElement>(null);

  const open = manual ?? streaming;

  // 流式时把最新的思考滚进视野 —— 不滚的话,长推理只能看到开头几行
  useEffect(() => {
    if (open && streaming) {
      bodyRef.current?.scrollTo({ top: bodyRef.current.scrollHeight });
    }
  }, [text, open, streaming]);

  if (!text) return null;

  return (
    <div className="thinking-trace">
      <button
        type="button"
        className="thinking-trace-summary"
        onClick={() => setManual(!open)}
        aria-expanded={open}
      >
        <Sparkles size={13} className={streaming ? "pulse" : ""} />
        <span>{streaming ? "正在思考…" : "分析思路"}</span>
        <ChevronDown size={13} className={open ? "is-open" : ""} />
      </button>
      {open ? (
        <div ref={bodyRef} className="thinking-trace-body">
          <p>{text}</p>
        </div>
      ) : null}
    </div>
  );
}
