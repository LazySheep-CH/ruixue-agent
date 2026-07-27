"use client";

import { useEffect, useRef } from "react";

/** 输入区:自增高、回车发送、流式中可停止。 */
export function Composer({
  value,
  onChange,
  onSend,
  onStop,
  sending,
}: {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  onStop: () => void;
  sending: boolean;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);

  // 随内容自增高(上限 140px 后内部滚动)
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 140) + "px";
  }, [value]);

  return (
    <div className="bg-gradient-to-b from-transparent to-wash px-6 pb-5 pt-3">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          onSend();
        }}
        className="mx-auto flex max-w-reading items-end gap-2 rounded-card border border-line bg-surface
          px-3.5 py-2.5 transition focus-within:border-brand/50
          focus-within:shadow-[0_0_0_3px_rgba(217,119,87,.10)]"
      >
        <textarea
          ref={ref}
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              onSend();
            }
          }}
          placeholder="问问地膜的降解、保墒、力学或用量……"
          className="max-h-[140px] flex-1 resize-none bg-transparent px-0.5 py-1 outline-none"
        />
        {sending ? (
          <button
            type="button"
            onClick={onStop}
            title="停止生成"
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-brand text-[11px] text-white"
          >
            ■
          </button>
        ) : (
          <button
            type="submit"
            disabled={!value.trim()}
            title="发送"
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-brand
              text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:bg-line
              disabled:text-muted"
          >
            ↑
          </button>
        )}
      </form>
      <p className="mx-auto mt-2 max-w-reading text-center text-xs text-muted">
        回车发送 · Shift+回车换行 · 结果由模型生成,仅供参考
      </p>
    </div>
  );
}
