"use client";

import { ArrowUp, Paperclip, Square } from "lucide-react";
import { AnimatePresence, m } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { uploadDataset, uploadKbDoc } from "~/core/api";


export function Composer({
  value,
  onChange,
  onSend,
  onStop,
  sending,
}: {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
  sending: boolean;
}) {
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);

  /** 上传成功后,把 dataset_id 写进输入框而不是直接发出去。
   *
   * 为什么不自动发送:上传是一个动作,提问是另一个。替用户决定"要问什么"
   * 会发出他没打算发的请求(而且要花钱)。把编号和一句可编辑的话填进去,
   * 他改两个字就能发 —— 主动权仍在他手上。
   */
  async function handleUpload(file: File) {
    setUploading(true);
    try {
      // 一个附件入口,按扩展名分流:表格是"待分析的数据",文档是"入库的知识",
      // 两者后端管线完全不同。让用户自己选类型是把内部结构暴露给他 —— 文件名
      // 已经说明了一切。
      if (/\.(pdf|txt|md)$/i.test(file.name)) {
        const d = await uploadKbDoc(file);
        toast.success(`已加入你的知识库《${d.filename}》`, {
          description: `解析为 ${d.n_chunks} 个片段,之后的提问会自动引用你的资料。`,
        });
        inputRef.current?.focus();
        return;
      }
      const s = await uploadDataset(file);
      const missed = s.unrecognized_columns.length
        ? `,未识别的列:${s.unrecognized_columns.join("、")}`
        : "";
      toast.success(`已上传《${s.filename}》`, {
        description: `${s.n_rows} 行,实测指标 ${s.targets.join("、") || "无"}${missed}`,
      });
      onChange(
        `${value ? value + "\n" : ""}我上传了实测数据(数据集编号 ${s.dataset_id}),` +
          `请帮我分析一下,和你们模型的预测差多少。`,
      );
      inputRef.current?.focus();
    } catch (e: unknown) {
      // 422 的 detail 是后端写给用户看的操作指引("请把列名改成……"),
      // 必须原样透出 —— 换成"上传失败"等于把最有用的信息扔了。
      toast.error("上传失败", {
        description: e instanceof Error ? e.message : "请稍后重试。",
        duration: 8000,
      });
    } finally {
      setUploading(false);
    }
  }

  useEffect(() => {
    const input = inputRef.current;
    if (!input) return;
    input.style.height = "22px";
    input.style.height = `${Math.min(input.scrollHeight, 88)}px`;
  }, [value]);

  return (
    <div className="composer-dock">
      <form
        className="task-composer"
        onSubmit={(event) => {
          event.preventDefault();
          onSend();
        }}
      >
        <textarea
          ref={inputRef}
          rows={1}
          name="task-request"
          autoComplete="off"
          maxLength={2000}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              onSend();
            }
          }}
          aria-label="输入农业问题或任务"
          placeholder="输入你的问题,或上传实测数据…"
        />
        <div className="composer-toolbar">
          {/* 隐藏的原生 input:样式没法直接改,所以用按钮触发它。
              accept 只是给文件选择器的提示,**不是校验** —— 真校验在后端
              (改个扩展名就能绕过前端的 accept)。 */}
          <input
            ref={fileRef}
            type="file"
            accept=".csv,.pdf,.txt,.md,text/csv,application/pdf,text/plain,text/markdown"
            hidden
            onChange={(event) => {
              const f = event.target.files?.[0];
              // 先清空 value 再处理:不清的话连续选同一个文件不会触发 change,
              // 用户会觉得"点了没反应"。
              event.target.value = "";
              if (f) void handleUpload(f);
            }}
          />
          <button
            type="button"
            className="composer-icon"
            disabled={uploading || sending}
            onClick={() => fileRef.current?.click()}
            aria-label="上传实测数据(CSV)或个人资料(PDF/TXT/MD)"
            title="上传实测数据(CSV)或个人资料(PDF/TXT/MD)"
          >
            <Paperclip size={16} />
          </button>
          <span className="composer-connection"><i />实时智能体</span>
          <span className="composer-hint">Enter 发送 · Shift + Enter 换行</span>
          {value.length > 1800 ? <span className="composer-count">{value.length}/2000</span> : null}
          <AnimatePresence mode="wait" initial={false}>
            {sending ? (
              <m.button
                key="stop"
                type="button"
                className="send-button"
                onClick={onStop}
                aria-label="停止接收（后台任务会继续）"
                initial={{ scale: 0.82, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.82, opacity: 0 }}
                whileTap={{ scale: 0.9 }}
              >
                <Square size={10} fill="currentColor" />
              </m.button>
            ) : (
              <m.button
                key="send"
                type="submit"
                className="send-button"
                disabled={!value.trim()}
                aria-label="发送"
                initial={{ scale: 0.82, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.82, opacity: 0 }}
                whileTap={{ scale: 0.9 }}
              >
                <ArrowUp size={16} />
              </m.button>
            )}
          </AnimatePresence>
        </div>
      </form>
    </div>
  );
}
