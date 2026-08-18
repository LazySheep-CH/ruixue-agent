"use client";

import { BookOpen, Plus, Trash2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { deleteKbDoc, listKbDocs, uploadKbDoc, type KbDoc } from "~/core/api";

/** 输入框回形针上传成功后广播这个事件,这里收到就刷新列表。
 * 两个入口(侧栏、输入框)共用一份数据,又不想为此上全局 store ——
 * 一个自定义事件就够了。 */
export const KB_CHANGED_EVENT = "ruixue:kb-changed";

/** 侧栏的"我的资料":用户自有知识库的可见入口。
 *
 * 之前只有输入框的回形针能传,传完资料就"消失"了 —— 用户不知道
 * 存在哪、有哪些、怎么删。数据库在后端是用户独有的,但看不见的
 * 隔离等于不存在,这个面板就是把它变成看得见摸得着的东西。
 */
export function KbDocs() {
  const [docs, setDocs] = useState<KbDoc[]>([]);
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(() => {
    listKbDocs()
      .then(setDocs)
      .catch(() => {
        // 列表拉不下来不弹错:侧栏是常驻区域,一次网络抖动不该弹窗打扰
      });
  }, []);

  useEffect(() => {
    refresh();
    window.addEventListener(KB_CHANGED_EVENT, refresh);
    return () => window.removeEventListener(KB_CHANGED_EVENT, refresh);
  }, [refresh]);

  async function handleUpload(file: File) {
    setBusy(true);
    try {
      const d = await uploadKbDoc(file);
      toast.success(`已加入你的知识库《${d.filename}》`, {
        description: `解析为 ${d.n_chunks} 个片段,之后的提问会自动引用你的资料。`,
      });
      refresh();
    } catch (e: unknown) {
      toast.error("上传失败", {
        description: e instanceof Error ? e.message : "请稍后重试。",
        duration: 8000,
      });
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(doc: KbDoc) {
    try {
      await deleteKbDoc(doc.doc_id);
      toast.success(`已删除《${doc.filename}》`);
      refresh();
    } catch (e: unknown) {
      toast.error("删除失败", {
        description: e instanceof Error ? e.message : "请稍后重试。",
      });
    }
  }

  return (
    <div className="kb-docs">
      <div className="sidebar-section-title kb-docs__title">
        <span>我的资料</span>
        <button
          type="button"
          disabled={busy}
          onClick={() => fileRef.current?.click()}
          aria-label="上传资料(PDF/TXT/MD)"
          title="上传资料(PDF/TXT/MD)"
        >
          <Plus size={13} />
        </button>
      </div>
      <input
        ref={fileRef}
        type="file"
        accept=".pdf,.txt,.md,application/pdf,text/plain,text/markdown"
        hidden
        onChange={(event) => {
          const f = event.target.files?.[0];
          event.target.value = "";
          if (f) void handleUpload(f);
        }}
      />
      <div className="record-list kb-docs__list">
        {docs.map((doc) => (
          <div key={doc.doc_id} className="record-row">
            <button type="button" title={`${doc.filename}(${doc.n_chunks} 个片段)`}>
              <BookOpen size={13} />
              <span>{doc.filename}</span>
              <small>{doc.n_chunks} 片段</small>
            </button>
            <button
              className="record-row__delete"
              onClick={() => void handleDelete(doc)}
              aria-label={`删除${doc.filename}`}
            >
              <Trash2 size={13} />
            </button>
          </div>
        ))}
        {docs.length === 0 ? (
          <p className="sidebar-empty">上传 PDF / TXT / MD，问答会自动引用你的资料</p>
        ) : null}
      </div>
    </div>
  );
}
