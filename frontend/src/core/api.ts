/** 后端 API 客户端:SSE 流式对话。
 *
 * 为什么不用 EventSource:它只支持 GET,而我们的 /chat/stream 是 POST
 * (要带 body 和 X-API-Key)。所以用 fetch + ReadableStream 手动解析 SSE。
 */

import { getToken } from "./auth";
import type { RunStatus, StreamEvent } from "./types";

/** 开发期走 Next 的 rewrites 代理到 FastAPI(见 next.config.mjs),故用相对路径。 */
const BASE = "/api";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

/** HTTP 状态 → 用户能看懂的中文提示。 */
function humanize(status: number): string {
  if (status === 401) return "登录已过期,请重新登录";
  if (status === 429) return "请求过于频繁,请稍后再试";
  if (status === 422) return "输入过长或格式不正确";
  if (status >= 500) return "服务暂时不可用,请稍后重试";
  return `请求失败(${status})`;
}

/**
 * 发起流式对话。每收到一个事件就回调 onEvent,由调用方增量渲染。
 * signal 用于取消(用户点"停止"或切走页面)。
 */
/**
 * 发起流式对话。每收到一个事件就回调 onEvent,由调用方增量渲染。
 * signal 用于取消(用户点"停止"或切走页面)。
 *
 * 注意:断开这条连接【不会】停掉后端的 agent —— 它在服务端后台跑完并落库。
 * onEvent 会先收到一个 {type:"run", run_id},调用方应存下 run_id:
 * 刷新页面后用 getRun/resumeRun 就能把结果取回来,不必重新提问、重新花钱。
 */
export async function streamChat(
  params: { threadId: string; message: string },
  onEvent: (e: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const resp = await fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${getToken()}`,
    },
    body: JSON.stringify({ thread_id: params.threadId, message: params.message }),
    signal,
  });
  await consumeSse(resp, onEvent);
}

/** 重连:补看某次运行的事件(刷新页面后恢复现场)。 */
export async function resumeRun(
  runId: string,
  onEvent: (e: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const resp = await fetch(`${BASE}/chat/runs/${runId}/stream`, {
    headers: { Authorization: `Bearer ${getToken()}` },
    signal,
  });
  await consumeSse(resp, onEvent);
}

/** 查询一次运行的状态与结果(刷新页面后先查它,已完成就直接显示答案)。 */
export async function getRun(runId: string): Promise<RunStatus> {
  const resp = await fetch(`${BASE}/chat/runs/${runId}`, {
    headers: { Authorization: `Bearer ${getToken()}` },
    cache: "no-store",
  });
  if (!resp.ok) throw new ApiError(resp.status, humanize(resp.status));
  return (await resp.json()) as RunStatus;
}

/** 下载这次运行的 PDF 报告。
 *
 * 为什么不用 `<a href>` 直接指过去:下载接口要带 Authorization 头,
 * 而浏览器发起的普通导航【带不了自定义头】—— 结果是 401。
 * 所以用 fetch 取回 blob,再用一个临时的 object URL 触发下载。
 *
 * 文件名以后端的 Content-Disposition 为准(它只含日期和 run_id,
 * 不拼用户提问 —— 见 ruixue_app/report.py::filename_for)。
 */
export async function downloadReport(runId: string): Promise<void> {
  const resp = await fetch(`${BASE}/chat/runs/${runId}/report.pdf`, {
    headers: { Authorization: `Bearer ${getToken()}` },
    cache: "no-store",
  });
  if (!resp.ok) throw new ApiError(resp.status, humanize(resp.status));

  const cd = resp.headers.get("Content-Disposition") ?? "";
  const name = /filename="([^"]+)"/.exec(cd)?.[1] ?? `ruixue-report-${runId.slice(0, 8)}.pdf`;

  const url = URL.createObjectURL(await resp.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  // 必须撤销,否则这份 blob 会一直占着内存直到刷新页面
  URL.revokeObjectURL(url);
}

export interface DatasetSummary {
  dataset_id: string;
  filename: string;
  n_rows: number;
  features: string[];
  targets: string[];
  unrecognized_columns: string[];
}

/** 上传一张实测数据表(CSV),返回数据集编号与概览。
 *
 * 注意【不要】手动设 Content-Type:交给浏览器,它会自动加上
 * multipart 的 boundary。手写 "multipart/form-data" 会漏掉 boundary,
 * 后端解析直接失败 —— 而报错信息("field required: file")指向的是
 * 完全无关的地方。
 */
export async function uploadDataset(file: File): Promise<DatasetSummary> {
  const form = new FormData();
  form.append("file", file);
  const resp = await fetch(`${BASE}/datasets`, {
    method: "POST",
    headers: { Authorization: `Bearer ${getToken()}` },
    body: form,
  });
  if (!resp.ok) {
    // 422 的 detail 是后端专门写给用户看的("没有找到实测值列……请改成……"),
    // 用通用文案替掉它等于把最有用的信息扔了。
    const detail = await resp
      .json()
      .then((b: { detail?: string }) => b.detail)
      .catch(() => undefined);
    throw new ApiError(resp.status, detail || humanize(resp.status));
  }
  return (await resp.json()) as DatasetSummary;
}

/** 上传个人资料(PDF/TXT/MD)入用户知识库。错误处理约定同 uploadDataset。 */
export async function uploadKbDoc(
  file: File,
): Promise<{ doc_id: string; filename: string; n_chunks: number }> {
  const form = new FormData();
  form.append("file", file);
  const resp = await fetch(`${BASE}/kb/docs`, {
    method: "POST",
    headers: { Authorization: `Bearer ${getToken()}` },
    body: form,
  });
  if (!resp.ok) {
    const detail = await resp
      .json()
      .then((b: { detail?: string }) => b.detail)
      .catch(() => undefined);
    throw new ApiError(resp.status, detail || humanize(resp.status));
  }
  return (await resp.json()) as { doc_id: string; filename: string; n_chunks: number };
}

export interface KbDoc {
  doc_id: string;
  filename: string;
  n_chunks: number;
  created_at: string;
}

/** 列出我的知识库资料。 */
export async function listKbDocs(): Promise<KbDoc[]> {
  const resp = await fetch(`${BASE}/kb/docs`, {
    headers: { Authorization: `Bearer ${getToken()}` },
  });
  if (!resp.ok) throw new ApiError(resp.status, humanize(resp.status));
  return ((await resp.json()) as { docs: KbDoc[] }).docs;
}

/** 删除一份资料(含全部切块与向量)。 */
export async function deleteKbDoc(docId: string): Promise<void> {
  const resp = await fetch(`${BASE}/kb/docs/${docId}`, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${getToken()}` },
  });
  if (!resp.ok) throw new ApiError(resp.status, humanize(resp.status));
}

/** 读一条 SSE 响应,逐事件回调。streamChat 与 resumeRun 共用。 */
async function consumeSse(resp: Response, onEvent: (e: StreamEvent) => void): Promise<void> {
  if (!resp.ok || !resp.body) {
    throw new ApiError(resp.status, humanize(resp.status));
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const emit = (chunk: string) => {
    const data = chunk
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n")
      .trim();
    if (!data) return;
    try {
      onEvent(JSON.parse(data) as StreamEvent);
    } catch {
      // 单帧格式错误不应使已在执行的后端任务失去前端连接。
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE 以空行分隔事件;最后一段可能不完整,留在 buffer 里等下一批
    const chunks = buffer.split(/\r?\n\r?\n/);
    buffer = chunks.pop() ?? "";
    chunks.forEach(emit);
  }

  buffer += decoder.decode();
  if (buffer.trim()) emit(buffer);
}

/** 健康检查:用于顶栏显示后端是否在线。 */
export async function checkHealth(): Promise<boolean> {
  try {
    const r = await fetch(`${BASE}/health`, { cache: "no-store" });
    return r.ok;
  } catch {
    return false;
  }
}
