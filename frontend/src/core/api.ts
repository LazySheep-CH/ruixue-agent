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

/** 读一条 SSE 响应,逐事件回调。streamChat 与 resumeRun 共用。 */
async function consumeSse(resp: Response, onEvent: (e: StreamEvent) => void): Promise<void> {
  if (!resp.ok || !resp.body) {
    throw new ApiError(resp.status, humanize(resp.status));
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE 以空行分隔事件;最后一段可能不完整,留在 buffer 里等下一批
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      const line = chunk.trim();
      // ": keepalive" 是心跳注释行(防代理掐掉空闲连接),不是事件
      if (!line.startsWith("data:")) continue;
      try {
        onEvent(JSON.parse(line.slice(5).trim()) as StreamEvent);
      } catch {
        // 单条解析失败不该中断整个流
      }
    }
  }
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
