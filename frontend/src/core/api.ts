/** 后端 API 客户端:SSE 流式对话。
 *
 * 为什么不用 EventSource:它只支持 GET,而我们的 /chat/stream 是 POST
 * (要带 body 和 X-API-Key)。所以用 fetch + ReadableStream 手动解析 SSE。
 */

import { getToken } from "./auth";
import type { StreamEvent } from "./types";

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
