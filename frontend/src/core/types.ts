/** 领域类型:一处定义,全站共用。 */

/** agent 调用过的一个工具(用于"让用户看见它在做什么")。 */
export interface ToolRun {
  name: string;
  done: boolean;
}

/** 一条消息。thinking 是模型的推理过程(可折叠),content 是正式回答。 */
export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  /** 本轮调用过的工具及其状态 */
  tools?: ToolRun[];
  /** 流式进行中(用于显示光标/禁用发送) */
  streaming?: boolean;
  /** 出错时的提示文案(与 content 互斥) */
  error?: string;
}

/** 一次会话。id 即后端的 thread_id,决定上下文记忆的归属。 */
export interface Thread {
  id: string;
  title: string;
  createdAt: number;
}

/** 后端 SSE 推送的事件(见 ruixue_app/main.py 的 /chat/stream)。 */
export type StreamEvent =
  | { type: "thinking"; text: string }
  | { type: "answer"; text: string }
  | { type: "tool_start"; name: string }
  | { type: "tool_end"; name: string }
  /** 后端在流的开头下发运行编号。存下它,断线/刷新后可凭它重连取回结果。 */
  | { type: "run"; run_id: string }
  /** 本次运行结束(正常或失败后)—— 收到即可停止渲染"生成中"。 */
  | { type: "done" }
  | { type: "error"; text: string };

/** 一次运行的状态(GET /chat/runs/{id})。刷新页面后用它把答案取回来。 */
export interface RunStatus {
  run_id: string;
  status: "running" | "succeeded" | "failed";
  question: string;
  answer: string | null;
  error: string | null;
}
