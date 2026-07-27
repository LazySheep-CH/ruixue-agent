/** 领域类型:一处定义,全站共用。 */

/** 一条消息。thinking 是模型的推理过程(可折叠),content 是正式回答。 */
export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  thinking?: string;
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
  | { type: "answer"; text: string };
