/** 认证:注册 / 登录 / 令牌存取。
 *
 * 令牌存 localStorage:实现简单、刷新不丢。
 * (更安全的做法是 httpOnly Cookie —— 能防 XSS 读取令牌,但需要后端配合设置
 *  Cookie 与 CSRF 防护。当前阶段先用 localStorage,已在 README 标注为待办。)
 */

const TOKEN_KEY = "ruixue_token";
const NAME_KEY = "ruixue_username";
const BASE = "/api";

export interface AuthResult {
  access_token: string;
  username: string;
}

export function getToken(): string {
  if (typeof window === "undefined") return "";
  return localStorage.getItem(TOKEN_KEY) ?? "";
}

export function getUsername(): string {
  if (typeof window === "undefined") return "";
  return localStorage.getItem(NAME_KEY) ?? "";
}

export function saveAuth(r: AuthResult): void {
  localStorage.setItem(TOKEN_KEY, r.access_token);
  localStorage.setItem(NAME_KEY, r.username);
}

export function clearAuth(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(NAME_KEY);
}

/** 注册或登录。失败时抛出后端给的中文提示。 */
export async function submitCredentials(
  mode: "login" | "register",
  username: string,
  password: string,
): Promise<AuthResult> {
  const resp = await fetch(`${BASE}/auth/${mode}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });

  if (!resp.ok) {
    let detail = mode === "login" ? "登录失败" : "注册失败";
    try {
      const body = await resp.json();
      // FastAPI 校验错误(422)的 detail 是数组,取第一条的 msg
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail) && body.detail[0]?.msg) {
        detail = "用户名至少 3 位,密码至少 6 位";
      }
    } catch {
      /* 保底用默认提示 */
    }
    throw new Error(detail);
  }

  const data = (await resp.json()) as AuthResult;
  saveAuth(data);
  return data;
}

/** 校验本地令牌是否仍有效(过期/被改 → false)。 */
export async function verifyToken(): Promise<boolean> {
  const token = getToken();
  if (!token) return false;
  try {
    const r = await fetch(`${BASE}/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });
    return r.ok;
  } catch {
    return false; // 网络不通时不误判为"未登录",由调用方决定是否放行
  }
}
