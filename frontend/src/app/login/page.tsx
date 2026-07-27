"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { submitCredentials } from "~/core/auth";

/** 登录 / 注册页(同一页切换两种模式,少一个页面少一份重复)。 */
export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const isRegister = mode === "register";

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      await submitCredentials(mode, username.trim(), password);
      router.replace("/"); // replace:登录后按返回键不该回到登录页
    } catch (err) {
      setError(err instanceof Error ? err.message : "操作失败");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-screen items-center justify-center bg-background px-4">
      <div className="w-full max-w-[380px]">
        <div className="mb-7 text-center">
          <div className="mx-auto mb-4 h-12 w-12 rounded-[14px] bg-primary" />
          <h1 className="text-[22px] font-semibold">瑞雪地膜智能助手</h1>
          <p className="mt-1 text-[13px] text-muted-foreground">知识问答 · 性能预测 · 用量估算</p>
        </div>

        <form
          onSubmit={onSubmit}
          className="rounded-[var(--radius)] border border-border bg-card p-6 shadow-sm"
        >
          <div className="mb-4 flex gap-1 rounded-[10px] bg-background p-1">
            {(["login", "register"] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => {
                  setMode(m);
                  setError("");
                }}
                className={`flex-1 rounded-lg py-1.5 text-sm transition
                  ${mode === m ? "bg-card font-medium shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
              >
                {m === "login" ? "登录" : "注册"}
              </button>
            ))}
          </div>

          <label className="mb-1.5 block text-[13px] text-muted-foreground">用户名</label>
          <input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            placeholder="至少 3 个字符"
            className="mb-3.5 w-full rounded-[10px] border border-border bg-background px-3 py-2.5 outline-none
              focus:border-primary/50 focus:bg-card"
          />

          <label className="mb-1.5 block text-[13px] text-muted-foreground">密码</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={isRegister ? "new-password" : "current-password"}
            placeholder="至少 6 个字符"
            className="w-full rounded-[10px] border border-border bg-background px-3 py-2.5 outline-none
              focus:border-primary/50 focus:bg-card"
          />

          {error && <p className="mt-3 text-[13px] text-destructive">{error}</p>}

          <button
            type="submit"
            disabled={busy || !username.trim() || !password}
            className="mt-5 w-full rounded-[10px] bg-primary py-2.5 font-medium text-primary-foreground transition
              hover:opacity-90 disabled:cursor-not-allowed disabled:bg-muted disabled:text-muted-foreground"
          >
            {busy ? "处理中…" : isRegister ? "注册并进入" : "登录"}
          </button>
        </form>

        <p className="mt-4 text-center text-xs text-muted-foreground">
          {isRegister ? "注册即创建你的专属会话空间,对话仅自己可见" : "还没有账号?点上方切换到注册"}
        </p>
      </div>
    </div>
  );
}
