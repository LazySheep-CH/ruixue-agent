"use client";

import { ArrowRight, Leaf, LockKeyhole, ShieldCheck, UserRound } from "lucide-react";
import { AnimatePresence, m } from "motion/react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { toast } from "sonner";

import { submitCredentials } from "~/core/auth";

function BrandMark() {
  return <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>;
}

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const isRegister = mode === "register";

  const onSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      await submitCredentials(mode, username.trim(), password);
      toast.success(isRegister ? "研究空间已创建" : "登录成功");
      router.replace("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "操作失败");
    } finally {
      setBusy(false);
    }
  };

  return (
    <m.main className="login-shell" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      <m.section
        className="login-panel"
        initial={{ opacity: 0, x: -18 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      >
        <m.div className="login-card" layout>
          <div className="login-brand">
            <BrandMark />
            <div><strong>瑞雪智研</strong><span>农业材料智能体</span></div>
          </div>

          <h1>{isRegister ? "创建研究空间" : "欢迎回来"}</h1>
          <p>{isRegister ? "建立你的专属会话与研究记录。" : "登录后继续你的材料研究与分析任务。"}</p>

          <div className="login-tabs">
            <button type="button" className={mode === "login" ? "active" : ""} onClick={() => { setMode("login"); setError(""); }}>登录</button>
            <button type="button" className={mode === "register" ? "active" : ""} onClick={() => { setMode("register"); setError(""); }}>注册</button>
          </div>

          <form onSubmit={onSubmit}>
            <label className="field">
              <span>用户名</span>
              <div className="field__control">
                <UserRound size={15} />
                <input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" placeholder="请输入用户名" />
              </div>
            </label>

            <label className="field">
              <span>密码</span>
              <div className="field__control">
                <LockKeyhole size={15} />
                <input
                  type="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  autoComplete={isRegister ? "new-password" : "current-password"}
                  placeholder={isRegister ? "至少 6 个字符" : "请输入密码"}
                />
              </div>
            </label>

            <AnimatePresence initial={false}>
              {error ? (
                <m.p
                  className="login-error"
                  role="alert"
                  initial={{ opacity: 0, height: 0, y: -4 }}
                  animate={{ opacity: 1, height: "auto", y: 0 }}
                  exit={{ opacity: 0, height: 0 }}
                >
                  {error}
                </m.p>
              ) : null}
            </AnimatePresence>

            <button type="submit" disabled={busy || !username.trim() || !password} className="login-submit">
              {busy ? "处理中…" : isRegister ? "注册并进入" : "进入工作台"}
              {!busy && <ArrowRight size={15} />}
            </button>
          </form>

          <div className="login-meta">
            <ShieldCheck size={13} />
            {isRegister ? "会话数据仅在你的专属空间内可见" : "登录会话受令牌校验保护"}
          </div>
        </m.div>
      </m.section>

      <section className="login-visual">
        <div className="login-visual__content">
          <div className="login-visual__eyebrow"><Leaf size={14} />RUIXUE MATERIAL INTELLIGENCE</div>
          <h2>让每一次材料决策，<br /><span>都有数据与文献依据。</span></h2>
          <p>面向生物降解地膜研发与应用，连接专业知识、环境数据和性能模型，把复杂分析组织成清晰、可追溯的研究过程。</p>
          <div className="login-visual__stats">
            <div><strong>1500+</strong><span>专业文献与标准</span></div>
            <div><strong>3 类</strong><span>核心性能预测</span></div>
            <div><strong>可追溯</strong><span>工具过程与引用</span></div>
          </div>
        </div>
      </section>
    </m.main>
  );
}
