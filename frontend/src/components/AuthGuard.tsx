"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { clearAuth, verifyToken } from "~/core/auth";

/**
 * 路由守卫:未登录/令牌过期 → 跳登录页。
 *
 * 注意:这是【前端体验层】的守卫,不是安全边界 —— 真正的拦截在后端
 * (每个受保护端点都 Depends(get_current_user))。前端守卫只是避免用户
 * 看到一个用不了的空界面。
 */
export function AuthGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [ok, setOk] = useState<boolean | null>(null);

  useEffect(() => {
    let alive = true;
    void verifyToken().then((status) => {
      if (!alive) return;
      if (status !== "invalid") {
        setOk(true);
      } else {
        clearAuth();
        router.replace("/login");
      }
    });
    return () => {
      alive = false;
    };
  }, [router]);

  if (ok === null) {
    return (
      <div className="auth-loading">
        <span className="auth-loading__mark" />
        正在进入研究空间…
      </div>
    );
  }
  return <>{children}</>;
}
