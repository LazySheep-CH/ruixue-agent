"use client";

import { LazyMotion, MotionConfig, domAnimation } from "motion/react";
import { Toaster } from "sonner";

/** 客户端体验基础设施:动效能力、系统级减弱动态偏好、全局短反馈。 */
export function AppProviders({ children }: { children: React.ReactNode }) {
  return (
    <LazyMotion features={domAnimation} strict>
      <MotionConfig
        reducedMotion="user"
        transition={{ duration: 0.24, ease: [0.16, 1, 0.3, 1] }}
      >
        {children}
        <Toaster
          position="bottom-right"
          visibleToasts={3}
          closeButton
          gap={8}
          toastOptions={{
            duration: 5000,
            classNames: {
              toast: "ruixue-toast",
              title: "ruixue-toast__title",
              description: "ruixue-toast__description",
              actionButton: "ruixue-toast__action",
              closeButton: "ruixue-toast__close",
            },
          }}
        />
      </MotionConfig>
    </LazyMotion>
  );
}
