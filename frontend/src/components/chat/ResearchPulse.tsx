"use client";

import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { useRef } from "react";

/**
 * 瑞雪的标志性任务动效:把“问题 → 工具/资料 → 结论”的工作过程画成一条研究轨迹。
 * 它只在任务运行时循环，并在系统要求减少动态效果时退化为静态流程图。
 */
export function ResearchPulse({ running }: { running: boolean }) {
  const root = useRef<HTMLDivElement>(null);

  useGSAP(
    () => {
      const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      gsap.set("[data-pulse-node]", { scale: 1, opacity: 1, transformOrigin: "center" });
      gsap.set("[data-pulse-signal]", { x: 0, opacity: running ? 1 : 0 });
      gsap.set("[data-pulse-path]", { strokeDashoffset: running ? 44 : 0 });

      if (!running || reduceMotion) return;

      const timeline = gsap.timeline({ repeat: -1, repeatDelay: 0.18 });
      timeline
        .to("[data-pulse-path]", {
          strokeDashoffset: 0,
          duration: 0.7,
          ease: "power2.out",
        })
        .to(
          "[data-pulse-signal]",
          { x: 154, duration: 1.05, ease: "power2.inOut" },
          0,
        )
        .to(
          "[data-pulse-node]",
          {
            scale: 1.18,
            duration: 0.22,
            stagger: 0.28,
            yoyo: true,
            repeat: 1,
            ease: "power2.out",
          },
          0.05,
        )
        .to("[data-pulse-signal]", { opacity: 0, duration: 0.16 }, ">-0.12");

      const onVisibility = () => {
        if (document.hidden) timeline.pause();
        else timeline.resume();
      };
      document.addEventListener("visibilitychange", onVisibility);
      return () => document.removeEventListener("visibilitychange", onVisibility);
    },
    { scope: root, dependencies: [running], revertOnUpdate: true },
  );

  return (
    <div ref={root} className={`research-pulse${running ? " is-running" : ""}`} aria-hidden="true">
      <svg viewBox="0 0 190 42" role="presentation">
        <path className="research-pulse__rail" d="M18 21H172" />
        <path data-pulse-path className="research-pulse__path" d="M18 21H172" />
        {[18, 95, 172].map((cx) => (
          <circle key={cx} data-pulse-node className="research-pulse__node" cx={cx} cy="21" r="5" />
        ))}
        <circle data-pulse-signal className="research-pulse__signal" cx="18" cy="21" r="3" />
      </svg>
      <div className="research-pulse__labels">
        <span>问题</span><span>工具与资料</span><span>结论</span>
      </div>
    </div>
  );
}
