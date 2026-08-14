import {
  BookOpen,
  CircleHelp,
  FlaskConical,
  Map,
  type LucideIcon,
} from "lucide-react";

export type WorkspaceModule = "overview" | "film" | "field" | "knowledge";

export type WorkspaceModuleDefinition = {
  id: WorkspaceModule;
  label: string;
  description: string;
  icon: LucideIcon;
};

export type TaskEntry = {
  id: string;
  request: string;
  result?: string;
};

export const workspaceModules: WorkspaceModuleDefinition[] = [
  {
    id: "overview",
    label: "农业问答",
    description: "种植、农资与田间问题",
    icon: CircleHelp,
  },
  {
    id: "film",
    label: "地膜选型",
    description: "材料推荐与性能分析",
    icon: FlaskConical,
  },
  {
    id: "field",
    label: "田间诊断",
    description: "症状判断与处置建议",
    icon: Map,
  },
  {
    id: "knowledge",
    label: "研究资料",
    description: "标准、文献与项目记录",
    icon: BookOpen,
  },
];

export const moduleLabels: Record<WorkspaceModule, string> = Object.fromEntries(
  workspaceModules.map((module) => [module.id, module.label]),
) as Record<WorkspaceModule, string>;
