# 瑞雪前端

Next.js 15 (App Router) + TypeScript + Tailwind + zustand。对标参考架构的前端组织方式。

## 跑起来

```bash
# 1) 后端(项目根目录)
uv run uvicorn ruixue_app.main:app --reload      # :8000

# 2) 前端(本目录)
npm install
npm run dev                                       # :3000
```

浏览器开 <http://127.0.0.1:3000>,右上角"设置 API Key"填 `demo-key-alice`。

开发期 `/api/*` 由 Next 的 rewrites 代理到 `127.0.0.1:8000`(见 `next.config.mjs`),
浏览器看来是同源 —— **不需要后端配 CORS**。换后端地址设 `NEXT_PUBLIC_API_BASE`。

## 结构

```
src/
├── app/              路由与页面(App Router)
│   ├── layout.tsx    根布局
│   └── page.tsx      聊天页(把各组件串起来)
├── components/chat/  UI 组件
│   ├── Sidebar.tsx   左侧图标条 + 会话面板
│   ├── TopBar.tsx    顶栏(后端在线状态 + API Key)
│   ├── MessageList.tsx  消息列表 + 空态建议
│   └── Composer.tsx  输入框(自增高/回车发送/停止)
├── core/             业务核心(与 UI 无关,可单测)
│   ├── types.ts      领域类型
│   ├── api.ts        SSE 流式客户端 + 错误中文化
│   └── store.ts      zustand 状态(会话/消息/Key,持久化 localStorage)
├── lib/markdown.ts   极简 markdown → HTML(先转义再套标签,防 XSS)
└── styles/globals.css 设计变量 + 全局样式
```

分层原则和后端一致:**`core/` 不认识 React**,组件只负责渲染 —— 换 UI 框架时核心逻辑不用动。

## 命令

```bash
npm run dev        # 开发(热更新)
npm run build      # 生产构建
npm run typecheck  # 类型检查
npm run check      # lint + 类型检查(提交前跑)
```

## 设计

配色/圆角等设计变量集中在 `src/styles/globals.css` 的 `:root`,
`tailwind.config.ts` 引用它们 —— **改配色只改一处**。

## 已知待办

- 会话历史存在浏览器 localStorage;后端已有 `thread_id` 持久化,尚未接"加载历史对话"接口
- 未做暗色模式与移动端深度适配(仅基本响应式)
