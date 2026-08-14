---
name: 瑞雪智研
description: 像原生桌面软件一样安静、紧凑的农业研究工作台
colors:
  graphite: "#1d1d1f"
  canvas: "#ececee"
  sidebar-material: "#f2f2f4"
  paper: "#ffffff"
  secondary-surface: "#f6f6f7"
  tertiary-surface: "#ededee"
  secondary-text: "#68686d"
  separator: "rgb(60 60 67 / 15%)"
  action-blue: "#0a73e8"
  success: "#248a3d"
typography:
  title:
    fontFamily: "-apple-system, BlinkMacSystemFont, SF Pro Text, PingFang SC, Microsoft YaHei UI, Segoe UI, sans-serif"
    fontSize: "21px"
    fontWeight: 650
    lineHeight: 1.38
    letterSpacing: "-0.02em"
  body:
    fontFamily: "-apple-system, BlinkMacSystemFont, SF Pro Text, PingFang SC, Microsoft YaHei UI, Segoe UI, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.72
  label:
    fontFamily: "-apple-system, BlinkMacSystemFont, SF Pro Text, PingFang SC, Microsoft YaHei UI, Segoe UI, sans-serif"
    fontSize: "11px"
    fontWeight: 550
    lineHeight: 1.4
rounded:
  compact: "7px"
  input: "12px"
spacing:
  compact: "8px"
  standard: "16px"
  section: "28px"
components:
  button-primary:
    backgroundColor: "{colors.action-blue}"
    textColor: "{colors.paper}"
    rounded: "{rounded.compact}"
    padding: "0 11px"
    height: "32px"
  button-secondary:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.graphite}"
    rounded: "{rounded.compact}"
    padding: "0 11px"
    height: "32px"
  command-field:
    backgroundColor: "{colors.secondary-surface}"
    textColor: "{colors.graphite}"
    rounded: "{rounded.input}"
    padding: "9px"
---

# Design System: 瑞雪智研

## Overview

**Creative North Star: “原生研究台”**

界面借鉴 macOS 生产力软件的结构和克制感，而不是把农业行业符号变成装饰主题。农业身份来自真实任务、字段和资料，应用外壳保持中性，让用户把注意力放在结论与行动上。

系统属于高密度 Operate 模式。左侧管理任务，中间保持一条连续工作流，右侧检查器只承载上下文。页面不使用仪表盘卡片、英雄标题或拟人化 AI 装饰。

**Key Characteristics:**

- 中性石墨色阶和唯一蓝色行动色。
- 细分隔线、紧凑控件和原生侧栏材质。
- 结果以普通文档、定义表和记录行呈现。
- 桌面三栏，窄屏单栏，检查器与侧栏按需出现。

## Colors

界面以冷中性灰建立原生应用层级，蓝色只用于主要行动和键盘焦点，语义色只用于状态。

### Primary

- **系统行动蓝**：主要按钮、焦点轮廓和正在执行的状态。

### Neutral

- **石墨文字**：标题与重要正文。
- **纸面白**：中央工作流的固定阅读表面。
- **侧栏材质**：导航区域的半透明冷灰。
- **系统分隔线**：面板与记录行之间的一像素边界。

### Named Rules

**The One Blue Rule.** 同一视图只允许一个主要蓝色行动，其余控件保持中性。

**The Content Carries Agriculture Rule.** 农业属性由任务内容表达，不使用农业绿色、叶片或田地图案装饰界面。

## Typography

**Display Font:** 系统 UI 字体栈  
**Body Font:** 系统 UI 字体栈  

**Character:** 使用操作系统原生或相近的工作型无衬线字体。标题只比正文高一到两个层级，避免营销页面式的夸张比例。

### Hierarchy

- **Title**（650，21px，1.38）：任务结论与工作区标题。
- **Section**（650，15px）：文档分区标题。
- **Body**（400，13px，1.72）：解释与建议，正文控制在约 72ch。
- **Label**（550，10–11px）：导航、元信息与检查器字段。

### Named Rules

**The Desktop Type Rule.** 控件和标题使用固定字号，不用随视口变化的流式大标题。

## Layout

桌面由 248px 导航侧栏、单一弹性工作区和 260px 检查器组成。正文最大宽度 820px，底部输入区与正文共享中心线。1120px 以下检查器成为覆盖面板，720px 以下导航成为抽屉，中央内容保持单栏；任何断点都不得产生页面级横向滚动。

## Elevation & Depth

常驻区域依靠色阶和细线分层。阴影只用于移动侧栏、覆盖检查器和底部输入区，表示它们暂时高于内容；侧栏的模糊仅用于模拟原生窗口材质。

### Named Rules

**The Flat Work Rule.** 文档、表格和记录行保持扁平，不能包装成独立悬浮卡片。

## Shapes

常规控制使用紧凑的 7px 圆角，任务输入区使用 12px 圆角。圆形保留给用户头像、状态点和 macOS 窗口控制点；标签和内容容器不做胶囊形。

## Components

### Buttons

- **Shape:** 32px 高、7px 小圆角。
- **Primary:** 系统行动蓝底色和白色文字，每个视图最多一个。
- **Hover / Focus:** 短促色彩变化和 2px 蓝色焦点轮廓。
- **Secondary:** 白色或透明背景，用一像素边界表达可操作性。

### Cards / Containers

- **Corner Style:** 工作内容不使用卡片；覆盖层和输入区才使用圆角。
- **Background:** 中央工作区为固定纸面，导航和检查器用第二级中性表面。
- **Shadow Strategy:** 常驻内容无阴影。
- **Border:** 使用一像素系统分隔线。

### Inputs / Fields

- **Style:** 第二级表面、细边界和 12px 圆角。
- **Focus:** 蓝色边界与低透明度外环。
- **Disabled:** 使用第三级表面与次要文字。

### Navigation

导航项为 30px 高的紧凑行，图标和文字共享中性色；当前项使用中性选中底色，不使用品牌色填充。移动端导航从左侧覆盖进入并提供遮罩。

### Command Field

任务输入区固定在工作流底部并与正文同宽。输入本身不悬浮成大面积对话卡，工具与发送动作收纳在同一紧凑表面内。

## Do's and Don'ts

### Do:

- **Do** 让结果像工作文档一样连续阅读。
- **Do** 通过侧栏、工具栏与检查器保持任务上下文。
- **Do** 使用系统字体、细线和紧凑间距建立原生感。
- **Do** 在不同系统缩放和窄屏下改变结构，而不是缩小全部元素。

### Don't:

- **Don't** 使用农业绿色主题、渐变发光和装饰性行业图形。
- **Don't** 使用指标卡阵列、大标题首页或相同圆角卡片堆叠。
- **Don't** 使用 AI 头像、星光图标和拟人化状态话术。
- **Don't** 把前端演示数据表达为真实生产结论。
