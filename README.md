# Service Agent 项目主页

单文件静态页 `index.html`（HTML + CSS + 内联 SVG，零外部依赖），按论文结构组织：
摘要 → 1 引言 → 2 相关工作 → 3 方法（§3.1–§3.6）→ 4 实验结果 → 5 讨论与局限 → 6 结论。

- **内容来源**：`../project_docs/01–06` 六篇项目文档；所有数字均来自官方评测或匹配对照实验，
  每张表附口径说明。图表为内联 SVG 手绘（图 1 全链路、图 2 评测架构、图 3 数据漏斗、
  图 4 RL 流水线、图 5 信用分配、图 6 多教师蒸馏、图 7 分域柱状图）。
- **本地预览**：直接用浏览器打开 `index.html`，或 `python3 -m http.server` 后访问。
- **部署**：页面发布到 `shihaofu2003/slime` 的 `gh-pages` 分支根目录。后续在仓库
  Settings → Pages 中选择 “Deploy from a branch”、`gh-pages` / `(root)` 即可启用。
- **仓库**：Hero 区 GitHub 按钮指向 `https://github.com/shihaofu2003/slime`。
