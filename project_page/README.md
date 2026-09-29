# Service Agent 项目主页

网站按定稿简历的六项工作展开：异步评测、SFT 数据治理、Banking 任务构造、异步 RL、状态进度信用分配、双教师 OPD。主线为 Raw Instruct → SFT4505 → 2026-09-25 Mix RL iter139 → OPD iter59。

## 文件与来源

- [index.html](index.html)：核心过程、示意图、实测结果与指标切换。
- [sources.html](sources.html)：每节的实验条件、来源标识、11 个完整模型的四域成绩与辅助诊断。
- [history/index.html](history/index.html)：保留 2026-09-23 旧页面，并标明历史版本。
- [data/results.json](data/results.json)、[CSV](data/results.csv)：最终实验 README 在 2026-09-29 19:58 CST 快照中的完整结果，从对应官方 summary 导出，保持原精度。
- [data/supporting-studies.json](data/supporting-studies.json)：专项实验绘图数字与对照条件。
- [scripts/build_figures.py](scripts/build_figures.py)：Matplotlib 绘制 14 张实测图表，分别导出 SVG / PNG。页面中的架构、算法例子和缓冲占用明确标为示意。

内容以 定稿简历、[项目文档导读](../project_docs/README.md) 和 [最终 OPD 实验](../output/experiments/tau2-opd-mix139-airline-20260929/README.md) 为依据。旧四教师 OPD、旧协议速度测试与当前结果分别说明，不混算。网页未沿用无法从当前数据处理记录直接复算的“10,821 条低质量样本”总数。

## 本地预览

从本目录运行：

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

打开 `http://127.0.0.1:8765/`。也可以直接用浏览器打开 `index.html`。HTML、CSS、JS 和图表均为本地资源，阅读网站无需安装依赖。

重绘图表需要 Python、Matplotlib、NumPy 和中文字体：

```bash
python3 scripts/build_figures.py --font /path/to/SourceHanSansSC-Regular.otf
```

## 发布位置

现有网站为 <https://shihaofu2003.github.io/slime/>，来自 `shihaofu2003/slime` 的 `gh-pages` 分支根目录。发布时复制本目录的页面、assets、data、scripts、history 和 README，并保留根目录 `.nojekyll`。网站源码同时保存在本仓库 `project_page/`；`gh-pages` 保存部署副本。后续更新以主分支源码为准。

## 本次验证

Chromium 检查通过：桌面 1440px、手机 390px/320px 无整页横向溢出；指标切换、图表载入、章节跳转、完整结果表和历史页返回正常；无 JavaScript 时正文及表格可读；浏览器无页面错误。所有本地 HTML 链接及锚点均有效。

预览截图：[桌面](preview/desktop.png)、[手机](preview/mobile.png)、[手机图表](preview/mobile-chart.png)。本目录包含页面与全部资源，可直接用于静态部署；zip 打包产物不纳入版本控制。
