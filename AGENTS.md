# Agent Workspace Rules

- Unless explicitly requested otherwise by the user, all Python scripts/commands must be executed using the `nc` Conda environment (`C:\Users\fired\anaconda3\envs\nc\python.exe`).
- Put all project-related literature/reference materials in `D:\HydroSynth\ref` by default, including Zotero exports, BibTeX/RIS files, PDFs, notes, literature reviews, and other non-code reference artifacts. The `ref` folder is intentionally not tracked by git to avoid uploading these materials to GitHub.

# Data & Workspace Architecture Rules (数据与工作空间管理规范)

- **代码仓库纯净原则**：代码仓库（`d:\HydroSynth`）必须保持纯净，仅存放代码、配置文件和文档。严禁在子项目文件夹内落盘生成原始数据、中间缓存（`cache/`）、模型权重（`.pt`/`.pth`）、日志（`logs/`）或实验结果（`results/`）。
- **统一路径调度规范 (`utils.paths`)**：所有子项目（如 `HydroGraph_S2S`、`FNO`、`U_Net_3D` 等）必须通过 `from utils.paths import SubprojectPaths` 动态解析路径：
  - 缓存目录：统一使用 `paths.cache_dir`（自动派生至 `<HYDRO_WORKSPACE>/cache/<子项目名>/`）。
  - 实验产物：统一使用 `paths.get_exp_dir("<exp_name>")`（自动派生至 `<HYDRO_WORKSPACE>/results/<子项目名>/<exp_name>/`，并自动创建 `checkpoints/`、`figures/`、`logs/`）。
  - 共享原始数据：统一通过 `paths.get_raw_data("<filename>")` 或 `.env` 环境变量引用，禁止修改原始数据。
- **环境配置解耦**：所有数据盘根路径统一在根目录 `.env` 中配置（如 `HYDRO_WORKSPACE`、`HYDRO_DATA_DIR`），严禁在 Python 源代码中硬编码本地绝对路径。

# Desktop Research & Document Asset Rules (桌面科研资产与文档流转规范)

- **核心资产库根路径**：所有学术论文、PPT 汇报、外审材料及调研报告的唯一桌面落盘中枢为 `C:\Users\fired\OneDrive\Desktop\文献与汇报\`。
- **严密二级拓扑架构**：
  - `01_外部文献与审稿/`：收纳所有精读的外部顶刊文献、课程研讨课件、阅读笔记，以及作为审稿人（Reviewer）进行同行评议的送审手稿与审稿意见。
  - `02_自研课题与论文/`：收纳课题组自主研发的所有课题资产（手稿初稿与演进、答辩与学术交流 PPT、基金与开题申报书、论文专用图表等）。
- **根目录纯净度红线**：`文献与汇报` 根目录及 `01`、`02` 下严禁散落孤立文件，所有文档必须严格归入具体课题/专题子目录中。未分类草案统归 `02_自研课题与论文/临时草稿与备份/`。

# Academic Paper & PPT Storage Rules (论文与演示文稿落盘禁忌)

- **代码仓库绝对纯净**：严禁在 `d:\HydroSynth` 或任何子项目代码目录下生成或保存学术论文手稿（`.docx`/`.tex`/`.pdf`）及汇报演示文稿（`.pptx`）。代码仓库仅保留纯净代码、配置文件与工程级纯技术文档。
- **自动对齐课题路径**：
  - 生成自研论文手稿、章节初稿、修改稿：直接落盘于 `C:\Users\fired\OneDrive\Desktop\文献与汇报\02_自研课题与论文/<对应自研课题>/`（涉及多版本演进的，归入其子目录 `手稿演进历程/`）。
  - 生成汇报与答辩演示文稿：自研成果汇报落盘于 `02_自研课题与论文/<对应自研课题>/`；外部文献分享或课程汇报落盘于 `01_外部文献与审稿/<对应专题>/`。

# Figure Dual-Preservation & Synchronization Rules (论文配图双重归档规范)

- **实验产物可复现性归档**：所有代码运行、模型评估输出的原生图像与图表，必须按 `utils.paths` 规范保存在实验产物路径 `<HYDRO_WORKSPACE>/results/<子项目名>/<exp_name>/figures/` 中，以保证实验记录完整与可复现。
- **论文配图自动桌面同步**：凡是选定用于学术论文、工作汇报或答辩 PPT 的配图，在生成或确定使用时，**必须同时复制一份**至桌面对应的自研课题文件夹中保留备份：
  - 目标路径：`C:\Users\fired\OneDrive\Desktop\文献与汇报\02_自研课题与论文/<对应自研课题>/figures/`（若配图数量较少或为单张核心流程图/架构图，亦可直接存放在该课题根目录下）。
- **图表格式与清晰度**：论文配图必须采用高清无损格式，优先保存矢量格式（`.svg`、`.pdf`）或高分辨率光栅图（`.png`，$\ge 300\text{ DPI}$），图名需反映学术内涵（如 `spatial_taylor_diagram.png` 或 `空间相关系数对比图.png`）。

# Naming Conventions & Organization Discipline (命名规范与条理化纪律)

- **20 字符长度红线（铁律）**：所有新建的文件夹名称以及非代码文件（Word 文档、PPT、PDF、说明文档）的 BaseName（不包含扩展名），**严格限制在 20 个汉字/字符以内**。
- **语义化命名称谓**：必须以简明能代表模型名称、核心方法或主要贡献的方式命名，严禁使用“新建文稿”、“草稿”、“ppt1”或自动时间戳长哈希。
- **多版本演进规范**：手稿修改过程必须具有可追溯性，统一采用 `课题名演进N_阶段说明.docx`（如 `ReMAP演进1_Stacking初稿.docx`）的标准命名结构。

# Work Report Rules

- Work Report Path: `C:\Users\fired\OneDrive\Desktop\工作汇报.docx`.
- Work Report Style: Strictly follow the academic, objective, and third-person narrative style of the previous reports.
- Text Formatting: All written text in paragraphs and tables must be plain text. Do NOT use bold, italics, custom colors, or other special text formatting in Word.
- Tables and Figures: Tables should use the standard Word "Table Grid" style with plain text. Figures should be added as inline pictures with captions (e.g., "图1 ...").
- Plan-First Workflow: Always present the proposed draft text, tables, and images in the implementation plan first. Wait for the user's explicit approval before modifying the original `.docx` file.
- 数值精度：所有数据如无特殊说明均须保留两位小数，包括表格中的数据。

# PPT Generation Rules (演示文稿视觉与结构化设计规范)

- **画布尺寸**：生成 PPT 时统一使用标准 16:9 宽屏画布（13.333 × 7.5 英寸 / 33.867 × 19.05 cm / 960 × 540 pt / 1280 × 720 px）。
- **底色与基调**：纯白底色（`#FFFFFF`），主板采用现代扁平卡片化（Card-based Modular UI）设计，严禁无结构的整页大文本框。
- **排版推荐工作流**：优先推荐使用 `ppt-master` 技能或基于 1280×720 SVG 的组件化渲染流程编译为 PPTX，确保容器阴影、圆角、标签对齐的像素级工业质感。
- **落盘路径约束**：严禁保存在代码仓库中，必须直接落盘于桌面上对应的项目文件夹（自研成果存入 `02_自研课题与论文/<对应课题名>/`，外部文献/课程汇报存入 `01_外部文献与审稿/<对应专题名>/`）。
- **单页文字预算红线（铁律）**：
  - 中文汇报：单页正文字数严格控制在 **250~350 汉字** 以内（不含图表标注与页脚）。
  - 英文汇报：单页正文字数严格控制在 **70~110 单词** 以内。
  - 单条 Bullet 长度红线：单条要点严禁超过 2 行（中文 ≤ 25 字，英文 ≤ 14 词）。
- **动宾/结论先导结构**：严禁出现整段自然语言解释、大段背景叙述或教科书式长句。必须采用 `【核心结论/关键技术】+ 简短论据/数据支撑` 的凝练格式，关键参数与对比数据加粗高亮。
- **信息图表化优先原则（拒绝纯文字铺开）**：任何单页幻灯片必须拥有明确的视觉重心（Visual Anchor）：
  - 凡对比必作表/图：多模型性能对比、消融实验一律使用“高对比度表格”或“原生矢量柱状图/折线图”呈现，禁止以列表文字念数据。
  - 凡机制必作流程轴/公式卡片：算法流程、海-云-气物理演变一律使用“带箭头流转卡片组”或“独立公式高亮胶囊”表达。
  - 核心指标必作 KPI 看板：突出关键成果时，必须提取大字号（28~36pt）彩色加粗数字徽章（如 `+107.80% ACC`、`p=99.60%`）。
- **页面四层组件系统**：
  - 顶部导航（Header）：左侧 6px 分类色标竖条 + 行为主标题（22~26pt 粗体陈述论点）+ 1 行结论性副标题（14~16pt 概括机制）+ 右上角分类状态胶囊（如 `P03 · 模式痛点`）。
  - 内容容器（Body Grid）：采用 2 列或 3 列卡片化网格，每个卡片底色采用极浅对比色（如 `#F8FAFC`、`#EFF6FF`、`#F0FDFA`，边框 `#CBD5E1`），卡片附带 4~8px 彩色重边条（痛点缺陷用红 `#DC2626`，机理自研用深蓝 `#0F2C59`，突破前沿用绿 `#0D9488` 或橙 `#D97706`）。
  - 底部启示栏（Footer Bar）：底部细分隔线 + 1 行加粗的「核心启示/Takeaway」（14~15pt 提取要旨）+ 汇报人/专题元数据。
  - 字号约束：页面内正文与说明文本的最小字号不得小于 14 号（`font-size >= 14`）。
- **多渠道图表资产自动装配流水线**：严禁在无图时退化为纯文本，按以下优先级装配：
  1. 第一优先级（课题原生图库）：主动检索桌面 `02_自研课题与论文/<课题>/figures/` 中已沉淀的高清实验评估图、架构图、Taylor 统计图。
  2. 第二优先级（手稿嵌入图提取）：若桌面存在相关课题 `.docx` 手稿，自动解压其 `word/media/` 提取高分辨率论文图。
  3. 第三优先级（原生代码矢量图）：性能时效对比、参数消融、技术路线图，直接在代码中生成原生 SVG/矢量形状与表格。
  4. 第四优先级（Nano Banana 概念生图）：仅在缺乏物理机制示意图或需要宏观科学概念封面图时，调用 `generate_image` 绘制简约学术插图。

# TECHNICAL_REPORT & TECHNICAL_DOCUMENTATION Rules

- 用语要学术化但是要足够详细，所有的技术词汇要详细解释，从概念开始逐步深入，要让外行人也能听懂。
- 模型架构讲解要格外非常详细，包括用到了哪些数据、数据如何输入输出、模型结构是怎么样的等所有概念
- 涉及到具体的操作步骤请给出命令或代码，每一个动作要给出如何分析问题、对应的思路
- 对于模型中的一些做法要给出具体的场景作为案例，说明其作用和效果。
