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

# PPT Generation Rules

- 画布尺寸：生成 PPT 时统一使用标准 16:9 宽屏画布（13.333 × 7.5 英寸 / 33.867 × 19.05 cm / 960 × 540 pt）。
- 背景与视觉风格：创作 PPT 时统一使用纯白色背景（`#FFFFFF`），采用极简风格（Minimalist Light），页面干净清晰。
- 文本内容精炼：不要堆砌过多文字，聚焦核心结论与关键论点，表达凝练有力。
- 字号约束：页面内所有文本的最小字号不得小于 14 号（`font-size >= 14`）。
- 图片生成工具：如需生成图片，直接使用 Nano Banana（即 `generate_image` 工具）。
- 落盘路径约束：生成的 PPT 严禁保存在代码仓库中，必须直接落盘于桌面上对应的项目文件夹（如自研课题存放在 `C:\Users\fired\OneDrive\Desktop\文献与汇报\02_自研课题与论文\<对应课题名>\`，外部文献汇报存放在 `01_外部文献与审稿\<对应专题名>\`）。

# TECHNICAL_REPORT & TECHNICAL_DOCUMENTATION Rules

- 用语要学术化但是要足够详细，所有的技术词汇要详细解释，从概念开始逐步深入，要让外行人也能听懂。
- 模型架构讲解要格外非常详细，包括用到了哪些数据、数据如何输入输出、模型结构是怎么样的等所有概念
- 涉及到具体的操作步骤请给出命令或代码，每一个动作要给出如何分析问题、对应的思路
- 对于模型中的一些做法要给出具体的场景作为案例，说明其作用和效果。
