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

# Work Report Rules

- Work Report Path: `C:\Users\fired\OneDrive\Desktop\工作汇报.docx`.
- Work Report Style: Strictly follow the academic, objective, and third-person narrative style of the previous reports.
- Text Formatting: All written text in paragraphs and tables must be plain text. Do NOT use bold, italics, custom colors, or other special text formatting in Word.
- Tables and Figures: Tables should use the standard Word "Table Grid" style with plain text. Figures should be added as inline pictures with captions (e.g., "图1 ...").
- Plan-First Workflow: Always present the proposed draft text, tables, and images in the implementation plan first. Wait for the user's explicit approval before modifying the original `.docx` file.
- 数值精度：所有数据如无特殊说明均须保留两位小数，包括表格中的数据。

# PPT Generation Rules

- 背景与视觉风格：创作 PPT 时统一使用纯白色背景（`#FFFFFF`），采用极简风格（Minimalist Light），页面干净清晰。
- 文本内容精炼：不要堆砌过多文字，聚焦核心结论与关键论点，表达凝练有力。
- 字号约束：字体要大，页面内所有文本的最小字号不得小于 20 号（`font-size >= 20`）。
- 图片生成工具：如需生成图片，直接使用 Nano Banana（即 `generate_image` 工具）。


