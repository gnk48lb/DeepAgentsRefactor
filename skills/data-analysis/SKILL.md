---
name: data-analysis
description: 在 Docker 沙箱中读取并分析用户工作区（/workspace）内的数据文件（CSV、Excel、JSON、Parquet 等）时使用。提供文件路径约定、编码探测、大数据集安全读取、中间结果存储及精炼汇报规范。
---

# 数据分析规范

## 路径与读写约定
- **/workspace**：只读挂载用户工作区（如 `/workspace/data.csv`）。严禁尝试修改、重命名或写入 `/workspace`，否则会触发只读文件系统错误。
- **/work**：沙箱当前会话可读写工作目录。中间数据、清洗后的临时表请存放在 `/work`（如 `/work/cleaned.parquet`），在当前会话跨多次调用均可读取。

## 执行方式与探查先行
1. **两种执行方式**：
   - 快速一次性分析：直接使用 `execute_python_code` 运行 Python 脚本；
   - 复杂长流程：用 `write_file` 写脚本到 `/work/analysis.py`，再用 `execute("python3 /work/analysis.py")` 执行，后续小改用 `edit_file`。
2. **快速预检命令**：
   - 在写复杂 Python 脚本前，可先通过 `execute` 运行 Linux 工具快速探测数据特征：
     - `execute("file -i /workspace/data.csv")`：快速识别 MIME 类型与字符集（如 utf-8、iso-8859-1）；
     - `execute("head -n 5 /workspace/data.csv")` 或 `execute("head -c 500 /workspace/data.csv")`：预览文件头部结构与分隔符；
     - `execute("wc -l /workspace/data.csv")`：快速查看数据行数与规模。

## 分析流程规范
1. **探查先行**：在进行深度统计或变换前，先查看数据的基本概况：`df.shape`、`df.dtypes` 以及 `df.head(5)`。
2. **编码处理（CSV）**：遇到中文乱码或解码错误时，按顺序尝试编码：`encoding="utf-8-sig"` → `encoding="gbk"` → `encoding="gb18030"`。
3. **Excel 处理**：
   - 首次读取先用 `pd.read_excel('/workspace/...', sheet_name=None)` 检查全部工作表名称。
   - 读取 `.xls` 老格式文件已预装 xlrd 引擎；`.xlsx` 默认使用 openpyxl。
4. **大文件防护**：
   - 数据量未知或超大文件（数十万行以上），先使用 `nrows=1000` 采样预览，或指定 `usecols=[...]` 仅加载必要列。
   - 避免打印整个大 DataFrame，终端输出请使用 `print(df.to_string(max_rows=20))` 精简展示。

## 汇报要求
- 汇报要紧凑明确，直接给出关键指标与数字（总样本数、分布均值、中位数、异常值占比、分组对比等）。
- 不要复述全部原始数据行，无需向用户汇报沙箱内部文件路径。
