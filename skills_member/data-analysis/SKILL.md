---
name: data-analysis
description: 在云端沙箱里分析数据（CSV、Excel、JSON 等）时使用。数据来自任务描述里给出的内容、代码生成的数据，以及（仅限沙箱可以联网时）下载到 /work 的公开文件。提供路径约定、数据落盘、编码探测、大数据集安全读取、中间结果存储及精炼汇报规范。
---

# 数据分析规范

## 数据从哪里来
- 沙箱里没有用户电脑上的任何文件：要分析的数据只有三种来源——任务描述里直接给出的数据、代码生成的数据，以及（仅当 CodeAgent 的运行环境说明写明沙箱可以联网时）从公开网址下载到 `/work` 的文件。
- 任务描述里给出的数据（几行表格、一串数字、一段 JSON），先用代码原样写成 `/work/data.csv` 或直接构造 DataFrame，再分析；数据缺失或明显不全时如实汇报缺什么，不要编造数据。
- 仅当运行环境说明写明沙箱可以联网时，才可以用 `requests` / `httpx` / `curl` 下载公开数据，下载后先看文件大小再读取；说明写明不能联网时，不要尝试下载或 `pip install`。

## 路径与读写约定
- **/work**：沙箱当前会话的可读写工作目录。原始数据、清洗后的临时表、中间结果都存放在这里（如 `/work/cleaned.parquet`），同一会话内跨多次调用可读取；沙箱闲置一段时间被回收后，里面的文件会随之丢失，重要的中间结果请写进汇报。
- **/work/outputs/**：专用产物导出目录，只放要发给用户的图片。

## 执行方式与探查先行
1. **两种执行方式**：
   - 快速一次性分析：直接使用 `execute_python_code` 运行 Python 脚本；
   - 复杂长流程：用 `write_file` 写脚本到 `/work/analysis.py`，再用 `execute("python3 /work/analysis.py")` 执行，后续小改用 `edit_file`。
2. **快速预检命令**：数据已经落到 `/work` 之后，可以先通过 `execute` 运行 Linux 工具探测数据特征：
   - `execute("file -i /work/data.csv")`：识别 MIME 类型与字符集（如 utf-8、iso-8859-1）；
   - `execute("head -n 5 /work/data.csv")`：预览文件头部结构与分隔符；
   - `execute("wc -l /work/data.csv")`：查看数据行数与规模。

## 分析流程规范
1. **探查先行**：在进行深度统计或变换前，先查看数据的基本概况：`df.shape`、`df.dtypes` 以及 `df.head(5)`。
2. **编码处理（CSV）**：读取下载来的文件遇到中文乱码或解码错误时，按顺序尝试编码：`encoding="utf-8-sig"` → `encoding="gbk"` → `encoding="gb18030"`。
3. **Excel 处理**：首次读取先用 `pd.read_excel('/work/...', sheet_name=None)` 检查全部工作表名称；`.xls` 老格式已预装 xlrd 引擎，`.xlsx` 默认使用 openpyxl。
4. **大文件防护**：数据量未知或超大文件（数十万行以上），先使用 `nrows=1000` 采样预览，或指定 `usecols=[...]` 仅加载必要列；避免打印整个大 DataFrame，请使用 `print(df.to_string(max_rows=20))` 精简展示。

## 汇报要求
- 汇报要紧凑明确，直接给出关键指标与数字（总样本数、分布均值、中位数、异常值占比、分组对比等）。
- 不要复述全部原始数据行，无需向用户汇报沙箱内部文件路径。
