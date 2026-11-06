---
name: matplotlib-chart
description: 在 Docker 沙箱里用 matplotlib 绘制图表（折线图、柱状图、直方图、散点图、饼图等）时使用。提供图片保存路径、统一样式规范与水印规则。只要任务需要输出图片就应先读取本技能；纯数值计算不需要。
---

# matplotlib 绘图规范

## 运行环境与执行方式
- 镜像已预置中文字体（Noto Sans CJK）与统一主题样式，直接 `import matplotlib.pyplot as plt` 即可正常显示中文和负号，无需手动设置 rcParams 或字体探测。
- 支持两种执行方式：
  1. 一次性绘图任务：直接调用 `execute_python_code` 一步完成准备数据、绘图与保存；
  2. 复杂脚本或多文件依赖：用 `write_file` 写脚本到 `/work/<name>.py`，再用 `execute("python3 /work/<name>.py")` 运行，局部调整用 `edit_file`。

## 保存与图表要求
1. **保存路径**：必须保存到 `/work/outputs/<英文文件名>.png`。例如 `fig.savefig("/work/outputs/revenue_chart.png")`，随后调用 `plt.close(fig)`。
2. **自动发送**：图片保存到 `/work/outputs/` 即由系统自动发送给用户，同一文件每个版本只发一次。
3. **严禁调用**：禁止调用 `plt.show()`（沙箱为无头环境，不支持 GUI 窗口展示）。
4. **出图数量**：默认一次任务只出一张图；用户明确要求多张图时才分别保存（单次至多 4 张）。

## 统计图绘制规范
- **柱状图**：必须在柱顶标注数值（使用 `ax.bar_label(bars)`）。
- **折线图**：单系列折线图标注出最大值点和最小值点数值；多系列折线图必须包含清晰图例（`ax.legend()`）。
- **饼图**：标注百分比格式 `autopct="%1.1f%%"`，不使用 explode 避免视觉分散。
- **直方图**：自行选择合理的 bins 区间，并在汇报中给出均值、标准差等核心统计量。
- **必加水印**：每张图右下角必须添加水印（供系统视觉校验）：
  `fig.text(0.99, 0.01, "GNK48-Agent · matplotlib-chart", ha="right", va="bottom", fontsize=8, alpha=0.5)`

## 汇报要求
- 用 1-3 句话总结图表核心发现，给出最大/最小值、核心趋势或统计指标。
- 切勿在汇报中出现沙箱内部路径（如 /work/outputs/），生成的图表将由系统自动发送给用户。
