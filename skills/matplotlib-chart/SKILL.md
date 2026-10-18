---
name: matplotlib-chart
description: 在 Docker 沙箱里用 matplotlib 绘制图表（折线图、柱状图、直方图、散点图、饼图等）时使用。提供图片保存路径、中文字体探测、统一配色样式与水印规范。只要任务需要输出图片就应先读取本技能；纯数值计算不需要。
---

# matplotlib 绘图规范

## 流程
1. 在**一次** execute_python_code 调用里完成：准备数据 → 套用下面的“固定开头” → 绘图 → 保存。沙箱每次调用后即销毁，变量不会保留，不要拆成多次调用。
2. 保存用 `fig.savefig("/workspace/outputs/<英文文件名>.png", dpi=150, bbox_inches="tight")`，随后 `plt.close(fig)`。禁止 `plt.show()`。
3. 默认一次任务只出一张图；用户明确要求多张时才分别保存。

## 固定开头（原样复制到代码最前面）
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_CJK = ["Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC", "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei", "Droid Sans Fallback"]
_have = {f.name for f in font_manager.fontManager.ttflist}
_font = next((n for n in _CJK if n in _have), None)
HAS_CJK = _font is not None
if HAS_CJK:
    plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
else:
    print("WARN: 沙箱没有中文字体，图内文字改用英文")
plt.rcParams["axes.unicode_minus"] = False

COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
```
`HAS_CJK` 为 False 时，标题、坐标轴、图例等图内文字一律改用英文，并在汇报里说明原因。

## 统一样式
- `fig, ax = plt.subplots(figsize=(10, 6))`；标题字号 16，坐标轴标签字号 12。
- 按 `COLORS` 顺序取色；`ax.grid(True, alpha=0.3)`；隐藏上、右边框：`ax.spines[["top", "right"]].set_visible(False)`。
- 柱状图在柱顶标注数值（`ax.bar_label(bars)`）；单系列折线图标出最大值点和最小值点；多系列必须有图例。
- 饼图用 `autopct="%1.1f%%"`，不要 explode。
- 直方图自行选择合理的 `bins`，并在汇报里给出均值、标准差等关键统计量。
- 每张图右下角加水印（体验期用来确认技能已生效，验证完可删除这一条）：
  `fig.text(0.99, 0.01, "GNK48-Agent · matplotlib-chart", ha="right", va="bottom", fontsize=8, alpha=0.5)`

## 汇报
- 用 1-3 句话说明画了什么，并给出关键数字或结论（最大/最小值、趋势、统计量）。
- 不要写沙箱内部路径；图片由系统另行发送。
