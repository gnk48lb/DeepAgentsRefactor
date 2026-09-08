import ctypes
import io
import base64
from mcp.server.fastmcp import FastMCP
import pyautogui
import pyperclip
from PIL import Image, ImageDraw, ImageFont

# 【核心防坑】强制声明 Per-Monitor DPI Aware v2，无视 175% 缩放
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception as e:
    print(f"DPI 设置失败 (非Windows 8.1+系统): {e}")

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.3 # 给 UI 响应时间

mcp = FastMCP("WindowsDesktopMCP")

# 针对 2256 x 1504 分辨率的最佳网格划分
# 2256 / 26 ≈ 86.7 像素，1504 / 16 = 94 像素 (每个格子足够大，且能精确命中图标)
GRID_COLS = 26
GRID_ROWS = 16

@mcp.tool()
def take_grid_screenshot() -> str:
    """获取当前屏幕(2256x1504)带 A1-Z16 网格的截图，返回 Base64"""
    # 由于 DPI 开启，这里截到的将是纯正的 2256x1504 物理像素
    screen = pyautogui.screenshot()
    width, height = screen.size
    
    draw = ImageDraw.Draw(screen, "RGBA")
    col_width = width / GRID_COLS
    row_height = height / GRID_ROWS
    
    # 画线
    for i in range(1, GRID_COLS):
        x = i * col_width
        draw.line([(x, 0), (x, height)], fill=(255, 0, 0, 100), width=1)
    for i in range(1, GRID_ROWS):
        y = i * row_height
        draw.line([(0, y), (width, y)], fill=(255, 0, 0, 100), width=1)
        
    # 标坐标 (尝试加载常见中文字体，失败则用默认)
    try:
        font = ImageFont.truetype("msyh.ttc", 20) # 微软雅黑
    except:
        font = ImageFont.load_default()

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            grid_id = f"{chr(65 + col)}{row + 1}"
            x = col * col_width + 5
            y = row * row_height + 5
            draw.text((x, y), grid_id, fill=(255, 0, 0, 220), font=font)
            
    buffered = io.BytesIO()
    screen.convert("RGB").save(buffered, format="JPEG", quality=75)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@mcp.tool()
def click_grid_center(grid_id: str, click_type: str = "single", button: str = "left") -> str:
    """根据类似 'C4' 的网格坐标点击中心物理位置"""
    width, height = pyautogui.size() # 由于 DPI 开启，这里返回 2256, 1504
    col_width = width / GRID_COLS
    row_height = height / GRID_ROWS
    
    grid_id = grid_id.upper()
    col_char = grid_id[0]
    row_str = grid_id[1:]
    
    col_idx = ord(col_char) - 65
    row_idx = int(row_str) - 1
    
    center_x = (col_idx * col_width) + (col_width / 2)
    center_y = (row_idx * row_height) + (row_height / 2)
    
    if click_type == "double":
        pyautogui.doubleClick(x=center_x, y=center_y, button=button)
    else:
        pyautogui.click(x=center_x, y=center_y, button=button)
    return f"Clicked {grid_id} at ({center_x:.0f}, {center_y:.0f})"

@mcp.tool()
def safe_input_text(text: str) -> str:
    """安全的中文输入"""
    pyperclip.copy(text)
    pyautogui.hotkey('ctrl', 'v')
    return f"Typed: {text}"

@mcp.tool()
def press_hotkey(keys: str) -> str:
    """按下快捷键"""
    key_list = keys.lower().split('+')
    pyautogui.hotkey(*key_list)
    return f"Pressed: {keys}"

if __name__ == "__main__":
    mcp.run()
