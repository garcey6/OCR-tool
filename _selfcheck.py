# -*- coding: utf-8 -*-
"""离线自检：不弹窗，用合成图片跑一遍识别流水线，验证依赖与算法链路"""
import time
from PIL import Image, ImageDraw, ImageFont
import ocr

# 合成一张带中英文的图，模拟截图内容
img = Image.new("RGB", (900, 260), "white")
d = ImageDraw.Draw(img)
font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 30)
d.text((30, 40), "屏幕识别工具 自检通过", fill="black", font=font)
d.text((30, 110), "DeepSeek Harness OCR", fill="black", font=font)
d.text((30, 180), "每三秒识别一次", fill="black", font=font)

t0 = time.perf_counter()
lines = ocr.recognize(img)
dt = time.perf_counter() - t0

print("耗时: %.2fs" % dt)
print("识别到 %d 行:" % len(lines))
for line in lines:
    print("  ", line)

assert lines, "识别结果为空"
joined = "".join(lines)
assert "自检通过" in joined, "中文识别失败: %r" % joined
print("PIPELINE OK")
