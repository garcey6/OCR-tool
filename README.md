# OCR 屏幕识别工具

一个置顶的半透明取景框 + 实时识别结果浮窗：把框拖到屏幕任意位置，框内内容每 3 秒自动识别一次，结果显示在右侧面板并可导出 TXT。

## 特性

- 取景框可整体拖动，四边/四角按 Windows 原生窗口的方式缩放
- 每 3 秒自动截屏识别，内容未变化时不重复追加
- 分块 OCR：大图按条带切分后缝合，避免长行被切断、小字漏检
- 自动屏蔽浅色水印（RGB 三通道同时高于阈值置白）
- 高频误识别纠错表 `CORRECT_MAP`，可按需扩充
- 结果面板支持暂停/继续、清空、导出 TXT

## 环境要求

- Windows（依赖 `PIL.ImageGrab` 与 `ctypes` 设置 DPI 感知）
- Python 3.9+

## 安装

```bash
pip install numpy PyQt5 opencv-python onnxruntime rapidocr_onnxruntime Pillow
```

可选 GPU 加速：把 `onnxruntime` 换成 `onnxruntime-directml`，脚本检测到 DirectML 会自动启用；
识别结果异常时可用环境变量 `OCR_FORCE_CPU=1` 强制走 CPU 对比。

## 运行

```bash
python ocr.py
```

## 自检

不需要弹窗，用合成图片跑一遍完整识别流水线：

```bash
python _selfcheck.py
```

## 可调参数

| 参数 | 说明 |
| --- | --- |
| `DEFAULT_SIZE` | 取景框默认边长 |
| `INTERVAL_MS` | 识别间隔（默认 3000ms） |
| `PULSE_MS` | 边框呼吸动画间隔，设为 0 关闭 |
| `CONF_THRES` | 置信度过滤阈值 |
| `WM_RGB_THRES` | 水印屏蔽阈值 |
| `CORRECT_MAP` | 误识别纠错表 |

---

作者：garcey6
