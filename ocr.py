# -*- coding: utf-8 -*-
"""
实时 OCR 识别框
- 一个可移动、可缩放的半透明识别框（置顶、无边框，默认 500x500）
- 鼠标放边框/四角可拖动缩放，框内区域实时进行 OCR
- 识别结果实时显示在旁边的面板中
"""

import sys
import time
import numpy as np
from PyQt5.QtCore import (Qt, QTimer, QThread, pyqtSignal, QRect, QRectF,
                          QPointF, QEvent)
from PyQt5.QtGui import QColor, QPainter, QPen, QPainterPath
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLabel, QPushButton, QCheckBox, QFileDialog, QFrame
)

# OCR 引擎（RapidOCR：模型随包附带，无需联网下载）
from rapidocr_onnxruntime import RapidOCR
from PIL import ImageGrab, Image

# ---------------- 配置 ----------------
DEFAULT_SIZE = 500        # 识别框默认边长（像素）
MIN_SIZE = 120            # 识别框最小边长
# ---- 仿 Windows 原生窗口的边缘缩放热区 ----
EDGE_GRIP = 10            # 四边缩放手柄厚度：Windows 约 8px，这里略放宽便于点选
CORNER_GRIP = 26          # 四角缩放手柄边长：角区优先于边区，与原生窗口一致
PANEL_MIN_W = 320         # 结果浮窗最小宽高：边缘缩放时不至于缩到不可用
PANEL_MIN_H = 260
# ---- 以下为纯外观参数：绘制范围必须落在 CAPTURE_MARGIN 以内，
# 否则边框像素会被截屏拍进去污染识别结果 ----
CORNER_LEN = 26           # 取景器四角 L 形标记长度（像素，仅影响外观）
CORNER_W = 4              # 取景器四角 L 形标记线宽
FRAME_INSET = 3           # 边框线到窗口边缘的距离
FRAME_RADIUS = 10         # 识别框圆角半径
PULSE_MS = 60             # 边框呼吸动画间隔（毫秒），设为 0 关闭动画
# ---- 取景框缩放热区：热区整圈移到框线外侧，识别区域内不含缩放手柄 ----
# 取景框缩放手柄：像 Windows 窗口一样压在可见框线上（框线外侧 16px + 内侧 8px），
# 框内其余位置整体拖动。内侧只占 8px，仍在截屏内缩 10px 之外，不会拍进识别画面。
FRAME_GRIP = 16           # 窗口向外扩出的透明留白宽度（缩放手柄的外侧部分）
FRAME_GRIP_IN = 8         # 缩放手柄向框内延伸的宽度
# 命中遮罩透明度：Windows 对分层窗口按逐像素 alpha 做命中测试，alpha=0 的区域
# 鼠标会直接穿透到下层窗口，导致框内无法拖动。铺一层 alpha=1 的遮罩即可整窗命中，
# 视觉上不可见（白底 255 仅变 254），对识别无影响。
FRAME_HIT_ALPHA = 1
# ---- 结果浮窗外观 ----
PANEL_SHADOW = 22         # 卡片外投影半径（同时作为窗口留白，避免投影被裁切）
PANEL_RADIUS = 14         # 卡片圆角
WATERMARK = "garcey6开发"   # 作者水印：显示在浮窗并写入导出文件

# 深色卡片主题：统一控件配色，避免原生控件在深色卡片上突兀
PANEL_QSS = """
* {
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
}
QFrame#card {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(28, 32, 41, 242), stop:1 rgba(20, 23, 30, 246));
    border: 1px solid rgba(255, 255, 255, 26);
    border-radius: 14px;
}
QLabel#winTitle  { color: #e8ecf3; font-size: 13px; font-weight: 600; }
QLabel#winSub    { color: #7c8698; font-size: 11px; }
QLabel#status    { color: #9fb6d4; font-size: 11px; padding: 3px 8px;
                   background: rgba(255, 255, 255, 14); border-radius: 9px; }
QLabel#watermark { color: #5c6779; font-size: 10px; }
QTextEdit {
    background: rgba(12, 14, 19, 200);
    color: #dfe6f0;
    border: 1px solid rgba(255, 255, 255, 22);
    border-radius: 10px;
    padding: 8px 10px;
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
    font-size: 12px;
    selection-background-color: rgba(0, 209, 255, 110);
}
QTextEdit:focus { border: 1px solid rgba(0, 209, 255, 130); }
QPushButton {
    background: rgba(255, 255, 255, 18);
    color: #d6deea;
    border: 1px solid rgba(255, 255, 255, 26);
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 12px;
}
QPushButton:hover   { background: rgba(0, 209, 255, 42);
                      border: 1px solid rgba(0, 209, 255, 120); color: #ffffff; }
QPushButton:pressed { background: rgba(0, 209, 255, 78); }
QPushButton#titleBtn, QPushButton#closeBtn {
    background: transparent; border: none; color: #8b95a6;
    padding: 2px 6px; font-size: 13px; border-radius: 7px;
}
QPushButton#titleBtn:hover { background: rgba(255, 255, 255, 26); color: #ffffff; }
QPushButton#closeBtn:hover { background: rgba(232, 72, 72, 210); color: #ffffff; }
QCheckBox { color: #9aa5b6; font-size: 12px; spacing: 6px; }
QCheckBox::indicator {
    width: 14px; height: 14px; border-radius: 4px;
    border: 1px solid rgba(255, 255, 255, 60);
    background: rgba(255, 255, 255, 14);
}
QCheckBox::indicator:checked {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #38d9f8, stop:1 #4f8bff);
    border: 1px solid rgba(0, 209, 255, 180);
}
QScrollBar:vertical { background: transparent; width: 9px; margin: 2px; }
QScrollBar::handle:vertical {
    background: rgba(255, 255, 255, 46); border-radius: 4px; min-height: 28px;
}
QScrollBar::handle:vertical:hover { background: rgba(0, 209, 255, 140); }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
 
"""
CAPTURE_MARGIN = 10       # 截屏时向内收缩，避免拍到红框自身
INTERVAL_MS = 3000        # 策略：每三秒识别一次（毫秒）
CONF_THRES = 0.5          # 置信度阈值（切块缝合稳定后误识减少，过滤边界噪声片段）
WM_RGB_THRES = 215        # 水印屏蔽阈值：R、G、B 三个通道都大于该值的像素置为纯白
                          # （水印 RGB≈244 会被屏蔽，正文深色文字保留）
DET_LIMIT_MAX = 1600      # 检测输入长边上限：超过则由横向分块切分，不整体缩小
TILE_SIZE = 1024          # 分块 OCR 的条带高度（像素）
TILE_OVERLAP = 160        # 条带重叠区：跨边界的小字不会被切断丢失
H_TILE_SIZE = 1400        # 横向切分宽度：超过该宽度检测会漏整行，切块后靠缝合还原长行
DET_BOX_THRES = 0.4       # 检测框阈值（默认 0.5）：调低提升小字召回率
DET_UNCLIP = 1.8          # 检测框扩张系数（默认 1.6）：小字框扩张更完整

# 截屏场景高频误识别纠错表（可按需扩充）
CORRECT_MAP = {
    "精空": "清空",
    "清空": "清空",
    # 识别模型把 n 读成 m：与 DeepSeek Harness 同时出现时几乎必错
    "Harmess": "Harness",
    "harmess": "Harness",
}


def band_starts(total, size, overlap):
    """条带切分的起点列表：相邻条带重叠 overlap 像素，最后一条贴齐末尾，避免漏掉尾部"""
    if total <= size:
        return [0]
    step = max(1, size - overlap)
    starts = list(range(0, total - size + 1, step))
    if starts[-1] + size < total:
        starts.append(total - size)
    return starts


def _box_rect(box):
    """四点框 → 外接矩形 (x1, y1, x2, y2)"""
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def _axis_overlap(a1, a2, b1, b2):
    """两个区间沿同一条轴的重叠长度"""
    return max(0.0, min(a2, b2) - max(a1, b1))


def _line_overlap_ratio(box_a, box_b):
    """垂直方向的重叠比例（相对较矮的框），用于判断两个片段是否同一行"""
    _, ay1, _, ay2 = _box_rect(box_a)
    _, by1, _, by2 = _box_rect(box_b)
    h = min(ay2 - ay1, by2 - by1)
    if h <= 0:
        return 0.0
    return _axis_overlap(ay1, ay2, by1, by2) / h


def _norm_text(text):
    """归一化文本：去掉空白并转小写，仅用于判断两段文字是否为同一内容"""
    return "".join(str(text).split()).lower()


def _is_cjk(ch):
    """是否中日韩全角字符：中文之间拼接时不加空格"""
    return ("\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f"
            or "\uff00" <= ch <= "\uffef")


def _join_segments(texts):
    """同一行内相邻片段拼接：中文直接相连，西文之间补一个空格"""
    out = ""
    for t in texts:
        if not t:
            continue
        if not out:
            out = t
        elif _is_cjk(out[-1]) or _is_cjk(t[0]):
            out += t
        else:
            out += " " + t
    return out


def reading_order_lines(results):
    """把检测框整理成阅读顺序的行：同一行的片段按 x 从左到右拼接，
    行与行按 y 从上到下排列，返回纯文本行列表"""
    if not results:
        return []
    items = []
    for box, text, _score in results:
        x1, y1, x2, y2 = _box_rect(box)
        items.append((x1, y1, x2, y2, str(text)))
    items.sort(key=lambda it: (it[1], it[0]))
    # 每行记 [顶端, 底端, [(x1, 文本), ...]]
    lines = []
    for x1, y1, x2, y2, text in items:
        target = None
        for line in lines:
            h = min(y2 - y1, line[1] - line[0])
            if h <= 0:
                continue
            if _axis_overlap(y1, y2, line[0], line[1]) / h < 0.5:
                continue
            # 垂直中心不能差太多：表格相邻行的框常常贴着，只靠重叠会被并成一行
            dy = abs((y1 + y2) / 2 - (line[0] + line[1]) / 2)
            if dy > max(y2 - y1, line[1] - line[0]) * 0.6:
                continue
            target = line
            break
        if target is None:
            lines.append([y1, y2, [(x1, text)]])
        else:
            target[0] = min(target[0], y1)
            target[1] = max(target[1], y2)
            target[2].append((x1, text))
    out = []
    for _, _, segs in lines:
        segs.sort(key=lambda s: s[0])
        out.append(_join_segments([t for _, t in segs]))
    return out


def _char_overlap(left, right):
    """left 尾部与 right 头部的最长公共字符数（>=1 才说明两段真的读到了同一批字）。
    只有字符级重叠才能证明是切断重复，仅框相交可能是同一行的两个不同词"""
    for n in range(min(len(left), len(right), 16), 0, -1):
        if left[-n:] == right[:n]:
            return n
    return 0


def merge_overlaps(results):
    """两轮整理切块产生的碎片：
    1) 同文重叠去重：文本归一化相同且水平、垂直都明显相交 → 同一框被两块各识一次
    2) 行内拼接缝合：同一行且水平重叠的相邻片段（长行被块边界切断）按字符数
       扣除重叠区后拼成一段，消除『展技术试点…』这类重复前缀
    注意：判断必须基于框的重叠而不是中心距离——表格相邻行常有完全相同的文本
    （如'计划或项目目标'），中心很近但并非同一框，误合并会导致内容错乱"""
    # ---- 1. 同文重叠去重 ----
    deduped = []
    for box, text, score in results:
        dup = False
        nt = _norm_text(text)
        for i, (mbox, mtext, mscore) in enumerate(deduped):
            if _norm_text(mtext) != nt:
                continue
            ax1, _, ax2, _ = _box_rect(box)
            bx1, _, bx2, _ = _box_rect(mbox)
            ho = _axis_overlap(ax1, ax2, bx1, bx2)
            if (ho >= 0.25 * min(ax2 - ax1, bx2 - bx1)
                    and _line_overlap_ratio(box, mbox) >= 0.5):
                if score > mscore:
                    deduped[i] = [box, text, score]
                dup = True
                break
        if not dup:
            deduped.append([box, text, score])
    # ---- 2. 行内重叠片段拼接 ----
    deduped.sort(key=lambda r: (min(p[1] for p in r[0]) + max(p[1] for p in r[0])) / 2)
    stitched = []
    for item in deduped:
        box, text, score = item
        for j in range(len(stitched) - 1, -1, -1):
            x1, y1, x2, y2 = _box_rect(box)
            pbox, ptext, pscore = stitched[j]
            qx1, qy1, qx2, qy2 = _box_rect(pbox)
            if _line_overlap_ratio(pbox, box) < 0.5:
                continue
            if x1 >= qx2 or qx1 >= x2:
                continue
            # 保证 left/right 与片段顺序一致：按从左到右拼接
            if x1 < qx1:
                box, pbox = pbox, box
                text, ptext = ptext, text
                score, pscore = pscore, score
                x1, y1, x2, y2 = _box_rect(box)
                qx1, qy1, qx2, qy2 = _box_rect(pbox)
            # 只在左片段尾部与右片段头部读到相同字符时才裁剪重叠区：
            # 按像素比例估算容易误删真实字符（如顿号），字符级重叠才是切断重复的铁证
            drop = _char_overlap(ptext, text)
            if drop == 0:
                continue
            new_text = ptext[:len(ptext) - drop] + text
            new_box = [[min(qx1, x1), min(qy1, y1)], [max(qx2, x2), min(qy1, y1)],
                       [max(qx2, x2), max(qy2, y2)], [min(qx1, x1), max(qy2, y2)]]
            stitched[j] = [new_box, new_text, max(pscore, score)]
            break
        else:
            stitched.append(item)
    return stitched


def ocr_with_tiles(img_arr):
    """分块 OCR：按整宽水平条带切分大图分别识别再合并。
    只纵向切条、不横向切断行——横向切断长行会让同一行变成两段重叠片段，
    既重复输出又在边界处错字（纵向切分至多整行重复，可被去重规则消除）"""
    h, w = img_arr.shape[:2]
    call_kw = dict(box_thresh=DET_BOX_THRES, unclip_ratio=DET_UNCLIP)
    xs = [0] if w <= H_TILE_SIZE else band_starts(w, H_TILE_SIZE, TILE_OVERLAP)
    ys = [0] if h <= TILE_SIZE else band_starts(h, TILE_SIZE, TILE_OVERLAP)
    if len(xs) == 1 and len(ys) == 1:
        result, _ = engine(img_arr, **call_kw)
        return merge_overlaps([[b, t, float(s)] for b, t, s in (result or [])])
    results = []
    for y in ys:
        for x in xs:
            tile = img_arr[y:min(y + TILE_SIZE, h), x:min(x + H_TILE_SIZE, w)]
            result, _ = engine(tile, **call_kw)
            if not result:
                continue
            for box, text, score in result:
                shifted = [[p[0] + x, p[1] + y] for p in box]
                results.append([shifted, text, float(score)])
    return merge_overlaps(results)
def remove_watermark(img):
    """基于 RGB 三通道阈值屏蔽浅色水印：
    R、G、B 都大于 WM_RGB_THRES 的像素（如 RGB≈244 的淡灰水印）统一置为纯白，
    深色正文文字保留，从而提升 OCR 对常驻水印截屏的识别效果"""
    arr = np.asarray(img.convert("RGB")).copy()
    mask = (arr[:, :, 0] > WM_RGB_THRES) & \
           (arr[:, :, 1] > WM_RGB_THRES) & \
           (arr[:, :, 2] > WM_RGB_THRES)
    arr[mask] = [255, 255, 255]
    return Image.fromarray(arr)


def prepare_for_ocr(img):
    """识别前预处理：屏蔽浅色水印。
    刻意不做放大——实测预放大（2~3 倍）会让检测框在词内断裂并误识
    （"Harness"→"Harmess"、"文字"→"又字"、顿号读成逗号），
    小字分辨率交给引擎的 det_limit_type=max 与分块策略保证"""
    return remove_watermark(img)


def recognize(img):
    """完整识别流水线：预处理 → 分块识别 → 置信度过滤 → 高频误识纠错 → 书写顺序整理"""
    result = ocr_with_tiles(np.array(prepare_for_ocr(img)))
    cleaned = []
    for box, text, score in result:
        if score < CONF_THRES:
            continue
        for wrong, right in CORRECT_MAP.items():
            if wrong in text:
                text = text.replace(wrong, right)
        cleaned.append([box, text, score])
    return reading_order_lines(cleaned)


import os

# 检测输入长边统一压到 DET_LIMIT_MAX（limit_type=max）：实测比 min/1280 更稳——
# 小图不会只被勉强放大到 1280 而漏字，大图也不会以超过 1600 的长边进模型导致整行漏检
# 优先用 DirectML（NVIDIA/Intel GPU）加速；环境不支持或设 OCR_FORCE_CPU=1 时用 CPU
ENGINE_KW = dict(text_score=0.4, det_limit_side_len=DET_LIMIT_MAX, det_limit_type="max")

_force_cpu = os.environ.get("OCR_FORCE_CPU") == "1"
try:
    import onnxruntime as _ort
    _dml_ok = "DmlExecutionProvider" in _ort.get_available_providers()
except Exception:
    _dml_ok = False

if _force_cpu or not _dml_ok:
    engine = RapidOCR(**ENGINE_KW)
    print("[OCR] CPU 推理模式" if _force_cpu else "[OCR] DirectML 不可用，CPU 推理")
else:
    try:
        engine = RapidOCR(det_use_dml=True, cls_use_dml=True, rec_use_dml=True,
                          **ENGINE_KW)
        print("[OCR] DirectML GPU 加速已启用（如识别失真：set OCR_FORCE_CPU=1 对比）")
    except Exception:
        engine = RapidOCR(**ENGINE_KW)
        print("[OCR] DirectML 初始化失败，已回退 CPU 推理")


class EdgeResizeMixin:
    """Windows 原生窗口式的边缘缩放 + 整体拖动。
    可见边缘在窗口内 edge_inset 处，缩放热区沿可见边缘内外各扩一圈；
    非热区按下即整体拖动，子类可用 edge_body_drag=False 关闭该行为。"""

    edge_inset = 0            # 可见边缘相对窗口的缩进（外圈透明留白宽度）
    edge_outward = 0          # 缩放热区向可见边缘外侧延伸的宽度
    edge_inward = None        # 缩放热区向内侧延伸的宽度，None 表示沿用 edge_grip
    edge_grip = EDGE_GRIP     # 四边热区厚度
    corner_grip = CORNER_GRIP  # 四角热区边长（角区优先于边区）
    edge_min_size = (MIN_SIZE, MIN_SIZE)
    edge_body_drag = True     # 非热区按下是否整体拖动窗口
    _hover_edge = (None, None)

    def _init_edge_resize(self, extra_watch=()):
        """初始化拖拽/缩放状态，并把会吞掉鼠标事件的子控件纳入事件监听"""
        self._edge_mode = None
        self._drag_origin = None
        self._press_geo = None
        self._press_pos = None
        self._hover_edge = (None, None)
        self._forced_cursor_widget = None
        self.setMouseTracking(True)
        for w in extra_watch:
            w.setMouseTracking(True)
            w.installEventFilter(self)

    def eventFilter(self, obj, event):
        """子控件上的鼠标事件映射回窗口坐标，复用同一套边缘逻辑"""
        et = event.type()
        if et in (QEvent.MouseMove, QEvent.MouseButtonPress,
                  QEvent.MouseButtonRelease):
            if (et == QEvent.MouseButtonPress
                    and event.button() != Qt.LeftButton):
                return False
            local = self.mapFromGlobal(event.globalPos())
            if et == QEvent.MouseButtonPress:
                if self._edge_press_at(local, event.globalPos(), event):
                    return True
                # 未命中边缘：交给子类判断是否作为标题栏拖动窗口
                return self.title_drag_press(local, event)
            if et == QEvent.MouseButtonRelease:
                return self._edge_release_at(local, event)
            return self._edge_move_at(local, event.globalPos(),
                                      event.buttons(), event, hover_widget=obj)
        return super().eventFilter(obj, event)

    def title_drag_press(self, local, event):
        """子控件内按下且不在边缘热区时的处理，默认不拦截子控件点击"""
        return False

    def _near_edge(self, dist, grip):
        """dist 是到可见边缘的带符号距离：负值表示落在边缘外侧的透明留白里。
        内侧上限可由 edge_inward 收窄，用于把缩放手柄整体挪到可见边缘之外。"""
        inner = grip if self.edge_inward is None else self.edge_inward
        return -self.edge_outward <= dist < inner

    def edge_zones(self, pos):
        """把窗口内坐标映射为缩放方向 (h, v)，h='l'/'r'，v='t'/'b'，None 表示该方向不缩放"""
        ins = self.edge_inset
        dl = pos.x() - ins
        dr = self.width() - 1 - ins - pos.x()
        dt = pos.y() - ins
        db = self.height() - 1 - ins - pos.y()
        # 四角优先：两个方向同时命中
        for dh, dd_h in (("l", dl), ("r", dr)):
            for dv, dd_v in (("t", dt), ("b", db)):
                if self._near_edge(dd_h, self.corner_grip) and \
                        self._near_edge(dd_v, self.corner_grip):
                    return dh, dv
        # 四条边：只沿一个方向缩放
        if self._near_edge(dl, self.edge_grip) or \
                self._near_edge(dr, self.edge_grip):
            return ("l", None) if dl < dr else ("r", None)
        if self._near_edge(dt, self.edge_grip) or \
                self._near_edge(db, self.edge_grip):
            return (None, "t") if dt < db else (None, "b")
        return None, None

    def edge_cursor(self, h, v):
        """光标形状与 Windows 一致：四角对角、上下水平/垂直、边内单向"""
        if h and v:
            return (Qt.SizeFDiagCursor if (h == "l") == (v == "t")
                    else Qt.SizeBDiagCursor)
        if h:
            return Qt.SizeHorCursor
        if v:
            return Qt.SizeVerCursor
        return self.body_cursor()
    def body_cursor(self):
        """非边缘区域的默认光标，子类可覆盖"""
        return Qt.ArrowCursor

    def on_edge_press(self, h, v):
        """缩放开始前的钩子，子类可覆盖"""

    def on_body_press(self):
        """整体拖动开始前的钩子，子类可覆盖"""

    def on_edge_release(self):
        """拖动/缩放结束的钩子，子类可覆盖"""

    def on_edge_hover(self, h, v):
        """悬停缩放方向变化时的钩子，子类可覆盖做视觉提示"""

    def body_drag_allowed(self, local):
        """非边缘区域按下时是否允许整体拖动，子类可按区域限制"""
        return True

    # ---------- 统一鼠标处理 ----------
    def edge_mouse_press(self, event):
        if event.button() != Qt.LeftButton:
            return False
        return self._edge_press_at(event.pos(), event.globalPos(), event)

    def edge_mouse_move(self, event):
        return self._edge_move_at(event.pos(), event.globalPos(),
                                  event.buttons(), event)

    def edge_mouse_release(self, event):
        return self._edge_release_at(event.pos(), event)

    def _edge_press_at(self, local, global_pos, event):
        h, v = self.edge_zones(local)
        if not (h or v):
            if not (self.edge_body_drag and self.body_drag_allowed(local)):
                return False
        self._press_geo = QRect(self.pos(), self.size())
        self._press_pos = global_pos
        if h or v:
            self._edge_mode = (h, v)
            self.on_edge_press(h, v)
        else:
            self._drag_origin = global_pos
            self.on_body_press()
        event.accept()
        return True

    def _edge_move_at(self, local, global_pos, buttons, event,
                      hover_widget=None):
        target = hover_widget or self
        if self._edge_mode is not None and self._press_geo is not None:
            self._apply_edge_resize(global_pos)
            event.accept()
            return True
        if self._drag_origin is not None and (buttons & Qt.LeftButton):
            self.move(self.pos() + (global_pos - self._drag_origin))
            self._drag_origin = global_pos
            event.accept()
            return True
        if not (buttons & Qt.LeftButton):
            h, v = self.edge_zones(local)
            if self._hover_edge != (h, v):
                # 手柄位置变化：通知子类刷新高亮，让用户看清该拖哪条边
                self._hover_edge = (h, v)
                self.on_edge_hover(h, v)
                self.update()
            if h or v:
                target.setCursor(self.edge_cursor(h, v))
                if target is not self:
                    self._forced_cursor_widget = target
            elif target is self:
                target.setCursor(self.body_cursor())
            elif self._forced_cursor_widget is target:
                # 离开边缘区：交还给子控件自己的光标（如按钮的手型）
                target.unsetCursor()
                self._forced_cursor_widget = None
        return False
    def _edge_release_at(self, local, event):
        if self._edge_mode is None and self._drag_origin is None:
            return False
        self._edge_mode = None
        self._drag_origin = None
        self._press_geo = None
        self._press_pos = None
        self.setCursor(self.edge_cursor(*self.edge_zones(local)))
        self._hover_edge = self.edge_zones(local)
        self.on_edge_hover(*self._hover_edge)
        self.on_edge_release()
        self.update()
        event.accept()
        return True

    def _apply_edge_resize(self, global_pos):
        """按 Windows 行为缩放：被拖的一边跟随鼠标，对边保持不动"""
        h, v = self._edge_mode
        geo = QRect(self._press_geo)
        delta = global_pos - self._press_pos
        min_w, min_h = self.edge_min_size
        if h == "l":
            # QRect.right() = left + width - 1，钳制时要还原这 1px 偏移
            geo.setLeft(min(geo.right() - min_w + 1, geo.left() + delta.x()))
        elif h == "r":
            geo.setRight(max(geo.left() + min_w - 1, geo.right() + delta.x()))
        if v == "t":
            geo.setTop(min(geo.bottom() - min_h + 1, geo.top() + delta.y()))
        elif v == "b":
            geo.setBottom(max(geo.top() + min_h - 1, geo.bottom() + delta.y()))
        self.setGeometry(geo)


class FrameWindow(EdgeResizeMixin, QWidget):
    """可移动、可缩放的透明识别框：框内任意位置可拖动，缩放热区整圈在框线外侧。
    窗口比识别区向外多出 FRAME_GRIP 的透明留白承载缩放手柄，识别区域本身不含任何缩放手柄。"""
    # 可见框线在窗口内 FRAME_GRIP 处，手柄跨框线：[-FRAME_GRIP, +FRAME_GRIP_IN)
    edge_inset = FRAME_GRIP
    edge_outward = FRAME_GRIP
    edge_inward = FRAME_GRIP_IN
    # 窗口每边比识别区多 FRAME_GRIP，最小窗口尺寸同样算上这一圈
    edge_min_size = (MIN_SIZE + FRAME_GRIP * 2, MIN_SIZE + FRAME_GRIP * 2)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("识别框")
        self.setWindowFlags(
            Qt.FramelessWindowHint |       # 无边框
            Qt.WindowStaysOnTopHint |      # 置顶
            Qt.Tool                        # 不显示在任务栏
        )
        # 半透明背景（能透出下方内容）
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 窗口 = 识别区 + 四周 FRAME_GRIP 透明留白（留白只用于缩放，不参与识别）
        outer = DEFAULT_SIZE + FRAME_GRIP * 2
        self.resize(outer, outer)
        self.setMinimumSize(MIN_SIZE + FRAME_GRIP * 2,
                            MIN_SIZE + FRAME_GRIP * 2)
        self._init_edge_resize()
        # 外观状态：active=识别中（青色呼吸），paused=已暂停（灰色静态）
        self.active = True
        self._pulse = 0.0
        self._pulse_dir = 1
        self._anim = QTimer(self)
        self._anim.setInterval(PULSE_MS)
        self._anim.timeout.connect(self._tick_pulse)
        if PULSE_MS:
            self._anim.start()

    def set_active(self, active):
        """切换识别中 / 暂停的配色，供结果面板联动调用"""
        self.active = active
        self.update()

    def _tick_pulse(self):
        """边框呼吸：来回渐变的不透明度，让『正在识别』更直观"""
        self._pulse += 0.04 * self._pulse_dir
        if self._pulse >= 1.0:
            self._pulse, self._pulse_dir = 1.0, -1
        elif self._pulse <= 0.0:
            self._pulse, self._pulse_dir = 0.0, 1
        self.update()
    def _accent(self):
        return QColor(0, 209, 255) if self.active else QColor(150, 158, 168)

    def paintEvent(self, event):
        """取景器风格边框：圆角细线 + 四角 L 形高亮标记 + 四角缩放热区微光。
        绘制范围收在窗口内侧，不会被截屏拍进识别画面"""
        painter = QPainter(self)
        # 关键：整窗铺一层几乎全透明的遮罩。分层窗口按逐像素 alpha 命中，
        # 不铺的话框内是 alpha=0 的空像素，鼠标会穿透下去，只能拖到画了像素的框线。
        painter.fillRect(self.rect(), QColor(0, 0, 0, FRAME_HIT_ALPHA))
        painter.setRenderHint(QPainter.Antialiasing)
        accent = self._accent()
        # 可见框线画在留白带内侧；留白带只有命中遮罩，肉眼仍是空的
        inset = FRAME_GRIP + FRAME_INSET
        rect = QRectF(inset, inset,
                      self.width() - inset * 2, self.height() - inset * 2)
        radius = min(FRAME_RADIUS, rect.width() / 2, rect.height() / 2)
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        # 1) 内侧 1px 深色描边：保证在白色文档上也看得见
        painter.setPen(QPen(QColor(0, 0, 0, 130), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
        # 2) 主边框：识别中随呼吸渐变，暂停时静态半透明
        alpha = 90 + int(120 * self._pulse) if self.active else 120
        line = QColor(accent)
        line.setAlpha(alpha)
        painter.setPen(QPen(line, 1.6))
        painter.drawPath(path)
        # 3) 四角 L 形标记（兼作缩放提示），比旧版黄色方块更轻
        corner_pen = QPen(accent, CORNER_W)
        corner_pen.setCapStyle(Qt.RoundCap)
        corner_pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(corner_pen)
        mark = min(CORNER_LEN, rect.width() / 2, rect.height() / 2)
        l, t = rect.left(), rect.top()
        r, b = rect.right(), rect.bottom()
        for sx, sy, dx, dy in ((l, t, 1, 1), (r, t, -1, 1), (l, b, 1, -1), (r, b, -1, -1)):
            painter.drawLine(QPointF(sx, sy), QPointF(sx + mark * dx, sy))
            painter.drawLine(QPointF(sx, sy), QPointF(sx, sy + mark * dy))

        # 4) 悬停高亮：鼠标落在哪条边/哪个角，该处框线加粗发亮，
        #    提示这里就是缩放手柄（离开后自动恢复）
        h, v = getattr(self, "_hover_edge", (None, None))
        if h or v:
            hot = QPen(accent, 4)
            hot.setCapStyle(Qt.RoundCap)
            painter.setPen(hot)
            pad = CORNER_LEN * 0.55      # 高亮线段两端留空，避免压住角标
            if h == "l":
                painter.drawLine(QPointF(l, t + pad), QPointF(l, b - pad))
            elif h == "r":
                painter.drawLine(QPointF(r, t + pad), QPointF(r, b - pad))
            if v == "t":
                painter.drawLine(QPointF(l + pad, t), QPointF(r - pad, t))
            elif v == "b":
                painter.drawLine(QPointF(l + pad, b), QPointF(r - pad, b))
    # ---------- 鼠标事件：整体拖动 + 四边/四角缩放（交由 EdgeResizeMixin）----------
    def body_cursor(self):
        """框内是取景区：用四向箭头提示可整体拖动"""
        return Qt.SizeAllCursor

    def on_body_press(self):
        self.setCursor(Qt.ClosedHandCursor)

    def mousePressEvent(self, event):
        self.edge_mouse_press(event)

    def mouseMoveEvent(self, event):
        self.edge_mouse_move(event)

    def mouseReleaseEvent(self, event):
        self.edge_mouse_release(event)

    def region(self):
        """当前识别区域（QRect）：扣掉外圈缩放留白，与可见框线严格一致"""
        g = self.geometry()
        return QRect(g.x() + FRAME_GRIP, g.y() + FRAME_GRIP,
                     g.width() - FRAME_GRIP * 2,
                     g.height() - FRAME_GRIP * 2)
class ResultPanel(EdgeResizeMixin, QWidget):
    """识别结果浮窗：无边框深色卡片，标题栏可拖动，四边/四角缩放对齐 Windows 窗口"""
    # 卡片本体缩进来一圈（外圈是透明投影），缩放热区要对齐可见边缘
    edge_inset = PANEL_SHADOW
    edge_min_size = (PANEL_MIN_W, PANEL_MIN_H)
    # 抓取带 = 整圈透明投影留白 + 卡片边缘向内 2px：
    # 像原生窗口的隐形边框，不会抢走标题栏拖动和按钮点击
    edge_grip = 2
    corner_grip = 2
    edge_outward = PANEL_SHADOW
    # 卡片内是文本区和按钮：正文按下交还子控件，只有标题栏可整体拖动
    edge_body_drag = False

    def __init__(self, frame_window: FrameWindow):
        super().__init__()
        self.frame_window = frame_window
        self.setWindowTitle("OCR 识别结果")
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # 窗口留白给手绘阴影用（QGraphicsDropShadowEffect 在透明窗口上会渲染黑底）
        outer = QVBoxLayout(self)
        outer.setContentsMargins(PANEL_SHADOW, PANEL_SHADOW, PANEL_SHADOW, PANEL_SHADOW)

        card = QFrame(self)
        card.setObjectName("card")
        outer.addWidget(card)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 11, 14, 12)
        layout.setSpacing(9)

        # 标题行
        head = QHBoxLayout()
        head.setSpacing(7)
        self.dot = QLabel()
        self.dot.setFixedSize(9, 9)
        head.addWidget(self.dot)
        title = QLabel("框内实时识别")
        title.setObjectName("winTitle")
        head.addWidget(title)
        head.addSpacing(4)
        sub = QLabel("实时截屏 · 自动去重")
        sub.setObjectName("winSub")
        head.addWidget(sub)
        head.addStretch(1)
        close_btn = QPushButton("✕")
        close_btn.setObjectName("closeBtn")
        close_btn.setFixedSize(24, 21)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setToolTip("退出程序")
        head.addWidget(close_btn)
        layout.addLayout(head)

        # 结果显示
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlaceholderText(
            "把取景框拖到屏幕任意位置，识别内容会实时出现在这里…")
        self.text_edit.setFrameShape(QFrame.NoFrame)
        layout.addWidget(self.text_edit)

        # 状态行
        self.status = QLabel()
        self.status.setObjectName("status")
        self.status.setMinimumWidth(206)   # 固定最小宽度：状态文案变化时不抖动
        self.status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        head_status = QHBoxLayout()
        head_status.addWidget(self.status)
        head_status.addStretch(1)
        layout.addLayout(head_status)

        # 控制行
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)
        self.auto_scroll = QCheckBox("自动滚动到底部")
        self.auto_scroll.setChecked(True)
        self.pause_btn = QPushButton("❚❚ 暂停")
        self.clear_btn = QPushButton("清空")
        self.export_btn = QPushButton("导出 TXT")
        for btn in (self.pause_btn, self.clear_btn, self.export_btn):
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFlat(True)
        self.export_btn.setDefault(False)
        ctrl_row.addWidget(self.auto_scroll)
        ctrl_row.addStretch(1)
        ctrl_row.addWidget(self.pause_btn)
        ctrl_row.addWidget(self.clear_btn)
        ctrl_row.addWidget(self.export_btn)
        layout.addLayout(ctrl_row)

        # 作者水印
        wm_row = QHBoxLayout()
        wm_row.setSpacing(0)
        wm_row.addStretch(1)
        watermark = QLabel(WATERMARK)
        watermark.setObjectName("watermark")
        wm_row.addWidget(watermark)
        layout.addLayout(wm_row)

        close_btn.clicked.connect(QApplication.quit)
        self._head_height = 34          # 标题行高度：仅此区域可拖动窗口
        self.paused = False
        # 卡片会吞掉鼠标事件，交给事件过滤器统一判定边缘缩放
        self._init_edge_resize(extra_watch=(card, self.text_edit))
        # 标题栏以外是文本区/按钮，按下交还给子控件，不触发窗口拖动
        self._panel_card = card
        self._set_status(f"就绪 · 每 {INTERVAL_MS // 1000} 秒识别一次")
        self._update_dot()
        # 后台识别线程
        self.worker = OcrWorker(self)
        self.worker.finished_ocr.connect(self.on_ocr_done)

        # 定时器：每 INTERVAL_MS 截屏一次
        self.timer = QTimer(self)
        self.timer.setInterval(INTERVAL_MS)
        self.timer.timeout.connect(self.tick)

        self.pause_btn.clicked.connect(self.toggle_pause)
        self.clear_btn.clicked.connect(self.clear_results)
        self.export_btn.clicked.connect(self.export_txt)

        self.timer.start()

    def tick(self):
        """截取识别框区域并启动识别"""
        if self.paused or self.worker.running:
            return
        r = self.frame_window.region()
        m = CAPTURE_MARGIN
        try:
            # 向内收缩 m 像素，避免把识别框红边拍进去；all_screens 支持多显示器
            img = ImageGrab.grab(
                bbox=(r.left() + m, r.top() + m,
                      r.left() + r.width() - m, r.top() + r.height() - m),
                all_screens=True)
        except Exception as e:
            self.status.setText(f"截屏失败: {e}")
            return
        self.worker.image = img
        self.worker.start()

    def on_ocr_done(self, lines, elapsed):
        """把识别结果（已按书写顺序整理）追加显示"""
        if lines:
            content = "\n".join(lines)
            # 内容与上次相同时不重复追加（避免常驻截屏刷屏）
            if content != getattr(self, "_last_text", None):
                self._last_text = content
                self.text_edit.append(content)
                self._set_status(f"识别完成 · {len(lines)} 行 · {elapsed:.2f}s")
            else:
                self._set_status(f"内容未变化 · {elapsed:.2f}s")
        if self.auto_scroll.isChecked():
            sb = self.text_edit.verticalScrollBar()
            sb.setValue(sb.maximum())

    def clear_results(self):
        """清空结果并重置去重记录"""
        self.text_edit.clear()
        self._last_text = None
        self._set_status("已清空，等待下一次识别")

    def toggle_pause(self):
        self.paused = not self.paused
        self._set_status("已暂停 · 点击继续" if self.paused else "已继续识别")
        self.pause_btn.setText("▶ 继续" if self.paused else "❚❚ 暂停")
        # 取景框配色随暂停状态联动
        self.frame_window.set_active(not self.paused)
        self._update_dot()

    # ---------- 外观 ----------
    def _set_status(self, text):
        self.status.setText("●  " + text)

    def _update_dot(self):
        """标题栏状态灯：识别中=青色，暂停=灰色"""
        color = "#3ad6f5" if not self.paused else "#8a93a3"
        self.dot.setStyleSheet(
            f"background: {color}; border-radius: 4px;"
            f"border: 1px solid rgba(255,255,255,60);")

    def paintEvent(self, event):
        """在留白区手绘柔和投影：让浮窗看起来是悬浮的圆角卡片"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        rect = QRectF(self.rect()).adjusted(PANEL_SHADOW, PANEL_SHADOW + 2,
                                            -PANEL_SHADOW, -PANEL_SHADOW + 2)
        for i in range(PANEL_SHADOW, 0, -2):
            layer = rect.adjusted(-(PANEL_SHADOW - i) * 0.5,
                                  -(PANEL_SHADOW - i) * 0.5,
                                  (PANEL_SHADOW - i) * 0.5,
                                  (PANEL_SHADOW - i) * 0.5)
            shadow = QColor(0, 0, 0, int(9 * i / PANEL_SHADOW))
            painter.setBrush(shadow)
            painter.drawRoundedRect(layer, PANEL_RADIUS + i * 0.5,
                                    PANEL_RADIUS + i * 0.5)
    def _in_title_bar(self, local_pos):
        return local_pos.y() <= PANEL_SHADOW + self._head_height

    def title_drag_press(self, local, event):
        """卡片内按下：只有标题栏区域拖动窗口，其余交还子控件（选字/点按钮）"""
        if event.button() == Qt.LeftButton and self._in_title_bar(local):
            # 复用 Mixin 的拖动位移计算，行为与取景框整体拖动完全一致
            self._drag_origin = event.globalPos()
            event.accept()
            return True
        return False

    def mousePressEvent(self, event):
        if self.edge_mouse_press(event):
            return
        self.title_drag_press(event.pos(), event)

    def mouseMoveEvent(self, event):
        self.edge_mouse_move(event)

    def mouseReleaseEvent(self, event):
        self.edge_mouse_release(event)

    def export_txt(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "导出识别结果", "ocr_result.txt", "Text Files (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as fp:
                fp.write(self.text_edit.toPlainText())
                fp.write("\n\n—— %s ——\n" % WATERMARK)


class OcrWorker(QThread):
    """后台识别线程：OCR 单次耗时几百毫秒以上，放主线程会让界面卡住"""
    finished_ocr = pyqtSignal(list, float)   # (文本行列表, 耗时秒)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.running = False

    def start(self, *args, **kwargs):
        """在进入 run() 之前就把 running 置位，避免定时器重复启动同一线程"""
        if self.running:
            return
        self.running = True
        super().start(*args, **kwargs)

    def run(self):
        img = self.image
        if img is None:
            self.running = False
            return
        try:
            t0 = time.perf_counter()
            lines = recognize(img)
            elapsed = time.perf_counter() - t0
            self.finished_ocr.emit(lines, elapsed)
        except Exception as e:
            print(f"[OCR] 识别失败: {e}")
        finally:
            self.running = False


def main():
    # Windows DPI 缩放适配：让 Qt 坐标与物理像素 1:1 对应，
    # 否则在高分屏缩放(125%/150%)下截屏区域会与边框错位
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()
    app = QApplication(sys.argv)
    app.setStyleSheet(PANEL_QSS)

    frame = FrameWindow()
    # 初始位置：屏幕右侧偏中
    screen = app.primaryScreen().availableGeometry()
    outer = DEFAULT_SIZE + FRAME_GRIP * 2
    frame.move(screen.width() - outer - 100,
               (screen.height() - outer) // 2)
    frame.show()

    panel = ResultPanel(frame)
    # resize 要算上四周的投影留白，卡片本体才是 420x400
    panel.resize(420 + PANEL_SHADOW * 2, 400 + PANEL_SHADOW * 2)
    # 定位基准是识别区（可见框线），不是含缩放留白的窗口外沿
    reg = frame.region()
    panel.move(max(10, reg.left() - 440 - PANEL_SHADOW),
               max(10, reg.top() - PANEL_SHADOW))
    panel.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()