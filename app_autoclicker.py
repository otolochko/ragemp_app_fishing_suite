import sys
import logging
try:
    import config_runtime as cfg
except Exception:
    class cfg:
        RESOLUTION_PROFILE = "1920x1080"
        LOG_LEVEL = logging.INFO
logging.basicConfig(level=getattr(cfg, 'LOG_LEVEL', logging.INFO))
logger = logging.getLogger(__name__)
import time
import threading
import ctypes
from ctypes import wintypes
import numpy as np
import mss
import pyautogui
import keyboard
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets

# High-DPI до створення QApplication
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

# -------------------- Налаштування --------------------
POLL_MS = 40         # ~25 Гц глобально (ЛКМ/ПКМ оверлеї)
CATCH_POLL_MS = 15   # зелений піксель — частіше (~66 Гц)

# ЛКМ/ПКМ — миттєвий тригер по зміні середнього кольору
LEFT_DELTA  = 25
RIGHT_DELTA = 25
LEFT_DEBOUNCE_MS  = 0
RIGHT_DEBOUNCE_MS = 0

# --------- ЗЕЛЕНИЙ (CATCH) ---------
CATCH_GREEN_LIST = np.array([
    [90, 208, 0],
    [91, 208, 0],
    [91, 209, 0],
    [89, 207, 0],
], dtype=np.float32)

# HSV-порог для яскраво-зеленого (~94°)
CATCH_H_MIN = 85   # градуси
CATCH_H_MAX = 105
CATCH_S_MIN = 0.85
CATCH_V_MIN = 0.60

# Параметри детекції
CATCH_TOLERANCE     = 32.0
CATCH_MIN_RATIO     = 0.0005
CATCH_MIN_PIX       = 3
CATCH_HOLD_FR       = 2
CATCH_HOLD_OFF_FR   = 6

# Послідовність після catch
CATCH_PRE_ESC_DELAY = 2   # чекати ПЕРЕД Esc (твій параметр)
ESC_TO_ONE_DELAY    = 0.7   # між Esc та "1"

# Failsafe: якщо зелений не з’являється довше N секунд
GREEN_TIMEOUT_SEC   = 90.0  # твій параметр

# --------- ЧЕРВОНИЙ (Перегружений інвентар) ---------
INV_RED_RGB   = np.array([254, 76, 40], dtype=np.float32)
INV_TOLERANCE = 18.0
INV_MIN_RATIO = 0.003
INV_MIN_PIX   = 4
INV_HOLD_FR   = 2

FADE_ALPHA   = 80
BORDER_WIDTH = 2
DEFAULT_HOTKEY = "f8"
# ------------------------------------------------------

# ---------- Надійна відправка клавіш (SendInput + сканкоди) ----------
user32 = ctypes.windll.user32
KEYEVENTF_KEYUP     = 0x0002
KEYEVENTF_SCANCODE  = 0x0008
SCAN_ESC = 0x01
SCAN_1   = 0x02
ULONG_PTR = wintypes.WPARAM

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk",wintypes.WORD),("wScan",wintypes.WORD),("dwFlags",wintypes.DWORD),
                ("time",wintypes.DWORD),("dwExtraInfo",ULONG_PTR)]
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx",wintypes.LONG),("dy",wintypes.LONG),("mouseData",wintypes.DWORD),
                ("dwFlags",wintypes.DWORD),("time",wintypes.DWORD),("dwExtraInfo",ULONG_PTR)]
class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg",wintypes.DWORD),("wParamL",wintypes.WORD),("wParamH",wintypes.WORD)]
class INPUT_UNION(ctypes.Union):
    _fields_ = [("ki",KEYBDINPUT),("mi",MOUSEINPUT),("hi",HARDWAREINPUT)]
class INPUT(ctypes.Structure):
    _fields_ = [("type",wintypes.DWORD),("ii",INPUT_UNION)]

def send_scan(scan_code, key_up=False):
    inp = INPUT()
    inp.type = 1
    inp.ii.ki = KEYBDINPUT(0, scan_code, KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if key_up else 0), 0, ULONG_PTR(0))
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

def press_top_1():
    send_scan(SCAN_1, False); time.sleep(0.01); send_scan(SCAN_1, True)
    try: pyautogui.press("1")
    except: pass
    try: keyboard.send("1")
    except: pass

def press_escape():
    send_scan(SCAN_ESC, False); time.sleep(0.01); send_scan(SCAN_ESC, True)
    try: pyautogui.press("esc")
    except: pass
    try: keyboard.send("esc")
    except: pass
# ---------------------------------------------------------------------

def mean_rgb(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return rgb.reshape(-1, 3).mean(axis=0)

def color_distance(a, b):
    return float(np.linalg.norm(a - b))

# ---- DPI scale для MSS (Windows) ----
def get_screen_scale():
    scr = QtWidgets.QApplication.primaryScreen()
    if scr is None:
        return 1.0
    try:
        logical_dpi = scr.logicalDotsPerInch()
        scale = logical_dpi / 96.0
        return float(scale if scale > 0 else 1.0)
    except Exception:
        return 1.0

# --------- Оверлей для зміни (ЛКМ/ПКМ) ---------
class RegionOverlay(QtWidgets.QWidget):
    colorChanged = QtCore.pyqtSignal(str)

    def __init__(self, which, color, *, trigger_delta=30, debounce_ms=0, parent=None):
        super().__init__(parent)
        self.which = which
        self.border_color = color
        self.trigger_delta = trigger_delta
        self.debounce_ms = debounce_ms

        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.Tool | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.setWindowFlag(QtCore.Qt.WindowDoesNotAcceptFocus, True)

        self.setMouseTracking(True)
        self.resize(120, 120)

        self._drag_pos = None
        self._resizing = False
        self._resize_margin = 10

        self._last_mean = None
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)

        self._last_emit_ts = 0.0

    @QtCore.pyqtSlot()
    def start(self): self._timer.start(POLL_MS)

    @QtCore.pyqtSlot()
    def stop(self):
        self._timer.stop()
        self._last_mean = None

    def paintEvent(self, e):
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = self.rect().adjusted(1, 1, -1, -1)
        fill = QtGui.QColor(self.border_color); fill.setAlpha(FADE_ALPHA)
        p.fillRect(r, fill); p.setPen(QtGui.QPen(self.border_color, BORDER_WIDTH)); p.drawRect(r)
        p.setPen(QtCore.Qt.white); f = p.font(); f.setBold(True); p.setFont(f)
        p.drawText(r.adjusted(6,6,-6,-6), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, f"{self.which.upper()} [change]")

    # drag/resize
    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            if self._on_resize_zone(e.pos()): self._resizing = True
            else: self._drag_pos = e.globalPos() - self.frameGeometry().topLeft()
            e.accept()
    def mouseMoveEvent(self, e):
        if self._drag_pos is not None:
            self.move(e.globalPos() - self._drag_pos); e.accept()
        elif self._resizing:
            self.resize(max(40, e.pos().x()), max(40, e.pos().y())); e.accept()
        else:
            cur = QtCore.Qt.SizeFDiagCursor if self._on_resize_zone(e.pos()) else QtCore.Qt.ArrowCursor
            self.setCursor(cur)
    def mouseReleaseEvent(self, e):
        self._drag_pos = None; self._resizing = False
    def _on_resize_zone(self, pos):
        return (self.width()-pos.x() <= self._resize_margin and self.height()-pos.y() <= self._resize_margin)

    def _grab_region_bgr(self):
        g = self.frameGeometry(); x,y,w,h = g.x(), g.y(), g.width(), g.height()
        if w <= 0 or h <= 0: return None
        scale = get_screen_scale()
        left = int(round(x * scale))
        top  = int(round(y * scale))
        width  = int(round(w * scale))
        height = int(round(h * scale))
        with mss.mss() as sct:
            try: shot = sct.grab({"left":left,"top":top,"width":width,"height":height})
            except Exception: return None
        return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

    def _tick(self):
        img = self._grab_region_bgr()
        if img is None: return
        m = mean_rgb(img)
        if self._last_mean is None:
            self._last_mean = m; return
        d = color_distance(m, self._last_mean)
        now = time.time()
        if d >= self.trigger_delta:
            if (now - self._last_emit_ts)*1000.0 >= self.debounce_ms:
                self._last_emit_ts = now
                self.colorChanged.emit(self.which)
        self._last_mean = m

# --------- Оверлей детекції кольору (HSV + RGB) ----------
class ColorDetectOverlay(QtWidgets.QWidget):
    detected = QtCore.pyqtSignal(str)

    def __init__(self, key, color, target_rgbs, tolerance, label_text,
                 min_ratio=0.005, min_pixels=5, hold_on_frames=2, hold_off_frames=6,
                 poll_ms=POLL_MS, use_hsv=False, hsv_params=None, parent=None):
        super().__init__(parent)
        self.key = key
        self.border_color = color
        self.target_rgbs = np.array(target_rgbs, dtype=np.float32).reshape(-1, 3)
        self.tolerance = float(tolerance)
        self.label_text = label_text
        self.min_ratio = float(min_ratio)
        self.min_pixels = int(min_pixels)

        self.hold_on_frames  = int(hold_on_frames)
        self.hold_off_frames = int(hold_off_frames)
        self._armed = True
        self._on_cnt = 0
        self._off_cnt = 0

        self._poll_ms = int(poll_ms)
        self._use_hsv = bool(use_hsv)
        self._hsv = hsv_params or {}

        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.Tool | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.setWindowFlag(QtCore.Qt.WindowDoesNotAcceptFocus, True)

        self.setMouseTracking(True)
        self.resize(40, 40)

        self._drag_pos = None
        self._resizing = False
        self._resize_margin = 10

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)

    @QtCore.pyqtSlot()
    def start_sensor(self):
        self._armed = True; self._on_cnt = 0; self._off_cnt = 0
        self._timer.start(self._poll_ms)

    @QtCore.pyqtSlot()
    def stop_sensor(self):
        self._timer.stop()
        self._armed = True; self._on_cnt = 0; self._off_cnt = 0

    def paintEvent(self, e):
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = self.rect().adjusted(1, 1, -1, -1)
        fill = QtGui.QColor(self.border_color); fill.setAlpha(FADE_ALPHA)
        p.fillRect(r, fill); p.setPen(QtGui.QPen(self.border_color, BORDER_WIDTH)); p.drawRect(r)
        p.setPen(QtCore.Qt.white); f = p.font(); f.setBold(True); p.setFont(f)
        state = "ARMED" if self._armed else "COOLDOWN"
        p.drawText(r.adjusted(6,6,-6,-6), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, f"{self.label_text} [{state}]")

    # drag/resize
    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            if self._on_resize_zone(e.pos()): self._resizing = True
            else: self._drag_pos = e.globalPos() - self.frameGeometry().topLeft()
            e.accept()
    def mouseMoveEvent(self, e):
        if self._drag_pos is not None:
            self.move(e.globalPos() - self._drag_pos); e.accept()
        elif self._resizing:
            self.resize(max(6, e.pos().x()), max(6, e.pos().y())); e.accept()
        else:
            cur = QtCore.Qt.SizeFDiagCursor if self._on_resize_zone(e.pos()) else QtCore.Qt.ArrowCursor
            self.setCursor(cur)
    def mouseReleaseEvent(self, e):
        self._drag_pos = None; self._resizing = False
    def _on_resize_zone(self, pos):
        return (self.width()-pos.x() <= self._resize_margin and self.height()-pos.y() <= self._resize_margin)

    def _grab_region_bgr(self):
        g = self.frameGeometry(); x,y,w,h = g.x(), g.y(), g.width(), g.height()
        if w <= 0 or h <= 0: return None
        scale = get_screen_scale()
        left = int(round(x * scale))
        top  = int(round(y * scale))
        width  = int(round(w * scale))
        height = int(round(h * scale))
        with mss.mss() as sct:
            try: shot = sct.grab({"left":left,"top":top,"width":width,"height":height})
            except Exception: return None
        return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

    def _tick(self):
        img = self._grab_region_bgr()
        if img is None: return

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        h, w, _ = rgb.shape
        total = h * w

        if self._use_hsv:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            H = (hsv[:, :, 0] * 2.0)
            S = hsv[:, :, 1] / 255.0
            V = hsv[:, :, 2] / 255.0

            hmin = float(self._hsv.get("hmin", CATCH_H_MIN))
            hmax = float(self._hsv.get("hmax", CATCH_H_MAX))
            smin = float(self._hsv.get("smin", CATCH_S_MIN))
            vmin = float(self._hsv.get("vmin", CATCH_V_MIN))

            hsv_mask = (H >= hmin) & (H <= hmax) & (S >= smin) & (V >= vmin)
        else:
            hsv_mask = np.zeros((h, w), dtype=bool)

        diffs = rgb[:, :, None, :] - self.target_rgbs[None, None, :, :]
        dists = np.linalg.norm(diffs, axis=3)
        mind  = np.min(dists, axis=2)
        rgb_mask = (mind <= self.tolerance)

        mask = hsv_mask | rgb_mask

        count = int(np.count_nonzero(mask))
        ratio = count / max(1, total)
        present = (count >= self.min_pixels) and (ratio >= self.min_ratio)

        if self.key == "catch_green":
            # Replaced verbose print with optional logging; keep silent by default.
            try:
                import logging
                logging.getLogger(__name__).debug(
                    "[CATCH][dbg] green_pixels=%s, ratio=%.5f", count, ratio
                )
            except Exception:
                # Swallow any logging import/config errors to avoid noisy stdout.
                pass

        if self._armed:
            if present:
                self._on_cnt += 1
                if self._on_cnt >= self.hold_on_frames:
                    self._armed = False
                    self._on_cnt = 0; self._off_cnt = 0
                    self.detected.emit(self.key)
            else:
                self._on_cnt = 0
        else:
            if not present:
                self._off_cnt += 1
                if self._off_cnt >= self.hold_off_frames:
                    self._armed = True
                    self._off_cnt = 0
            else:
                self._off_cnt = 0

# --------------------- Головне вікно ---------------------
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fishing Helper")
        self.setFixedSize(420, 470)
        self.auto_on = False
        self.hotkey = DEFAULT_HOTKEY
        self.hotkey_handle = None
        self.await_hotkey = False

        # чи біжить послідовність
        self._sequence_running = False

        # Таймер 15с контролю пошуку зеленого
        self._green_timeout_timer = QtCore.QTimer(self)
        self._green_timeout_timer.setSingleShot(True)
        self._green_timeout_timer.timeout.connect(self._green_search_timed_out)

        # ОКРЕМІ таймери для ESC та "1" (щоб можна було їх стопити)
        self._esc_timer = QtCore.QTimer(self)
        self._esc_timer.setSingleShot(True)
        self._esc_timer.timeout.connect(self._press_esc)

        self._one_timer = QtCore.QTimer(self)
        self._one_timer.setSingleShot(True)
        self._one_timer.timeout.connect(self._press_one_and_resume)

        # --------- UI ---------
        self.btn_toggle = QtWidgets.QPushButton("Toggle Auto-Click")
        self.btn_regions = QtWidgets.QPushButton("Toggle Regions")
        self.btn_lock = QtWidgets.QPushButton("Lock (click-through): ON")
        self.lbl_status = QtWidgets.QLabel("Auto-click: OFF")
        self.lbl_hotkey = QtWidgets.QLabel(f"Current hotkey: {self.hotkey}")
        self.btn_change_hotkey = QtWidgets.QPushButton("Change Hotkey")

        self.res_label = QtWidgets.QLabel("Resolution:")
        self.res_combo = QtWidgets.QComboBox()
        self.res_combo.addItems(["2560x1440", "1920x1080", "1680x1050"])
        # Default from config
        try:
            default_res = getattr(cfg, 'RESOLUTION_PROFILE', '1920x1080') or '1920x1080'
        except Exception:
            default_res = '1920x1080'
        if default_res not in ["2560x1440", "1920x1080", "1680x1050"]:
            default_res = '1920x1080'
        self.res_combo.setCurrentText(default_res)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(self.res_label, 0, 0)
        grid.addWidget(self.res_combo, 0, 1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(grid)
        layout.addWidget(self.btn_toggle)
        layout.addWidget(self.btn_regions)
        layout.addWidget(self.btn_lock)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.lbl_hotkey)
        layout.addWidget(self.btn_change_hotkey, alignment=QtCore.Qt.AlignCenter)

        # --------- Overlays ---------
        self.left_overlay = RegionOverlay("left", QtGui.QColor(66,194,255),
                                          trigger_delta=LEFT_DELTA, debounce_ms=LEFT_DEBOUNCE_MS)
        self.right_overlay = RegionOverlay("right", QtGui.QColor(255,102,102),
                                           trigger_delta=RIGHT_DELTA, debounce_ms=RIGHT_DEBOUNCE_MS)

        self.catch_green_overlay = ColorDetectOverlay(
            "catch_green", QtGui.QColor(120,220,120),
            CATCH_GREEN_LIST, CATCH_TOLERANCE, "CATCH [green]",
            min_ratio=CATCH_MIN_RATIO, min_pixels=CATCH_MIN_PIX,
            hold_on_frames=CATCH_HOLD_FR, hold_off_frames=CATCH_HOLD_OFF_FR,
            poll_ms=CATCH_POLL_MS,
            use_hsv=True,
            hsv_params=dict(hmin=CATCH_H_MIN, hmax=CATCH_H_MAX, smin=CATCH_S_MIN, vmin=CATCH_V_MIN)
        )

        self.inv_red_overlay = ColorDetectOverlay(
            "inv_red", QtGui.QColor(220,80,80),
            [INV_RED_RGB], INV_TOLERANCE, "INVENTORY [red]",
            min_ratio=INV_MIN_RATIO, min_pixels=INV_MIN_PIX,
            hold_on_frames=INV_HOLD_FR, hold_off_frames=2
        )

        # Сигнали
        self.left_overlay.colorChanged.connect(self._do_action)
        self.right_overlay.colorChanged.connect(self._do_action)
        self.catch_green_overlay.detected.connect(self._color_detected)
        self.inv_red_overlay.detected.connect(self._color_detected)

        # Кнопки
        self.btn_toggle.clicked.connect(self.toggle_autoclick)
        self.btn_regions.clicked.connect(self.toggle_regions)
        self.btn_change_hotkey.clicked.connect(self.change_hotkey)
        self.btn_lock.clicked.connect(self.toggle_clickthrough)
        self.res_combo.currentIndexChanged.connect(self._apply_resolution)

        self._register_hotkey(self.hotkey)

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.0

        self._clickthrough = True
        self._apply_clickthrough(True)

        self._apply_resolution()

    # --- Таймаут пошуку зеленого ---
    def _arm_green_timeout(self):
        self._green_timeout_timer.stop()
        self._green_timeout_timer.start(int(GREEN_TIMEOUT_SEC * 1000))
        logger.warning("[CATCH] Green search timeout (%.1fs). Forcing ESC -> 1.", GREEN_TIMEOUT_SEC)

    def _disarm_green_timeout(self):
        if self._green_timeout_timer.isActive():
            self._green_timeout_timer.stop()
            print("[CATCH] Green search timeout disarmed.")

    # --- click-through ---
    def _apply_clickthrough(self, enabled: bool):
        self._clickthrough = enabled
        for ov in (self.left_overlay, self.right_overlay,
                   self.catch_green_overlay, self.inv_red_overlay):
            ov.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, enabled)
        self.btn_lock.setText(f"Lock (click-through): {'ON' if enabled else 'OFF'}")

    @QtCore.pyqtSlot()
    def toggle_clickthrough(self):
        self._apply_clickthrough(not self._clickthrough)

    # --- хоткей ---
    def _hotkey_action(self):
        QtCore.QMetaObject.invokeMethod(self, "toggle_autoclick", QtCore.Qt.QueuedConnection)

    def _register_hotkey(self, key):
        if self.hotkey_handle: keyboard.remove_hotkey(self.hotkey_handle)
        self.hotkey_handle = keyboard.add_hotkey(key, self._hotkey_action)
        self.lbl_hotkey.setText(f"Current hotkey: {key}")

    def change_hotkey(self):
        if self.await_hotkey: return
        self.await_hotkey = True
        self.lbl_hotkey.setText("Press new hotkey...")
        threading.Thread(target=self._wait_hotkey_worker, daemon=True).start()

    def _wait_hotkey_worker(self):
        try: newk = keyboard.read_key(suppress=False)
        except Exception: newk = None
        if newk:
            self.hotkey = newk
            QtCore.QMetaObject.invokeMethod(self, "_apply_new_hotkey",
                                            QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, newk))
        else:
            QtCore.QMetaObject.invokeMethod(self, "_apply_new_hotkey",
                                            QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, self.hotkey))

    @QtCore.pyqtSlot(str)
    def _apply_new_hotkey(self, newk):
        self._register_hotkey(newk); self.await_hotkey = False

    # --- логіка ---
    @QtCore.pyqtSlot()
    def toggle_autoclick(self):
        self.auto_on = not self.auto_on
        if self.auto_on:
            # Повний «чистий» старт
            self._sequence_running = False
            self._esc_timer.stop()
            self._one_timer.stop()
            print("[STATE] Auto-click: ON")
            self.lbl_status.setText("Auto-click: ON")
            self.left_overlay.start()
            self.right_overlay.start()
            self.catch_green_overlay.start_sensor()
            self.inv_red_overlay.start_sensor()
            self._apply_clickthrough(True)
            self._arm_green_timeout()
        else:
            print("[STATE] Auto-click: OFF")
            self.lbl_status.setText("Auto-click: OFF")
            self.left_overlay.stop()
            self.right_overlay.stop()
            self.catch_green_overlay.stop_sensor()
            self.inv_red_overlay.stop_sensor()
            self._apply_clickthrough(False)
            self._disarm_green_timeout()
            # Скинути стан і таймери
            self._sequence_running = False
            self._esc_timer.stop()
            self._one_timer.stop()

    def toggle_regions(self):
        visible = self.left_overlay.isVisible()
        for ov in (self.left_overlay, self.right_overlay,
                   self.catch_green_overlay, self.inv_red_overlay):
            ov.hide() if visible else ov.show()

    def _scale_coords(self, x, y, from_res, to_res):
        fx = to_res[0] / from_res[0]
        fy = to_res[1] / from_res[1]
        return int(round(x * fx)), int(round(y * fy))

    def _apply_resolution(self):
        res_text = self.res_combo.currentText()
        if res_text == "2560x1440":
            base = (2560,1440); target = (2560,1440)
        elif res_text == "1920x1080":
            base = (2560,1440); target = (1920,1080)
        else:  # 1680x1050
            base = (2560,1440); target = (1680,1050)

        def place(widget, x, y, w, h):
            sx, sy = self._scale_coords(x, y, base, target)
            sw, sh = self._scale_coords(w, h, base, target)
            widget.setGeometry(sx, sy, sw, sh)

        place(self.left_overlay,  228-50, 1201-50, 100, 100)
        place(self.right_overlay, 2324-50, 1194-50, 100, 100)
        place(self.catch_green_overlay, 1630-20, 134-20, 40, 40)

        x1, x2, y = 1096-10, 1461+10, 1316
        height = 20
        place(self.inv_red_overlay, x1, y - height//2, (x2 - x1), height)

        self.current_resolution = target
        logger.info("[UI] Resolution applied: %s", res_text)

    @QtCore.pyqtSlot(str)
    def _do_action(self, which: str):
        if not self.auto_on:
            return
        def _run():
            try:
                if which == "left":
                    pyautogui.click(button="left")
                elif which == "right":
                    pyautogui.click(button="right")
            except Exception:
                pass
        threading.Thread(target=_run, daemon=True).start()

    # ---- ESC → 0.7s → "1" через керовані таймери ----
    def _press_esc(self):
        if not self.auto_on:
            return
        logger.info("[CATCH] Pressing ESC")
        press_escape()
        self.left_overlay.stop()
        self.right_overlay.stop()
        # плануємо натискання "1"
        self._one_timer.stop()
        self._one_timer.start(int(ESC_TO_ONE_DELAY*1000))

    def _press_one_and_resume(self):
        if not self.auto_on:
            return
        logger.info("[CATCH] Pressing 1")
        press_top_1()
        self.left_overlay.start()
        self.right_overlay.start()
        self.catch_green_overlay.start_sensor()
        self._arm_green_timeout()
        self._sequence_running = False
        logger.info("[CATCH] Sequence finished, sensors resumed.")

    # ---- Таймаут пошуку зеленого ----
    def _green_search_timed_out(self):
        if not self.auto_on or self._sequence_running:
            return
        logger.warning("[CATCH] Green search timeout (%.1fs). Forcing ESC -> 1.", GREEN_TIMEOUT_SEC)
        self.catch_green_overlay.stop_sensor()
        # очистимо можливі старі відкладені натискання
        self._esc_timer.stop()
        self._one_timer.stop()
        # одразу ESC
        self._esc_timer.start(0)

    @QtCore.pyqtSlot(str)
    def _color_detected(self, key: str):
        if not self.auto_on:
            return

        if key == "inv_red":
            print("[INVENTORY] Red pixel detected → pressing ESC and stopping...")
            def _inv_stop():
                try: press_escape()
                finally:
                    QtCore.QMetaObject.invokeMethod(self, "_stop_all", QtCore.Qt.QueuedConnection)
            threading.Thread(target=_inv_stop, daemon=True).start()
            return

        if key == "catch_green":
            if self._sequence_running:
                return
            self._sequence_running = True

            print("[CATCH] Green pixel detected! Stop green sensor and schedule ESC in "
                  f"{CATCH_PRE_ESC_DELAY:.1f}s.")
            self.catch_green_overlay.stop_sensor()
            self._disarm_green_timeout()

            # плануємо ESC через CATCH_PRE_ESC_DELAY
            self._esc_timer.stop()
            self._esc_timer.start(int(CATCH_PRE_ESC_DELAY*1000))

    @QtCore.pyqtSlot()
    def _stop_all(self):
        self.auto_on = False
        self.lbl_status.setText("Auto-click: OFF (inventory red)")
        self.left_overlay.stop()
        self.right_overlay.stop()
        self.catch_green_overlay.stop_sensor()
        self.inv_red_overlay.stop_sensor()
        self._apply_clickthrough(False)
        self._disarm_green_timeout()
        # критично: зупинити відкладені дії та скинути стан
        self._esc_timer.stop()
        self._one_timer.stop()
        self._sequence_running = False
        print("[STATE] Auto-click: OFF (inventory red)")

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
