# Авто-адаптація під 2560×1440 (база) та 1920×1080 (та інші кратні формати).
# Координати задані в базових пікселях 2560×1440 і масштабуються під поточний екран.
#
# Гарячі клавіші:
#   F8 — старт/стоп
#   F2 — калібрувати пустий слот (3 семпли)
#   F6 — діагностика
#   F7 — очистити пам’ять FULL
#   F3 — TURBO (швидше перетягування та перевірки)
#
# FULL-пам’ять — «липка» за замовчанням: не забуває автоматично між циклами.

import time
import threading
from typing import List, Tuple, Optional, Set
from collections import deque

import ctypes
import pyautogui
import keyboard
import numpy as np
import mss
import cv2
import math

# ========= БАЗОВА РОЗДІЛЬНА ДЛЯ КООРДИНАТ =========
BASE_W, BASE_H = 2560, 1440

def _screen_size() -> Tuple[int,int]:
    try:
        return pyautogui.size()
    except Exception:
        # fallback: вважаємо базу
        return (BASE_W, BASE_H)

def _scale_xy(x: int, y: int) -> Tuple[int,int]:
    sw, sh = _screen_size()
    sx, sy = sw / BASE_W, sh / BASE_H
    return int(round(x * sx)), int(round(y * sy))

def _scale_points(arr: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    return [_scale_xy(x, y) for (x, y) in arr]

def _scale_len(v: int) -> int:
    sw, sh = _screen_size()
    # середній масштаб (для радіусів/патчів)
    s = (sw / BASE_W + sh / BASE_H) / 2.0
    return max(1, int(round(v * s)))

# ========= ГАРЯЧІ КЛАВІШІ =========
HOTKEY_TOGGLE     = 'f8'   # старт/стоп
HOTKEY_CALIB      = 'f2'   # калібрувати пустий слот (3 семпли)
HOTKEY_DIAG       = 'f6'   # діагностика
HOTKEY_CLEAR_FULL = 'f7'   # скинути пам'ять "повних" сумок у багажнику
HOTKEY_FAST       = 'f3'   # FAST MODE вкл/викл

# ========= ТАЙМІНГИ =========
# звичайний режим
PAUSE_SHORT = 0.06
PAUSE_DRAG  = 0.18
HOLD_DOWN   = 0.08

# FAST MODE
F_PAUSE_SHORT = 0.005
F_PAUSE_DRAG  = 0.02
F_HOLD_DOWN   = 0.01

# стабілізація вимірів
STABLE_N_SLOW = 3
STABLE_N_FAST = 1

pyautogui.PAUSE = 0.02
pyautogui.FAILSAFE = True

# ========= ЗОНА АНАЛІЗУ СЛОТА (база → масштабуємо нижче) =========
PATCH_BASE = 85       # в базових пікселях 2560×1440
INNER_MARGIN = 0

# ========= ПОРИГИ ДЕТЕКТОРА =========
MAD_H_THR = 22.0
MAD_S_THR = 42.0
LAP_THR   = 110.0
STDV_THR  = 45.0
LAP_BOOST  = 1.2
LAP_EXTRA  = 50.0
LAP_CAP    = 400.0
STDV_EXTRA = 8.0
STDV_CAP   = 15.0

SIM_THR_STRICT  = 0.88
SIM_THR_LOOSE   = 0.82
EDGE_THR_STRICT = 0.03
EDGE_THR_LOOSE  = 0.12

# ========= КОЛЬОРОВІ СИГНАТУРИ ІКОНКИ =========
ICON_COLORS_RGB = [
    (161, 75, 249),  # фіолетовий
    (12, 109, 209),  # темно-синій
]
HSV_H_TOL = 14.0
S_MIN     = 60.0
V_MIN     = 90.0
LAB_THR   = 20.0
MIN_AREA_BASE  = 48   # мін. площа кластера в базовому PATCH_BASE×PATCH_BASE
MIN_AREA_PIXELS = MIN_AREA_BASE  # перераховується у _apply_scaling()

VERBOSE = True

# ========= КООРДИНАТИ (БАЗА 2560×1440) =========
BAG_SLOT_BASE = (975, 352)

TRUNK_SLOTS_BASE: List[Tuple[int, int]] = [
    (2113,348),(2245,344),(2383,348),
    (1587,482),(1721,482),(1854,475),(1977,477),(2105,479),(2244,474),(2378,480),
    (1579,609),(1717,612),(1841,608),(1985,612),(2117,607),(2243,612),(2374,616),
    (1568,747),(1723,746),(1851,746),(1986,745),(2104,745),(2248,751),(2373,751),
    (1577,888),(1720,880),(1857,883),(1972,876),(2123,876),(2248,878),(2388,883),
    (1585,1012),(1726,1006),(1837,1006),(1975,1007),(2119,1013),(2229,1009),(2386,1012),
    (1576,1147),(1724,1148)
]

PRIORITY_ZONES_BASE: List[Tuple[int, int]] = [
    (181,676),(314,679),(450,679),(569,678),(716,673),
    (854,678),(985,678),(1119,680),(1258,674),(1383,676)
]

SECONDARY_ZONES_BASE: List[Tuple[int, int]] = [
    (173,876),(317,879),(448,879),(581,881),(720,881),
    (845,873),(979,877),(1118,878),(1257,879),(1375,881),
    (1384,999),(1255,1010),(1093,1013),(965,1011),(857,1013),
    (719,1018),(569,1015),(437,1013),(318,1016),(176,1012),
    (176,1150),(306,1149),(446,1146),(586,1150),(709,1149),
    (842,1147),(967,1151),(1108,1147),(1234,1150),(1379,1153)
]

# захищені точки у Основній сумці (не чіпати) — база
PROTECTED_POINTS_BASE: List[Tuple[int,int]] = [(175,878), (310,875)]
PROTECTED_RADIUS_BASE = 10  # px (база)

# ========= ЕТАЛОН ПУСТОГО =========
empty_ref_hsv: Optional[np.ndarray] = None
empty_ref_lap: Optional[float] = None
empty_ref_stdv: Optional[float] = None
empty_ref_tiny: Optional[np.ndarray] = None  # 16x16 розмитий вектор (L2-norm)

# попередньо прораховані цілі для кольорів іконки
TARGETS_HSV: Optional[List[np.ndarray]] = None
TARGETS_LAB: Optional[List[np.ndarray]] = None

# ========= СТАН =========
running = False
working_now = False
FAST_MODE = False
lock = threading.Lock()
main_bag_pos: Optional[Tuple[int,int]] = None

# 🧠 пам'ять повних сумок у багажнику
full_bag_slots: Set[Tuple[int,int]] = set()

# Якщо False — FULL-пам’ять не очищується автоматично між циклами
FULL_MEMORY_AUTO_FORGET = False

# Коротка історія останніх місць для Основної
_last_main_bag_spots = deque(maxlen=4)

def log(msg: str):
    if VERBOSE:
        print(msg)

# ========= МАСШТАБОВАНІ ГЛОБАЛИ (заповнюються в _apply_scaling) =========
PATCH = PATCH_BASE
PROTECTED_RADIUS = PROTECTED_RADIUS_BASE
BAG_SLOT: Tuple[int,int] = BAG_SLOT_BASE
TRUNK_SLOTS: List[Tuple[int,int]] = TRUNK_SLOTS_BASE.copy()
PRIORITY_ZONES: List[Tuple[int,int]] = PRIORITY_ZONES_BASE.copy()
SECONDARY_ZONES: List[Tuple[int,int]] = SECONDARY_ZONES_BASE.copy()
PROTECTED_POINTS: List[Tuple[int,int]] = PROTECTED_POINTS_BASE.copy()
_last_scaled_wh: Optional[Tuple[int,int]] = None  # щоб переприміняти скейл, якщо роздільну змінять на льоту

def _apply_scaling():
    """Перерахувати всі координати/розміри під поточний екран."""
    global PATCH, PROTECTED_RADIUS, BAG_SLOT, TRUNK_SLOTS, PRIORITY_ZONES, SECONDARY_ZONES, PROTECTED_POINTS
    global MIN_AREA_PIXELS, _last_scaled_wh

    sw, sh = _screen_size()
    PATCH = max(48, _scale_len(PATCH_BASE))  # не робимо занадто маленьким
    PROTECTED_RADIUS = max(6, _scale_len(PROTECTED_RADIUS_BASE))
    BAG_SLOT = _scale_xy(*BAG_SLOT_BASE)
    TRUNK_SLOTS = _scale_points(TRUNK_SLOTS_BASE)
    PRIORITY_ZONES = _scale_points(PRIORITY_ZONES_BASE)
    SECONDARY_ZONES = _scale_points(SECONDARY_ZONES_BASE)
    PROTECTED_POINTS = _scale_points(PROTECTED_POINTS_BASE)

    # Площа кластера іконки масштабується квадратично з PATCH
    scale_area = (PATCH / max(1, PATCH_BASE)) ** 2
    MIN_AREA_PIXELS = max(12, int(round(MIN_AREA_BASE * scale_area)))

    _last_scaled_wh = (sw, sh)
    print(f"🔧 Resolution detected: {sw}x{sh} | PATCH={PATCH} | RAD={PROTECTED_RADIUS} | MIN_AREA={MIN_AREA_PIXELS}")

def _ensure_scaling_current():
    """Якщо роздільну змінено (через VAR3), перепримінити скейл без рестарту."""
    sw, sh = _screen_size()
    global _last_scaled_wh
    if _last_scaled_wh != (sw, sh):
        _apply_scaling()

# Ініціальна прив’язка до поточної роздільної
_apply_scaling()

# ========= ШВИДКИЙ БЕКЕНД МИШІ (Windows SendInput) =========
user32 = ctypes.windll.user32

MOUSEEVENTF_MOVE       = 0x0001
MOUSEEVENTF_LEFTDOWN   = 0x0002
MOUSEEVENTF_LEFTUP     = 0x0004
MOUSEEVENTF_ABSOLUTE   = 0x8000

def _to_absolute(x, y):
    sw, sh = _screen_size()
    ax = int(x * 65535 / max(1, (sw - 1)))
    ay = int(y * 65535 / max(1, (sh - 1)))
    return ax, ay

def fast_move(x, y):
    ax, ay = _to_absolute(x, y)
    user32.mouse_event(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, ax, ay, 0, 0)

def fast_down():
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

def fast_up():
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

# ========= КОРИСНІ ФУНКЦІЇ ДЛЯ ЗОБРАЖЕНЬ =========
def grab_patch(x: int, y: int, size: int = PATCH) -> np.ndarray:
    half = size // 2
    monitor = {"top": y - half, "left": x - half, "width": size, "height": size}
    with mss.mss() as sct:
        img = np.array(sct.grab(monitor))
    return img[:, :, :3]  # BGR

def bgr2hsv(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

def laplacian_var(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(lap.var())

def center_roi(img: np.ndarray, margin: int = INNER_MARGIN) -> np.ndarray:
    if margin <= 0:
        return img
    h, w = img.shape[:2]
    return img[margin:h-margin, margin:w-margin]

def tiny_vector(gray_img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray_img, (9,9), 2.5)
    tiny = cv2.resize(blur, (16,16), interpolation=cv2.INTER_AREA).astype(np.float32)
    v = tiny.reshape(-1)
    n = np.linalg.norm(v) + 1e-6
    return v / n

def edge_density(gray_img: np.ndarray) -> float:
    g = cv2.GaussianBlur(gray_img, (5,5), 1.2)
    edges = cv2.Canny(g, 40, 100)
    return float((edges > 0).mean())

def is_near_protected(x: int, y: int) -> bool:
    for (px, py) in PROTECTED_POINTS:
        if math.hypot(x - px, y - py) <= PROTECTED_RADIUS:
            return True
    return False

def hue_diff(a: np.ndarray, b: float) -> np.ndarray:
    d = np.abs(a - b)
    return np.minimum(d, 180.0 - d)

def precompute_color_targets():
    global TARGETS_HSV, TARGETS_LAB
    TARGETS_HSV, TARGETS_LAB = [], []
    for (r, g, b) in ICON_COLORS_RGB:
        arr = np.uint8([[[b, g, r]]])  # BGR для OpenCV
        hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)[0,0].astype(np.float32) # type: ignore
        lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)[0,0].astype(np.float32) # type: ignore
        TARGETS_HSV.append(hsv)
        TARGETS_LAB.append(lab)

precompute_color_targets()

# ========= ДЕТЕКТОР ІКОНКИ =========
def icon_present_by_color(roi_bgr: np.ndarray) -> bool:
    """Повертає True, якщо в центрі слота є кластер «іконки» достатнього розміру."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    for thsv, tlab in zip(TARGETS_HSV, TARGETS_LAB):
        h0, s0, v0 = thsv

        # HSV-маска: близький H + достатні S і V
        dh = hue_diff(H, float(h0))
        mask_hsv = (dh <= HSV_H_TOL) & (S >= max(S_MIN, s0 - 20)) & (V >= max(V_MIN, v0 - 40))

        # Lab-маска: ΔE76
        d = lab - tlab
        dE = np.sqrt(np.sum(d*d, axis=2))
        mask_lab = dE <= LAB_THR

        mask = (mask_hsv | mask_lab).astype(np.uint8) * 255

        # Морфологія + компоненти
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]  # без фону
            if int(areas.max()) >= MIN_AREA_PIXELS:
                return True
    return False

# ========= КАЛІБРУВАННЯ ПУСТОГО =========
def calibrate_empty_from_cursor():
    _ensure_scaling_current()
    global empty_ref_hsv, empty_ref_lap, empty_ref_stdv, empty_ref_tiny
    x0, y0 = pyautogui.position()
    laps, stdvs, tinys = [], [], []
    hsv_keep = None
    for dx, dy in [(0,0),(2,1),(-2,-1)]:
        x, y = x0 + dx, y0 + dy
        patch_bgr = grab_patch(x, y, size=PATCH)
        hsv = bgr2hsv(patch_bgr).astype(np.float32)
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        roi_hsv  = center_roi(hsv)
        roi_gray = center_roi(gray)
        if hsv_keep is None:
            hsv_keep = hsv
        laps.append(laplacian_var(roi_gray))
        stdvs.append(float(roi_hsv[:,:,2].std()))
        tinys.append(tiny_vector(roi_gray))
    empty_ref_hsv  = hsv_keep
    empty_ref_lap  = float(np.median(laps))
    empty_ref_stdv = float(np.median(stdvs))
    v = np.mean(np.stack(tinys, axis=0), axis=0)
    empty_ref_tiny = v / (np.linalg.norm(v) + 1e-6)
    print(f"✅ Еталон збережено @({x0},{y0}) | lapMed={empty_ref_lap:.1f}, stdVMed={empty_ref_stdv:.1f}")

# ========= ДЕТЕКТОР ПУСТОГО СЛОТА =========
def is_empty_cell(x: int, y: int) -> bool:
    patch_bgr = grab_patch(x, y, size=PATCH)
    hsv = bgr2hsv(patch_bgr).astype(np.float32)
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

    roi_bgr  = center_roi(patch_bgr)
    roi_hsv  = center_roi(hsv)
    roi_gray = center_roi(gray)

    # 1) Якщо бачимо іконку — ЗАЙНЯТО
    if icon_present_by_color(roi_bgr):
        return False

    # 2) Анти-блюр пустоти
    V = roi_hsv[:,:,2]
    std_v = float(V.std())
    lap = laplacian_var(roi_gray)
    edens = edge_density(roi_gray)

    if empty_ref_hsv is None or empty_ref_tiny is None:
        return (lap < 0.7*LAP_THR) and (std_v < 0.7*STDV_THR) and (edens < 0.7*EDGE_THR_LOOSE)

    ref_center = center_roi(empty_ref_hsv)
    mad_h = float(np.mean(np.abs(roi_hsv[:,:,0] - ref_center[:,:,0])))
    mad_s = float(np.mean(np.abs(roi_hsv[:,:,1] - ref_center[:,:,1])))

    lap_thr_dyn  = min(max(LAP_THR,  empty_ref_lap  * LAP_BOOST + LAP_EXTRA), empty_ref_lap  + LAP_CAP)
    stdv_thr_dyn = min(max(STDV_THR, empty_ref_stdv + STDV_EXTRA),             empty_ref_stdv + STDV_CAP)

    sim = float(np.dot(tiny_vector(roi_gray), empty_ref_tiny))

    strict = (sim >= SIM_THR_STRICT) and (edens <= EDGE_THR_STRICT) and (lap <= lap_thr_dyn) \
             and (mad_h <= MAD_H_THR) and (mad_s <= MAD_S_THR) and (std_v <= stdv_thr_dyn)

    loose  = (sim >= SIM_THR_LOOSE)  and (edens <= EDGE_THR_LOOSE)  and (lap <= lap_thr_dyn) \
             and (std_v <= stdv_thr_dyn)

    return strict or loose

def is_filled_cell(x: int, y: int) -> bool:
    return not is_empty_cell(x, y)

# ========= СТАБІЛІЗАЦІЯ =========
def stable_check(fn, x: int, y: int, n: int = None, delay: float = 0.02) -> bool:
    if n is None:
        n = STABLE_N_FAST if FAST_MODE else STABLE_N_SLOW
    ok = 0
    for _ in range(n):
        if fn(x, y):
            ok += 1
        time.sleep(delay)
    return ok >= (n // 2 + 1)

def stable_empty(x: int, y: int, n: int = None, delay: float = 0.02) -> bool:
    return stable_check(is_empty_cell, x, y, n=n, delay=delay)

def stable_filled(x: int, y: int, n: int = None, delay: float = 0.02) -> bool:
    return stable_check(is_filled_cell, x, y, n=n, delay=delay)

# ========= КЕРУВАННЯ FAST MODE =========
def _with_fast_temporarily(off: bool):
    class _Ctx:
        def __enter__(self):
            self.prev = FAST_MODE
            if off and self.prev:
                toggle_fast()
        def __exit__(self, exc_type, exc, tb):
            if off and self.prev != FAST_MODE:
                toggle_fast()
    return _Ctx()

# ========= DRAG З ВЕРИФІКАЦІЄЮ =========
def drag_verified(src: Tuple[int,int], dst: Tuple[int,int], require_dst_empty: bool = True) -> bool:
    sx, sy = src; dx, dy = dst

    if is_near_protected(sx, sy):
        log(f"⤷ skip drag: src {src} у зоні захисту")
        return False
    if is_near_protected(dx, dy):
        log(f"⤷ skip drag: dst {dst} у зоні захисту")
        return False

    if src != BAG_SLOT and not stable_filled(sx, sy):
        log(f"⤷ skip drag: src {src} не зайнятий")
        return False
    if require_dst_empty and dst != BAG_SLOT and not stable_empty(dx, dy):
        log(f"⤷ skip drag: dst {dst} не пустий")
        return False

    if FAST_MODE:
        p_short, p_drag, p_hold = F_PAUSE_SHORT, F_PAUSE_DRAG, F_HOLD_DOWN
    else:
        p_short, p_drag, p_hold = PAUSE_SHORT, PAUSE_DRAG, HOLD_DOWN

    # Спроба 1: швидкий флік (SendInput)
    if FAST_MODE:
        try:
            log(f"⤷ drag(fast) {src} → {dst}")
            fast_move(sx, sy); time.sleep(p_short)
            fast_down();       time.sleep(p_hold)
            fast_move(dx, dy); time.sleep(p_drag)
            fast_up();         time.sleep(0.005)
            ok_src = (src == BAG_SLOT) or stable_empty(sx, sy)
            ok_dst = (dst == BAG_SLOT) or stable_filled(dx, dy)
            if ok_src or ok_dst:
                log("   ✔ підтверджено fast")
                return True
        except Exception as e:
            log(f"   ⚠ fast backend fail: {e}")

    # Спроба 2: швидкий pyautogui
    try:
        log(f"⤷ drag(pyauto) {src} → {dst}")
        pyautogui.moveTo(sx, sy, duration=(0 if FAST_MODE else p_short))
        time.sleep(p_short if not FAST_MODE else 0.0)
        pyautogui.mouseDown(); time.sleep(p_hold)
        pyautogui.moveTo(dx, dy, duration=(0 if FAST_MODE else p_drag))
        pyautogui.mouseUp(); time.sleep(0.01 if FAST_MODE else 0.08)

        ok_src = (src == BAG_SLOT) or stable_empty(sx, sy)
        ok_dst = (dst == BAG_SLOT) or stable_filled(dx, dy)
        if ok_src or ok_dst:
            log("   ✔ підтверджено pyauto")
            return True
    except Exception as e:
        log(f"   ⚠ pyautogui fail: {e}")

    # Спроба 3: повільний fallback
    try:
        log(f"   ↻ retry(slow) {src}→{dst}")
        pyautogui.moveTo(sx, sy, duration=0.05); time.sleep(0.03)
        pyautogui.mouseDown(); time.sleep(max(p_hold, 0.06))
        pyautogui.moveTo(dx, dy, duration=0.15); time.sleep(0.03)
        pyautogui.mouseUp(); time.sleep(0.1)

        ok_src = (src == BAG_SLOT) or stable_empty(sx, sy, n=STABLE_N_SLOW)
        ok_dst = (dst == BAG_SLOT) or stable_filled(dx, dy, n=STABLE_N_SLOW)
        log(f"   {'✔' if (ok_src or ok_dst) else '✖'} після ретраю")
        return ok_src or ok_dst
    except Exception as e:
        log(f"   ❌ fallback fail: {e}")
        return False

# ========= ПОШУК ПУСТОГО/ПОВНОГО =========
def first_empty(coords: List[Tuple[int,int]], exclude: Set[Tuple[int,int]] = set()) -> Optional[Tuple[int,int]]:
    for (x, y) in coords:
        if is_near_protected(x, y):
            continue
        if (x, y) in exclude:
            continue
        if is_empty_cell(x, y):
            return (x, y)
    return None

def first_filled(coords: List[Tuple[int,int]], exclude: Set[Tuple[int,int]] = set()) -> Optional[Tuple[int,int]]:
    for (x, y) in coords:
        if is_near_protected(x, y):
            continue
        if (x, y) in exclude:
            continue
        if not is_empty_cell(x, y):
            return (x, y)
    return None

# ========= СТАН ОСНОВНОЇ СУМКИ =========
def main_has_any_priority() -> bool:
    return first_filled(PRIORITY_ZONES) is not None

def main_has_any_secondary() -> bool:
    return first_filled(SECONDARY_ZONES) is not None

def main_is_clean() -> bool:
    return (not main_has_any_priority()) and (not main_has_any_secondary())

# ========= SECONDARY → PRIORITY у Основній =========
def refill_priority_from_secondary_in_main_bag() -> None:
    moves = 0
    MAX_MOVES = 200
    bad_pr_targets: Set[Tuple[int,int]] = set()
    while running and moves < MAX_MOVES:
        pr_empty = first_empty(PRIORITY_ZONES, exclude=bad_pr_targets)
        if pr_empty is None:
            log("   ✔ Пріоритетні у сумці заповнені / нема куди класти")
            return
        sec_filled = first_filled(SECONDARY_ZONES)
        if sec_filled is None:
            log("   ℹ Другорядні у сумці порожні")
            return
        if not stable_filled(*sec_filled):
            continue
        if not stable_empty(*pr_empty):
            bad_pr_targets.add(pr_empty); continue
        if drag_verified(sec_filled, pr_empty, require_dst_empty=True):
            moves += 1
        else:
            bad_pr_targets.add(pr_empty); moves += 1

# ========= ПАМ’ЯТЬ ПОВНИХ СУМОК =========
def clear_full_bag_memory():
    full_bag_slots.clear()
    print("🧹 Пам'ять повних сумок очищено.")

def maybe_forget_full_if_now_empty(x: int, y: int):
    if not FULL_MEMORY_AUTO_FORGET:
        return
    if (x, y) in full_bag_slots and is_empty_cell(x, y):
        full_bag_slots.discard((x, y))
        log(f"🔄 Слот {(x,y)} більше не повний — розблокував.")

def compact_full_bag_memory() -> None:
    if not FULL_MEMORY_AUTO_FORGET:
        return
    stale = []
    for (x, y) in list(full_bag_slots):
        if is_empty_cell(x, y):
            stale.append((x, y))
    for s in stale:
        full_bag_slots.discard(s)

# ========= УТИЛІТИ ДЛЯ ОСНОВНОЇ СУМКИ/ПОВЕРНЕННЯ =========
def take_main_bag_back_to_hands() -> None:
    global main_bag_pos
    if main_bag_pos is None:
        return
    x, y = main_bag_pos
    with _with_fast_temporarily(off=True):
        if not is_empty_cell(x, y):
            if drag_verified((x, y), BAG_SLOT, require_dst_empty=False):
                time.sleep(0.08)
    main_bag_pos = None

def return_bag_to_slot_or_fallback(src: Tuple[int,int]) -> None:
    if drag_verified(BAG_SLOT, src, require_dst_empty=True):
        time.sleep(0.03 if FAST_MODE else 0.06)
        return
    alt = first_empty(TRUNK_SLOTS)
    if alt is not None and drag_verified(BAG_SLOT, alt, require_dst_empty=True):
        time.sleep(0.03 if FAST_MODE else 0.06)
        return
    if main_bag_pos is not None and not is_empty_cell(*main_bag_pos):
        drag_verified(BAG_SLOT, main_bag_pos, require_dst_empty=False)
        time.sleep(0.03 if FAST_MODE else 0.06)

def place_main_bag_in_trunk_safely() -> Optional[Tuple[int,int]]:
    global main_bag_pos

    compact_full_bag_memory()  # підчистити FULL (якщо увімкнено політикою)

    candidates: List[Tuple[int,int]] = []
    for xy in TRUNK_SLOTS:
        if xy in full_bag_slots:
            continue
        if xy in _last_main_bag_spots:
            continue
        if stable_empty(*xy, n=STABLE_N_SLOW, delay=0.03):
            candidates.append(xy)

    if not candidates:
        for xy in TRUNK_SLOTS:
            if xy in full_bag_slots:
                continue
            if stable_empty(*xy, n=STABLE_N_SLOW, delay=0.03):
                candidates.append(xy)
                break

    if not candidates:
        print("ДЯДЯ, немає стабільно порожнього місця під Основну сумку.")
        return None

    with _with_fast_temporarily(off=True):
        for slot in candidates:
            if not stable_empty(*slot, n=max(4, STABLE_N_SLOW+1), delay=0.03):
                continue

            ok = drag_verified(BAG_SLOT, slot, require_dst_empty=True)
            if not ok:
                continue

            time.sleep(0.10)
            if stable_filled(*slot, n=max(3, STABLE_N_SLOW), delay=0.03):
                main_bag_pos = slot
                _last_main_bag_spots.append(slot)
                return slot

    print("ДЯДЯ, не вийшло надійно покласти Основну в багажник (усі кандидати відхилено).")
    return None

# ========= ОСНОВНИЙ ПРОХІД ПО БАГАЖНИКУ =========
def pass_once_trunk_flow() -> bool:
    """True — потрібен новий цикл; False — Основна сумка чиста (лише захищені)."""
    global main_bag_pos

    # якщо роздільна змінилась — перепримінити скейл
    _ensure_scaling_current()

    # 1) покласти Основну сумку у багажник
    spot = place_main_bag_in_trunk_safely()
    if spot is None:
        return not main_is_clean()

    # 2) обхід сумок у багажнику
    for src in TRUNK_SLOTS:
        if not running:
            break
        if main_bag_pos is not None and src == main_bag_pos:
            continue

        if src in full_bag_slots:
            maybe_forget_full_if_now_empty(*src)
            if src in full_bag_slots:
                log(f"⤷ skip FULL {src}")
                continue

        if is_empty_cell(*src):
            continue

        # беремо сумку в руки
        if not drag_verified(src, BAG_SLOT, require_dst_empty=False):
            continue
        time.sleep(0.03 if FAST_MODE else 0.05)

        # 3) розкладаємо PRIO → SECONDARY
        moves = 0
        MAX_MOVES = 300
        bad_sec_targets: Set[Tuple[int,int]] = set()

        while running and moves < MAX_MOVES:
            pr_filled = first_filled(PRIORITY_ZONES)
            if pr_filled is None or not stable_filled(*pr_filled):
                pr_filled = first_filled(PRIORITY_ZONES)
                if pr_filled is None or not stable_filled(*pr_filled):
                    # повернути цю сумку
                    return_bag_to_slot_or_fallback(src)
                    # забрати Основну
                    take_main_bag_back_to_hands()
                    # у Основній: SECONDARY → PRIORITY
                    refill_priority_from_secondary_in_main_bag()
                    # вирішити, чи потрібен новий цикл
                    if main_has_any_priority():
                        print("♻️ Основна має предмети у Пріоритетних — починаю наступний цикл.")
                        return True
                    if main_has_any_secondary():
                        refill_priority_from_secondary_in_main_bag()
                        if main_has_any_priority():
                            print("♻️ Після повторного наповнення PRIO — новий цикл.")
                            return True
                    compact_full_bag_memory()
                    print("✅ Готово: Основна сумка чиста (лише захищені).")
                    return False

            sec_free = first_empty(SECONDARY_ZONES, exclude=bad_sec_targets)
            if sec_free is None:
                if pr_filled is not None:
                    full_bag_slots.add(src)
                    log(f"🧠 Запам'ятав ПОВНУ сумку в {src}")
                log("   ↪ SECONDARY немає робочих вільних — повертаю сумку на місце")
                return_bag_to_slot_or_fallback(src)
                break

            if not stable_filled(*pr_filled):
                continue
            if not stable_empty(*sec_free):
                bad_sec_targets.add(sec_free); continue

            if drag_verified(pr_filled, sec_free, require_dst_empty=True):
                moves += 1
            else:
                bad_sec_targets.add(sec_free); moves += 1

    # Якщо пріоритет не звільнили — повертаємось через Основну
    take_main_bag_back_to_hands()
    refill_priority_from_secondary_in_main_bag()

    if main_has_any_priority():
        print("♻️ Основна має предмети у Пріоритетних — починаю наступний цикл.")
        return True
    if main_has_any_secondary():
        refill_priority_from_secondary_in_main_bag()
        if main_has_any_priority():
            print("♻️ Після повторного наповнення PRIO — новий цикл.")
            return True

    compact_full_bag_memory()
    print("✅ Готово: Основна сумка чиста (лише захищені).")
    return False

# ========= ДІАГНОСТИКА =========
def diag_all(max_each: int = 6):
    def metrics(x, y):
        bgr = grab_patch(x, y, size=PATCH)
        hsv = bgr2hsv(bgr).astype(np.float32)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        roi_bgr  = center_roi(bgr)
        roi_hsv  = center_roi(hsv)
        roi_gray = center_roi(gray)
        lap     = laplacian_var(roi_gray)
        edens   = edge_density(roi_gray)
        sim = float('nan'); mad_h = float('nan'); mad_s = float('nan')
        if empty_ref_hsv is not None and empty_ref_tiny is not None:
            sim = float(np.dot(tiny_vector(roi_gray), empty_ref_tiny))
            mad_h = float(np.mean(np.abs(roi_hsv[:,:,0] - center_roi(empty_ref_hsv)[:,:,0])))
            mad_s = float(np.mean(np.abs(roi_hsv[:,:,1] - center_roi(empty_ref_hsv)[:,:,1])))
        stdv = float(roi_hsv[:,:,2].std())
        icon = icon_present_by_color(roi_bgr)
        return mad_h, mad_s, lap, stdv, edens, sim, icon

    groups = [("TRUNK", TRUNK_SLOTS),("PRIO ", PRIORITY_ZONES),("SECON", SECONDARY_ZONES)]
    print("=== ДІАГНОСТИКА ===")
    print(f"🧠 FULL slots: {sorted(list(full_bag_slots))}")
    for name, coords in groups:
        print(f"--- {name} ---")
        for (x, y) in coords[:max_each]:
            mad_h, mad_s, lap, stdv, ed, sim, icon = metrics(x, y)
            state = "EMPTY" if is_empty_cell(x, y) else "FILLED"
            prot = " [PROT]" if is_near_protected(x, y) else ""
            full = " [FULL]" if (x, y) in full_bag_slots else ""
            print(f"({x},{y}): madH={mad_h:.1f}, madS={mad_s:.1f}, lap={lap:.1f}, stdV={stdv:.1f}, edge={ed:.3f}, sim={sim:.2f}, icon={icon} -> {state}{prot}{full}")

# ========= ПЕРЕМИКАЧІ РЕЖИМІВ =========
def toggle_running():
    global running
    running = not running
    print('▶️ УВІМКНЕНО' if running else '⏹ ВИМКНЕНО')

def toggle_fast():
    global FAST_MODE
    FAST_MODE = not FAST_MODE
    print('⚡ FAST MODE: ON' if FAST_MODE else '🐢 FAST MODE: OFF')

# ========= ГОЛОВНИЙ ЦИКЛ =========
def main_loop():
    global running, working_now
    print('Гарячі клавіші: F8 — увімк/вимк; F2 — калібрувати; F6 — діагностика; F7 — очистити FULL; F3 — FAST MODE.')
    keyboard.add_hotkey(HOTKEY_TOGGLE, toggle_running)
    keyboard.add_hotkey(HOTKEY_CALIB, calibrate_empty_from_cursor)
    keyboard.add_hotkey(HOTKEY_DIAG, diag_all)
    keyboard.add_hotkey(HOTKEY_CLEAR_FULL, clear_full_bag_memory)
    keyboard.add_hotkey(HOTKEY_FAST, toggle_fast)
    try:
        while True:
            if running and not working_now:
                with lock:
                    working_now = True
                try:
                    need_more = True
                    while running and need_more:
                        need_more = pass_once_trunk_flow()
                finally:
                    working_now = False
                    running = False
                    # гарантуємо, що Основна у руках, а не "забута" у багажнику
                    take_main_bag_back_to_hands()
                    print("⏹ ВИМКНЕНО (цикл(и) завершено, Основна сумка у BAG_SLOT)")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main_loop()