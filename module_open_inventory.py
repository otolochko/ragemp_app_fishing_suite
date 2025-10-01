# open inventory via SendInput
# 1) pull camera down (away from you) with relative mouse moves (SendInput)
# 2) press physical X via scan-code (layout independent)
# 3) click on "Inventory" (authored at 2560x1440 → auto-scale)

import time
import ctypes
from typing import Tuple, Optional

# ---------- BASE FOR SCALING ----------
BASE_W, BASE_H = 2560, 1440
INVENTORY_CLICK_BASE: Tuple[int, int] = (1485, 668)  # "Inventory" point for 2560x1440

# ---------- CAMERA PULL PARAMETERS ----------
PULL_PIXELS_BASE = 3000   # how much to pull down (in base pixels)
PULL_STEPS       = 18     # number of micro-steps (games detect move)
PULL_STEP_DELAY  = 0.03   # pause between steps
HOLD_RMB         = True   # hold RMB during pull (often required)

# ---------- PAUSES ----------
COUNTDOWN_SEC      = 2  # countdown before start
AFTER_PULL_PAUSE   = 0.10
AFTER_X_PAUSE      = 0.20
AFTER_CLICK_PAUSE  = 0.15

# ---------- CONSTANTS & ctypes setup ----------
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

INPUT_MOUSE    = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_MOVE       = 0x0001
MOUSEEVENTF_LEFTDOWN   = 0x0002
MOUSEEVENTF_LEFTUP     = 0x0004
MOUSEEVENTF_RIGHTDOWN  = 0x0008
MOUSEEVENTF_RIGHTUP    = 0x0010
MOUSEEVENTF_ABSOLUTE   = 0x8000

KEYEVENTF_SCANCODE     = 0x0008
KEYEVENTF_KEYUP        = 0x0002

# Scan-code for key X (Set 1): 0x2D = 45
SC_X = 0x2D

ULONG_PTR = ctypes.POINTER(ctypes.c_ulong)

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ULONG_PTR)]

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ULONG_PTR)]

class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT),
                ("ki", KEYBDINPUT)]

class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", INPUT_UNION)]

def is_admin() -> bool:
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def get_screen_size() -> Tuple[int,int]:
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

def scale_point(base_xy: Tuple[int,int]) -> Tuple[int,int]:
    sw, sh = get_screen_size()
    sx, sy = sw/BASE_W, sh/BASE_H
    x, y = base_xy
    return int(round(x*sx)), int(round(y*sy))

def scale_len(v: int) -> int:
    sw, sh = get_screen_size()
    s = (sw/BASE_W + sh/BASE_H) / 2.0
    return max(1, int(round(v*s)))

def set_default_resolution(width: int = 1920, height: int = 1080):
    global BASE_W, BASE_H
    BASE_W, BASE_H = 2560, 1440  # base for authored coords remains 2560x1440
    # function kept for compatibility; scaling uses current screen size dynamically
    # no-op; included to align API if needed elsewhere

def send_inputs(inputs):
    n = len(inputs)
    arr = (INPUT * n)(*inputs)
    sent = user32.SendInput(n, ctypes.byref(arr), ctypes.sizeof(INPUT))
    return sent == n

def mouse_move_rel(dx: int, dy: int) -> bool:
    inp = INPUT(type=INPUT_MOUSE, ii=INPUT_UNION(mi=MOUSEINPUT(dx=dx, dy=dy, mouseData=0,
                                                               dwFlags=MOUSEEVENTF_MOVE, time=0, dwExtraInfo=None)))
    return send_inputs([inp])

def mouse_left_click_at(x: int, y: int) -> bool:
    # Move cursor absolutely (SetCursorPos) — stable for UI click
    user32.SetCursorPos(x, y)
    down = INPUT(type=INPUT_MOUSE, ii=INPUT_UNION(mi=MOUSEINPUT(0,0,0,MOUSEEVENTF_LEFTDOWN,0,None)))
    up   = INPUT(type=INPUT_MOUSE, ii=INPUT_UNION(mi=MOUSEINPUT(0,0,0,MOUSEEVENTF_LEFTUP,0,None)))
    return send_inputs([down, up])

def mouse_right_down() -> bool:
    inp = INPUT(type=INPUT_MOUSE, ii=INPUT_UNION(mi=MOUSEINPUT(0,0,0,MOUSEEVENTF_RIGHTDOWN,0,None)))
    return send_inputs([inp])

def mouse_right_up() -> bool:
    inp = INPUT(type=INPUT_MOUSE, ii=INPUT_UNION(mi=MOUSEINPUT(0,0,0,MOUSEEVENTF_RIGHTUP,0,None)))
    return send_inputs([inp])

def key_tap_scancode(sc: int, hold: float = 0.02) -> bool:
    down = INPUT(type=INPUT_KEYBOARD, ii=INPUT_UNION(ki=KEYBDINPUT(wVk=0, wScan=sc, dwFlags=KEYEVENTF_SCANCODE, time=0, dwExtraInfo=None)))
    up   = INPUT(type=INPUT_KEYBOARD, ii=INPUT_UNION(ki=KEYBDINPUT(wVk=0, wScan=sc, dwFlags=KEYEVENTF_SCANCODE|KEYEVENTF_KEYUP, time=0, dwExtraInfo=None)))
    ok1 = send_inputs([down])
    time.sleep(hold)
    ok2 = send_inputs([up])
    return ok1 and ok2

def pull_camera_down(pixels_base: int = PULL_PIXELS_BASE, steps: int = PULL_STEPS, hold_rmb: bool = HOLD_RMB):
    dy_total = scale_len(pixels_base)
    step = max(1, dy_total // max(1, steps))
    if hold_rmb:
        mouse_right_down()
    try:
        moved = 0
        for _ in range(steps):
            if not mouse_move_rel(0, step):
                # If SendInput fails — short sleep and continue
                time.sleep(0.005)
            moved += step
            time.sleep(PULL_STEP_DELAY)
        # finish fractional remainder precisely
        rest = dy_total - moved
        if rest > 0:
            mouse_move_rel(0, rest)
    finally:
        if hold_rmb:
            mouse_right_up()
    time.sleep(AFTER_PULL_PAUSE)

def click_inventory(inv_point_base: Tuple[int,int] = INVENTORY_CLICK_BASE):
    ix, iy = scale_point(inv_point_base)
    mouse_left_click_at(ix, iy)
    time.sleep(AFTER_CLICK_PAUSE)

def open_inventory(inv_point_base: Tuple[int,int] = INVENTORY_CLICK_BASE) -> bool:
    print(f"🖥️ Screen={get_screen_size()} | admin={'YES' if is_admin() else 'NO'}")
    # 1) pull camera (to ensure X appears)
    print("↘️  Pulling camera…")
    pull_camera_down()

    # 2) press physical X (scan-code)
    print("⌨️  Pressing X…")
    key_tap_scancode(SC_X, hold=0.03)
    time.sleep(AFTER_X_PAUSE)

    # 3) click on "Inventory"
    print("🖱️ Clicking ""Inventory""…")
    click_inventory(inv_point_base)

    print("✅ Done.")
    return True

# ---- optional custom coordinate via "x,y" arg ----
def parse_point_arg(arg: str) -> Optional[Tuple[int,int]]:
    try:
        xs, ys = arg.split(",")
        return int(xs.strip()), int(ys.strip())
    except Exception:
        return None

if __name__ == "__main__":
    import sys
    inv_pt = INVENTORY_CLICK_BASE
    if len(sys.argv) >= 2:
        maybe = parse_point_arg(sys.argv[1])
        if maybe:
            inv_pt = maybe
            print(f"↪ Using custom base point: {inv_pt} (base 2560×1440)")
    print(f"▶️ Starting in {COUNTDOWN_SEC:.1f}s… switch to the game (Borderless/Windowed recommended).")
    time.sleep(COUNTDOWN_SEC)
    open_inventory(inv_pt)
