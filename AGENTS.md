Agent Notes

Scope: entire repository.

Purpose
- This repo contains a Windows-only helper suite for RAGE:MP fishing that integrates an autoclicker, an inventory opener (SendInput), and a transfer module. Primary entry is `app_fishing_suite.py`.

Environment
- OS: Windows 10/11.
- Python: 3.10+ recommended.
- Dependencies: PyQt5, pyautogui, mss, numpy, opencv-python, keyboard.
- No network access required for normal operation.

Conventions
- Base authored coordinates are for 2560×1440; logic auto‑scales for 1920×1080 and other DPI settings where possible.
- Keep hotkeys consistent:
  - F8: toggle autoclicker/transfer loops
  - F3: TURBO/FAST mode
  - Transfer module extras: F2 (calibrate empty), F6 (diag), F7 (clear FULL)
- Logging should honor optional `config_runtime.py` (e.g., `LOG_LEVEL`). Avoid adding global side effects at import time.

Coding Style
- Prefer small, focused changes; avoid refactors that alter public behavior or hotkeys.
- Keep UI text and flow consistent with the integrated suite’s help message.
- When adding coordinates, define them at 2560×1440 and pass through the existing scaling helpers.

Testing Tips
- Use a borderless/windowed game mode to test SendInput behavior.
- If SendInput fails in dev VMs, run as Administrator or on a real desktop session.
- For transfer vision, verify scaling via `calibrate_empty_from_cursor()` (F2) and use `diag_all()` (F6).

Do Not
- Change hotkeys without updating README and in‑app help.
- Introduce blocking sleeps on the UI thread; schedule via timers/threads as done in `app_fishing_suite.py`.

