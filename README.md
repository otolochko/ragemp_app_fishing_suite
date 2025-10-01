RAGE:MP Fishing Suite (Autoclicker + Inventory + Transfer)

A Windows-only helper suite for the RAGE:MP fishing gameplay loop that integrates:

- Autoclicker and color detectors (green catch cue, inventory red cue)
- One-click inventory opening via SendInput (camera pull + press X + click Inventory)
- Smart item transfer between trunk/bag/priority zones with optional TURBO mode

The integrated app coordinates these modules to automate the “catch → open inventory → transfer → resume” loop on 2560×1440 and 1920×1080 displays (auto‑scaled).


## Components

- `app_fishing_suite.py` — PyQt5 UI that ties everything together. Adds TURBO toggle (F3) and orchestrates the flow once the inventory red cue is detected.
- `app_autoclicker.py` — PyQt5 autoclicker with on‑screen color overlays/sensors. Default hotkey F8 to start/stop.
- `module_open_inventory.py` — Uses Windows SendInput to pull camera, press physical `X` (scan‑code), and click the Inventory UI. Auto‑scales from 2560×1440 authored coordinates.
- `module_transfer.py` — Vision‑assisted drag & drop between trunk/priority/secondary zones. Supports FAST/TURBO timings and calibration from cursor.
- `config_runtime.py` — Optional runtime overrides, for example `LOG_LEVEL` or resolution profile used by autoclicker.


## Features at a Glance

- 1440p and 1080p support (auto‑scaled from 2560×1440 base)
- Global hotkeys: F8 (toggle autoclicker), F3 (TURBO/FAST), F2/F6/F7 inside transfer module
- Automatic pipeline after inventory red cue:
  - ESC → pause → open inventory → transfer → ESC → pause → press `E` → wait → press `1` → double F8 → resume
- Transfer TURBO toggleable from the suite window and via F3


## Requirements

- Windows 10/11 desktop (SendInput + Win32 APIs)
- Python 3.10+ recommended
- Packages: `PyQt5`, `pyautogui`, `mss`, `numpy`, `opencv-python`, `keyboard`

Install packages:

```
python -m pip install PyQt5 pyautogui mss numpy opencv-python keyboard
```

Run from an elevated terminal if SendInput or global hotkeys require it in your setup.


## Usage

1) Start the integrated suite:

```
python app_fishing_suite.py
```

2) In the window:
- Choose resolution profile if exposed by the autoclicker UI (1080p/1440p).
- Use “Help / Instructions” for a quick guide.
- Toggle TURBO with F3 or the TURBO button.
- Start/stop autoclicker with F8.

3) During play:
- The green catch cue and inventory red cue sensors drive the flow.
- On red inventory cue, the suite auto‑runs: open inventory → transfer → restore fishing state.

To run modules standalone (advanced):

- Open Inventory only:
  - `python module_open_inventory.py`
  - Optional base click override: `python module_open_inventory.py 1485,668`

- Transfer loop only:
  - `python module_transfer.py`
  - Inside: F8 toggle run, F2 calibrate empty from cursor, F3 FAST, F6 diagnostics, F7 clear FULL memory.

- Autoclicker only:
  - `python app_autoclicker.py`


## Hotkeys (summary)

- Global/integrated:
  - `F8` — Toggle autoclicker
  - `F3` — TURBO/FAST mode (transfer speed)

- Transfer module window (when run standalone):
  - `F8` — Start/stop main loop
  - `F2` — Calibrate empty cell from cursor
  - `F6` — Diagnostics
  - `F7` — Clear remembered FULL slots


## Tips

- Use Windowed/Borderless in game so the cursor and SendInput behave predictably.
- First transfer run may prompt a short one‑time calibration while hovering an empty slot.
- If something stalls, press ESC once and toggle F8 to re‑sync; the suite also double‑taps F8 automatically after a cycle.


## Disclaimer

This tool automates inputs via SendInput and global hotkeys. Use at your own risk and ensure it complies with the rules of the servers/games you play on.

