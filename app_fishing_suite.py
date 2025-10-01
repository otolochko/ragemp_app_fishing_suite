# app_fishing_suite.py (renamed from fishing_suite_integrated_2560_turbo.py)
# -*- coding: utf-8 -*-
"""
1440p / 1080p integrator: autoclicker + open inventory + transfer.
Added TURBO (F3) to speed up transfer.

After transfer: ESC → pause → E (scan 0x12) → 2.5 s → "1" → double F8.
"""

import sys
import time
import threading
import keyboard  # global F3

# ---- local module imports ----
try:
    import app_autoclicker as var3mod
except Exception:
    print("❌ Could not find app_autoclicker.py nearby.")
    raise

try:
    from module_open_inventory import open_inventory
except Exception:
    print("❌ Could not find module_open_inventory.py nearby.")
    raise

try:
    # if located elsewhere — adjust import or path
    import module_transfer as perekidka
except Exception:
    print("❌ Could not find module_transfer.py nearby.")
    raise

from PyQt5 import QtCore, QtWidgets
import logging
try:
    import config_runtime as cfg
except Exception:
    class cfg:
        LOG_LEVEL = logging.INFO
logging.basicConfig(level=getattr(cfg, 'LOG_LEVEL', logging.INFO))
logger = logging.getLogger(__name__)

# ----- delays -----
PRE_NEXT_DELAY_SEC = 3.0
try:
    POST_RESUME_DELAY_SEC = float(getattr(var3mod, "ESC_TO_ONE_DELAY", 1.0))
except Exception:
    POST_RESUME_DELAY_SEC = 1.0
AFTER_E_PAUSE_SEC = 4.0

# ----- scan-codes -----
SC_E  = 0x12  # "E"
SC_F8 = 0x42

class IntegratedWindow(var3mod.MainWindow):
    """
1440p / 1080p integrator: autoclicker + open inventory + transfer.
Added TURBO (F3) to speed up transfer.

After transfer: ESC → pause → E (scan 0x12) → 2.5 s → "1" → double F8.
"""
    def __init__(self):
        super().__init__()

        # allow 2560×1440 or 1920×1080 (scaling handled by autoclicker)
        try:
            if hasattr(self, 'res_combo'):
                # keep list as in autoclicker
                # ensure signal connected to _apply_resolution
                try:
                    self.res_combo.currentIndexChanged.disconnect(self._apply_resolution)
                except Exception:
                    pass
                self.res_combo.currentIndexChanged.connect(self._apply_resolution)
            # apply current value immediately
            self._apply_resolution()
        except Exception:
            pass

        # ---- states ----
        self._pipeline_running = False
        self._sequence_running = False
        self._suppress_green  = False
        self._calibrated_once = False
        self._hotkey_guard    = False
        self._turbo_on        = bool(getattr(perekidka, "FAST_MODE", False))

        # ---- UI: Help and Turbo ----
        self.help_btn  = QtWidgets.QPushButton("Help / Instructions")
        self.btn_turbo = QtWidgets.QPushButton(self._turbo_label())
        self.layout().addWidget(self.help_btn)
        self.layout().addWidget(self.btn_turbo)
        self.help_btn.clicked.connect(self._show_help)
        self.btn_turbo.clicked.connect(self.toggle_turbo)

        # ---- detector interceptors ----
        try:
            self.catch_green_overlay.detected.disconnect(self._color_detected)
        except Exception:
            pass
        try:
            self.inv_red_overlay.detected.disconnect(self._color_detected)
        except Exception:
            pass
        self.catch_green_overlay.detected.connect(self._on_detected_intercept)
        self.inv_red_overlay.detected.connect(self._on_detected_intercept)

        # ---- global hotkey F3 for TURBO ----
        keyboard.add_hotkey('f3',
            lambda: QtCore.QMetaObject.invokeMethod(self, "toggle_turbo", QtCore.Qt.QueuedConnection)
        )
        logger.info("[TURBO] init: %s", 'ON' if self._turbo_on else 'OFF')

    # ----- Turbo helpers -----
    def _turbo_label(self) -> str:
        return f"TURBO: {'ON' if self._turbo_on else 'OFF'} (F3)"

    @QtCore.pyqtSlot()
    def toggle_turbo(self):
        # toggle transfer speed
        self._turbo_on = not self._turbo_on
        try:
            # reflect flag into transfer module
            perekidka.FAST_MODE = self._turbo_on
        except Exception:
            pass
        self.btn_turbo.setText(self._turbo_label())
        logger.info('[TURBO MODE]: %s', 'ON' if self._turbo_on else 'OFF')

    # ---------- HELP ----------
    def _show_help(self):
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Help - quick guide")
        msg.setText(
            "• Toggle Auto-Click — start/stop autoclicker (default F8)\n"
            "• Resolution: 2560×1440 or 1920×1080 (choose in autoclicker)\n"
            "• F3 — TURBO for transfer (faster drags/checks inside transfer module)\n"
            "• Flow: red inventory → open_inventory → transfer → ESC → pause → E → 2.5 s → \"1\" → double F8 → resume."
        )
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()

    # ---------- detector event interceptors ----------
    @QtCore.pyqtSlot(str)
    def _on_detected_intercept(self, key: str):
        if not self.auto_on:
            return

        if key == "catch_green":
            if self._pipeline_running or self._suppress_green:
                return
            if self._sequence_running:
                return
            self._sequence_running = True
            #print("[CATCH] Green detected → ESC plan...")
            self.catch_green_overlay.stop_sensor()
            self._disarm_green_timeout()
            self._esc_timer.stop()
            self._esc_timer.start(int(var3mod.CATCH_PRE_ESC_DELAY * 1000))
            return

        if key == "inv_red":
            if self._pipeline_running:
                logger.info("[INVENTORY] Already in pipeline — skip.")
                return
            logger.info("[INVENTORY] Red detected -> start pipeline.")
            try:
                var3mod.press_escape()
            except Exception:
                pass
            try:
                self._esc_timer.stop(); self._one_timer.stop()
            except Exception:
                pass

            self._suppress_green   = True
            self._pipeline_running = True
            self._sequence_running = True

            # Pause sensors/timers
            self.left_overlay.stop()
            self.right_overlay.stop()
            self.catch_green_overlay.stop_sensor()
            self.inv_red_overlay.stop_sensor()
            self._disarm_green_timeout()

            threading.Thread(target=self._pipeline_worker, daemon=True).start()

    # ---------- pipeline ----------
    def _pipeline_worker(self):
        try:
            # 0) delay before open_inventory
            time.sleep(PRE_NEXT_DELAY_SEC)

            # 1) Open inventory
            open_inventory()
            time.sleep(0.15)

            # 2) Calibrate empty slot (once per session)
            if not self._calibrated_once:
                logger.info("[CALIB] Hover EMPTY slot — calibrating ~1.5s...")
                for _ in range(15):
                    if not self.auto_on:
                        return
                    time.sleep(0.1)
                try:
                    perekidka.calibrate_empty_from_cursor()
                    self._calibrated_once = True
                except Exception as e:
                    logger.warning("[CALIB] Calibration failed: %s — continue in loose mode.", e)

            # 3) Transfer (without F8), using current TURBO state
            try:
                perekidka.running = True
                # ensure speed flag is current
                try:
                    perekidka.FAST_MODE = self._turbo_on
                except Exception:
                    pass

                need_more = True
                while self.auto_on and perekidka.running and need_more:
                    need_more = perekidka.pass_once_trunk_flow()
                perekidka.running = False
                logger.info("[PIPE] Transfer finished.")
            except Exception as e:
                logger.error("[PIPE] Transfer error: %s", e)

            # 4) Close inventory (ESC)
            try:
                var3mod.press_escape()
            except Exception:
                pass
            time.sleep(0.1)

        finally:
            # 5) ESC → pause → E → 2.5 s → "1" → double F8 → resume
            if self.auto_on:
                time.sleep(POST_RESUME_DELAY_SEC)
                try:
                    var3mod.send_scan(SC_E, False); time.sleep(0.03); var3mod.send_scan(SC_E, True)
                    logger.info("[STATE] Pressed E (scan 0x12)")
                except Exception:
                    pass

                time.sleep(AFTER_E_PAUSE_SEC)

                try:
                    var3mod.press_top_1()
                    logger.info("[STATE] Resume: pressed 1")
                except Exception:
                    pass

                try:
                    self._hotkey_guard = True
                    for _ in range(2):
                        try:
                            var3mod.send_scan(SC_F8, False); time.sleep(0.01); var3mod.send_scan(SC_F8, True)
                        except Exception:
                            pass
                        time.sleep(0.08)
                    logger.info("[STATE] Sent F8 twice")
                finally:
                    self._hotkey_guard = False

                QtCore.QMetaObject.invokeMethod(self, "_resume_var3_after_cycle", QtCore.Qt.QueuedConnection)
                logger.info("[STATE] VAR3 restored after transfer.")

            self._sequence_running = False
            self._pipeline_running = False
            self._suppress_green = False

    @QtCore.pyqtSlot()
    def _resume_var3_after_cycle(self):
        if not self.auto_on:
            return
        try:
            self.left_overlay.start()
            self.right_overlay.start()
            self.catch_green_overlay.start_sensor()
            self.inv_red_overlay.start_sensor()
            self._apply_clickthrough(True)
            self._disarm_green_timeout()
            self._arm_green_timeout()
            logger.info("[STATE] VAR3 restored (UI-thread).")
        finally:
            self._suppress_green = False
            self._sequence_running = False
            self._pipeline_running = False

    # --- suppress ESC/1 during pipeline ---
    def _press_esc(self):
        if getattr(self, "_pipeline_running", False) or getattr(self, "_suppress_green", False):
            return
        super()._press_esc()

    def _press_one_and_resume(self):
        if getattr(self, "_pipeline_running", False) or getattr(self, "_suppress_green", False):
            return
        super()._press_one_and_resume()

    # --- start/stop autoclicker ---
    def toggle_autoclick(self):  # override
        if getattr(self, "_hotkey_guard", False):
            logger.info("[HOTKEY] Ignored F8 (guard active).")
            return
        new_state = not self.auto_on
        if new_state:
            self._suppress_green = False
            self._pipeline_running = False
            self._sequence_running = False
            try:
                self._esc_timer.stop(); self._one_timer.stop()
            except Exception:
                pass
            logger.info("[STATE] Auto-click: ON")
            self.lbl_status.setText("Auto-click: ON")
            self.left_overlay.start()
            self.right_overlay.start()
            self.catch_green_overlay.start_sensor()
            self.inv_red_overlay.start_sensor()
            self._apply_clickthrough(True)
            self._arm_green_timeout()
            self.auto_on = True
        else:
            self._suppress_green = True
            self._pipeline_running = False
            try:
                perekidka.running = False
            except Exception:
                pass
            try:
                var3mod.press_escape()
                logger.info("[STATE] Auto-click: OFF -> ESC.")
            except Exception:
                pass
            self.left_overlay.stop()
            self.right_overlay.stop()
            self.catch_green_overlay.stop_sensor()
            self.inv_red_overlay.stop_sensor()
            self._apply_clickthrough(False)
            self._disarm_green_timeout()
            try:
                self._esc_timer.stop(); self._one_timer.stop()
            except Exception:
                pass
            self._sequence_running = False
            self.auto_on = False
            self.lbl_status.setText("Auto-click: OFF")
            logger.info("[STATE] Auto-click: OFF (manual)")

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    w = IntegratedWindow()
    w.setWindowTitle("Fishing Suite — 1440p / 1080p + TURBO (F3)")
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()


