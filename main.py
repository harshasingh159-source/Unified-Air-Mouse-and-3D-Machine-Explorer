"""
AIR MOUSE + FX + 3D MACHINE EXPLORER  (everything in one file)
==============================================================
Uses MediaPipe HandLandmarker (VIDEO mode), OpenCV, NumPy and PyAutoGUI.
Optional: pynput (enables the global pause hotkey).

MODES  (toggle with a key; the default mode is the air mouse)
  F ........ FX mode        E ........ 3D MACHINE EXPLORER mode

AIR MOUSE (default)  -- any one hand:
  Move cursor ........ point; the cursor follows your index FINGERTIP
  Left click ......... pinch thumb + RING finger and release (a real click)
  Drag ............... keep the thumb + ring pinch held for 0.25s, move, open to drop
  Right click ........ pinch thumb + PINKY finger
  Scroll ............. OK sign (thumb + index pinched, middle/ring/pinky up),
                       then move your hand UP/DOWN
  Freeze (clutch) .... make a tight fist -> cursor stops
  Alt+Tab ............ OPEN PALM (all 4 fingers up; the cursor freezes), then
                       swipe RIGHT = next window, LEFT = previous window.
                       Keep swiping to keep cycling; close your hand to pick.
                       (To move the cursor, curl at least one finger.)
  Zoom ............... TWO HANDS: pinch thumb + index on both, then spread
                       apart = zoom in, bring together = zoom out (Ctrl+wheel
                       on the window under the cursor; Z turns it on/off)
  Calibrate .......... press C: pinch ring, pinky, index when asked. Your
                       personal thresholds are measured and saved to a file.

CUSTOM GESTURES (air mouse and FX mode):
  Record ............. press G, hold your own hand pose for 2 seconds, then press a
                       number key to pick its action: 1 play/pause  2 next track
                       3 previous track  4 volume up  5 volume down  6 mute
                       7 copy  8 paste  9 undo  0 screenshot
  Use it ............. make the pose and hold it for half a second
  Remove ............. U deletes the most recent gesture
  More actions ....... edit custom_gestures.json (key combos, open a site/app, type text)
  Not allowed ........ open palm, pointing (1-2 fingers), plain fist and pinches,
                       because those already do something

FX MODE (F):
  Mouse hand ......... works exactly as above
  Peace sign (hold 1s) Opens YouTube in Chrome (mouse hand)
  Other hand, palm up  Rasengan overlay on your palm (needs rasengan.mp4)
  Both hands ......... Absolute Cinema (needs absolute_cinema.mp4)

3D MACHINE EXPLORER MODE (E):  the mouse is switched off; your hands drive a 3D viewer
  Built-in machines .. engine, gear reducer, robot arm, sedan, Bugatti Chiron-style hypercar
  Rotate ............. pinch THUMB + INDEX, then move your hand
  Zoom ............... pinch THUMB + RING, then move your hand up (in) / down (out)
  Select a part ...... point at it, then TAP thumb + MIDDLE (tap again to deselect)
  Isolate / show all . TAP thumb + PINKY (needs a selected part)
  Explode ............ TWO HANDS: move them apart = explode, together = assemble
  Reset view ......... hold a FIST for 1 second
  Next / previous .... swipe an OPEN PALM right / left to change machine
  Physical mouse ..... drag = rotate, click = select, wheel = zoom
  Viewer keys ........ N/B machine  Space animate  X explode  [ ] explode less/more
                       Tab next part  I isolate  + - zoom  A D W S rotate  R reset
                       T auto-rotate  L labels  H hide help  G quality
  Your own models .... put folders in  machines/<name>/  (next to the program)
                       with .stl, .obj or .glb/.gltf files
                       (a car from Sketchfab etc. as .glb works; textures are not shown)

Keys (camera window focused):
  F fx mode   E 3D explorer   P pause   D debug   C calibrate (Esc cancels)
  S swipe on/off   Z two-hand zoom on/off   G record gesture   U remove last gesture
  Q quit (or close the window)
GLOBAL PAUSE: Ctrl+Alt+P works from ANY window (needs: pip install pynput)
EMERGENCY STOP: slam your physical mouse into any screen corner
(PyAutoGUI fail-safe) and the program exits and releases all buttons.
"""

import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import textwrap
import threading
import time
import urllib.request
import webbrowser
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# ============================================================
# Paths
#   resource_path() -> read-only files bundled with the app (PyInstaller _MEIPASS)
#   app_dir()       -> writable folder next to the program (calibration, logs,
#                      the machines/ folder, a downloaded hand model)
# ============================================================
def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(relative_path):
    """ Get absolute path to a bundled resource, works for dev and for PyInstaller """
    base_path = getattr(sys, "_MEIPASS", app_dir())
    return os.path.join(base_path, relative_path)


def setup_logging():
    """In a windowed (no console) build stdout/stderr are None: send output to a log file."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        log = open(os.path.join(app_dir(), "air_mouse.log"), "a", buffering=1, encoding="utf-8")
        log.write(f"\n===== started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        if sys.stdout is None:
            sys.stdout = log
        if sys.stderr is None:
            sys.stderr = log
    except OSError:
        pass


_BUNDLED_MODEL = resource_path("hand_landmarker.task")
MODEL_PATH = (_BUNDLED_MODEL if os.path.exists(_BUNDLED_MODEL)
              else os.path.join(app_dir(), "hand_landmarker.task"))
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ============================================================
# Tunable settings - change these first if something feels off
# ============================================================
@dataclass
class Config:
    cam_index: int = 0
    cam_w: int = 640
    cam_h: int = 480
    max_read_failures: int = 30      # consecutive failed camera reads before giving up

    # Active region: the part of the camera view that maps to the whole screen.
    # Smaller = less arm movement needed, but less precision.
    region_width: float = 0.60       # fraction of frame width
    region_center: tuple = (0.5, 0.5)

    # Which landmark drives the cursor. 8 = index fingertip, 5 = index knuckle.
    pointer_landmark: int = 8

    # One Euro filter. Lower min_cutoff = steadier when slow.
    # Higher beta = less lag when moving fast.
    min_cutoff: float = 1.5
    beta: float = 15.0

    # Pinch distances are divided by hand size (wrist -> middle knuckle).
    # "close" = pinch registers, "open" = pinch releases (the gap stops flicker).
    # Read the T-I / T-R / T-P numbers in the debug line while you pinch, and
    # set "close" just above what you see (and well below your resting value).
    pinch_close: float = 0.45          # LEFT click  : thumb-ring
    pinch_open: float = 0.70
    right_pinch_close: float = 0.55    # RIGHT click : thumb-pinky
    right_pinch_open: float = 0.85
    scroll_pinch_close: float = 0.40   # SCROLL sign : thumb-index
    scroll_pinch_open: float = 0.60
    click_confirm_frames: int = 2      # frames a pinch must persist to count
    drag_hold_s: float = 0.25          # hold a left pinch this long to start a drag

    # Fist = ALL four fingertips closer to the wrist than this * hand size.
    fist_tip_ratio: float = 1.1

    # Finger counts as extended if tip-to-wrist > pip-to-wrist * this
    extend_ratio: float = 1.2
    pose_confirm_frames: int = 3     # frames a pose must persist to switch mode

    scroll_confirm_frames: int = 3   # frames the OK sign must be held to scroll
    scroll_gain: float = 1500.0      # larger = faster scrolling
    scroll_deadzone: float = 0.003   # ignore tiny hand jitter (frame fraction)

    lost_grace_frames: int = 6       # tolerate this many dropped frames

    # Two-hand pinch zoom (air mouse mode, Z toggles): pinch thumb + index on BOTH
    # hands, then spread apart = Ctrl+wheel up (zoom in), together = zoom out.
    # Uses the calibrated thumb-index thresholds (scroll_pinch_close / _open).
    zoom_enabled: bool = True
    zoom_confirm_frames: int = 3     # both pinches must be held this many frames
    zoom_step: float = 0.06          # change in hand gap (fraction of frame width) per zoom step
    zoom_max_steps: int = 2          # at most this many zoom steps per frame

    # Custom gestures (press G to record one; saved in custom_gestures.json next to the program)
    gesture_match_dist: float = 0.25   # lower = stricter matching, fewer accidental triggers
    gesture_hold_s: float = 0.5        # hold the pose this long before it fires
    gesture_repeat_s: float = 0.35     # repeat interval for repeating actions (volume up/down)

    # Global pause hotkey (works from any window; needs the pynput package)
    global_pause_hotkey: str = "<ctrl>+<alt>+p"

    # FX mode. MediaPipe labels hands from the mirrored image. If mouse and
    # Rasengan are on the wrong hands, swap these two values.
    mouse_hand_label: str = "Left"
    fx_hand_label: str = "Right"
    cinema_confirm_frames: int = 5   # both hands must be seen this many frames
    rasengan_video: str = resource_path("rasengan.mp4")
    cinema_video: str = resource_path("absolute_cinema.mp4")

    # Peace sign -> YouTube (FX mode only)
    youtube_url: str = "https://www.youtube.com"
    peace_hold_s: float = 1.0        # hold the sign this long
    peace_spread: float = 0.35       # min gap between the two fingertips / hand size
    yt_cooldown_s: float = 3.0

    # Open-palm swipe -> Alt+Tab. All distances are fractions of frame width.
    swipe_enabled: bool = True
    swipe_dist: float = 0.16           # hand must travel at least this far...
    swipe_window_s: float = 0.40       # ...within this time window...
    swipe_min_speed: float = 0.70      # ...at least this fast (frame widths / second)...
    swipe_horizontal_ratio: float = 2.2  # ...and mostly sideways
    swipe_cooldown_s: float = 0.50     # pause after a swipe
    swipe_reverse_lock_s: float = 1.20   # the hand coming BACK is not a swipe the other way,
    swipe_reverse_speed: float = 1.5     # unless it is a deliberate, fast flick like this
    alt_hold_s: float = 1.25           # keep Alt held this long after the last swipe

    # 3D machine explorer mode (press E). Your own models go in machines/<name>/ (.stl .obj .glb .gltf)
    # This folder lives NEXT TO the program so users can add models to a packaged build.
    machines_dir: str = os.path.join(app_dir(), "machines")
    models_z_up: bool = True         # CAD files are usually Z-up; set False if yours look sideways


# ============================================================
# Global pause hotkey (optional: needs pynput)
# ============================================================
class GlobalHotkey:
    """Listens for a key combo system-wide on a background thread."""

    def __init__(self, combo):
        self._hit = threading.Event()
        self.listener = None
        try:
            from pynput import keyboard
            self.listener = keyboard.GlobalHotKeys({combo: self._hit.set})
            self.listener.daemon = True
            self.listener.start()
            print(f"Global pause hotkey: {combo}")
        except Exception as e:
            print(f"Global pause hotkey unavailable ({e}). Install pynput to enable it.")

    def poll(self):
        """True once per press (safe to call every frame from the main thread)."""
        if self._hit.is_set():
            self._hit.clear()
            return True
        return False

    def stop(self):
        if self.listener is not None:
            try:
                self.listener.stop()
            except Exception:
                pass


# ============================================================
# Open YouTube in Chrome (falls back to the default browser)
# ============================================================
def open_youtube_in_chrome(url):
    try:
        if sys.platform == "darwin":
            if subprocess.run(["open", "-a", "Google Chrome", url]).returncode == 0:
                return True
        else:
            if sys.platform.startswith("win"):
                candidates = []
                for var in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
                    base = os.environ.get(var)
                    if base:
                        candidates.append(os.path.join(base, "Google", "Chrome", "Application", "chrome.exe"))
            else:
                candidates = [shutil.which(n) for n in
                              ("google-chrome", "google-chrome-stable", "chromium-browser", "chromium")]
            for path in candidates:
                if path and os.path.exists(path):
                    subprocess.Popen([path, url])
                    return True
    except Exception as e:
        print(f"Could not launch Chrome directly: {e}")
    print("Chrome not found, opening your default browser instead.")
    webbrowser.open(url)
    return False


# ============================================================
# Calibration (measure YOUR pinch distances, save them to a file)
# ============================================================
CALIB_PATH = os.path.join(app_dir(), "air_mouse_calibration.json")
CALIB_KEYS = ("pinch_close", "pinch_open", "right_pinch_close", "right_pinch_open",
              "scroll_pinch_close", "scroll_pinch_open")


def load_calibration(cfg):
    try:
        with open(CALIB_PATH) as f:
            data = json.load(f)
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"Could not read {CALIB_PATH}: {e}")
        return False
    for k in CALIB_KEYS:
        if k in data:
            setattr(cfg, k, float(data[k]))
    print(f"Loaded calibration from {CALIB_PATH}")
    return True


def save_calibration(cfg):
    try:
        with open(CALIB_PATH, "w") as f:
            json.dump({k: getattr(cfg, k) for k in CALIB_KEYS}, f, indent=2)
        print(f"Calibration saved to {CALIB_PATH}")
    except Exception as e:
        print(f"Could not save calibration: {e}")


def hand_ratios(landmarks, w, h):
    """(thumb-index, thumb-ring, thumb-pinky), each divided by hand size."""
    pts = np.array([[l.x * w, l.y * h] for l in landmarks])
    scale = np.linalg.norm(pts[9] - pts[0]) + 1e-6
    t = pts[4]
    return (np.linalg.norm(t - pts[8]) / scale,
            np.linalg.norm(t - pts[16]) / scale,
            np.linalg.norm(t - pts[20]) / scale)


class Calibrator:
    STEPS = [
        ("REST", "Relax your hand and POINT with your index finger"),
        ("RING", "PINCH thumb + RING finger and hold"),
        ("PINKY", "PINCH thumb + PINKY finger and hold"),
        ("INDEX", "OK SIGN: pinch thumb + INDEX, other 3 fingers up"),
    ]
    PREP_S = 2.5        # time to get into the pose
    CAPTURE_S = 2.0     # time spent measuring
    MIN_SAMPLES = 15    # frames with a visible hand needed per step

    def __init__(self):
        self.active = False
        self.step = 0
        self.t0 = 0.0
        self.samples = []
        self.med = {}
        self.live = ""
        self.message = ""
        self.message_until = 0.0

    def _say(self, text, now, secs=4.0):
        self.message = text
        self.message_until = now + secs
        print(text)

    def start(self, now):
        self.active = True
        self.step = 0
        self.t0 = now
        self.samples = []
        self.med = {}
        self.message_until = 0.0

    def cancel(self, now):
        self.active = False
        self._say("Calibration cancelled - nothing changed", now)

    def display(self, now):
        el = now - self.t0
        text = self.STEPS[self.step][1]
        if el < self.PREP_S:
            return text, f"Get ready... {self.PREP_S - el:.1f}s", False, el / self.PREP_S
        return text, "HOLD IT - measuring", True, min((el - self.PREP_S) / self.CAPTURE_S, 1.0)

    def update(self, ratios, now):
        """Feed one frame. Returns a dict of new thresholds when finished, else None."""
        if ratios is not None:
            self.live = f"T-I {ratios[0]:.2f}  T-R {ratios[1]:.2f}  T-P {ratios[2]:.2f}"
        else:
            self.live = "no hand seen"
        el = now - self.t0
        if el < self.PREP_S:
            return None
        if el < self.PREP_S + self.CAPTURE_S:
            if ratios is not None:
                self.samples.append(ratios)
            return None
        if len(self.samples) < self.MIN_SAMPLES:
            self.samples = []
            self.t0 = now
            print("Hand not visible enough - repeating this step")
            return None
        self.med[self.STEPS[self.step][0]] = np.median(np.array(self.samples), axis=0)
        self.samples = []
        self.step += 1
        self.t0 = now
        if self.step < len(self.STEPS):
            return None
        return self._finish(now)

    @staticmethod
    def _thresholds(pinched, rest):
        span = rest - pinched
        if span < 0.15:          # pinch isn't clearly different from resting
            return None
        return pinched + 0.40 * span, pinched + 0.70 * span

    def _finish(self, now):
        self.active = False
        rest = self.med["REST"]
        specs = [
            ("ring", "pinch_close", "pinch_open", self.med["RING"][1], rest[1]),
            ("pinky", "right_pinch_close", "right_pinch_open", self.med["PINKY"][2], rest[2]),
            ("index", "scroll_pinch_close", "scroll_pinch_open", self.med["INDEX"][0], rest[0]),
        ]
        out, good, bad = {}, [], []
        for name, kc, ko, pinched, resting in specs:
            th = self._thresholds(pinched, resting)
            if th is None:
                bad.append(name)
            else:
                out[kc], out[ko] = round(float(th[0]), 3), round(float(th[1]), 3)
                good.append(name)
        if bad and not good:
            self._say("Not saved: pinches too close to resting. Pinch harder and retry (C)", now, 6.0)
            return None
        if bad:
            self._say(f"Saved: {', '.join(good)} | NOT saved (retry with C): {', '.join(bad)}", now, 6.0)
        else:
            self._say("Calibration saved!", now, 4.0)
        return out


# ============================================================
# Custom gestures: record your own hand pose (G) and give it an action
# ============================================================
GESTURES_PATH = os.path.join(app_dir(), "custom_gestures.json")
_MOD = "command" if sys.platform == "darwin" else "ctrl"

# The actions offered after recording a pose (pick with a number key).
# More action types are possible by editing custom_gestures.json:
#   {"type": "key",    "keys": ["playpause"]}           press key(s) one after another
#   {"type": "hotkey", "keys": ["mod", "shift", "t"]}   a key combo ("mod" = Ctrl, or Cmd on a Mac)
#   {"type": "open",   "target": "https://..." or a file / app path}
#   {"type": "type",   "text": "hello"}                 types text
#   {"type": "screenshot"}                              saves a PNG in screenshots/
GESTURE_PRESETS = [
    ("1", "Play / Pause", {"type": "key", "keys": ["playpause"]}, False),
    ("2", "Next track", {"type": "key", "keys": ["nexttrack"]}, False),
    ("3", "Previous track", {"type": "key", "keys": ["prevtrack"]}, False),
    ("4", "Volume up", {"type": "key", "keys": ["volumeup"]}, True),
    ("5", "Volume down", {"type": "key", "keys": ["volumedown"]}, True),
    ("6", "Mute", {"type": "key", "keys": ["volumemute"]}, False),
    ("7", "Copy", {"type": "hotkey", "keys": ["mod", "c"]}, False),
    ("8", "Paste", {"type": "hotkey", "keys": ["mod", "v"]}, False),
    ("9", "Undo", {"type": "hotkey", "keys": ["mod", "z"]}, False),
    ("0", "Screenshot", {"type": "screenshot"}, False),
]


_NOTICE = {"text": "", "until": 0.0, "ok": True}


def notify(text, ok=True, secs=4.0):
    """Print a message AND show it in the camera window for a few seconds."""
    print(text)
    _NOTICE.update(text=text, ok=ok, until=time.perf_counter() + secs)


def take_screenshot(path):
    """Save a PNG of the whole screen to `path`. Tries several methods so it works
    on most systems without extra setup; raises with the reasons if all fail."""
    errors = []
    if sys.platform == "darwin":                      # built into macOS
        try:
            r = subprocess.run(["screencapture", "-x", path], capture_output=True, timeout=10)
            if r.returncode == 0 and os.path.exists(path):
                return
            errors.append("macOS blocked it - allow Screen Recording for this app in "
                          "System Settings > Privacy & Security")
        except Exception as e:
            errors.append(f"screencapture: {e}")
    try:                                              # fast, all monitors (pip install mss)
        import mss
        import mss.tools
        with mss.mss() as sct:
            shot = sct.grab(sct.monitors[0])
            mss.tools.to_png(shot.rgb, shot.size, output=path)
        if os.path.exists(path):
            return
    except ImportError:
        errors.append("mss not installed")
    except Exception as e:
        errors.append(f"mss: {e}")
    try:                                              # Pillow (pip install pillow)
        from PIL import ImageGrab
        ImageGrab.grab(all_screens=True).save(path)
        if os.path.exists(path):
            return
    except ImportError:
        errors.append("Pillow not installed")
    except Exception as e:
        errors.append(f"Pillow: {e}")
    if sys.platform.startswith("linux"):              # desktop screenshot tools
        for cmd in (["gnome-screenshot", "-f", path], ["spectacle", "-b", "-n", "-o", path],
                    ["scrot", path], ["grim", path]):
            if shutil.which(cmd[0]):
                try:
                    subprocess.run(cmd, capture_output=True, timeout=10)
                    if os.path.exists(path):
                        return
                except Exception as e:
                    errors.append(f"{cmd[0]}: {e}")
    raise RuntimeError("; ".join(errors) or "no screenshot method available")


def run_action(action, name=""):
    """Perform a custom gesture's action. Errors are printed, never fatal."""
    kind = action.get("type")
    try:
        if kind == "key":
            pyautogui.press(list(action["keys"]))
        elif kind == "hotkey":
            pyautogui.hotkey(*[_MOD if k == "mod" else k for k in action["keys"]])
        elif kind == "type":
            pyautogui.write(str(action["text"]), interval=0.01)
        elif kind == "open":
            target = str(action["target"])
            if target.startswith(("http://", "https://")):
                webbrowser.open(target)
            elif sys.platform.startswith("win"):
                os.startfile(target)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", target])
            else:
                subprocess.Popen(["xdg-open", target])
        elif kind == "screenshot":
            folder = os.path.join(app_dir(), "screenshots")
            os.makedirs(folder, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
            path = os.path.join(folder, f"screenshot_{stamp}.png")
            n = 2
            while os.path.exists(path):                      # never overwrite an earlier shot
                path = os.path.join(folder, f"screenshot_{stamp}_{n}.png")
                n += 1
            take_screenshot(path)
            notify(f"Screenshot saved: {path}")
            return
        else:
            notify(f"Gesture '{name}': unknown action type {kind!r}", ok=False)
            return
        print(f"Gesture: {name}")
    except Exception as e:
        notify(f"Gesture '{name}' failed: {e}", ok=False, secs=7.0)


def pose_shape(pts):
    """Hand pose as 20 points relative to the wrist, scaled by hand size and turned
    so the palm points 'up'. The same pose gives the same shape wherever the hand
    is in the frame, at any distance, and tilted sideways."""
    p = np.asarray(pts, float)[:, :2] - pts[0][:2]
    v = p[9]
    s = float(np.linalg.norm(v)) + 1e-6
    u = v / s                                   # wrist -> middle knuckle
    r = np.array([-u[1], u[0]])
    return np.stack([p @ r, -(p @ u)], 1)[1:] / s


_TIP_ROWS = [3, 7, 11, 15, 19]                  # fingertips (landmarks 4, 8, 12, 16, 20)


def shape_distance(a, b):
    """0 = identical pose. Mixes the average point difference with the worst
    fingertip difference, so a single finger (e.g. a raised thumb) counts."""
    d = np.linalg.norm(a - b, axis=1)
    return 0.5 * float(d.mean()) + 0.5 * float(d[_TIP_ROWS].max())


def looks_like_click(ratios, fingers, cfg):
    """True if this hand would trigger a built-in click, right click or scroll
    (the same rules AirMouse.on_hand uses)."""
    ti, tr, tp = ratios
    index, middle, ring, pinky = fingers
    return ((not ring and tr < cfg.pinch_close and tr <= tp)              # left click
            or (not pinky and tp < cfg.right_pinch_close and tp < tr)     # right click
            or (ti < cfg.scroll_pinch_close and middle and ring and pinky))  # scroll (OK sign)


class CustomGesture:
    def __init__(self, name, action, repeat, pattern, samples):
        self.name = name
        self.action = action
        self.repeat = bool(repeat)
        self.pattern = tuple(bool(x) for x in pattern)      # index, middle, ring, pinky extended
        self.samples = np.asarray(samples, float).reshape(-1, 20, 2)
        mirrored = self.samples.copy()
        mirrored[:, :, 0] *= -1                             # works with either hand
        self.all = np.concatenate([self.samples, mirrored])

    def distance(self, shape):
        return min(shape_distance(shape, s) for s in self.all)

    def to_json(self):
        return {"name": self.name, "action": self.action, "repeat": self.repeat,
                "pattern": [int(x) for x in self.pattern],
                "samples": np.round(self.samples.reshape(len(self.samples), -1), 3).tolist()}


class GestureStore:
    """Your recorded gestures, saved in custom_gestures.json next to the program."""

    def __init__(self, path):
        self.path = path
        self.items = []
        self.load()

    def load(self):
        try:
            with open(self.path) as f:
                data = json.load(f)
        except FileNotFoundError:
            return
        except Exception as e:
            print(f"Could not read {self.path}: {e}")
            return
        for g in data.get("gestures", []):
            try:
                self.items.append(CustomGesture(g["name"], g["action"], g.get("repeat", False),
                                                g["pattern"], g["samples"]))
            except Exception as e:
                print(f"Skipping a broken gesture in {self.path}: {e}")
        if self.items:
            print(f"Loaded {len(self.items)} custom gesture(s): " + ", ".join(g.name for g in self.items))

    def save(self):
        try:
            with open(self.path, "w") as f:
                json.dump({"version": 1, "gestures": [g.to_json() for g in self.items]}, f, indent=1)
        except Exception as e:
            print(f"Could not save gestures: {e}")

    def add(self, gesture):
        self.items.append(gesture)
        self.save()

    def remove_last(self):
        if not self.items:
            return None
        g = self.items.pop()
        self.save()
        return g

    def match(self, pts, pattern, max_dist):
        """Best gesture for this hand, or (None, None). The fingers that are up must
        match exactly, then the overall shape must be close enough."""
        pattern = tuple(bool(x) for x in pattern)
        cands = [g for g in self.items if g.pattern == pattern]
        if not cands:
            return None, None
        shape = pose_shape(pts)
        best, best_d = None, max_dist
        for g in cands:
            d = g.distance(shape)
            if d < best_d:
                best, best_d = g, d
        return (best, best_d) if best is not None else (None, None)


class GestureRecorder:
    """G -> hold your pose -> press a number to pick its action."""
    PREP_S = 2.5
    CAPTURE_S = 2.0
    MIN_SAMPLES = 12
    KEEP_SAMPLES = 15

    def __init__(self, store):
        self.store = store
        self.state = "idle"            # idle | pose | choose
        self.t0 = 0.0
        self.shapes, self.patterns, self.checks = [], [], []
        self.pending = None
        self.message = ""
        self.message_until = 0.0

    @property
    def active(self):
        return self.state != "idle"

    @property
    def choosing(self):
        return self.state == "choose"

    def _say(self, text, now, secs=4.0):
        self.message = text
        self.message_until = now + secs
        print(text)

    def start(self, now):
        self.state, self.t0 = "pose", now
        self.shapes, self.patterns, self.checks = [], [], []
        self.pending = None
        self.message_until = 0.0

    def cancel(self, now):
        if self.active:
            self.state = "idle"
            self._say("Gesture recording cancelled", now)

    def update(self, landmarks, w, h, now, cfg):
        if self.state != "pose":
            return
        el = now - self.t0
        if el < self.PREP_S:
            return
        if el < self.PREP_S + self.CAPTURE_S:
            if landmarks is not None:
                pts = np.array([[l.x * w, l.y * h] for l in landmarks])
                scale = np.linalg.norm(pts[9] - pts[0]) + 1e-6
                self.shapes.append(pose_shape(pts))
                self.patterns.append(tuple(fingers_extended(pts, cfg.extend_ratio)))
                tight = all(np.linalg.norm(pts[t] - pts[0]) < cfg.fist_tip_ratio * scale
                            for t in (8, 12, 16, 20))
                tucked = np.linalg.norm(pts[4] - pts[6]) < 0.6 * scale
                clicky = looks_like_click(hand_ratios(landmarks, w, h), self.patterns[-1], cfg)
                self.checks.append((clicky, tight and tucked))
            return
        self._finish_pose(now, cfg)

    def _finish_pose(self, now, cfg):
        if len(self.shapes) < self.MIN_SAMPLES:
            self.t0 = now
            self.shapes, self.patterns, self.checks = [], [], []
            print("Hand not visible enough - hold the pose again")
            return
        # keep only the frames that show the most common finger pattern
        pattern = max(set(self.patterns), key=self.patterns.count)
        keep = [i for i, p in enumerate(self.patterns) if p == pattern]
        if len(keep) < self.MIN_SAMPLES:
            self.state = "idle"
            self._say("Pose was not steady - press G and hold it still", now, 5.0)
            return
        clicky = sum(self.checks[i][0] for i in keep) > len(keep) / 2
        plain_fist = sum(self.checks[i][1] for i in keep) > len(keep) / 2
        problem = None
        if all(pattern):
            problem = "an open palm is the Alt+Tab swipe"
        elif pattern in ((True, False, False, False), (True, True, False, False)):
            problem = "one or two fingers up is used for pointing"
        elif plain_fist:
            problem = "a plain fist is the freeze gesture"
        elif clicky:
            problem = "the thumb touches your ring / pinky finger (that is a click)"
        if problem:
            self.state = "idle"
            self._say(f"Not saved: {problem}. Try a different pose (G).", now, 6.0)
            return
        idx = np.linspace(0, len(keep) - 1, min(self.KEEP_SAMPLES, len(keep))).round().astype(int)
        self.pending = (pattern, [self.shapes[keep[i]] for i in idx])
        self.state = "choose"

    def choose(self, key, now):
        """Number key pressed while choosing. Returns True if handled."""
        for k, name, action, repeat in GESTURE_PRESETS:
            if key == ord(k):
                pattern, samples = self.pending
                self.store.add(CustomGesture(name, action, repeat, pattern, samples))
                self.state = "idle"
                self._say(f"Saved gesture '{name}' ({len(self.store.items)} total)", now)
                return True
        return False

    def display(self, now):
        el = now - self.t0
        if el < self.PREP_S:
            return f"Get ready... {self.PREP_S - el:.1f}s", False, el / self.PREP_S
        return "HOLD IT - recording", True, min((el - self.PREP_S) / self.CAPTURE_S, 1.0)


# ============================================================
# One Euro filter: smooth when slow, responsive when fast
# ============================================================
class OneEuro:
    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


# ============================================================
# Core controller
# ============================================================
class AirMouse:
    def __init__(self, cfg, frame_w, frame_h):
        self.cfg = cfg
        self.sw, self.sh = pyautogui.size()

        # Active region with the same aspect ratio as the screen.
        rw = cfg.region_width
        rh = rw * (frame_w / frame_h) * (self.sh / self.sw)
        if rh > 0.9:
            rw *= 0.9 / rh
            rh = 0.9
        cx = min(max(cfg.region_center[0], rw / 2), 1 - rw / 2)
        cy = min(max(cfg.region_center[1], rh / 2), 1 - rh / 2)
        self.region = (cx - rw / 2, cy - rh / 2, rw, rh)  # normalized x, y, w, h

        self.fx = OneEuro(cfg.min_cutoff, cfg.beta)
        self.fy = OneEuro(cfg.min_cutoff, cfg.beta)

        self.paused = False
        self.fx_mode = False   # set from main; enables the peace-sign macro
        self.status = "NO HAND"

        self.lost = 0
        self.pose = "NONE"
        self._cand = "NONE"
        self._cand_n = 0

        self.left_down = False
        self.left_pending = False   # pinch seen: release = click, hold = drag
        self.left_pending_t = 0.0
        self.left_n = 0
        self.right_latched = False
        self.right_n = 0

        self.debug = True
        self.dbg = ""

        # OK sign (scroll)
        self.ok_n = 0
        self.ok_miss = 0
        self.scrolling = False
        self.scroll_y = None
        self.scroll_acc = 0.0

        # Peace sign macro
        self.peace_start = None
        self.peace_miss = 0
        self.peace_armed = True
        self.last_yt = -1e9
        self.peace_pct = 0

        # Open-palm swipe -> Alt+Tab
        self.swipe_enabled = cfg.swipe_enabled
        self.swipe_hist = deque()
        self.last_swipe = -1e9
        self.last_swipe_dir = 0
        self.alt_down = False
        self.alt_last = 0.0
        self.mod = "command" if sys.platform == "darwin" else "alt"
        self.flash_text = ""
        self.flash_until = 0.0

        # Two-hand pinch zoom (set from main; only used by the HUD)
        self.zoom = None

        # Custom gestures (set from main)
        self.gestures = None
        self.cg = None            # gesture currently being held
        self.cg_start = 0.0
        self.cg_fired = None
        self.cg_miss = 0
        self.cg_pct = 0

    # ---------- state helpers ----------
    def release_all(self):
        if self.left_down:
            pyautogui.mouseUp()
        self.left_down = False
        self.left_pending = False
        self.left_n = 0
        self.right_latched = False
        self.right_n = 0
        self.ok_n = 0
        self.ok_miss = 0
        self.scrolling = False
        self.scroll_y = None
        self.scroll_acc = 0.0
        self.peace_start = None
        self.peace_miss = 0
        self.peace_armed = True
        self._release_alt()
        self.swipe_hist.clear()
        self.cg = None
        self.cg_miss = 0

    def reset_filters(self):
        self.fx.reset()
        self.fy.reset()

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.release_all()
            self.reset_filters()
            self.status = "PAUSED"
        else:
            self.status = "NO HAND"
        print("Air mouse", "PAUSED" if self.paused else "resumed")

    # ---------- no hand ----------
    def on_no_hand(self):
        if self.paused:
            return
        self.lost += 1
        self.status = "NO HAND"
        if self.lost > self.cfg.lost_grace_frames:
            self.release_all()
            self.reset_filters()
            self.pose = "NONE"
            self._cand, self._cand_n = "NONE", 0

    # ---------- hand present ----------
    def _fingers(self, pts):
        cfg = self.cfg
        wrist = pts[0]

        def ext(tip, pip):
            return bool(np.linalg.norm(pts[tip] - wrist)
                        > np.linalg.norm(pts[pip] - wrist) * cfg.extend_ratio)

        return ext(8, 6), ext(12, 10), ext(16, 14), ext(20, 18)

    def _to_screen(self, x, y, now):
        rx, ry, rw, rh = self.region
        u = min(max((x - rx) / rw, 0.0), 1.0)
        v = min(max((y - ry) / rh, 0.0), 1.0)
        u = self.fx(u, now)
        v = self.fy(v, now)
        m = 5  # stay off the corners so our own moves never trip the fail-safe
        sx = m + u * (self.sw - 1 - 2 * m)
        sy = m + v * (self.sh - 1 - 2 * m)
        return int(sx), int(sy)

    # ---------- Alt+Tab swipe ----------
    def _release_alt(self):
        if self.alt_down:
            pyautogui.keyUp(self.mod)
            self.alt_down = False

    def _alt_tab(self, direction, now):
        if not self.alt_down:
            pyautogui.keyDown(self.mod)       # hold Alt: the window switcher stays open
            self.alt_down = True
        if direction < 0:
            pyautogui.keyDown("shift")        # Shift+Tab = move left
            pyautogui.press("tab")
            pyautogui.keyUp("shift")
        else:
            pyautogui.press("tab")            # Tab = move right
        self.alt_last = now
        self.flash_text = "ALT-TAB  <<" if direction < 0 else "ALT-TAB  >>"
        self.flash_until = now + 0.6

    def _track_swipe(self, pts, w, h, now):
        cfg = self.cfg
        palm = (pts[0] + pts[5] + pts[9] + pts[13] + pts[17]) / 5.0
        cx, cy = palm[0] / w, palm[1] / h
        hist = self.swipe_hist
        hist.append((now, cx, cy))
        while hist and now - hist[0][0] > cfg.swipe_window_s:
            hist.popleft()
        if len(hist) < 4 or now - self.last_swipe < cfg.swipe_cooldown_s:
            return
        t_new, x_new, y_new = hist[-1]
        best = None
        for t0, x0, y0 in hist:                      # try every start point in the window
            dt = t_new - t0
            if dt < 0.08:
                continue
            dx, dy = x_new - x0, y_new - y0
            hz, vt = abs(dx), abs(dy)
            if hz < cfg.swipe_dist or hz < vt * cfg.swipe_horizontal_ratio:
                continue
            speed = hz / dt
            if speed < cfg.swipe_min_speed:
                continue
            if best is None or speed > best[2]:      # judge by the fastest qualifying stroke
                best = (hz, dx, speed)
        if best is None:
            return
        # The frame is mirrored, so +x on screen = your hand moving right.
        direction = 1 if best[1] > 0 else -1
        if (direction == -self.last_swipe_dir and now - self.last_swipe < cfg.swipe_reverse_lock_s
                and best[2] < cfg.swipe_reverse_speed):
            return                                   # just the hand returning
        self._alt_tab(direction, now)
        self.last_swipe, self.last_swipe_dir = now, direction
        hist.clear()

    def tick(self, now):
        """Call every frame, even with no hand: lets go of Alt after a pause."""
        if self.alt_down and now - self.alt_last > self.cfg.alt_hold_s:
            self._release_alt()

    def _custom_gesture(self, pts, pattern, raw, click_like, now):
        """Track a held custom gesture; fire its action after gesture_hold_s.
        Returns True while a custom gesture is being held (cursor + clicks pause)."""
        cfg = self.cfg
        g = None
        if (self.gestures is not None and self.gestures.items and raw != "PALM"
                and not click_like and not self.left_down and not self.left_pending
                and not self.right_latched and self.ok_n == 0):
            g, _ = self.gestures.match(pts, pattern, cfg.gesture_match_dist)
        if g is not None:
            self.cg_miss = 0
            if self.cg is not g:
                self.cg, self.cg_start, self.cg_fired = g, now, None
            held = now - self.cg_start
            self.cg_pct = int(min(held / cfg.gesture_hold_s, 1.0) * 100)
            if held >= cfg.gesture_hold_s and (
                    self.cg_fired is None or (g.repeat and now - self.cg_fired >= cfg.gesture_repeat_s)):
                run_action(g.action, g.name)
                self.cg_fired = now
        elif self.cg is not None:
            self.cg_miss += 1
            if self.cg_miss > 3:                    # pose really ended: re-arm
                self.cg = None
        return self.cg is not None

    def on_hand(self, landmarks, w, h, now):
        if self.paused:
            return
        cfg = self.cfg
        self.lost = 0

        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
        scale = np.linalg.norm(pts[9] - pts[0]) + 1e-6  # wrist -> middle knuckle
        thumb = pts[4]
        idx_ratio = np.linalg.norm(thumb - pts[8]) / scale     # thumb-index : OK sign (scroll)
        ring_ratio = np.linalg.norm(thumb - pts[16]) / scale   # thumb-ring  : left click
        pinky_ratio = np.linalg.norm(thumb - pts[20]) / scale  # thumb-pinky : right click
        spread = np.linalg.norm(pts[8] - pts[12]) / scale      # index-middle gap

        index, middle, ring, pinky = self._fingers(pts)
        # Tight fist: every fingertip pulled in close to the wrist. A pinch with
        # loosely curled fingers is NOT a fist (the pinching fingertip stays out).
        tight = all(np.linalg.norm(pts[t] - pts[0]) < cfg.fist_tip_ratio * scale
                    for t in (8, 12, 16, 20))
        # Open palm: all four fingers up and NOT pinching anything.
        all_up = index and middle and ring and pinky
        pinching_any = (idx_ratio < cfg.scroll_pinch_close or ring_ratio < cfg.pinch_close
                        or pinky_ratio < cfg.right_pinch_close)
        if tight:
            raw = "FIST"
        elif (self.swipe_enabled and all_up and not pinching_any and self.ok_n == 0
              and not self.left_pending and not self.left_down):
            raw = "PALM"
        else:
            raw = "POINT"
        self.dbg = (f"T-I{idx_ratio:.2f} T-R{ring_ratio:.2f} T-P{pinky_ratio:.2f} "
                    f"fing {int(index)}{int(middle)}{int(ring)}{int(pinky)} {raw}")

        # Debounced pose (FIST freezes the cursor).
        if raw == self._cand:
            self._cand_n += 1
        else:
            self._cand, self._cand_n = raw, 1
        if self._cand_n >= cfg.pose_confirm_frames:
            if self.pose != self._cand and self._cand != "POINT":
                self.reset_filters()
            self.pose = self._cand

        # ---------- Open palm swipe -> Alt+Tab ----------
        if raw == "PALM" and self.pose == "PALM":
            self._track_swipe(pts, w, h, now)
        else:
            self.swipe_hist.clear()
            if self.pose != "PALM":
                self._release_alt()       # hand closed: pick the highlighted window

        # ---------- Custom gestures (recorded with G) ----------
        click_like = looks_like_click((idx_ratio, ring_ratio, pinky_ratio), (index, middle, ring, pinky), cfg)
        custom_active = self._custom_gesture(pts, (index, middle, ring, pinky), raw, click_like, now)

        # ---------- Peace sign -> YouTube (FX mode only) ----------
        peace_shape = index and middle and not ring and not pinky and spread > cfg.peace_spread
        peace_active = self.fx_mode and peace_shape
        if peace_active:
            self.peace_miss = 0
            if self.peace_start is None:
                self.peace_start = now
            held = now - self.peace_start
            self.peace_pct = int(min(held / cfg.peace_hold_s, 1.0) * 100)
            if (self.peace_armed and held >= cfg.peace_hold_s
                    and now - self.last_yt > cfg.yt_cooldown_s):
                open_youtube_in_chrome(cfg.youtube_url)
                self.peace_armed = False
                self.last_yt = now
        elif self.peace_start is not None:
            self.peace_miss += 1
            if self.peace_miss > 3:          # gesture really ended: re-arm
                self.peace_start = None
                self.peace_armed = True
                self.peace_miss = 0

        # ---------- OK sign (thumb-index) -> scroll ----------
        # Left click uses the RING finger now, so this sign never clashes with it.
        thr = cfg.scroll_pinch_open if self.ok_n > 0 else cfg.scroll_pinch_close
        ok_shape = (not self.left_down and raw != "FIST" and idx_ratio < thr
                    and middle and ring and pinky)
        if ok_shape:
            self.ok_miss = 0
            self.ok_n += 1
        elif self.ok_n > 0:
            self.ok_miss += 1
            if self.ok_miss > 2:
                self.ok_n = 0
                self.ok_miss = 0
        scrolling = self.ok_n >= cfg.scroll_confirm_frames

        # ---------- Buttons ----------
        # A fist or a peace sign often touches the thumb to the fingers:
        # never treat that as a click.
        clicks_allowed = raw != "FIST" and not peace_shape and self.ok_n == 0 and not custom_active
        # Require the finger being pinched to actually be curled, so a naturally
        # close thumb/finger distance (hand rotated toward the camera) is not a click.
        want_left = (clicks_allowed and not ring
                     and ring_ratio < cfg.pinch_close
                     and ring_ratio <= pinky_ratio)
        want_right = (clicks_allowed and not pinky
                      and pinky_ratio < cfg.right_pinch_close
                      and pinky_ratio < ring_ratio)

        # Left button: pinch + release = click. Pinch held = drag.
        if self.left_down:
            if ring_ratio > cfg.pinch_open:
                pyautogui.mouseUp()
                self.left_down = False
                self.left_n = 0
        elif self.left_pending:
            if ring_ratio > cfg.pinch_open:
                pyautogui.click()                 # quick pinch: a real click
                self.left_pending = False
                self.left_n = 0
            elif now - self.left_pending_t >= cfg.drag_hold_s:
                pyautogui.mouseDown()             # held: start dragging
                self.left_down = True
                self.left_pending = False
        else:
            self.left_n = self.left_n + 1 if want_left else 0
            if self.left_n >= cfg.click_confirm_frames and not self.right_latched:
                self.left_pending = True
                self.left_pending_t = now

        # Right click: fires once per pinch, re-arms when released.
        if self.right_latched:
            if pinky_ratio > cfg.right_pinch_open:
                self.right_latched = False
                self.right_n = 0
        elif not self.left_down and not self.left_pending:
            self.right_n = self.right_n + 1 if want_right else 0
            if self.right_n >= cfg.click_confirm_frames:
                pyautogui.rightClick()
                self.right_latched = True

        # ---------- Scroll ----------
        if scrolling:
            y = pts[9][1] / h
            if self.scroll_y is not None:
                dy = y - self.scroll_y
                if abs(dy) < cfg.scroll_deadzone:
                    dy = 0.0
                self.scroll_acc += -dy * cfg.scroll_gain
                n = int(self.scroll_acc)
                if n:
                    pyautogui.scroll(n)
                    self.scroll_acc -= n
            self.scroll_y = y
            self.status = "SCROLL"
            return
        self.scroll_y = None
        self.scroll_acc = 0.0

        # ---------- Cursor movement ----------
        # Freeze while a pinch is forming so the cursor doesn't drift off the target.
        pinch_forming = ((want_left or want_right) and not self.left_down) or self.left_pending
        moving = self.left_down or (
            self.pose == "POINT" and not pinch_forming and not self.right_latched
            and self.ok_n == 0 and not peace_active and not custom_active
        )

        if moving:
            p = landmarks[cfg.pointer_landmark]
            sx, sy = self._to_screen(p.x, p.y, now)
            pyautogui.moveTo(sx, sy)
            self.status = "DRAGGING" if self.left_down else "MOVE"
        elif custom_active:
            self.status = (f"GESTURE: {self.cg.name} {self.cg_pct}%" if self.cg_fired is None
                           else f"GESTURE: {self.cg.name} (done)")
        elif peace_active:
            self.status = (f"YOUTUBE {self.peace_pct}%" if self.peace_armed
                           else "YOUTUBE OPENED")
        elif self.right_latched:
            self.status = "RIGHT CLICK"
        elif self.ok_n > 0:
            self.status = "OK SIGN..."
        elif self.pose == "PALM":
            self.status = self.flash_text if now < self.flash_until else "PALM: swipe L / R"
        elif self.pose == "FIST":
            self.status = "FROZEN (fist)"
        else:
            self.status = "CLICKING"


# ============================================================
# FX helpers (Rasengan / Absolute Cinema)
# ============================================================
class TwoHandZoom:
    """Two-hand pinch zoom for the air mouse.

    Pinch thumb + index on BOTH hands: the cursor freezes and zoom takes over.
    Moving the pinches apart sends Ctrl+wheel up (zoom in); moving them together
    sends Ctrl+wheel down (zoom out). Browsers, maps, image viewers and Office
    zoom on Ctrl+wheel, applied to the window under the (frozen) cursor.
    """

    WHEEL_NOTCH = 120          # one standard mouse-wheel click

    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = cfg.zoom_enabled
        self.reset()
        self.t_last = -1e9

    def reset(self):
        self.n = 0                 # frames with both hands pinched
        self.active = False
        self.ref = None            # hand gap at the last zoom step
        self.miss = 0
        self.direction = 0         # last zoom direction, for the HUD
        self.pins = None           # the two pinch points, for drawing

    def _pinch(self, lm, w, h):
        """(is_pinched, pinch_point) for one hand, with hysteresis."""
        pts = np.array([[l.x * w, l.y * h] for l in lm])
        scale = np.linalg.norm(pts[9] - pts[0]) + 1e-6
        ratio = np.linalg.norm(pts[4] - pts[8]) / scale
        limit = self.cfg.scroll_pinch_open if self.active else self.cfg.scroll_pinch_close
        return ratio < limit, (pts[4] + pts[8]) / 2.0

    def _send(self, steps):
        mod = "command" if sys.platform == "darwin" else "ctrl"
        pyautogui.keyDown(mod)
        try:
            pyautogui.scroll(steps * self.WHEEL_NOTCH)
        finally:
            pyautogui.keyUp(mod)

    def update(self, hands, w, h, now):
        """Returns True while zoom owns the hands (the mouse should do nothing)."""
        if now - self.t_last > 0.3:          # not called for a while (other mode): start fresh
            self.reset()
        self.t_last = now
        if not self.enabled:
            self.reset()
            return False

        if len(hands) >= 2:
            (p1, a), (p2, b) = self._pinch(hands[0], w, h), self._pinch(hands[1], w, h)
            both = p1 and p2
        else:
            both = False

        if not both:
            if self.active and self.miss < self.cfg.lost_grace_frames:
                self.miss += 1                # brief tracking drop: hold the zoom
                return True
            self.reset()
            return False

        self.miss = 0
        self.n += 1
        self.pins = (a, b)
        gap = np.linalg.norm(a - b) / w
        if not self.active:
            if self.n < self.cfg.zoom_confirm_frames:
                return True                   # pinches forming: keep the mouse still
            self.active, self.ref = True, gap
            return True

        steps = int((gap - self.ref) / self.cfg.zoom_step)
        if steps:
            steps = max(-self.cfg.zoom_max_steps, min(self.cfg.zoom_max_steps, steps))
            self._send(steps)
            self.ref += steps * self.cfg.zoom_step
            self.direction = 1 if steps > 0 else -1
        return True

    @property
    def status(self):
        if not self.active:
            return "ZOOM: hold both pinches..."
        return {1: "ZOOM IN  +", -1: "ZOOM OUT  -"}.get(self.direction, "ZOOM: spread / squeeze")

    def draw(self, frame):
        if self.pins is None:
            return
        a, b = (tuple(int(v) for v in p) for p in self.pins)
        color = (0, 255, 255) if self.active else (180, 180, 180)
        cv2.line(frame, a, b, color, 2)
        for p in (a, b):
            cv2.circle(frame, p, 9, color, 2)


class LoopingVideo:
    def __init__(self, path):
        cap = cv2.VideoCapture(path) if os.path.exists(path) else None
        self.cap = cap if (cap is not None and cap.isOpened()) else None

    @property
    def ok(self):
        return self.cap is not None

    def read(self):
        if self.cap is None:
            return None
        ret, f = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, f = self.cap.read()
        return f if ret else None

    def release(self):
        if self.cap is not None:
            self.cap.release()


def fingers_extended(pts, ratio):
    wrist = pts[0]
    pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    return [
        np.linalg.norm(pts[t] - wrist) > np.linalg.norm(pts[p] - wrist) * ratio
        for t, p in pairs
    ]


def overlay_rasengan(frame, video_frame, cx, cy, size):
    """Chroma-key a green-screen video onto the frame around (cx, cy)."""
    h, w = frame.shape[:2]
    if video_frame is None or video_frame.size == 0:
        return frame

    # Center-crop to a square first so the clip isn't stretched.
    vh, vw = video_frame.shape[:2]
    side = min(vh, vw)
    y0, x0 = (vh - side) // 2, (vw - side) // 2
    video_frame = video_frame[y0:y0 + side, x0:x0 + side]

    size = int(np.clip(size, 80, 240))
    r = cv2.resize(video_frame, (size, size))

    x1, y1 = int(cx - size // 2), int(cy - size // 2)
    x2, y2 = x1 + size, y1 + size
    sx1, sy1 = max(0, -x1), max(0, -y1)
    sx2, sy2 = size - max(0, x2 - w), size - max(0, y2 - h)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x1 >= x2 or y1 >= y2:
        return frame

    r = r[sy1:sy2, sx1:sx2]
    roi = frame[y1:y2, x1:x2]
    if r.size == 0 or roi.shape[:2] != r.shape[:2]:
        return frame

    hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 30, 30]), np.array([85, 255, 255]))
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    bg = cv2.bitwise_and(roi, roi, mask=mask)
    fg = cv2.bitwise_and(r, r, mask=cv2.bitwise_not(mask))
    frame[y1:y2, x1:x2] = cv2.add(bg, fg)
    return frame


# ============================================================
# Drawing
# ============================================================
def draw_hand(frame, landmarks, w, h, pointer=8):
    pts = [(int(l.x * w), int(l.y * h)) for l in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 3, (0, 0, 255), -1)
    cv2.circle(frame, pts[pointer], 8, (0, 255, 255), 2)  # pointer


def draw_hud(frame, am, fps, fx_mode=False):
    h, w = frame.shape[:2]
    rx, ry, rw, rh = am.region
    cv2.rectangle(
        frame,
        (int(rx * w), int(ry * h)),
        (int((rx + rw) * w), int((ry + rh) * h)),
        (255, 255, 255), 1,
    )
    color = (0, 255, 255) if not am.paused else (0, 165, 255)
    cv2.rectangle(frame, (8, 8), (w - 8, 82 if am.debug else 62), (20, 20, 20), -1)
    cv2.putText(frame, am.status, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    n_g = len(am.gestures.items) if am.gestures is not None else 0
    cv2.putText(frame, f"FPS {fps:.0f} P:pause F:fx({'ON' if fx_mode else 'off'}) D:debug C:calib "
                       f"S:swipe({'ON' if am.swipe_enabled else 'off'}) "
                       f"Z:zoom({'ON' if am.zoom is not None and am.zoom.enabled else 'off'}) "
                       f"G:gesture({n_g}) E:3D Q:quit", (16, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    if am.debug:
        cv2.putText(frame, am.dbg, (16, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
    if time.perf_counter() < _NOTICE["until"]:          # gesture results, e.g. "Screenshot saved"
        lines = textwrap.wrap(_NOTICE["text"], 84)[:3]
        y0 = 92 if am.debug else 72
        cv2.rectangle(frame, (8, y0), (w - 8, y0 + 8 + 16 * len(lines)), (20, 20, 20), -1)
        color = (0, 255, 0) if _NOTICE["ok"] else (0, 0, 255)
        for i, ln in enumerate(lines):
            cv2.putText(frame, ln, (16, y0 + 18 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)


def draw_calibration(frame, calib, now):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    if calib.active:
        text, status, capturing, frac = calib.display(now)
        color = (0, 255, 0) if capturing else (0, 200, 255)
        cv2.rectangle(frame, (8, h - 112), (w - 8, h - 8), (20, 20, 20), -1)
        cv2.putText(frame, f"CALIBRATION {calib.step + 1}/{len(calib.STEPS)}   (Esc = cancel)",
                    (16, h - 92), font, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, text, (16, h - 68), font, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, status, (16, h - 46), font, 0.5, color, 1)
        cv2.putText(frame, calib.live, (16, h - 28), font, 0.42, (0, 255, 0), 1)
        cv2.rectangle(frame, (16, h - 20), (16 + int((w - 32) * frac), h - 14), color, -1)
    elif now < calib.message_until:
        cv2.rectangle(frame, (8, h - 40), (w - 8, h - 8), (20, 20, 20), -1)
        cv2.putText(frame, calib.message, (16, h - 18), font, 0.45, (0, 255, 255), 1)


def draw_recorder(frame, rec, now):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    if rec.state == "pose":
        status, capturing, frac = rec.display(now)
        color = (0, 255, 0) if capturing else (0, 200, 255)
        cv2.rectangle(frame, (8, h - 96), (w - 8, h - 8), (20, 20, 20), -1)
        cv2.putText(frame, "NEW GESTURE   (Esc = cancel)", (16, h - 76), font, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, "Make your pose with the mouse hand and hold it still",
                    (16, h - 52), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, status, (16, h - 30), font, 0.5, color, 1)
        cv2.rectangle(frame, (16, h - 20), (16 + int((w - 32) * frac), h - 14), color, -1)
    elif rec.state == "choose":
        cv2.rectangle(frame, (8, h - 150), (w - 8, h - 8), (20, 20, 20), -1)
        cv2.putText(frame, "Pose recorded! Press a number for its action   (Esc = cancel)",
                    (16, h - 130), font, 0.48, (0, 255, 255), 1)
        for i, (k, name, _, _) in enumerate(GESTURE_PRESETS):
            col, row = i // 5, i % 5
            cv2.putText(frame, f"{k}  {name}", (24 + col * 300, h - 104 + row * 21),
                        font, 0.5, (255, 255, 255), 1)
    elif now < rec.message_until:
        cv2.rectangle(frame, (8, h - 40), (w - 8, h - 8), (20, 20, 20), -1)
        cv2.putText(frame, rec.message, (16, h - 18), font, 0.45, (0, 255, 255), 1)


# ============================================================
# 3D MACHINE EXPLORER (software renderer: OpenCV + NumPy only)
# ============================================================
# Defaults for gesture thresholds (the air_mouse Config overrides these,
# including your calibrated values)
DEFAULTS = dict(
    pinch_close=0.45, pinch_open=0.70,
    right_pinch_close=0.55, right_pinch_open=0.85,
    scroll_pinch_close=0.40, scroll_pinch_open=0.60,
    extend_ratio=1.2, fist_tip_ratio=1.1,
    swipe_dist=0.20, swipe_window_s=0.35, swipe_min_speed=0.9,
)


# ============================================================
# Small math helpers
# ============================================================
def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


I3 = np.eye(3)
Z3 = np.zeros(3)


# ============================================================
# Mesh builders. A mesh is (V, F): V (n,3) float, F (m,3) int,
# triangles wound counter-clockwise when seen from OUTSIDE.
# ============================================================
def merge(*meshes):
    Vs, Fs, off = [], [], 0
    for V, F in meshes:
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    return np.vstack(Vs), np.vstack(Fs)


def place(mesh, R=None, T=None):
    V, F = mesh
    if R is not None:
        V = V @ np.asarray(R).T
    if T is not None:
        V = V + np.asarray(T, float)
    return V, F


def box(sx, sy, sz):
    x, y, z = sx / 2, sy / 2, sz / 2
    V = np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
                  [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]], float)
    F = np.array([[4, 5, 6], [4, 6, 7], [0, 3, 2], [0, 2, 1], [1, 2, 6], [1, 6, 5],
                  [0, 4, 7], [0, 7, 3], [3, 7, 6], [3, 6, 2], [0, 1, 5], [0, 5, 4]])
    return V, F


def _orient(V, axis):
    if axis == "y":
        return V @ Rx(-math.pi / 2).T      # z axis -> y axis
    if axis == "x":
        return V @ Ry(math.pi / 2).T       # z axis -> x axis
    return V


def _ring_solid(ang, ro, ri, h, full):
    """Extruded ring / disc / sector along z. ro may be an array (gears)."""
    k = len(ang)
    n = k if full else k - 1
    ro = np.broadcast_to(np.asarray(ro, float), (k,))
    c, s = np.cos(ang), np.sin(ang)
    zb, zt = -h / 2, h / 2
    verts = []

    def add(arr):
        base = sum(len(a) for a in verts)
        verts.append(arr)
        return base

    ob = add(np.stack([ro * c, ro * s, np.full(k, zb)], 1))
    ot = add(np.stack([ro * c, ro * s, np.full(k, zt)], 1))
    if ri > 0:
        ib = add(np.stack([ri * c, ri * s, np.full(k, zb)], 1))
        it = add(np.stack([ri * c, ri * s, np.full(k, zt)], 1))
    else:
        cb = add(np.array([[0.0, 0.0, zb]]))
        ct = add(np.array([[0.0, 0.0, zt]]))
    F = []
    for i in range(n):
        j = (i + 1) % k if full else i + 1
        F += [[ob + i, ob + j, ot + j], [ob + i, ot + j, ot + i]]            # outer wall
        if ri > 0:
            F += [[ib + i, it + j, ib + j], [ib + i, it + i, it + j]]        # inner wall
            F += [[ot + i, ot + j, it + j], [ot + i, it + j, it + i]]        # top ring
            F += [[ob + i, ib + j, ob + j], [ob + i, ib + i, ib + j]]        # bottom ring
        else:
            F += [[ot + i, ot + j, ct]]
            F += [[ob + i, cb, ob + j]]
    if not full and ri > 0:                                                  # cut faces
        a, b = 0, k - 1
        F += [[ib + a, ob + a, ot + a], [ib + a, ot + a, it + a]]
        F += [[ib + b, ot + b, ob + b], [ib + b, it + b, ot + b]]
    return np.vstack(verts), np.array(F)


def tube(ro, ri=0.0, h=1.0, n=24, a0=0.0, a1=2 * math.pi, axis="z"):
    full = abs((a1 - a0) - 2 * math.pi) < 1e-9
    k = n if full else n + 1
    ang = a0 + (a1 - a0) * np.arange(k) / n
    V, F = _ring_solid(ang, ro, ri, h, full)
    return _orient(V, axis), F


def gear(teeth, root, tip, h, phase=0.0):
    pa = 2 * math.pi / teeth
    ang, rad = [], []
    for t in range(teeth):
        c = phase + t * pa
        ang += [c - 0.36 * pa, c - 0.14 * pa, c + 0.14 * pa, c + 0.36 * pa]
        rad += [root, tip, tip, root]
    V, F = _ring_solid(np.array(ang), np.array(rad), 0.0, h, True)
    return V, F


def prism(profile, width):
    """Convex 2D side profile (x, y), counter-clockwise, extruded along z."""
    P = np.asarray(profile, float)
    n, h = len(P), width / 2.0
    V = np.vstack([np.column_stack([P, np.full(n, -h)]),
                   np.column_stack([P, np.full(n, h)])])
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i]]            # side walls
    for i in range(1, n - 1):
        F += [[n, n + i, n + i + 1], [0, i + 1, i]]        # front / back caps
    return V, np.array(F)


def slab(length, thick, width, angle, center):
    """Thin box whose long (y) axis is tilted by `angle` about z (slanted glass / pillars)."""
    return place(box(thick, length, width), R=Rz(angle), T=center)


def section(yb, yt, wb, wm, wt):
    """Convex cross-section ring (z, y): half-width wb at the bottom, wm at ~55% height, wt at the top."""
    s = np.array([0.0, 0.12, 0.28, 0.45, 0.62, 0.82, 1.0])
    w = np.where(s <= 0.55,
                 wb + (wm - wb) * np.sin(np.pi / 2 * np.minimum(s / 0.55, 1.0)),
                 wm - (wm - wt) * (np.maximum(s - 0.55, 0.0) / 0.45) ** 1.5)
    y = yb + s * (yt - yb)
    return np.vstack([np.column_stack([w, y]), np.column_stack([-w[::-1], y[::-1]])])


def loft(stations):
    """Closed solid skinned through cross-section rings.
    stations: [(x, ring)], ring = (n,2) array of (z, y) points, the same n for every station."""
    n = len(stations[0][1])
    S = len(stations)
    V = np.array([[x, y, z] for x, ring in stations for z, y in ring], float)
    quads = []
    for i in range(S - 1):
        for k in range(n):
            k2 = (k + 1) % n
            quads.append((i * n + k, i * n + k2, (i + 1) * n + k2, (i + 1) * n + k))

    def tris(flip):
        out = []
        for a, b, d, c in quads:
            out += [[a, d, b], [a, c, d]] if flip else [[a, b, d], [a, d, c]]
        return out

    a, b, d, _ = quads[0]                      # pick the winding that points away from the axis
    nrm = np.cross(V[b] - V[a], V[d] - V[a])
    ref = (V[:n].mean(0) + V[n:2 * n].mean(0)) / 2
    flip = float(np.dot(nrm, (V[a] + V[b] + V[d]) / 3 - ref)) < 0
    F = tris(flip)
    for ring_i, outward in ((0, np.sign(stations[0][0] - stations[1][0])),
                            (S - 1, np.sign(stations[-1][0] - stations[-2][0]))):
        base = ring_i * n
        for k in range(1, n - 1):
            t = [base, base + k, base + k + 1]
            if np.cross(V[t[1]] - V[t[0]], V[t[2]] - V[t[0]])[0] * outward < 0:
                t = [t[0], t[2], t[1]]
            F.append(t)
    return V, np.array(F)


def signed_volume(mesh):
    V, F = mesh
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def refine_mesh(mesh, limit, max_iter=8):
    """Split triangles whose longest edge exceeds `limit` (longest-edge bisection).
    Big triangles break the painter's-algorithm depth sort, small ones don't."""
    V, F = mesh
    V, F = V.copy(), F.copy()
    for _ in range(max_iter):
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        e = np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                      np.linalg.norm(a - c, axis=1)], 1)
        longest = e.max(1)
        sel = longest > limit
        if not sel.any():
            break
        k = e.argmax(1)[sel]
        f = F[sel]
        r = np.arange(len(f))
        i0, i1, i2 = f[r, k], f[r, (k + 1) % 3], f[r, (k + 2) % 3]
        mid = (V[i0] + V[i1]) / 2
        mi = len(V) + r
        V = np.vstack([V, mid])
        F = np.vstack([F[~sel], np.stack([i0, mi, i2], 1), np.stack([mi, i1, i2], 1)])
    return V, F


def refine_parts(parts, limit):
    for p in parts:
        p.V, p.F = refine_mesh((p.V, p.F), limit)
        p.F = p.F.astype(np.int32)
    return parts


# ============================================================
# Parts and machines
# ============================================================
_SHINE = {}      # colour -> shininess, filled in below the colour list


class Part:
    def __init__(self, name, mesh, color, desc="", explode=(0, 0, 0), two_sided=False, shine=None):
        self.name = name
        self.V = np.asarray(mesh[0], float)
        self.F = np.asarray(mesh[1], np.int32)
        self.color = color                    # BGR
        self.desc = desc
        self.explode = np.asarray(explode, float)
        self.two_sided = two_sided
        self.shine = _SHINE.get(tuple(color), 0.25) if shine is None else shine


class Machine:
    def __init__(self, name, parts, pose_fn=None, info="", view=(0.55, 0.30)):
        self.name = name
        self.view = view                  # default (yaw, pitch) of the camera
        self.parts = parts
        self.pose_fn = pose_fn
        self.animated = pose_fn is not None
        self.info = info
        self._static = [(I3, Z3)] * len(parts)
        self._finalize()

    def pose(self, t):
        return self.pose_fn(t) if self.pose_fn else self._static

    def _finalize(self):
        Vs, Fs, vp, fp, vs, off = [], [], [], [], [], 0
        for i, p in enumerate(self.parts):
            Vs.append(p.V)
            Fs.append(p.F + off)
            vp.append(np.full(len(p.V), i))
            fp.append(np.full(len(p.F), i))
            vs.append((off, off + len(p.V)))
            off += len(p.V)
        self.V = np.vstack(Vs)
        self.F = np.vstack(Fs)
        self.vp = np.concatenate(vp)
        self.fp = np.concatenate(fp)
        self.vs = vs
        self.col = np.array([p.color for p in self.parts], float)
        self.ts = np.array([p.two_sided for p in self.parts])
        self.ex = np.array([p.explode for p in self.parts])
        self.shine = np.array([p.shine for p in self.parts], float)
        # bounds over the animation, assembled (e=0) and fully exploded (e=1)
        times = [0.0, 0.7, 1.4, 2.1, 2.8, 3.5, 4.2] if self.animated else [0.0]
        lo = {0: np.full(3, 1e9), 1: np.full(3, 1e9)}
        hi = {0: np.full(3, -1e9), 1: np.full(3, -1e9)}
        for e in (0, 1):
            for t in times:
                Vw = self.world(t, e)
                lo[e] = np.minimum(lo[e], Vw.min(0))
                hi[e] = np.maximum(hi[e], Vw.max(0))
        self.c0, self.c1 = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
        self.r0 = self._fit_radius(hi[0] - lo[0])
        self.r1 = self._fit_radius(hi[1] - lo[1])
        self.ymin0, self.ymin1 = float(lo[0][1]), float(lo[1][1])

    @staticmethod
    def _fit_radius(ext):
        # height matters most on a landscape screen; width/depth count a bit less
        return 0.5 * float(max(ext[1], 0.8 * ext[0], 0.8 * ext[2]))

    def world(self, t, e):
        poses = self.pose(t)
        R = np.stack([p[0] for p in poses])
        T = np.stack([p[1] for p in poses]) + e * self.ex
        return np.einsum("nij,nj->ni", R[self.vp], self.V) + T[self.vp]


# ------------------------------------------------------------
# Colours (BGR)
# ------------------------------------------------------------
STEEL = (178, 172, 166)
IRON = (104, 100, 98)
ALU = (214, 208, 202)
COPPER = (60, 125, 200)
RED = (60, 60, 205)
BLUE = (200, 125, 55)
DARK = (58, 56, 56)
GOLD = (60, 175, 225)
CERAMIC = (226, 226, 216)
ORANGE = (30, 135, 245)
_SHINE.update({STEEL: 0.45, ALU: 0.40, GOLD: 0.50, CERAMIC: 0.35, COPPER: 0.40,
               IRON: 0.20, DARK: 0.15, ORANGE: 0.20, RED: 0.30, BLUE: 0.30})
CHIRON_BLUE = (168, 88, 22)      # deep "French racing" blue
CARBON = (44, 42, 42)
TINT = (96, 78, 58)              # tinted glass
_SHINE.update({CHIRON_BLUE: 0.60, CARBON: 0.35, TINT: 0.70})


# ============================================================
# Built-in machine 1: single-cylinder 4-stroke engine
# ============================================================
def make_engine():
    r, L = 0.45, 1.5
    cut0, cut1 = -math.pi / 2 + 1.0, 3 * math.pi / 2 - 1.0     # opening faces the viewer (+z)

    # --- static parts (built in world coordinates) ---
    block = merge(
        place(tube(0.80, 0.58, 1.95, 44, cut0, cut1, "y"), T=(0, 1.625, 0)),
        *[place(tube(0.95, 0.78, 0.06, 44, cut0, cut1, "y"), T=(0, y, 0))
          for y in (0.9, 1.2, 1.5, 1.8, 2.1, 2.4)])
    head = merge(
        place(tube(0.85, 0, 0.40, 40, axis="y"), T=(0, 2.8, 0)),
        *[place(tube(1.0, 0.84, 0.05, 40, axis="y"), T=(0, y, 0)) for y in (2.68, 2.82, 2.96)])
    plug = merge(
        place(tube(0.12, 0, 0.12, 6, axis="y"), T=(0, 3.06, 0)),
        place(tube(0.07, 0, 0.40, 16, axis="y"), T=(0, 3.32, 0)),
        place(tube(0.035, 0, 0.18, 12, axis="y"), T=(0, 3.60, 0)))
    crankcase = place(tube(1.15, 1.03, 1.5, 40, math.pi, 2 * math.pi, "z"))

    # --- moving parts (built around their own pivot) ---
    piston = merge(
        place(tube(0.55, 0, 0.70, 32, axis="y"), T=(0, 0.12, 0)),
        *[place(tube(0.575, 0.52, 0.035, 32, axis="y"), T=(0, y, 0)) for y in (0.22, 0.30, 0.38)])
    pin = tube(0.09, 0, 0.95, 14, axis="z")
    rod = merge(
        place(tube(0.23, 0.11, 0.30, 24, axis="z"), T=(0, -L / 2, 0)),
        place(tube(0.14, 0.095, 0.30, 20, axis="z"), T=(0, L / 2, 0)),
        place(box(0.15, L - 0.32, 0.10), T=(0, 0.04, 0)))
    crank = merge(
        place(tube(0.13, 0, 2.3, 16, axis="z"), T=(0, 0, -0.2)),
        place(box(0.30, 0.95, 0.10), T=(0, 0.125, 0.2)),
        place(box(0.30, 0.95, 0.10), T=(0, 0.125, -0.2)),
        place(tube(0.10, 0, 0.5, 16, axis="z"), T=(0, r, 0)),
        place(tube(0.20, 0, 0.12, 20, axis="z"), T=(0, 0, 0.65)),
        place(tube(0.20, 0, 0.12, 20, axis="z"), T=(0, 0, -0.65)))
    cw_a0, cw_a1 = -math.pi / 2 - 0.9, -math.pi / 2 + 0.9
    counter = merge(
        place(tube(0.55, 0.14, 0.12, 20, cw_a0, cw_a1, "z"), T=(0, 0, 0.33)),
        place(tube(0.55, 0.14, 0.12, 20, cw_a0, cw_a1, "z"), T=(0, 0, -0.33)))
    spokes = [place(place(box(0.16, 0.55, 0.10), T=(0, 0.55, 0)), R=Rz(k * math.pi / 2))
              for k in range(4)]
    fly = merge(
        place(tube(0.95, 0.80, 0.22, 48, axis="z"), T=(0, 0, -1.05)),
        place(tube(0.30, 0, 0.22, 24, axis="z"), T=(0, 0, -1.05)),
        *[place(sp, T=(0, 0, -1.05)) for sp in spokes])

    def valve():
        return merge(
            place(tube(0.035, 0, 0.8, 10, axis="y"), T=(0, 0.4, 0)),
            place(tube(0.17, 0, 0.04, 18, axis="y"), T=(0, -0.02, 0)),
            place(tube(0.09, 0.07, 0.30, 14, axis="y"), T=(0, 0.55, 0)))

    parts = [
        Part("Crankcase", crankcase, IRON,
             "Lower housing (shown cut away) that supports the crankshaft and holds the engine oil.",
             (0, -1.8, 0)),
        Part("Cylinder Block", block, STEEL,
             "Main body containing the cylinder bore. Cooling fins shed heat. Cut away so you can see the piston.",
             (0, 0.6, 0)),
        Part("Cylinder Head", head, ALU,
             "Seals the top of the cylinder and carries the valves and the spark plug.",
             (0, 2.4, 0)),
        Part("Spark Plug", plug, CERAMIC,
             "Ignites the compressed air-fuel mixture with an electric spark.",
             (0, 3.6, 0)),
        Part("Intake Valve", valve(), BLUE,
             "Opens during the intake stroke to let fresh air-fuel mixture into the cylinder.",
             (-1.7, 2.6, 0)),
        Part("Exhaust Valve", valve(), RED,
             "Opens at the end of the cycle to let burnt gases out.",
             (1.7, 2.6, 0)),
        Part("Piston", piston, ALU,
             "Slides up and down in the bore. Combustion pressure pushes it down, which turns the crank.",
             (-2.2, 0.8, 0)),
        Part("Wrist Pin", pin, GOLD,
             "Joins the piston to the connecting rod while letting the rod swing.",
             (-2.2, 1.7, 0)),
        Part("Connecting Rod", rod, STEEL,
             "Transfers the piston's straight push into rotation at the crankshaft.",
             (2.2, 0.2, 0)),
        Part("Crankshaft", crank, DARK,
             "Converts the piston's up-and-down motion into rotation.",
             (0, -0.6, 1.8)),
        Part("Counterweights", counter, COPPER,
             "Balance the weight of the piston and rod to reduce vibration.",
             (-1.8, -1.2, 1.2)),
        Part("Flywheel", fly, IRON,
             "Stores rotational energy to smooth out the power pulses between strokes.",
             (0, 0, -2.6)),
    ]

    def pose(t):
        th = t * 2.2
        pin_pos = np.array([r * math.sin(th), r * math.cos(th), 0.0])
        yp = pin_pos[1] + math.sqrt(L * L - pin_pos[0] ** 2)
        top = np.array([0.0, yp, 0.0])
        d = top - pin_pos
        rod_R = Rz(math.atan2(-d[0], d[1]))
        ph = th % (4 * math.pi)
        lift_i = 0.10 * math.sin(math.pi * (ph - 0.15 * math.pi) / (0.75 * math.pi)) \
            if 0.15 * math.pi < ph < 0.90 * math.pi else 0.0
        lift_e = 0.10 * math.sin(math.pi * (ph - 3.15 * math.pi) / (0.75 * math.pi)) \
            if 3.15 * math.pi < ph < 3.90 * math.pi else 0.0
        spin = Rz(-th)
        return [
            (I3, Z3), (I3, Z3), (I3, Z3), (I3, Z3),
            (I3, np.array([-0.28, 2.58 - lift_i, 0.0])),
            (I3, np.array([0.28, 2.58 - lift_e, 0.0])),
            (I3, top), (I3, top),
            (rod_R, (pin_pos + top) / 2),
            (spin, Z3), (spin, Z3), (spin, Z3),
        ]

    return Machine("Single-cylinder 4-stroke engine", parts, pose,
                   "Intake, compression, power, exhaust")


# ============================================================
# Built-in machine 2: two-stage gear reducer
# ============================================================
def make_gearbox():
    n1, n2, mod = 12, 20, 0.2
    r1, r2 = n1 * mod / 2, n2 * mod / 2
    c1, c2 = np.array([-1.1, 0, 0]), np.array([2.1, 0, 0])

    pinion = gear(n1, r1 - 0.25, r1 + 0.20, 0.5, 0.0)
    big = gear(n2, r2 - 0.25, r2 + 0.20, 0.5, math.pi / n2)
    key_box = place(box(0.10, 0.10, 0.28), T=(0.25, 0, 0.78))
    shaft_in = merge(place(tube(0.22, 0, 1.54, 20, axis="z"), T=(0, 0, 0.13)), key_box)
    shaft_out = merge(place(tube(0.30, 0, 1.54, 24, axis="z"), T=(0, 0, 0.13)),
                      place(box(0.12, 0.12, 0.28), T=(0.33, 0, 0.78)))
    back = place(box(6.6, 5.4, 0.22), T=(0.5, -0.05, -0.75))

    def bearing(c, z, ri):
        return place(tube(ri + 0.2, ri, 0.28, 28, axis="z"), T=(c[0], c[1], z))

    parts = [
        Part("Back Plate", back, IRON,
             "Rigid mounting plate that keeps the bearings, and so the shafts, in exact alignment.",
             (0, 0, -2.4)),
        Part("Input Shaft", shaft_in, STEEL,
             "Carries the pinion and receives torque from the motor. Spins fast.",
             (-0.9, 0, 2.8)),
        Part("Pinion Gear (12 teeth)", pinion, GOLD,
             "Small driving gear. It turns faster than the large gear it meshes with.",
             (-1.2, 0, 1.2)),
        Part("Output Shaft", shaft_out, STEEL,
             "Delivers the slower, higher-torque rotation to the load.",
             (0.9, 0, 3.4)),
        Part("Output Gear (20 teeth)", big, COPPER,
             "Large driven gear. 20 teeth / 12 teeth gives a 1.67 : 1 speed reduction and 1.67x the torque.",
             (1.2, 0, 1.9)),
        Part("Bearing A (input, front)", bearing(c1, 0.5, 0.30), BLUE,
             "Supports the input shaft and lets it spin with low friction.", (-1.4, 1.5, 1.2)),
        Part("Bearing B (input, rear)", bearing(c1, -0.5, 0.30), BLUE,
             "Supports the input shaft at the plate.", (-1.4, -1.5, -1.2)),
        Part("Bearing C (output, front)", bearing(c2, 0.5, 0.38), RED,
             "Supports the output shaft and lets it spin with low friction.", (1.4, 1.5, 1.2)),
        Part("Bearing D (output, rear)", bearing(c2, -0.5, 0.38), RED,
             "Supports the output shaft at the plate.", (1.4, -1.5, -1.2)),
    ]

    refine_parts(parts, 0.6)

    def pose(t):
        th = t * 1.4
        a = Rz(th)
        b = Rz(-th * n1 / n2)
        return [(I3, Z3), (a, c1), (a, c1), (b, c2), (b, c2),
                (I3, Z3), (I3, Z3), (I3, Z3), (I3, Z3)]

    return Machine("Gear reducer (12:20)", parts, pose, "Speed down, torque up")


# ============================================================
# Built-in machine 3: 4-axis robot arm
# ============================================================
def make_robot():
    # Joint heights (world, arm upright): turret top 1.13, shoulder 1.55, elbow +1.75, wrist +1.4
    base = merge(place(tube(1.15, 0.5, 0.08, 40, axis="y"), T=(0, 0.04, 0)),
                 place(tube(1.0, 0.5, 0.35, 40, axis="y"), T=(0, 0.255, 0)))
    turret = place(tube(0.62, 0, 0.70, 32, axis="y"), T=(0, 0.78, 0))
    shoulder = merge(place(tube(0.42, 0, 0.16, 28, axis="z"), T=(0, 1.55, 0.42)),
                     place(tube(0.42, 0, 0.16, 28, axis="z"), T=(0, 1.55, -0.42)),
                     place(tube(0.12, 0, 0.84, 16, axis="z"), T=(0, 1.55, 0)))
    upper = place(box(0.34, 1.75, 0.50), T=(0, 0.875, 0))
    elbow = merge(place(tube(0.30, 0, 0.12, 24, axis="z"), T=(0, 1.75, 0.33)),
                  place(tube(0.30, 0, 0.12, 24, axis="z"), T=(0, 1.75, -0.33)),
                  place(tube(0.10, 0, 0.78, 16, axis="z"), T=(0, 1.75, 0)))
    fore = place(box(0.28, 1.2, 0.38), T=(0, 0.6, 0))
    wrist = place(tube(0.20, 0, 0.56, 24, axis="z"), T=(0, 1.4, 0))
    palm = place(box(0.55, 0.22, 0.24), T=(0, 0.31, 0))
    finger = place(box(0.10, 0.5, 0.14), T=(0, 0.67, 0))

    parts = [
        Part("Base", base, DARK, "Heavy fixed foundation that keeps the arm stable.", (0, -1.2, 0)),
        Part("Turret (joint 1)", turret, ORANGE,
             "Rotates the whole arm left and right.", (0, 0.2, 0)),
        Part("Shoulder Joint (joint 2)", shoulder, DARK,
             "Pivot that raises and lowers the upper arm.", (-1.6, 0.9, 0)),
        Part("Upper Arm", upper, ORANGE, "First long link of the arm.", (1.6, 1.2, 0)),
        Part("Elbow Joint (joint 3)", elbow, DARK,
             "Bends the forearm relative to the upper arm.", (-1.6, 1.9, 0)),
        Part("Forearm", fore, ORANGE, "Second long link; carries the wrist.", (1.6, 2.5, 0)),
        Part("Wrist (joint 4)", wrist, DARK, "Tilts the gripper.", (-1.5, 3.1, 0)),
        Part("Gripper Palm", palm, STEEL, "Mount for the gripper fingers.", (0, 3.6, 0)),
        Part("Gripper Finger L", finger, GOLD, "Closes to grip an object.", (-1.1, 4.0, 0)),
        Part("Gripper Finger R", finger, GOLD, "Closes to grip an object.", (1.1, 4.0, 0)),
    ]
    refine_parts(parts, 0.5)

    def T4(R, t):
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = t
        return M

    def pose(t):
        a1 = 0.6 * math.sin(0.5 * t)
        a2 = 0.35 + 0.35 * math.sin(0.7 * t)
        a3 = 0.9 + 0.6 * math.sin(0.9 * t)
        a4 = 0.5 * math.sin(1.3 * t)
        op = 0.10 + 0.10 * (1 + math.sin(2.0 * t))
        M0 = T4(Ry(a1), (0, 0, 0))
        M1 = M0 @ T4(Rz(a2), (0, 1.55, 0))
        M2 = M1 @ T4(Rz(a3), (0, 1.75, 0))
        M3 = M2 @ T4(Rz(a4), (0, 1.4, 0))

        def p(M, off=(0, 0, 0)):
            return (M[:3, :3], M[:3, 3] + M[:3, :3] @ np.asarray(off, float))
        return [(I3, Z3), p(M0), p(M0), p(M1), p(M1), p(M2), p(M2), p(M3),
                p(M3, (-(0.17 + op), 0, 0)), p(M3, (0.17 + op, 0, 0))]

    return Machine("4-axis robot arm", parts, pose, "Base, shoulder, elbow, wrist")

# ============================================================
# Built-in machine 4: sedan car
# ============================================================
def make_car():
    # Layout: length along x (front = +x), up = +y, width along z.
    PAINT = BLUE
    GLASS = (235, 205, 160)

    # ---------- chassis & powertrain ----------
    chassis = merge(place(box(4.1, 0.10, 1.5), T=(0, 0.35, 0)),
                    place(box(4.1, 0.10, 0.10), T=(0, 0.28, 0.6)),
                    place(box(4.1, 0.10, 0.10), T=(0, 0.28, -0.6)))
    engine = merge(place(box(1.0, 0.45, 0.8), T=(1.35, 0.65, 0)),
                   *[place(tube(0.12, 0, 0.2, 14, axis="y"), T=(x, 0.98, 0))
                     for x in (1.05, 1.25, 1.45, 1.65)])
    trans = merge(place(tube(0.17, 0, 0.7, 20, axis="x"), T=(0.55, 0.58, 0)),
                  place(tube(0.05, 0, 2.0, 12, axis="x"), T=(-0.8, 0.58, 0)))
    exhaust = merge(place(tube(0.04, 0, 3.0, 10, axis="x"), T=(0.0, 0.46, 0.5)),
                    place(tube(0.11, 0, 0.7, 16, axis="x"), T=(-1.6, 0.46, 0.5)))
    tank = place(box(0.7, 0.25, 1.0), T=(-1.5, 0.58, 0))
    axle_f = place(tube(0.06, 0, 1.9, 12, axis="z"), T=(1.4, 0.4, 0))
    axle_r = place(tube(0.06, 0, 1.9, 12, axis="z"), T=(-1.4, 0.4, 0))

    # ---------- body panels (side profiles, extruded across the car) ----------
    lower = prism([(-2.2, 0.40), (2.2, 0.40), (2.25, 0.72), (1.95, 0.90),
                   (-1.95, 0.92), (-2.25, 0.78)], 1.7)
    hood = prism([(0.85, 0.92), (1.95, 0.90), (1.95, 0.97), (0.85, 0.99)], 1.66)
    trunk = prism([(-1.95, 0.92), (-1.15, 0.92), (-1.15, 0.99), (-1.95, 0.99)], 1.66)
    door_l = place(box(1.6, 0.5, 0.06), T=(-0.05, 0.68, 0.88))
    door_r = place(box(1.6, 0.5, 0.06), T=(-0.05, 0.68, -0.88))
    roof = place(box(1.2, 0.06, 1.5), T=(-0.17, 1.48, 0))
    pillars = merge(
        *[slab(0.658, 0.06, 0.06, 0.753, (0.625, 1.21, z)) for z in (0.75, -0.75)],     # A pillars
        *[slab(0.625, 0.06, 0.06, -0.695, (-0.95, 1.21, z)) for z in (0.75, -0.75)],    # C pillars
        *[place(box(0.08, 0.48, 0.06), T=(-0.10, 1.21, z)) for z in (0.75, -0.75)])     # B pillars
    windows = merge(slab(0.658, 0.03, 1.44, 0.753, (0.625, 1.21, 0)),                   # windshield
                    slab(0.625, 0.03, 1.44, -0.695, (-0.95, 1.21, 0)),                  # rear glass
                    place(box(1.55, 0.42, 0.03), T=(-0.175, 1.21, 0.72)),               # side glass
                    place(box(1.55, 0.42, 0.03), T=(-0.175, 1.21, -0.72)))
    seats = merge(*[place(box(0.5, 0.12, 0.5), T=(-0.10, 1.03, z)) for z in (0.35, -0.35)],
                  *[place(box(0.10, 0.34, 0.5), T=(-0.40, 1.20, z)) for z in (0.35, -0.35)])
    wheel_in_car = place(tube(0.17, 0.13, 0.05, 20, axis="x"), T=(0.40, 1.20, 0.35))
    bump_f = place(box(0.14, 0.28, 1.76), T=(2.30, 0.52, 0))
    bump_r = place(box(0.14, 0.28, 1.76), T=(-2.30, 0.52, 0))
    heads = merge(*[place(box(0.08, 0.14, 0.35), T=(2.26, 0.74, z)) for z in (0.55, -0.55)])
    tails = merge(*[place(box(0.08, 0.14, 0.35), T=(-2.26, 0.74, z)) for z in (0.55, -0.55)])

    # ---------- wheels (built at the origin, posed to each corner) ----------
    tire = tube(0.40, 0.26, 0.30, 28, axis="z")
    rim = merge(tube(0.27, 0, 0.20, 24, axis="z"),
                tube(0.09, 0, 0.30, 12, axis="z"),
                *[place(box(0.07, 0.52, 0.24), R=Rz(k * 2 * math.pi / 5)) for k in range(5)])
    corners = [("Front Left", (1.4, 0.4, 0.95), 1), ("Front Right", (1.4, 0.4, -0.95), 1),
               ("Rear Left", (-1.4, 0.4, 0.95), 0), ("Rear Right", (-1.4, 0.4, -0.95), 0)]

    parts = [
        Part("Chassis", chassis, IRON,
             "Load-bearing floor and side rails. Everything else bolts to this frame.", (0, -1.5, 0)),
        Part("Engine", engine, ORANGE,
             "Burns fuel to make power. Sits under the hood and drives the gearbox.",
             (1.6, 1.0, 0)),
        Part("Transmission + Driveshaft", trans, STEEL,
             "Gearbox changes torque and speed; the driveshaft carries power to the rear wheels.",
             (0.4, 0.3, 0)),
        Part("Exhaust", exhaust, DARK,
             "Carries burnt gases away; the muffler quietens them.", (-0.5, -1.0, 1.4)),
        Part("Fuel Tank", tank, COPPER, "Stores the fuel for the engine.", (-1.5, 0.3, 0)),
        Part("Front Axle", axle_f, STEEL, "Connects the front wheels and lets them steer.", (1.0, -1.1, 0)),
        Part("Rear Axle", axle_r, STEEL, "Connects the rear wheels and delivers drive torque.", (-1.0, -1.1, 0)),
        Part("Body Shell", lower, PAINT,
             "Main painted body that wraps the mechanicals and passenger floor.", (0, 0.5, 0), shine=0.55),
        Part("Hood", hood, PAINT, "Lid over the engine bay.", (1.0, 1.8, 0), shine=0.55),
        Part("Trunk Lid", trunk, PAINT, "Lid over the luggage space.", (-1.0, 1.6, 0), shine=0.55),
        Part("Door (left)", door_l, PAINT, "Side door with hinges at the front edge.", (0, 0, 1.8), shine=0.55),
        Part("Door (right)", door_r, PAINT, "Side door with hinges at the front edge.", (0, 0, -1.8), shine=0.55),
        Part("Roof", roof, PAINT, "Panel closing the passenger cabin.", (0, 2.2, 0), shine=0.55),
        Part("Pillars", pillars, DARK, "A, B and C pillars hold the roof up and protect occupants.", (0, 1.3, 0)),
        Part("Windows", windows, GLASS, "Windshield, side and rear glass.", (0, 1.7, 0), shine=0.60),
        Part("Seats", seats, DARK, "Front seats.", (-0.3, 0.9, 0)),
        Part("Steering Wheel", wheel_in_car, STEEL, "Driver's steering input.", (1.0, 1.6, 0.6)),
        Part("Front Bumper", bump_f, DARK, "Absorbs low-speed impacts.", (1.4, -0.2, 0)),
        Part("Rear Bumper", bump_r, DARK, "Absorbs low-speed impacts.", (-1.4, -0.2, 0)),
        Part("Headlights", heads, CERAMIC, "Light the road ahead.", (1.6, 0.8, 0)),
        Part("Tail Lights", tails, RED, "Show your position and braking to following drivers.", (-1.6, 0.8, 0)),
    ]
    n_static = len(parts)
    for name, c, _ in corners:
        parts.append(Part(f"Tire {name}", tire, DARK, "Rubber tire gripping the road.",
                          (np.sign(c[0]) * 1.0, 0, np.sign(c[2]) * 1.8)))
    for name, c, _ in corners:
        parts.append(Part(f"Rim {name}", rim, ALU, "Alloy wheel the tire is mounted on.",
                          (np.sign(c[0]) * 1.0, 0, np.sign(c[2]) * 2.4)))
    refine_parts(parts, 0.7)

    def pose(t):
        spin = t * 3.0
        steer = 0.25 * math.sin(0.8 * t)
        poses = [(I3, Z3)] * n_static
        wheels = []
        for _, c, front in corners:
            R = (Ry(steer) if front else I3) @ Rz(-spin)
            wheels.append((R, np.array(c, float)))
        return poses + wheels + wheels          # tires first, then rims (same motion)

    return Machine("Sedan car", parts, pose, "Chassis, powertrain, body, wheels", view=(-0.7, 0.22))


# ============================================================
# Built-in machine 5: Bugatti Chiron-style hypercar
# (an original, simplified model built from the car's real proportions and
#  signature features - NOT the manufacturer's CAD data)
# ============================================================
def _prof(points):
    pts = sorted(points)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return lambda x: float(np.interp(x, xs, ys))


def make_chiron():
    # +x = nose, +y = up, z = across the car. 1 unit = 1 metre (real car: 4.54 x 2.04 x 1.21 m).
    FA, RA = 1.40, -1.40             # axle positions
    RF, RR = 0.37, 0.385             # front / rear tire radius
    ARCH = 0.44                      # wheel-arch radius

    ytop = _prof([(-2.27, .74), (-2.0, .88), (-1.5, 1.0), (-0.95, .97), (-0.75, .93), (0.85, .92),
                  (1.0, .92), (1.4, .92), (1.9, .86), (2.15, .70), (2.27, .60)])
    wmax = _prof([(-2.27, .78), (-1.9, .92), (-1.4, 1.02), (0.5, 1.02), (1.4, 1.02),
                  (1.7, .99), (2.0, .88), (2.27, .66)])

    def ybot(x):
        yb = 0.14
        for ax, r in ((FA, RF), (RA, RR)):
            dx = abs(x - ax)
            if dx < ARCH:
                yb = max(yb, r + math.sqrt(ARCH * ARCH - dx * dx))
        return yb

    def arch_xs(ax):
        return [ax + ARCH * math.cos(t) for t in np.linspace(0, math.pi, 15)] + [ax + ARCH + 0.004, ax - ARCH - 0.004]

    def body(xs):
        st = []
        for x in sorted({round(float(v), 4) for v in xs}, reverse=True):
            yb = ybot(x)
            yt = max(ytop(x), yb + 0.06)
            wm = wmax(x)
            st.append((x, section(yb, yt, 0.80 * wm, wm, 0.78 * wm)))
        return loft(st)

    body_front = body(list(np.linspace(0.95, 2.27, 10)) + arch_xs(FA))
    body_mid = body(list(np.linspace(-0.95, 0.95, 11)))
    body_rear = body(list(np.linspace(-2.27, -0.95, 10)) + arch_xs(RA))

    # ---------- greenhouse, roof, spine ----------
    cab_top = _prof([(0.85, .945), (0.60, 1.03), (0.30, 1.15), (0.10, 1.205), (-0.45, 1.21),
                     (-0.70, 1.16), (-0.92, 1.0)])
    cab_xs = [0.85, 0.75, 0.60, 0.45, 0.30, 0.15, 0.0, -0.2, -0.45, -0.60, -0.75, -0.85, -0.92]
    cabin = loft([(x, section(ytop(x), max(cab_top(x), ytop(x) + 0.03), 0.79, 0.80, 0.50)) for x in cab_xs])
    roof = loft([(x, section(cab_top(x) - 0.012, cab_top(x) + 0.02, 0.47, 0.47, 0.42))
                 for x in (0.18, 0.05, -0.1, -0.3, -0.45, -0.58)])
    spine = prism([(-2.15, 0.80), (-0.55, 0.80), (-0.55, 1.215), (-2.15, 1.0)], 0.06)
    scoops = merge(*[place(box(0.55, 0.12, 0.30), T=(-1.05, 1.02, z)) for z in (0.50, -0.50)])

    # ---------- front end ----------
    grille = place(tube(0.22, 0.15, 0.05, 28, -0.55, math.pi + 0.55, axis="x"), T=(2.285, 0.37, 0))
    grille_mesh = place(box(0.03, 0.28, 0.34), T=(2.275, 0.36, 0))
    badge = place(tube(0.035, 0, 0.02, 16, axis="x"), T=(2.31, 0.44, 0))
    heads = merge(*[place(box(0.03, 0.06, 0.30), T=(2.285, 0.49, z)) for z in (0.45, -0.45)],
                  *[place(box(0.03, 0.04, 0.22), T=(2.285, 0.36, z)) for z in (0.50, -0.50)])
    splitter = place(box(0.45, 0.03, 1.55), T=(2.30, 0.125, 0))
    front_rad = place(box(0.10, 0.40, 0.95), T=(1.95, 0.38, 0))

    # ---------- sides ----------
    cline_L = place(tube(0.32, 0.29, 0.02, 28, 0.9, 2 * math.pi - 0.9, axis="z"), T=(-0.35, 0.56, 1.035))
    cline_R = place(tube(0.32, 0.29, 0.02, 28, 0.9, 2 * math.pi - 0.9, axis="z"), T=(-0.35, 0.56, -1.035))
    intake_L = place(box(0.30, 0.22, 0.03), T=(-0.30, 0.56, 1.03))
    intake_R = place(box(0.30, 0.22, 0.03), T=(-0.30, 0.56, -1.03))
    mirror_L = place(box(0.16, 0.09, 0.12), T=(0.48, 0.99, 1.03))
    mirror_R = place(box(0.16, 0.09, 0.12), T=(0.48, 0.99, -1.03))
    side_rad = merge(*[place(box(0.50, 0.28, 0.10), T=(-0.55, 0.52, z)) for z in (0.70, -0.70)])

    # ---------- rear end ----------
    tail = place(box(0.06, 0.05, 1.45), T=(-2.275, 0.64, 0))
    diff_fins = merge(place(box(0.8, 0.03, 1.4), R=Rz(0.12)),
                      *[place(box(0.8, 0.10, 0.02), R=Rz(0.12), T=(0, 0.06, z)) for z in (-0.45, -0.22, 0, 0.22, 0.45)])
    diffuser = place(diff_fins, T=(-1.95, 0.16, 0))
    exhaust = merge(*[place(tube(0.055, 0.04, 0.30, 16, axis="x"), T=(-2.30, y, z))
                      for y in (0.30, 0.46) for z in (-0.13, 0.13)])
    wing_blade = merge(prism([(0.22, 0), (0.10, 0.035), (-0.20, 0.03), (-0.26, 0), (-0.20, -0.02), (0.10, -0.025)], 1.70),
                       *[place(box(0.50, 0.14, 0.02), T=(-0.02, 0.0, z)) for z in (0.86, -0.86)])
    wing_pivot = np.array([-2.06, 1.10, 0.0])
    wing_struts = merge(*[place(box(0.05, 0.27, 0.06), T=(-2.06, 0.975, z)) for z in (0.40, -0.40)])

    # ---------- interior ----------
    mono = place(box(2.0, 0.50, 1.40), T=(-0.05, 0.56, 0))
    seats = merge(*[place(box(0.5, 0.10, 0.45), T=(-0.20, 0.52, z)) for z in (0.36, -0.36)],
                  *[place(place(box(0.10, 0.48, 0.42), R=Rz(0.2)), T=(-0.45, 0.78, z)) for z in (0.36, -0.36)],
                  *[place(box(0.08, 0.16, 0.26), T=(-0.52, 1.07, z)) for z in (0.36, -0.36)])
    dash = place(box(0.45, 0.16, 1.55), T=(0.55, 0.96, 0))
    steering = place(tube(0.15, 0.125, 0.035, 20, axis="x"), R=Rz(0.4), T=(0.32, 1.04, -0.36))

    # ---------- powertrain: 8.0 L W16, four turbos, transaxle ----------
    banks = [(-22.5, -0.17), (-7.5, -0.06), (7.5, 0.06), (22.5, 0.17)]
    cyl = [place(tube(0.085, 0, 0.34, 12, axis="y"), R=Rx(math.radians(a)),
                 T=(-1.30 + 0.20 * k + (0.10 if i % 2 else 0.0), 0.74, zo))
           for i, (a, zo) in enumerate(banks) for k in range(4)]
    engine = merge(place(box(0.95, 0.28, 0.50), T=(-1.0, 0.45, 0)),
                   place(box(0.90, 0.10, 0.34), T=(-1.0, 1.00, 0)), *cyl)
    crank = merge(place(tube(0.07, 0, 1.2, 16, axis="x"), T=(-1.0, 0, 0)),
                  place(tube(0.20, 0, 0.06, 24, axis="x"), T=(-0.40, 0, 0)),
                  place(box(0.04, 0.09, 0.07), T=(-0.40, 0.17, 0)))
    turbos = merge(*[place(tube(0.08, 0.04, 0.16, 16, axis="z"), T=(-1.62, 0.86, z)) for z in (-0.34, -0.12, 0.12, 0.34)])
    gearbox = merge(place(box(0.55, 0.42, 0.80), T=(-1.80, 0.40, 0)),
                    place(tube(0.14, 0, 0.5, 20, axis="x"), T=(-1.45, 0.40, 0)))
    axle_r = merge(place(tube(0.045, 0, 1.32, 12, axis="z"), T=(RA, RR, 0)),
                   place(box(0.28, 0.24, 0.30), T=(RA, RR, 0)))
    axle_f = place(tube(0.045, 0, 1.36, 12, axis="z"), T=(FA, RF, 0))

    parts, posers = [], []

    def add(name, mesh, color, desc, explode, pose=None, **kw):
        parts.append(Part(name, mesh, color, desc, explode, **kw))
        posers.append(pose)

    add("Front Body", body_front, CHIRON_BLUE,
        "Nose, hood and front fenders. The low, wide nose helps downforce and cooling.", (2.0, 0.9, 0))
    add("Center Body", body_mid, CHIRON_BLUE,
        "Doors and sills wrapped around the carbon monocoque passenger cell.", (0, 1.3, 0))
    add("Rear Body", body_rear, CHIRON_BLUE,
        "Engine cover and tail. The body tapers to keep air attached over the rear.", (-2.0, 0.9, 0))
    add("Cabin Glass", cabin, TINT, "Tinted windshield, side and rear glass of the low greenhouse.", (0, 2.4, 0))
    add("Carbon Roof", roof, CARBON, "Lightweight carbon-fibre roof panel.", (0, 3.0, 0))
    add("Dorsal Spine", spine, CARBON,
        "The central fin running to the tail. It stabilises airflow and ties the two-tone design together.",
        (-1.2, 2.8, 0))
    add("Engine Air Scoops", scoops, CARBON, "Feed air to the engine and turbochargers.", (-1.2, 2.2, 0))
    add("Horseshoe Grille", grille, ALU, "Bugatti's signature horseshoe, in polished metal.", (3.0, 0.5, 0), shine=0.6)
    add("Grille Mesh", grille_mesh, CARBON, "Mesh behind the horseshoe that lets cooling air in.", (2.6, 0.3, 0))
    add("Badge", badge, RED, "The oval marque badge.", (3.4, 0.9, 0))
    add("Headlights", heads, CERAMIC, "Slim LED headlight strips.", (2.4, 1.0, 0))
    add("Front Splitter", splitter, CARBON, "Carbon lip that creates front downforce.", (2.2, -1.0, 0))
    add("Front Radiator", front_rad, DARK, "Cools the engine coolant.", (2.6, -0.4, 0))
    add("C-Line Trim (left)", cline_L, ALU, "The signature C-shaped line that wraps the side air intake.", (0, 0.4, 2.2), shine=0.6)
    add("C-Line Trim (right)", cline_R, ALU, "The signature C-shaped line that wraps the side air intake.", (0, 0.4, -2.2), shine=0.6)
    add("Side Intake (left)", intake_L, CARBON, "Feeds air to the side radiator and engine.", (0, -0.4, 1.8))
    add("Side Intake (right)", intake_R, CARBON, "Feeds air to the side radiator and engine.", (0, -0.4, -1.8))
    add("Side Radiators", side_rad, DARK, "Extra cooling for the engine, turbo and transmission.", (0, -0.9, 0))
    add("Mirror (left)", mirror_L, CHIRON_BLUE, "Side mirror.", (0, 0.8, 1.5), shine=0.55)
    add("Mirror (right)", mirror_R, CHIRON_BLUE, "Side mirror.", (0, 0.8, -1.5), shine=0.55)
    add("LED Tail Light", tail, RED, "A single LED light bar across the whole tail.", (-2.8, 0.9, 0), shine=0.5)
    add("Rear Diffuser", diffuser, CARBON, "Accelerates air under the car to add downforce and cut drag.", (-2.2, -1.0, 0))
    add("Quad Exhaust", exhaust, STEEL, "Four tailpipes carrying exhaust from the engine.", (-3.2, -0.6, 0))
    add("Rear Wing Struts", wing_struts, DARK, "Fixed supports for the active rear wing.", (-1.5, 1.2, 0))
    add("Active Rear Wing", wing_blade, CARBON,
        "Changes its angle for downforce, braking and top speed. Watch it tilt.", (-1.6, 2.4, 0),
        pose=lambda t: (Rz(0.10 * math.sin(1.2 * t)), wing_pivot))
    add("Carbon Monocoque", mono, CARBON, "The rigid carbon-fibre passenger cell that everything bolts to.", (0, -0.8, 0))
    add("Carbon Seats", seats, DARK, "Two lightweight bucket seats.", (0.3, 1.9, 0))
    add("Dashboard", dash, DARK, "Instrument panel and controls.", (1.0, 1.7, 0))
    add("Steering Wheel", steering, STEEL, "Driver's steering input.", (1.3, 2.0, -0.4))
    add("W16 Engine (8.0 L)", engine, STEEL,
        "Quad-turbo 8.0 litre engine with sixteen cylinders in four banks, in a W layout.", (-1.5, -0.3, 0))
    add("Crankshaft + Flywheel", crank, DARK, "Turns the pistons' push into rotation.", (-2.4, -0.3, 0),
        pose=lambda t: (Rx(-t * 5.0), np.array([0.0, 0.40, 0.0])))
    add("Turbochargers", turbos, COPPER, "Four turbos force extra air into the engine.", (-1.8, 0.9, 0))
    add("Transaxle", gearbox, IRON, "Dual-clutch gearbox that sends power to all four wheels.", (-2.6, -0.9, 0))
    add("Rear Axle + Differential", axle_r, STEEL, "Splits drive torque between the rear wheels.", (-0.8, -1.4, 0))
    add("Front Axle", axle_f, STEEL, "Carries the front wheels and steering.", (1.2, -1.4, 0))

    # ---------- wheels: tire, rim, brake disc, caliper at each corner ----------
    def wheel_meshes(r_t, width):
        tire = tube(r_t, r_t - 0.125, width, 26, axis="z")
        spokes = [place(place(box(0.055, 0.17, 0.10), T=(0, 0.14, 0.03)), R=Rz(k * math.pi / 5)) for k in range(10)]
        rim = merge(place(tube(0.255, 0.215, 0.20, 24, axis="z"), T=(0, 0, 0.0)),
                    place(tube(0.07, 0, 0.22, 14, axis="z"), T=(0, 0, 0.03)), *spokes)
        disc = place(tube(0.21, 0.08, 0.03, 24, axis="z"), T=(0, 0, -0.04))
        caliper = place(box(0.16, 0.12, 0.08), T=(0, 0.18, -0.04))
        return tire, rim, disc, caliper

    for nm, ax, side, front in (("Front Left", FA, 1, True), ("Front Right", FA, -1, True),
                                ("Rear Left", RA, 1, False), ("Rear Right", RA, -1, False)):
        r_t, wd, track = (RF, 0.285, 0.86) if front else (RR, 0.355, 0.84)
        c = np.array([ax, r_t, side * track])
        tire, rim, disc, caliper = wheel_meshes(r_t, wd)
        sx = 1.0 if ax > 0 else -1.0

        def spin_pose(t, c=c, front=front):
            R = (Ry(0.30 * math.sin(0.8 * t)) if front else I3) @ Rz(-t * 3.0)
            return (R, c)

        def steer_pose(t, c=c, front=front):
            return ((Ry(0.30 * math.sin(0.8 * t)) if front else I3), c)

        add(f"Tire {nm}", tire, DARK, "Rubber tire gripping the road.", (sx * 1.0, 0, side * 2.0), pose=spin_pose)
        add(f"Rim {nm}", rim, ALU, "Alloy wheel the tire is mounted on.", (sx * 1.0, 0, side * 2.8), pose=spin_pose, shine=0.5)
        add(f"Brake Disc {nm}", disc, IRON, "Carbon-ceramic disc that the caliper squeezes to slow the car.",
            (sx * 1.0, 0, side * 1.2), pose=spin_pose)
        add(f"Brake Caliper {nm}", caliper, RED, "Clamps the brake pads onto the spinning disc.",
            (sx * 1.0, 0.6, side * 1.5), pose=steer_pose)

    refine_parts(parts, 0.55)

    def pose(t):
        return [p(t) if p else (I3, Z3) for p in posers]

    return Machine("Bugatti Chiron-style hypercar", parts, pose,
                   "Monocoque, W16 engine, active wing (simplified, original model)", view=(-0.7, 0.22))


def builtin_machines():
    return [make_engine(), make_gearbox(), make_robot(), make_car(), make_chiron()]


# ============================================================
# Loading your own STL / OBJ / glTF models
# ============================================================
def load_stl(path):
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack("<I", data[80:84])[0] if len(data) >= 84 else 0
    if len(data) == 84 + 50 * n and n > 0:
        dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        arr = np.frombuffer(data, dtype=dt, count=n, offset=84)
        V = arr["v"].reshape(-1, 3).astype(float)
    else:
        txt = data.decode("utf-8", errors="ignore")
        nums = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", txt)
        V = np.array(nums, dtype=float).reshape(-1, 3)
    F = np.arange(len(V)).reshape(-1, 3)
    return V, F


def load_obj(path):
    """Returns a list of (name, V, F); every o/g group becomes a part."""
    verts, groups, name, faces = [], [], os.path.splitext(os.path.basename(path))[0], []

    def flush():
        nonlocal faces
        if faces:
            groups.append((name, faces))
        faces = []

    with open(path, errors="ignore") as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "v" and len(t) >= 4:
                verts.append([float(t[1]), float(t[2]), float(t[3])])
            elif t[0] in ("o", "g"):
                flush()
                name = " ".join(t[1:]) or name
            elif t[0] == "f":
                idx = []
                for p in t[1:]:
                    i = int(p.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    flush()
    allv = np.array(verts, float)
    out = []
    for nm, fc in groups:
        fc = np.array(fc)
        used, inv = np.unique(fc.ravel(), return_inverse=True)
        out.append((nm, allv[used], inv.reshape(-1, 3)))
    return out


_PALETTE = [STEEL, ORANGE, BLUE, GOLD, COPPER, ALU, RED, IRON]


MAX_IMPORT_FACES = 16000       # keeps heavy models interactive (lower it on a slow PC)


def decimate_mesh(mesh, target):
    """Vertex-clustering simplification down to roughly `target` triangles."""
    V, F = mesh
    if len(F) <= target:
        return V, F
    lo, hi = V.min(0), V.max(0)
    ext = max(float((hi - lo).max()), 1e-9)

    def cluster(n):
        q = np.floor((V - lo) / (ext / n)).astype(np.int64)
        K = n + 2
        uniq, inv = np.unique(q[:, 0] + q[:, 1] * K + q[:, 2] * K * K, return_inverse=True)
        inv = inv.ravel()
        Fm = inv[F]
        keep = (Fm[:, 0] != Fm[:, 1]) & (Fm[:, 1] != Fm[:, 2]) & (Fm[:, 0] != Fm[:, 2])
        return len(uniq), inv, Fm[keep]

    lo_n, hi_n, best = 4, 1024, None
    while hi_n - lo_n > 1:
        mid = (lo_n + hi_n) // 2
        res = cluster(mid)
        if len(res[2]) <= target:
            lo_n, best = mid, res
        else:
            hi_n = mid
    if best is None:
        best = cluster(lo_n)
    nu, inv, Fk = best
    cnt = np.maximum(np.bincount(inv, minlength=nu), 1).astype(float)
    Vn = np.stack([np.bincount(inv, weights=V[:, d], minlength=nu) / cnt for d in range(3)], 1)
    used, inv2 = np.unique(Fk.ravel(), return_inverse=True)
    return Vn[used], inv2.reshape(-1, 3)


# ------------------------------------------------------------
# glTF / GLB (the format most downloadable car models come in)
# ------------------------------------------------------------
_GL_DTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
             5125: np.uint32, 5126: np.float32}
_GL_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def _gl_read(path):
    import base64
    from urllib.parse import unquote
    with open(path, "rb") as f:
        data = f.read()
    base = os.path.dirname(path)
    bins, js = [], None
    if data[:4] == b"glTF":                                   # binary .glb container
        off = 12
        while off + 8 <= len(data):
            clen, ctype = struct.unpack("<II", data[off:off + 8])
            chunk = data[off + 8:off + 8 + clen]
            off += 8 + clen
            if ctype == 0x4E4F534A:
                js = json.loads(chunk.decode("utf-8"))
            elif ctype == 0x004E4942:
                bins.append(chunk)
    else:
        js = json.loads(data.decode("utf-8"))
    if js is None:
        raise ValueError("no JSON chunk found")
    buffers = []
    for b in js.get("buffers", []):
        uri = b.get("uri")
        if uri is None:
            buffers.append(bins[0] if bins else b"")
        elif uri.startswith("data:"):
            buffers.append(base64.b64decode(uri.split(",", 1)[1]))
        else:
            with open(os.path.join(base, unquote(uri)), "rb") as f:
                buffers.append(f.read())
    return js, buffers


def _gl_accessor(js, buffers, idx):
    acc = js["accessors"][idx]
    dt = np.dtype(_GL_DTYPE[acc["componentType"]])
    nc = _GL_NCOMP[acc["type"]]
    count = acc["count"]
    if "bufferView" not in acc:
        return np.zeros((count, nc), dt)
    bv = js["bufferViews"][acc["bufferView"]]
    off = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride") or nc * dt.itemsize
    arr = np.ndarray((count, nc), dtype=dt, buffer=buffers[bv["buffer"]], offset=off,
                     strides=(stride, dt.itemsize))
    return np.array(arr)


def _gl_node_matrix(n):
    if "matrix" in n:
        return np.array(n["matrix"], float).reshape(4, 4).T          # column-major
    M = np.eye(4)
    if "rotation" in n:
        x, y, z, w = n["rotation"]
        M[:3, :3] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    if "scale" in n:
        M[:3, :3] = M[:3, :3] @ np.diag(n["scale"])
    if "translation" in n:
        M[:3, 3] = n["translation"]
    return M


def load_gltf(path, max_parts=80):
    """Returns [(name, V, F, color_bgr_or_None)]. glTF is Y-up, like this viewer."""
    js, buffers = _gl_read(path)
    nodes, mats = js.get("nodes", []), js.get("materials", [])

    def mat_info(mi):
        if mi is None or mi >= len(mats):
            return None, "default"
        m = mats[mi]
        pbr = m.get("pbrMetallicRoughness", {})
        f = np.array(pbr.get("baseColorFactor", [1, 1, 1, 1]), float)[:3]
        if "baseColorTexture" in pbr and f.min() > 0.95:
            return (170, 168, 165), m.get("name", f"material{mi}")       # texture only: neutral grey
        c = (np.clip(f, 0, 1) ** (1 / 2.2) * 255)[::-1]                 # linear RGB -> sRGB BGR
        return tuple(int(v) for v in c), m.get("name", f"material{mi}")

    out = []

    def visit(ni, parent):
        n = nodes[ni]
        M = parent @ _gl_node_matrix(n)
        if "mesh" in n:
            mesh = js["meshes"][n["mesh"]]
            Vs, Fs, off, best = [], [], 0, (None, "default", -1)
            for prim in mesh["primitives"]:
                if prim.get("mode", 4) != 4 or "POSITION" not in prim["attributes"]:
                    continue
                P = _gl_accessor(js, buffers, prim["attributes"]["POSITION"]).astype(float)
                if "indices" in prim:
                    I = _gl_accessor(js, buffers, prim["indices"]).astype(np.int64).ravel()
                else:
                    I = np.arange(len(P))
                F = I[:len(I) // 3 * 3].reshape(-1, 3)
                if np.linalg.det(M[:3, :3]) < 0:
                    F = F[:, [0, 2, 1]]
                Vs.append(P @ M[:3, :3].T + M[:3, 3])
                Fs.append(F + off)
                off += len(P)
                col, mname = mat_info(prim.get("material"))
                if len(F) > best[2]:
                    best = (col, mname, len(F))
            if Vs:
                name = n.get("name") or mesh.get("name") or f"part {len(out) + 1}"
                out.append((name, np.vstack(Vs), np.vstack(Fs), best[0], best[1]))
        for c in n.get("children", []):
            visit(c, M)

    if js.get("scenes"):
        roots = js["scenes"][js.get("scene", 0)].get("nodes", [])
    else:
        # No scene list: start only from ROOT nodes, otherwise every child
        # mesh would be visited twice (once directly, once via its parent).
        children = {c for n in nodes for c in n.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in children]
    for r in roots:
        visit(r, np.eye(4))

    if len(out) > max_parts:                       # too many tiny parts: group them by material
        groups = {}
        for name, V, F, col, mname in out:
            g = groups.setdefault(mname, [[], [], 0, col])
            g[0].append(V)
            g[1].append(F + g[2])
            g[2] += len(V)
        print(f"{os.path.basename(path)}: {len(out)} objects grouped into {len(groups)} materials")
        return [(k, np.vstack(g[0]), np.vstack(g[1]), g[3]) for k, g in groups.items()]
    return [(n, V, F, c) for n, V, F, c, _ in out]


def load_machine_folder(folder, z_up=True):
    raw = []                                    # (name, V, F, color or None)
    to_y_up = Rx(-math.pi / 2)
    for fn in sorted(os.listdir(folder)):
        path = os.path.join(folder, fn)
        ext = os.path.splitext(fn)[1].lower()
        try:
            if ext == ".stl":
                V, F = load_stl(path)
                raw.append((os.path.splitext(fn)[0], V @ to_y_up.T if z_up else V, F, None))
            elif ext == ".obj":
                raw += [(n, V @ to_y_up.T if z_up else V, F, None) for n, V, F in load_obj(path)]
            elif ext in (".glb", ".gltf"):
                raw += load_gltf(path)          # already Y-up
        except Exception as e:
            print(f"Could not load {path}: {e}")
    raw = [r for r in raw if len(r[1]) and len(r[2])]
    if not raw:
        return None
    total = sum(len(r[2]) for r in raw)
    if total > MAX_IMPORT_FACES:
        raw = [(n, *decimate_mesh((V, F), max(150, int(MAX_IMPORT_FACES * len(F) / total))), c)
               for n, V, F, c in raw]
        print(f"{os.path.basename(folder)}: simplified {total} -> {sum(len(r[2]) for r in raw)} triangles")
    allv = np.vstack([r[1] for r in raw])
    lo, hi = allv.min(0), allv.max(0)
    ctr = (lo + hi) / 2
    scale = 2.2 / max(float(np.linalg.norm(allv - ctr, axis=1).max()), 1e-9)
    parts = []
    for i, (n, V, F, col) in enumerate(raw):
        V = (V - ctr) * scale
        V, F = refine_mesh((V, F), 0.5, max_iter=3)          # big flat faces sort badly otherwise
        pc = V.mean(0)
        d = pc if np.linalg.norm(pc) > 1e-6 else np.array([0, 1.0, 0])
        d = d / np.linalg.norm(d) * 2.0
        # closed, consistently wound meshes can use back-face culling (twice as fast)
        vol = signed_volume((V, F))
        bbox = float(np.prod(np.maximum(V.max(0) - V.min(0), 1e-9)))
        two_sided = abs(vol) < 0.05 * bbox
        if not two_sided and vol < 0:
            F = F[:, [0, 2, 1]]
        parts.append(Part(n, (V, F), col if col is not None else _PALETTE[i % len(_PALETTE)],
                          f"Imported part: {n}", d, two_sided=two_sided))
    return Machine(os.path.basename(folder.rstrip("/\\")), parts, None, "Imported model")


def load_machines(root, z_up=True):
    out = []
    if not root or not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if os.path.isdir(sub):
            try:
                m = load_machine_folder(sub, z_up)
            except Exception as e:
                print(f"Could not load machine folder '{name}': {e}")
                continue
            if m:
                out.append(m)
                print(f"Loaded machine '{m.name}' with {len(m.parts)} parts")
    return out


# ============================================================
# The explorer: camera, renderer, picking, gestures
# ============================================================
class Explorer:
    def __init__(self, cfg=None, machines=None, size=(1000, 640), ss=1.25):
        self.cfg = cfg
        self.W, self.H = size
        self.machines = machines or builtin_machines()
        self.active = False
        self.mi = 0
        self.t = 0.0
        self.anim = True
        self._t_last = time.perf_counter()
        self.ss = float(ss)               # supersampling: 1.0 fast, 1.25 balanced, 1.5 smooth
        self._bg_ss = self._make_bg(self.ss)

        self.pointer = None            # viewer pixels, or None when no hand
        self.hover = -1
        self.gesture = ""
        self.flash = ""
        self.flash_until = 0.0
        self.labels = True
        self.show_help = True
        self.autorot = False
        self.vyaw = self.vpitch = 0.0     # rotation inertia (rad/s)
        self._dragging = False
        self._md = None
        self._mt = 0.0
        self._mouse_t = -1e9              # last time the physical mouse moved over the viewer
        self._hand_t = 0.0
        self._fps = 30.0

        # hand state
        self.pinch = None
        self._cand, self._cand_n = None, 0
        self.pinch_t0 = 0.0
        self.pinch_prev = (0.0, 0.0)
        self.pinch_move = 0.0
        self.pinch_hover = -1
        self.two_active = False
        self.d0 = self.e0 = 0.0
        self.explode_range = 0.45      # hand distance (frame widths) for full explode
        self.fist_t = None
        self.fist_done = False
        self.swipe_hist = deque()
        self.last_swipe = -1e9
        self._lost = 0

        self.set_machine(0)

    # ---------- config helpers ----------
    def th(self, name):
        return getattr(self.cfg, name, DEFAULTS[name])

    @property
    def machine(self):
        return self.machines[self.mi]

    def _make_bg(self, S=1.0):
        w, h = int(self.W * S), int(self.H * S)
        g = np.linspace(0, 1, h)[:, None, None]
        top, bot = np.array([54, 44, 38], float), np.array([22, 18, 16], float)
        return np.repeat(top * (1 - g) + bot * g, w, axis=1).astype(np.uint8)

    # ---------- state ----------
    def set_machine(self, i):
        self.mi = i % len(self.machines)
        m = self.machine
        self.t = 0.0
        self.sel = -1
        self.hover = -1
        self.isolate = False
        self.explode = 0.0
        self.explode_target = 0.0
        self.yaw, self.pitch = m.view
        self.zoom = 1.0
        self.tgt = m.c0.copy()
        self.rad = m.r0
        self.pinch = None
        self.vyaw = self.vpitch = 0.0

    def reset_view(self):
        self.sel, self.isolate = -1, False
        self.explode_target = 0.0
        self.yaw, self.pitch = self.machine.view
        self.zoom = 1.0
        self.vyaw = self.vpitch = 0.0
        self._say("VIEW RESET")

    def _say(self, text, secs=1.2):
        self.flash = text
        self.flash_until = time.perf_counter() + secs

    def select(self, i):
        if i < 0 or i == self.sel:
            self.sel, self.isolate = -1, False
        else:
            self.sel = i

    def toggle_isolate(self):
        if self.sel < 0:
            self._say("Select a part first")
            return
        self.isolate = not self.isolate
        self._say("ISOLATED" if self.isolate else "SHOW ALL")

    # ---------- keyboard ----------
    def key(self, k):
        c = chr(k).lower() if 0 <= k < 256 else ""
        if c == "n":
            self.set_machine(self.mi + 1)
        elif c == "b":
            self.set_machine(self.mi - 1)
        elif c == " ":
            self.anim = not self.anim
        elif c == "x":
            self.explode_target = 0.0 if self.explode_target > 0.5 else 1.0
        elif c == "[":
            self.explode_target = max(0.0, self.explode_target - 0.1)
        elif c == "]":
            self.explode_target = min(1.0, self.explode_target + 0.1)
        elif k == 9:
            n = len(self.machine.parts)
            self.sel = (self.sel + 1) % n
        elif c == "i":
            self.toggle_isolate()
        elif c and c in "+=":
            self.zoom = min(4.0, self.zoom * 1.1)
        elif c == "-":
            self.zoom = max(0.35, self.zoom / 1.1)
        elif c == "a":
            self.yaw -= 0.08
        elif c == "d":
            self.yaw += 0.08
        elif c == "w":
            self.pitch = max(-1.45, self.pitch - 0.08)
        elif c == "s":
            self.pitch = min(1.45, self.pitch + 0.08)
        elif c == "t":
            self.autorot = not self.autorot
            self._say("AUTO-ROTATE " + ("ON" if self.autorot else "OFF"))
        elif c == "l":
            self.labels = not self.labels
        elif c == "h":
            self.show_help = not self.show_help
        elif c == "g":
            levels = [1.0, 1.25, 1.5]
            self.ss = levels[(levels.index(min(levels, key=lambda v: abs(v - self.ss))) + 1) % 3]
            self._bg_ss = self._make_bg(self.ss)
            self._say("QUALITY: " + {1.0: "FAST", 1.25: "BALANCED", 1.5: "SMOOTH"}[self.ss])
        elif c == "r":
            self.reset_view()

    # ---------- hand gestures ----------
    def update(self, hands, w, h, now):
        """hands: list of landmark lists (normalized x, y) from MediaPipe."""
        n = len(hands)
        if n >= 2:
            self._two_hands(hands, w, h)
            return
        self.two_active = False
        if n == 0:
            # keep the pointer if the physical mouse is driving the viewer
            if now - self._mouse_t > 0.5:
                self.pointer = None
            self._lost += 1
            self.gesture = ""
            if self._lost > 6:
                self._end_pinch(now, cancel=True)
                self.fist_t, self.fist_done = None, False
                self.swipe_hist.clear()
            return
        self._lost = 0
        self._one_hand(hands[0], w, h, now)

    @staticmethod
    def _palm(lm):
        return ((lm[0].x + lm[5].x + lm[17].x) / 3.0, (lm[0].y + lm[5].y + lm[17].y) / 3.0)

    def _two_hands(self, hands, w, h):
        self._end_pinch(time.perf_counter(), cancel=True)
        self.pointer = None
        a, b = self._palm(hands[0]), self._palm(hands[1])
        d = math.hypot(a[0] - b[0], (a[1] - b[1]) * h / w)
        if not self.two_active:
            self.two_active, self.d0, self.e0 = True, d, self.explode_target
        self.explode_target = float(np.clip(self.e0 + (d - self.d0) / self.explode_range, 0.0, 1.0))
        self.gesture = f"EXPLODE {int(self.explode_target * 100)}%"

    def _one_hand(self, lm, w, h, now):
        pts = np.array([[l.x * w, l.y * h] for l in lm])
        scale = np.linalg.norm(pts[9] - pts[0]) + 1e-6
        thumb = pts[4]
        ratio = {k: np.linalg.norm(thumb - pts[i]) / scale
                 for k, i in (("index", 8), ("middle", 12), ("ring", 16), ("pinky", 20))}
        palm = self._palm(lm)
        dt_h = max(now - self._hand_t, 1e-3)
        self._hand_t = now

        # cursor follows the index fingertip
        u = float(np.clip((lm[8].x - 0.15) / 0.70, 0, 1))
        v = float(np.clip((lm[8].y - 0.10) / 0.75, 0, 1))
        tgt = np.array([u * self.W, v * self.H])
        if self.pointer is None:
            self.pointer = tgt
        else:
            self.pointer = self.pointer + (tgt - self.pointer) * 0.55

        wrist = pts[0]
        fist = all(np.linalg.norm(pts[t] - wrist) < self.th("fist_tip_ratio") * scale
                   for t in (8, 12, 16, 20))
        if fist:
            self._end_pinch(now, cancel=True)
            self.swipe_hist.clear()
            if self.fist_t is None:
                self.fist_t = now
            held = now - self.fist_t
            self.gesture = f"RESET {int(min(held, 1.0) * 100)}%"
            if held >= 1.0 and not self.fist_done:
                self.reset_view()
                self.fist_done = True
            return
        self.fist_t, self.fist_done = None, False

        specs = {"index": ("scroll_pinch_close", "scroll_pinch_open", "rotate"),
                 "middle": ("pinch_close", "pinch_open", "select"),
                 "ring": ("pinch_close", "pinch_open", "zoom"),
                 "pinky": ("right_pinch_close", "right_pinch_open", "isolate")}

        if self.pinch is None:
            best, bv = None, 9.0
            for k, r in ratio.items():
                val = r / self.th(specs[k][0])
                if val < 1.0 and val < bv:
                    best, bv = k, val
            if best is None:
                self._cand, self._cand_n = None, 0
            else:
                self._cand_n = self._cand_n + 1 if best == self._cand else 1
                self._cand = best
                if self._cand_n >= 2:
                    self.pinch = best
                    self.pinch_t0, self.pinch_prev = now, palm
                    self.pinch_move, self.pinch_hover = 0.0, self.hover
        else:
            k = self.pinch
            if ratio[k] > self.th(specs[k][1]):
                self._end_pinch(now)

        if self.pinch is not None:
            kind = specs[self.pinch][2]
            dx, dy = palm[0] - self.pinch_prev[0], palm[1] - self.pinch_prev[1]
            self.pinch_prev = palm
            self.pinch_move += math.hypot(dx, dy)
            if math.hypot(dx, dy) < 0.0012:
                dx = dy = 0.0
            if kind == "rotate":
                self.yaw += dx * 4.0
                self.pitch = float(np.clip(self.pitch + dy * 3.0, -1.45, 1.45))
                self.vyaw = float(np.clip(0.5 * self.vyaw + 0.5 * dx * 4.0 / dt_h, -3, 3))
                self.vpitch = float(np.clip(0.5 * self.vpitch + 0.5 * dy * 3.0 / dt_h, -3, 3))
                self.gesture = "ROTATING"
            elif kind == "zoom":
                self.zoom = float(np.clip(self.zoom * math.exp(-dy * 4.0), 0.35, 4.0))
                self.gesture = f"ZOOM {self.zoom:.2f}x"
            elif kind == "select":
                self.gesture = "SELECT (tap)"
            else:
                self.gesture = "ISOLATE (tap)"
            self.swipe_hist.clear()
            return

        # open palm swipe -> change machine
        ext = [np.linalg.norm(pts[t] - wrist) > np.linalg.norm(pts[p] - wrist) * self.th("extend_ratio")
               for t, p in ((8, 6), (12, 10), (16, 14), (20, 18))]
        free = all(ratio[k] > self.th(specs[k][0]) for k in ratio)
        if all(ext) and free:
            self.gesture = "PALM: swipe L / R" if len(self.machines) > 1 else "OPEN PALM"
            self._track_swipe(palm, now)
        else:
            self.swipe_hist.clear()
            self.gesture = "POINTING"

    def _track_swipe(self, palm, now):
        if len(self.machines) < 2:
            return                      # nothing to switch to
        hist = self.swipe_hist
        hist.append((now, palm[0], palm[1]))
        while hist and now - hist[0][0] > self.th("swipe_window_s"):
            hist.popleft()
        if now - self.last_swipe < 0.9 or len(hist) < 3:
            return
        t0, x0, y0 = hist[0]
        dt = now - t0
        if dt < 0.06:
            return
        dx, dy = palm[0] - x0, palm[1] - y0
        if (abs(dx) >= self.th("swipe_dist") and abs(dx) > 2 * abs(dy)
                and abs(dx) / dt >= self.th("swipe_min_speed")):
            self.set_machine(self.mi + (1 if dx > 0 else -1))
            self._say(self.machine.name)
            self.last_swipe = now
            hist.clear()

    def _end_pinch(self, now, cancel=False):
        if self.pinch is None:
            return
        dur = now - self.pinch_t0
        if not cancel and dur < 0.7 and self.pinch_move < 0.06:
            if self.pinch == "middle":
                self.select(self.pinch_hover)
            elif self.pinch == "pinky":
                self.toggle_isolate()
        self.pinch = None
        self._cand, self._cand_n = None, 0

    # ---------- physical mouse (connected in main via cv2.setMouseCallback) ----------
    def on_mouse(self, event, x, y, flags, _):
        now = time.perf_counter()
        if event == cv2.EVENT_LBUTTONDOWN:
            self._md = (x, y, x, y)
            self._dragging = True
            self.vyaw = self.vpitch = 0.0
            self._mt = now
        elif event == cv2.EVENT_MOUSEMOVE:
            self.pointer = np.array([x, y], float)
            self._mouse_t = now
            if self._dragging and self._md:
                x0, y0, px, py = self._md
                dyaw, dpit = (x - px) * 0.008, (y - py) * 0.008
                self.yaw += dyaw
                self.pitch = float(np.clip(self.pitch + dpit, -1.45, 1.45))
                dtm = max(now - self._mt, 1e-3)
                self._mt = now
                self.vyaw = float(np.clip(0.5 * self.vyaw + 0.5 * dyaw / dtm, -4, 4))
                self.vpitch = float(np.clip(0.5 * self.vpitch + 0.5 * dpit / dtm, -4, 4))
                self._md = (x0, y0, x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._md:
            x0, y0, _, _ = self._md
            if abs(x - x0) + abs(y - y0) < 6:
                self.select(self.hover)
            self._md = None
            self._dragging = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.zoom = float(np.clip(self.zoom * (1.1 if flags > 0 else 1 / 1.1), 0.35, 4.0))

    # ---------- rendering ----------
    def render(self, now=None, preview=None):
        if now is None:
            now = time.perf_counter()
        dt = min(max(now - self._t_last, 1e-3), 0.1)
        self._t_last = now
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        m = self.machine
        W, H, S = self.W, self.H, self.ss

        if self.anim and m.animated:
            self.t += dt
        self.explode += (self.explode_target - self.explode) * (1 - math.exp(-dt * 8.0))
        e = self.explode

        # rotation inertia after you let go, plus optional auto-rotate
        if self.pinch != "index" and not self._dragging:
            self.yaw += self.vyaw * dt
            self.pitch = float(np.clip(self.pitch + self.vpitch * dt, -1.45, 1.45))
            damp = math.exp(-dt * 4.5)
            self.vyaw *= damp
            self.vpitch *= damp
            if self.autorot:
                self.yaw += 0.35 * dt

        Vw = m.world(self.t, e)

        sel = self.sel
        if self.isolate and sel >= 0:
            s0, s1 = m.vs[sel]
            pv = Vw[s0:s1]
            pc = pv.mean(0)
            tgt_des = pc
            rad_des = max(float(np.linalg.norm(pv - pc, axis=1).max()) * 1.1, 0.3)
        else:
            tgt_des = m.c0 + (m.c1 - m.c0) * e
            rad_des = m.r0 + (m.r1 - m.r0) * e
        k = 1 - math.exp(-dt * 6.0)
        self.tgt = self.tgt + (tgt_des - self.tgt) * k
        self.rad += (rad_des - self.rad) * k
        D = self.rad * 3.2 / self.zoom

        Rc = Rx(self.pitch) @ Ry(self.yaw)
        Vc = (Vw - self.tgt) @ Rc.T
        Vc[:, 2] -= D
        depth = -Vc[:, 2]
        f = 1.25 * H
        cxp, cyp = 0.50 * W, 0.43 * H
        dd = np.maximum(depth, 1e-3)
        pts = np.stack([(cxp + f * Vc[:, 0] / dd) * S, (cyp - f * Vc[:, 1] / dd) * S], 1)

        buf = self._bg_ss.copy()
        self._grid(buf, m, Rc, D, f * S, cxp * S, cyp * S, e, 2 if S > 1.2 else 1)

        F = m.F
        a, b, c = Vc[F[:, 0]], Vc[F[:, 1]], Vc[F[:, 2]]
        n = np.cross(b - a, c - a)
        n /= np.linalg.norm(n, axis=1)[:, None] + 1e-12
        facing = np.einsum("ij,ij->i", n, -a) > 0
        vis = facing | m.ts[m.fp]
        n = np.where(facing[:, None], n, -n)
        dmin = np.minimum(np.minimum(depth[F[:, 0]], depth[F[:, 1]]), depth[F[:, 2]])
        vis &= dmin > 0.1
        if self.isolate and sel >= 0:
            vis &= (m.fp == sel)
        idx = np.nonzero(vis)[0]

        hover = -1
        if len(idx):
            zmean = (depth[F[idx, 0]] + depth[F[idx, 1]] + depth[F[idx, 2]]) / 3.0
            order = idx[np.argsort(-zmean)]            # far -> near (painter's algorithm)
            Fo, fp_s, ns = F[order], m.fp[order], n[order]
            L1 = np.array([0.45, 0.75, 0.50]); L1 /= np.linalg.norm(L1)
            L3 = np.array([-0.6, -0.4, 0.3]); L3 /= np.linalg.norm(L3)
            Hh = L1 + np.array([0.0, 0.0, 1.0]); Hh /= np.linalg.norm(Hh)
            shade = (0.20 + 0.46 * np.maximum(0, ns @ L1) + 0.26 * np.maximum(0, ns[:, 2])
                     + 0.10 * np.maximum(0, ns @ L3))
            spec = np.maximum(0.0, ns @ Hh) ** 24 * m.shine[fp_s]
            col = m.col[fp_s] * shade[:, None] + 140.0 * spec[:, None]
            if self.hover >= 0 and self.hover != sel:
                hm = (fp_s == self.hover)[:, None]
                col = np.where(hm, col * 0.72 + 255 * 0.28, col)
            if sel >= 0:
                sm = (fp_s == sel)[:, None]
                col = np.where(sm, col * 0.70 + np.array([0, 191, 255]) * 0.30, col)
            col = np.clip(col, 0, 255).astype(np.uint8).tolist()

            P = pts[Fo]                                    # (K,3,2) float
            tri = np.round(P).astype(np.int32)
            fill = cv2.fillConvexPoly
            is_sel = (fp_s == sel) if sel >= 0 else None
            if is_sel is not None and not self.isolate and is_sel.any():
                # X-ray: everything else turns ghostly, the selected part stays solid on top
                base = buf.copy()
                for i in np.nonzero(~is_sel)[0]:
                    fill(buf, tri[i], col[i])
                buf = cv2.addWeighted(buf, 0.62, base, 0.38, 0)
                for i in np.nonzero(is_sel)[0]:
                    fill(buf, tri[i], col[i])
            else:
                for i in range(len(order)):
                    fill(buf, tri[i], col[i])

            if self.pointer is not None:
                hit = self._hit(P, float(self.pointer[0]) * S, float(self.pointer[1]) * S)
                if hit >= 0:
                    hover = int(fp_s[hit])
        self.hover = hover

        img = cv2.resize(buf, (W, H), interpolation=cv2.INTER_AREA) if S != 1.0 else buf
        self._labels(img, m, Vw, Rc, D, f, cxp, cyp, e)
        self._hud(img, m, e, preview)
        return img

    @staticmethod
    def _hit(P, px, py):
        """Index of the nearest (last-drawn) triangle containing (px, py), or -1."""
        x0, y0 = P[:, 0, 0], P[:, 0, 1]
        x1, y1 = P[:, 1, 0], P[:, 1, 1]
        x2, y2 = P[:, 2, 0], P[:, 2, 1]
        d1 = (px - x1) * (y0 - y1) - (x0 - x1) * (py - y1)
        d2 = (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)
        d3 = (px - x0) * (y2 - y0) - (x2 - x0) * (py - y0)
        neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        hits = np.nonzero(~(neg & pos))[0]
        return int(hits[-1]) if len(hits) else -1

    def _grid(self, img, m, Rc, D, f, cx, cy, e, thick=1):
        yf = m.ymin0 + (m.ymin1 - m.ymin0) * e - 0.35
        G = int(math.ceil(self.rad * 1.8))
        rng = range(-G, G + 1)
        A = [(-G, yf, i) for i in rng] + [(i, yf, -G) for i in rng]
        B = [(G, yf, i) for i in rng] + [(i, yf, G) for i in rng]
        PA = (np.array(A, float) - self.tgt) @ Rc.T
        PB = (np.array(B, float) - self.tgt) @ Rc.T
        PA[:, 2] -= D
        PB[:, 2] -= D
        for p, q in zip(PA, PB):
            if -p[2] < 0.2 or -q[2] < 0.2:
                continue
            x1, y1 = cx + f * p[0] / -p[2], cy - f * p[1] / -p[2]
            x2, y2 = cx + f * q[0] / -q[2], cy - f * q[1] / -q[2]
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (72, 64, 58), thick, cv2.LINE_AA)

    # ---------- labels + HUD ----------
    def _panel(self, img, x0, y0, x1, y1, alpha=0.62):
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.W, x1), min(self.H, y1)
        roi = img[y0:y1, x0:x1]
        roi[:] = (roi * (1 - alpha) + np.array([20, 16, 14]) * alpha).astype(np.uint8)

    def _labels(self, img, m, Vw, Rc, D, f, cx, cy, e):
        """Part names in two tidy side columns with leader lines (shown once the model is exploded).
        If there are more parts than fit, the smallest ones are skipped - hover over them instead."""
        if not self.labels or e < 0.3:
            return
        W, H = self.W, self.H
        FONT, AA = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
        gap = 15
        L_LO, L_HI, R_LO, R_HI = 92, H - 140, 160, H - 200       # keep clear of the HUD panels
        cap_l, cap_r = (L_HI - L_LO) // gap + 1, (R_HI - R_LO) // gap + 1
        items = []
        for i, (s0, s1) in enumerate(m.vs):
            if self.isolate and self.sel >= 0 and i != self.sel:
                continue
            pv = Vw[s0:s1]
            ctr = pv.mean(0)
            rad = float(np.linalg.norm(pv - ctr, axis=1).max())
            c = (ctr - self.tgt) @ Rc.T
            c[2] -= D
            dep = -c[2]
            if dep < 0.3:
                continue
            x, y = cx + f * c[0] / dep, cy - f * c[1] / dep
            if 0 <= x < W and 0 <= y < H:
                items.append([1e9 if i == self.sel else rad, x, y, i])
        items.sort(key=lambda it: -it[0])
        items = items[:cap_l + cap_r]
        items.sort(key=lambda it: it[1])                          # left-to-right on screen
        n_left = min(cap_l, max(round(len(items) * 0.55), len(items) - cap_r))
        for group, lo, hi, side in ((items[:n_left], L_LO, L_HI, 0), (items[n_left:], R_LO, R_HI, 1)):
            group.sort(key=lambda it: it[2])
            ty = [min(max(it[2], lo), hi) for it in group]
            for k in range(1, len(ty)):
                ty[k] = max(ty[k], ty[k - 1] + gap)
            if ty and ty[-1] > hi:
                ty[-1] = hi
                for k in range(len(ty) - 2, -1, -1):
                    ty[k] = min(ty[k], ty[k + 1] - gap)
            for it, t in zip(group, ty):
                name = m.parts[it[3]].name
                tw = cv2.getTextSize(name, FONT, 0.40, 1)[0][0]
                tx = 16 if side == 0 else W - 16 - tw
                ax = tx + tw + 5 if side == 0 else tx - 5
                cv2.line(img, (int(it[1]), int(it[2])), (int(ax), int(t)), (200, 196, 190), 1, AA)
                cv2.circle(img, (int(it[1]), int(it[2])), 3, (255, 255, 255), -1, AA)
                cv2.putText(img, name, (tx, int(t) + 4), FONT, 0.40, (0, 0, 0), 3, AA)
                cv2.putText(img, name, (tx, int(t) + 4), FONT, 0.40, (240, 240, 240), 1, AA)

    def _hud(self, img, m, e, preview):
        W, H = self.W, self.H
        F = cv2.FONT_HERSHEY_SIMPLEX
        AA = cv2.LINE_AA
        white, grey, amber = (245, 245, 245), (175, 170, 165), (0, 200, 255)

        self._panel(img, 10, 10, 500, 78)
        cv2.putText(img, m.name, (20, 36), F, 0.62, white, 1, AA)
        state = ("ANIMATING" if self.anim else "PAUSED") if m.animated else "STATIC"
        cv2.putText(img, f"{self.mi + 1}/{len(self.machines)}   {state}   Explode {int(e * 100)}%"
                         f"   Zoom {self.zoom:.2f}x   {self._fps:.0f} fps",
                    (20, 62), F, 0.45, grey, 1, AA)

        shown = self.sel if self.sel >= 0 else self.hover
        if shown >= 0:
            p = m.parts[shown]
            lines = textwrap.wrap(p.desc, 34)
            hgt = 54 + 19 * len(lines) + 28
            self._panel(img, W - 330, 10, W - 10, 10 + hgt)
            tag = ("ISOLATED" if self.isolate else "SELECTED") if self.sel == shown else "HOVER"
            cv2.putText(img, tag, (W - 320, 30), F, 0.4, amber if self.sel == shown else grey, 1, AA)
            cv2.putText(img, p.name, (W - 320, 54), F, 0.55, white, 1, AA)
            for i, ln in enumerate(lines):
                cv2.putText(img, ln, (W - 320, 78 + 19 * i), F, 0.44, (210, 205, 200), 1, AA)
            cv2.putText(img, f"Part {shown + 1} of {len(m.parts)}   {len(p.F)} triangles",
                        (W - 320, 78 + 19 * len(lines) + 8), F, 0.4, grey, 1, AA)

        # explode bar
        bx0, bx1, by = int(W * 0.50) - 150, int(W * 0.50) + 150, H - 20
        cv2.rectangle(img, (bx0, by), (bx1, by + 8), (70, 64, 60), -1)
        cv2.rectangle(img, (bx0, by), (bx0 + int(300 * e), by + 8), amber, -1)
        cv2.putText(img, "ASSEMBLED", (bx0 - 96, by + 9), F, 0.38, grey, 1, AA)
        cv2.putText(img, "EXPLODED", (bx1 + 10, by + 9), F, 0.38, grey, 1, AA)

        # gesture status / flash
        now = time.perf_counter()
        msg = self.flash if now < self.flash_until else self.gesture
        if msg:
            sz = cv2.getTextSize(msg, F, 0.8, 2)[0]
            cv2.putText(img, msg, (int(W * 0.50) - sz[0] // 2, 112), F, 0.8, amber, 2, AA)

        # legend
        if self.show_help:
            self._panel(img, 10, H - 128, W - 262, H - 34, 0.5)
            legend = ["THUMB+INDEX pinch + move: ROTATE      THUMB+RING pinch + up/down: ZOOM",
                      "THUMB+MIDDLE tap: SELECT part        THUMB+PINKY tap: ISOLATE / SHOW ALL",
                      "TWO HANDS apart / together: EXPLODE   FIST 1s: RESET   PALM swipe: NEXT MACHINE",
                      "Keys: E exit 3D  N/B machine  Space animate  X explode  I isolate  Tab part",
                      "      T auto-rotate  L labels  G quality  H hide help   Mouse: drag/click/wheel"]
            for i, ln in enumerate(legend):
                cv2.putText(img, ln, (20, H - 108 + 19 * i), F, 0.4, (200, 196, 190), 1, AA)
        else:
            cv2.putText(img, "H = help", (20, H - 40), F, 0.4, grey, 1, AA)

        # hand cursor
        if self.pointer is not None:
            px, py = int(self.pointer[0]), int(self.pointer[1])
            colr = {"index": (0, 255, 0), "ring": (255, 200, 0), "middle": (0, 200, 255),
                    "pinky": (255, 120, 255)}.get(self.pinch, white)
            cv2.circle(img, (px, py), 11, colr, 2, AA)
            cv2.circle(img, (px, py), 2, colr, -1, AA)
            if self.hover >= 0 and self.pinch is None:
                cv2.putText(img, m.parts[self.hover].name, (px + 16, py - 10), F, 0.45, (0, 0, 0), 3, AA)
                cv2.putText(img, m.parts[self.hover].name, (px + 16, py - 10), F, 0.45, white, 1, AA)

        # camera preview
        if preview is not None:
            pw, ph = 240, 180
            img[H - ph - 10:H - 10, W - pw - 10:W - 10] = cv2.resize(preview, (pw, ph))
            cv2.rectangle(img, (W - pw - 10, H - ph - 10), (W - 10, H - 10), (200, 200, 200), 1)


# ============================================================
# Setup helpers
# ============================================================
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print(f"{MODEL_PATH} not found, downloading...")
    tmp = MODEL_PATH + ".part"
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        if os.path.getsize(tmp) < 1_000_000:          # the real model is several MB
            raise ValueError("downloaded file is too small - probably an error page")
        os.replace(tmp, MODEL_PATH)                   # only appears once it is complete
    except Exception as e:
        try:
            os.remove(tmp)
        except OSError:
            pass
        print(f"Download failed: {e}\nDownload it manually from:\n{MODEL_URL}\n"
              f"and put it here: {MODEL_PATH}")
        sys.exit(1)


def open_camera(cfg):
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(cfg.cam_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cfg.cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_h)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # less lag
    return cap


def window_closed(window):
    """True if the user closed the window with its X button."""
    try:
        return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def main():
    setup_logging()
    cfg = Config()
    load_calibration(cfg)
    ensure_model()

    pyautogui.PAUSE = 0.0
    pyautogui.FAILSAFE = True  # corner-slam = emergency exit

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = open_camera(cfg)
    if not cap.isOpened():
        print("ERROR: could not open the camera. Is another app using it?")
        detector.close()
        return

    ok, frame = cap.read()
    if not ok:
        print("ERROR: camera returned no frames.")
        cap.release()
        detector.close()
        return
    h, w = frame.shape[:2]
    am = AirMouse(cfg, w, h)

    window = "Air Mouse"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 640, 480)
    cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)

    rasengan = LoopingVideo(cfg.rasengan_video)
    cinema = LoopingVideo(cfg.cinema_video)
    fx_mode = False
    two_hands_n = 0
    calib = Calibrator()
    gestures = GestureStore(GESTURES_PATH)
    recorder = GestureRecorder(gestures)
    am.gestures = gestures
    zoom = TwoHandZoom(cfg)
    am.zoom = zoom
    os.makedirs(cfg.machines_dir, exist_ok=True)   # so users can see where models go
    explorer = Explorer(cfg, builtin_machines() + load_machines(cfg.machines_dir, cfg.models_z_up))
    explore_mode = False           # 3D machine explorer mode (E), toggled just like FX mode (F)

    # The physical mouse drives the 3D viewer (drag = rotate, click = select, wheel = zoom).
    def on_mouse(event, x, y, flags, param):
        if explore_mode:
            explorer.on_mouse(event, x, y, flags, param)
    cv2.setMouseCallback(window, on_mouse)

    hotkey = GlobalHotkey(cfg.global_pause_hotkey)

    t0 = time.perf_counter()
    last_ts = -1
    fps, fps_n, fps_t = 0.0, 0, time.perf_counter()
    read_failures = 0

    print(__doc__)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # Webcams hiccup (USB glitches, power saving). Don't quit on one bad frame.
                read_failures += 1
                am.on_no_hand()
                am.tick(time.perf_counter())
                if read_failures >= cfg.max_read_failures:
                    print("ERROR: the camera stopped sending frames. Exiting.")
                    break
                if cv2.waitKey(30) & 0xFF == ord("q") or window_closed(window):
                    break
                continue
            read_failures = 0
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            if hotkey.poll():
                am.toggle_pause()

            now = time.perf_counter()
            ts = int((now - t0) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_image, ts)

            hands = result.hand_landmarks or []
            labels = [c[0].category_name for c in result.handedness] if result.handedness else []

            two_hands_n = two_hands_n + 1 if (fx_mode and len(hands) >= 2) else 0

            if explore_mode:
                # ---- 3D machine explorer: the mouse is off, your hands drive the viewer ----
                for hl in hands:
                    draw_hand(frame, hl, w, h, cfg.pointer_landmark)
                am.status = "3D EXPLORER"
                explorer.update(hands, w, h, now)

            elif recorder.active:
                # ---- Recording a custom gesture: mouse control is off ----
                rl = hands[0] if hands else None
                if rl is not None:
                    draw_hand(frame, rl, w, h, cfg.pointer_landmark)
                am.release_all()
                am.reset_filters()
                am.status = "RECORDING GESTURE"
                recorder.update(rl, w, h, now, cfg)

            elif calib.active:
                # ---- Calibration: mouse control is off while we measure ----
                cl = hands[0] if hands else None
                cratios = hand_ratios(cl, w, h) if cl is not None else None
                if cl is not None:
                    draw_hand(frame, cl, w, h, cfg.pointer_landmark)
                am.release_all()
                am.reset_filters()
                am.status = "CALIBRATING"
                new = calib.update(cratios, now)
                if new:
                    for k, v in new.items():
                        setattr(cfg, k, v)
                    save_calibration(cfg)

            elif fx_mode and two_hands_n >= cfg.cinema_confirm_frames:
                # ---- Absolute Cinema: both hands ----
                am.release_all()
                am.reset_filters()
                am.status = "ABSOLUTE CINEMA"
                cf = cinema.read()
                if cf is not None:
                    frame = cv2.resize(cf, (w, h))

            elif fx_mode and len(hands) >= 2:
                am.on_no_hand()  # two hands appearing, wait for confirmation

            elif not fx_mode and not am.paused and zoom.update(hands, w, h, now):
                # ---- Two-hand pinch zoom: the cursor stays frozen ----
                for hl in hands[:2]:
                    draw_hand(frame, hl, w, h, cfg.pointer_landmark)
                zoom.draw(frame)
                am.release_all()
                am.reset_filters()
                am.lost = 0
                am.status = zoom.status

            elif hands:
                mouse_lm = fx_lm = None
                if fx_mode:
                    # Assign hands by MediaPipe's label. A hand with neither label
                    # (or a missing label) falls back to being the mouse hand.
                    for i, lm in enumerate(hands):
                        label = labels[i] if i < len(labels) else ""
                        if label == cfg.fx_hand_label:
                            fx_lm = lm
                        elif label == cfg.mouse_hand_label or mouse_lm is None:
                            mouse_lm = lm
                else:
                    mouse_lm = hands[0]

                if mouse_lm is not None:
                    draw_hand(frame, mouse_lm, w, h, cfg.pointer_landmark)
                    am.on_hand(mouse_lm, w, h, now)
                else:
                    am.on_no_hand()

                # ---- Rasengan: FX hand with an open / cupped palm ----
                if fx_lm is not None:
                    fpts = np.array([[l.x * w, l.y * h] for l in fx_lm])
                    for a, b in HAND_CONNECTIONS:
                        cv2.line(frame, tuple(fpts[a].astype(int)), tuple(fpts[b].astype(int)), (255, 255, 0), 2)
                    if sum(fingers_extended(fpts, cfg.extend_ratio)) >= 3 and rasengan.ok:
                        rf = rasengan.read()
                        palm_center = (fpts[5] + fpts[9] + fpts[17]) / 3.0
                        palm_w = np.linalg.norm(fpts[5] - fpts[17])
                        frame = overlay_rasengan(frame, rf, palm_center[0], palm_center[1], palm_w * 2.0)
                        am.status = "RASENGAN!"
            else:
                am.on_no_hand()

            am.tick(now)

            fps_n += 1
            if now - fps_t >= 1.0:
                fps = fps_n / (now - fps_t)
                fps_n, fps_t = 0, now

            if explore_mode:
                cv2.imshow(window, explorer.render(now, preview=frame))
            else:
                draw_hud(frame, am, fps, fx_mode)
                draw_calibration(frame, calib, now)
                draw_recorder(frame, recorder, now)
                cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or window_closed(window):
                break
            if key == ord("e"):
                explore_mode = not explore_mode
                explorer.active = explore_mode
                if calib.active:
                    calib.cancel(time.perf_counter())
                recorder.cancel(time.perf_counter())
                two_hands_n = 0
                am.release_all()
                am.reset_filters()
                cv2.resizeWindow(window, *((explorer.W, explorer.H) if explore_mode else (640, 480)))
                print("3D explorer mode:", "ON" if explore_mode else "off")
                continue
            if explore_mode:
                if key != 255:
                    explorer.key(key)          # in 3D mode the other keys control the viewer
                continue
            if recorder.active:
                # while recording, keys only pick an action or cancel
                if key == 27 or key == ord("g"):
                    recorder.cancel(time.perf_counter())
                elif recorder.choosing and key != 255:
                    recorder.choose(key, time.perf_counter())
                continue
            if key == ord("g") and not calib.active:
                am.release_all()
                recorder.start(time.perf_counter())
                continue
            if key == ord("u"):
                g = gestures.remove_last()
                recorder._say(f"Removed gesture '{g.name}' ({len(gestures.items)} left)" if g
                              else "No custom gestures to remove", time.perf_counter())
            if key == ord("p"):
                am.toggle_pause()
            if key == ord("d"):
                am.debug = not am.debug
            if key == ord("z"):
                zoom.enabled = not zoom.enabled
                zoom.reset()
                print("Two-hand pinch zoom:", "ON" if zoom.enabled else "off")
            if key == ord("s"):
                am.swipe_enabled = not am.swipe_enabled
                if not am.swipe_enabled:
                    am._release_alt()
            if key == ord("c"):
                if calib.active:
                    calib.cancel(time.perf_counter())
                else:
                    am.release_all()
                    calib.start(time.perf_counter())
            if key == 27 and calib.active:      # Esc
                calib.cancel(time.perf_counter())
            if key == ord("f"):
                fx_mode = not fx_mode
                am.fx_mode = fx_mode
                two_hands_n = 0
                am.release_all()
                am.reset_filters()
                for name, vid in (("rasengan.mp4", rasengan), ("absolute_cinema.mp4", cinema)):
                    if fx_mode and not vid.ok:
                        print(f"FX mode: {name} not found (effect will be skipped).")

    except pyautogui.FailSafeException:
        print("Fail-safe triggered (mouse in a corner). Exiting.")
    finally:
        pyautogui.FAILSAFE = False  # so cleanup can't raise again
        am.release_all()
        hotkey.stop()
        detector.close()
        rasengan.release()
        cinema.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()