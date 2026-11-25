# Unified Air Mouse and 3D Machine Explorer

Control your computer with hand gestures through an ordinary webcam. No extra hardware is needed.
It tracks your hand with [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
and drives the real mouse and keyboard with PyAutoGUI.

There are three modes in one program:

- **Air Mouse** (default): move, click, drag, scroll, right-click and Alt+Tab with one hand.
- **FX mode**: video effects drawn on your hand, plus a gesture that opens YouTube.
- **3D Machine Explorer**: a 3D viewer for engines, gearboxes, robots and cars that you rotate,
  zoom and take apart with your hands. You can load your own `.stl`, `.obj` and `.glb` models.

You can also record **your own gestures** and give each one an action
(play/pause, volume, copy/paste, screenshot and more).

---

## Quick start

Requires **Python 3.10+** (developed on 3.13) and a webcam.

```bash
git clone https://github.com/harshasingh159-source/Unified-Air-Mouse-and-3D-Machine-Explorer.git
cd Unified-Air-Mouse-and-3D-Machine-Explorer
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python main.py
```

The hand-tracking model (`hand_landmarker.task`) is included in the repo. If it is missing,
the program downloads it on first run.

### First launch: the tutorial

The first time you open the app, a short interactive tutorial starts. It walks through
calibration and every gesture (move, click, right click, drag, scroll, freeze, two-hand zoom
and Alt+Tab) on a **practice board** drawn over the camera view. During the tutorial, gestures
move a virtual cursor only, so nothing on your real desktop gets clicked.

- `T` turns the tutorial on and off any time, just like `F` does for FX mode.
  The top bar shows `T:tutorial(ON)` or `T:tutorial(off)`.
- `SPACE` skips a lesson. `Esc` also closes the tutorial.

> **Emergency stop:** push your physical mouse into any screen corner. The PyAutoGUI
> fail-safe quits the program and releases all buttons.

---

## Air Mouse gestures (one hand)

| Action | Gesture |
|---|---|
| Move cursor | Point. The cursor follows your **index fingertip** |
| Left click | Quick **tap**: touch **thumb + index** fingertips and let go |
| Right click | Pinch **thumb + middle finger** |
| Drag | Pinch **thumb + ring finger** and hold, move, then open your fingers to drop |
| Scroll | **OK sign** (thumb + index pinched, other fingers up), hold it for about 0.3 s, then move your hand up/down |
| Freeze cursor | Make a tight **fist** |
| Alt+Tab | **Open palm**, then swipe right/left to cycle windows. Close your hand to pick one |
| Zoom | **Two hands**: pinch **thumb + index** on both, then spread apart = zoom in, bring together = zoom out |

Zoom sends Ctrl+mouse wheel to the window under the cursor, so point at what you want to zoom
first. It works in browsers, maps, image viewers and Office. Press `Z` to turn it on or off.
It is off in FX mode, where two hands trigger Absolute Cinema.

**Calibrate** with `C`: touch your thumb to your index, middle and ring fingertips when asked. Your personal thresholds are saved
to `air_mouse_calibration.json`.

## Custom gestures

1. Press `G` and hold any hand pose for 2 seconds.
2. Press a number key to choose its action:

   | Key | Action | Key | Action |
   |---|---|---|---|
   | 1 | Play / pause | 6 | Mute |
   | 2 | Next track | 7 | Copy |
   | 3 | Previous track | 8 | Paste |
   | 4 | Volume up | 9 | Undo |
   | 5 | Volume down | 0 | Screenshot |

3. To use it, make the pose and hold it for half a second.

Press `U` to delete the most recent gesture. Gestures are saved in `custom_gestures.json`, where you
can also set other actions: any key combo, opening a website or app, or typing text.
Poses that already do something (open palm, pointing, plain fist, pinches) can't be recorded.

## FX mode (`F`)

- The mouse hand works as normal.
- **Peace sign** held for 1 s opens YouTube in Chrome.
- **Other hand, palm up**: Rasengan effect on your palm (`rasengan.mp4`).
- **Snap your fingers (either hand)**: you dissolve into dust and **vanish**, with the room
  painted over you. Snap again to come back. "SNAP READY" shows when your thumb and middle finger
  are pressed together. For a clean vanish, press `B` and step out of view during the 3-second
  countdown so it learns the empty room. The bottom line shows how much of the room it knows.
  In FX mode, thumb + middle is the snap, so right-click is off there.
- **Other hand, fist then open it fast**: **power burst**, with a shockwave, sparks and a screen shake.
- **Other hand, thumb and index spread apart** (other fingers curled): **lightning** crackles
  between the two fingertips.
- **Both hands**: "Absolute Cinema" overlay (`absolute_cinema.mp4`).

## 3D Machine Explorer (`E`)

The mouse is switched off in this mode, and your hands control the viewer instead.
Built-in models: engine, gear reducer, robot arm, sedan and a hypercar.

| Action | Gesture |
|---|---|
| Rotate | Pinch **thumb + index**, then move your hand |
| Zoom | Pinch **thumb + ring**, then move up (in) / down (out) |
| Select a part | Point at it, tap **thumb + middle** |
| Isolate / show all | Tap **thumb + pinky** (with a part selected) |
| Explode / assemble | **Two hands**: move apart / together |
| Reset view | Hold a **fist** for 1 s |
| Next / previous machine | Swipe an **open palm** right / left |

The physical mouse works too: drag to rotate, click to select, wheel to zoom.
Viewer keys: `N`/`B` machine, `Space` animate, `X` explode, `[` `]` explode amount,
`Tab` next part, `I` isolate, `+`/`-` zoom, `A` `D` `W` `S` rotate, `R` reset, `T` auto-rotate,
`L` labels, `H` help, `G` quality.

**Your own models:** create `machines/<name>/` next to the program and put `.stl`, `.obj`,
`.glb` or `.gltf` files in it. Each file or object becomes a selectable part.
Textures are not shown.

---

## Keyboard shortcuts (camera window focused)

| Key | Action | Key | Action |
|---|---|---|---|
| `F` | FX mode | `S` | Swipe (Alt+Tab) on/off |
| `E` | 3D explorer | `G` | Record gesture |
| `P` | Pause | `U` | Remove last gesture |
| `D` | Debug overlay | `Q` | Quit |
| `C` | Calibrate (`Esc` cancels) | `Z` | Two-hand zoom on/off |
| `T` | Tutorial on/off | `B` | FX mode: learn the empty room (vanish trick) |

**Global pause:** `Ctrl+Alt+P` works from any window (needs `pynput`).

All the tuning values (smoothing, pinch thresholds, gesture timing and so on) are in the `Config`
class near the top of [`main.py`](main.py).

---

## Build a Windows .exe

```bash
pip install pyinstaller
pyinstaller AirMouse.spec
```

The app is created in `dist/AirMouse/`. Run `AirMouse.exe` from that folder.
It is a windowed app, so errors go to `air_mouse.log` next to the .exe.
Calibration, custom gestures and the `machines/` folder are also stored next to the .exe.

## Project files

| File | Purpose |
|---|---|
| `main.py` | The whole application |
| `AirMouse.spec` | PyInstaller build configuration |
| `hand_landmarker.task` | MediaPipe hand-tracking model |
| `selfie_segmenter.tflite` | MediaPipe person-segmentation model (FX mode vanish effect) |
| `rasengan.mp4`, `absolute_cinema.mp4` | FX mode videos |
| `air_mouse_calibration.json` | Your calibration thresholds (created when you calibrate, not in git) |
| `custom_gestures.json` | Your recorded gestures (created when you record one, not in git) |
| `air_mouse_settings.json` | Remembers that the tutorial was shown (created by the app, not in git) |
| `requirements.txt` | Python dependencies |
