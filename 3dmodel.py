"""
MACHINE EXPLORER - explore machines in 3D with your hands
=========================================================
Pure OpenCV + NumPy (no OpenGL, nothing extra to install).

Gestures (one hand unless stated; a small cursor follows your index fingertip):
  Rotate ............. pinch THUMB + INDEX, then move your hand
  Zoom ............... pinch THUMB + RING, then move your hand up (in) / down (out)
  Select a part ...... point at it, then TAP thumb + MIDDLE (tap again to deselect)
  Isolate / show all . TAP thumb + PINKY (needs a selected part)
  Explode ............ TWO HANDS: move them apart = explode, together = assemble
  Reset view ......... hold a FIST for 1 second
  Next / previous .... swipe an OPEN PALM right / left to change machine

Keyboard (viewer window focused):
  N / B next / previous machine     Space  pause animation
  E  explode on/off                 [ ]    explode less / more
  Tab  next part                    I      isolate selected part
  + -  zoom                         A D W S  rotate      R  reset

Your own models: put folders in  machines/<name>/  containing .stl or .obj
files (one file or one OBJ object per part) and they appear in the list.

Run this file directly for a no-camera demo:
  python machine_explorer.py            (mouse: drag = rotate, wheel = zoom,
                                         click = select; keyboard as above)
  python machine_explorer.py --shots out_folder   (save preview images)
"""

import math
import os
import re
import struct
import sys
import textwrap
import time
from collections import deque

import cv2
import numpy as np

# ------------------------------------------------------------
# Defaults for gesture thresholds (the air_mouse Config overrides these,
# including your calibrated values)
# ------------------------------------------------------------
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
class Part:
    def __init__(self, name, mesh, color, desc="", explode=(0, 0, 0), two_sided=False):
        self.name = name
        self.V = np.asarray(mesh[0], float)
        self.F = np.asarray(mesh[1], np.int32)
        self.color = color                    # BGR
        self.desc = desc
        self.explode = np.asarray(explode, float)
        self.two_sided = two_sided


class Machine:
    def __init__(self, name, parts, pose_fn=None, info=""):
        self.name = name
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
    crankcase = place(tube(1.15, 1.03, 1.0, 40, math.pi, 2 * math.pi, "z"), T=(0, 0, 0.15))

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
        place(tube(0.13, 0, 2.3, 16, axis="z"), T=(0, 0, 0.2)),
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
        place(tube(0.95, 0.80, 0.22, 48, axis="z"), T=(0, 0, 1.05)),
        place(tube(0.30, 0, 0.22, 24, axis="z"), T=(0, 0, 1.05)),
        *[place(sp, T=(0, 0, 1.05)) for sp in spokes])

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
             (0, -0.6, -1.8)),
        Part("Counterweights", counter, COPPER,
             "Balance the weight of the piston and rod to reduce vibration.",
             (-1.8, -1.2, -1.2)),
        Part("Flywheel", fly, IRON,
             "Stores rotational energy to smooth out the power pulses between strokes.",
             (0, 0, 2.6)),
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
    base = merge(place(tube(1.15, 0, 0.08, 40, axis="y"), T=(0, 0.04, 0)),
                 place(tube(1.0, 0, 0.35, 40, axis="y"), T=(0, 0.255, 0)))
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


def builtin_machines():
    return [make_engine(), make_gearbox(), make_robot()]


# ============================================================
# Loading your own STL / OBJ models
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


def load_machine_folder(folder, z_up=True):
    raw = []
    for fn in sorted(os.listdir(folder)):
        path = os.path.join(folder, fn)
        ext = os.path.splitext(fn)[1].lower()
        try:
            if ext == ".stl":
                V, F = load_stl(path)
                raw.append((os.path.splitext(fn)[0], V, F))
            elif ext == ".obj":
                raw += load_obj(path)
        except Exception as e:
            print(f"Could not load {path}: {e}")
    raw = [r for r in raw if len(r[1]) and len(r[2])]
    if not raw:
        return None
    if z_up:                                   # CAD files are usually Z-up
        R = Rx(-math.pi / 2)
        raw = [(n, V @ R.T, F) for n, V, F in raw]
    allv = np.vstack([r[1] for r in raw])
    lo, hi = allv.min(0), allv.max(0)
    ctr = (lo + hi) / 2
    scale = 2.2 / max(float(np.linalg.norm(allv - ctr, axis=1).max()), 1e-9)
    parts = []
    for i, (n, V, F) in enumerate(raw):
        V = (V - ctr) * scale
        pc = V.mean(0)
        d = pc if np.linalg.norm(pc) > 1e-6 else np.array([0, 1.0, 0])
        d = d / np.linalg.norm(d) * 2.0
        parts.append(Part(n, (V, F), _PALETTE[i % len(_PALETTE)],
                          f"Imported part: {n}", d, two_sided=True))
    return Machine(os.path.basename(folder.rstrip("/\\")), parts, None, "Imported model")


def load_machines(root, z_up=True):
    out = []
    if not root or not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if os.path.isdir(sub):
            m = load_machine_folder(sub, z_up)
            if m:
                out.append(m)
                print(f"Loaded machine '{m.name}' with {len(m.parts)} parts")
    return out


# ============================================================
# The explorer: camera, renderer, picking, gestures
# ============================================================
class Explorer:
    def __init__(self, cfg=None, machines=None, size=(1000, 640)):
        self.cfg = cfg
        self.W, self.H = size
        self.machines = machines or builtin_machines()
        self.active = False
        self.mi = 0
        self.t = 0.0
        self.anim = True
        self._t_last = time.perf_counter()
        self._bg = self._make_bg()

        self.pointer = None            # viewer pixels, or None when no hand
        self.hover = -1
        self.gesture = ""
        self.flash = ""
        self.flash_until = 0.0

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

    def _make_bg(self):
        g = np.linspace(0, 1, self.H)[:, None, None]
        top, bot = np.array([54, 44, 38], float), np.array([22, 18, 16], float)
        col = top * (1 - g) + bot * g
        return np.repeat(col, self.W, axis=1).astype(np.uint8)

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
        self.yaw, self.pitch, self.zoom = 0.55, 0.30, 1.0
        self.tgt = m.c0.copy()
        self.rad = m.r0
        self.pinch = None

    def reset_view(self):
        self.sel, self.isolate = -1, False
        self.explode_target = 0.0
        self.yaw, self.pitch, self.zoom = 0.55, 0.30, 1.0
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
        elif c == "e":
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
        elif c in "+=":
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
            self.gesture = "PALM: swipe L / R"
            self._track_swipe(palm, now)
        else:
            self.swipe_hist.clear()
            self.gesture = "POINTING"

    def _track_swipe(self, palm, now):
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

    # ---------- mouse (demo) ----------
    def on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._md = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.pointer = np.array([x, y], float)
            if flags & cv2.EVENT_FLAG_LBUTTON and getattr(self, "_md", None):
                x0, y0, px, py = self._md
                self.yaw += (x - px) * 0.008
                self.pitch = float(np.clip(self.pitch + (y - py) * 0.008, -1.45, 1.45))
                self._md = (x0, y0, x, y)
        elif event == cv2.EVENT_LBUTTONUP and getattr(self, "_md", None):
            x0, y0, _, _ = self._md
            if abs(x - x0) + abs(y - y0) < 6:
                self.select(self.hover)
            self._md = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.zoom = float(np.clip(self.zoom * (1.1 if flags > 0 else 1 / 1.1), 0.35, 4.0))

    # ---------- rendering ----------
    def render(self, now=None, preview=None):
        if now is None:
            now = time.perf_counter()
        dt = min(max(now - self._t_last, 1e-3), 0.1)
        self._t_last = now
        m = self.machine
        W, H = self.W, self.H

        if self.anim and m.animated:
            self.t += dt
        self.explode += (self.explode_target - self.explode) * (1 - math.exp(-dt * 8.0))
        e = self.explode

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
        cx, cy = 0.50 * W, 0.43 * H
        dd = np.maximum(depth, 1e-3)
        sx = cx + f * Vc[:, 0] / dd
        sy = cy - f * Vc[:, 1] / dd
        pts = np.stack([sx, sy], 1)

        img = self._bg.copy()
        self._grid(img, m, Rc, D, f, cx, cy, e)

        F = m.F
        a, b, c = Vc[F[:, 0]], Vc[F[:, 1]], Vc[F[:, 2]]
        n = np.cross(b - a, c - a)
        n /= np.linalg.norm(n, axis=1)[:, None] + 1e-12
        facing = np.einsum("ij,ij->i", n, -a) > 0
        two = m.ts[m.fp]
        vis = facing | two
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
            Fo = F[order]
            fp_s = m.fp[order]
            ns = n[order]
            L1 = np.array([0.45, 0.75, 0.50]); L1 /= np.linalg.norm(L1)
            L3 = np.array([-0.6, -0.4, 0.3]); L3 /= np.linalg.norm(L3)
            shade = (0.20 + 0.46 * np.maximum(0, ns @ L1) + 0.26 * np.maximum(0, ns[:, 2])
                     + 0.10 * np.maximum(0, ns @ L3))
            col = m.col[fp_s] * shade[:, None]
            if sel >= 0 and not self.isolate:
                col = np.where((fp_s == sel)[:, None], col, col * 0.62)
            if self.hover >= 0:
                hm = (fp_s == self.hover)[:, None]
                col = np.where(hm, col * 0.72 + 255 * 0.28, col)
            if sel >= 0:
                sm = (fp_s == sel)[:, None]
                col = np.where(sm, col * 0.68 + np.array([0, 191, 255]) * 0.32, col)
            col = np.clip(col, 0, 255).astype(np.uint8).tolist()

            P = pts[Fo]                                    # (K,3,2) float
            tri = np.round(P).astype(np.int32)
            edge = (fp_s == sel) if sel >= 0 else None
            for i in range(len(order)):
                cv2.fillConvexPoly(img, tri[i], col[i])
                if edge is not None and edge[i]:
                    cv2.polylines(img, [tri[i]], True, (0, 225, 255), 1)

            if self.pointer is not None:
                hit = self._hit(P, float(self.pointer[0]), float(self.pointer[1]))
                if hit >= 0:
                    hover = int(fp_s[hit])
        self.hover = hover

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
        inside = ~(neg & pos)
        hits = np.nonzero(inside)[0]
        return int(hits[-1]) if len(hits) else -1

    def _grid(self, img, m, Rc, D, f, cx, cy, e):
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
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (72, 64, 58), 1, cv2.LINE_AA)

    # ---------- HUD ----------
    def _panel(self, img, x0, y0, x1, y1, alpha=0.62):
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.W, x1), min(self.H, y1)
        roi = img[y0:y1, x0:x1]
        roi[:] = (roi * (1 - alpha) + np.array([20, 16, 14]) * alpha).astype(np.uint8)

    def _hud(self, img, m, e, preview):
        W, H = self.W, self.H
        F = cv2.FONT_HERSHEY_SIMPLEX
        AA = cv2.LINE_AA
        white, grey, amber = (245, 245, 245), (175, 170, 165), (0, 200, 255)

        self._panel(img, 10, 10, 470, 78)
        cv2.putText(img, f"{m.name}", (20, 36), F, 0.62, white, 1, AA)
        cv2.putText(img, f"{self.mi + 1}/{len(self.machines)}   "
                         f"{'ANIMATING' if (self.anim and m.animated) else 'PAUSED' if m.animated else 'STATIC'}"
                         f"   Explode {int(e * 100)}%   Zoom {self.zoom:.2f}x",
                    (20, 62), F, 0.45, grey, 1, AA)

        shown = self.sel if self.sel >= 0 else self.hover
        if shown >= 0:
            p = m.parts[shown]
            lines = textwrap.wrap(p.desc, 34)
            hgt = 54 + 19 * len(lines) + 8
            self._panel(img, W - 330, 10, W - 10, 10 + hgt)
            tag = ("ISOLATED" if self.isolate else "SELECTED") if self.sel == shown else "HOVER"
            cv2.putText(img, tag, (W - 320, 30), F, 0.4, amber if self.sel == shown else grey, 1, AA)
            cv2.putText(img, p.name, (W - 320, 54), F, 0.55, white, 1, AA)
            for i, ln in enumerate(lines):
                cv2.putText(img, ln, (W - 320, 78 + 19 * i), F, 0.44, (210, 205, 200), 1, AA)

        # explode bar
        bx0, bx1, by = int(W * 0.50) - 150, int(W * 0.50) + 150, H - 26
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
        self._panel(img, 10, H - 112, W - 262, H - 40, 0.5)
        legend = ["THUMB+INDEX pinch + move: ROTATE      THUMB+RING pinch + up/down: ZOOM",
                  "THUMB+MIDDLE tap: SELECT part        THUMB+PINKY tap: ISOLATE / SHOW ALL",
                  "TWO HANDS apart / together: EXPLODE   FIST 1s: RESET   PALM swipe: NEXT MACHINE",
                  "Keys: M exit 3D   N/B machine   Space animate   E explode   I isolate   Tab part"]
        for i, ln in enumerate(legend):
            cv2.putText(img, ln, (20, H - 92 + 19 * i), F, 0.4, (200, 196, 190), 1, AA)

        # hand cursor
        if self.pointer is not None:
            px, py = int(self.pointer[0]), int(self.pointer[1])
            colr = {"index": (0, 255, 0), "ring": (255, 200, 0), "middle": (0, 200, 255),
                    "pinky": (255, 120, 255)}.get(self.pinch, white)
            cv2.circle(img, (px, py), 11, colr, 2, AA)
            cv2.circle(img, (px, py), 2, colr, -1, AA)
            if self.hover >= 0 and self.pinch is None:
                cv2.putText(img, m.parts[self.hover].name, (px + 16, py - 10), F, 0.45, white, 1, AA)

        # camera preview
        if preview is not None:
            pw, ph = 240, 180
            pv = cv2.resize(preview, (pw, ph))
            img[H - ph - 10:H - 10, W - pw - 10:W - 10] = pv
            cv2.rectangle(img, (W - pw - 10, H - ph - 10), (W - 10, H - 10), (200, 200, 200), 1)


# ============================================================
# Stand-alone demo / preview images
# ============================================================
def _demo():
    ex = Explorer()
    ex.active = True
    win = "Machine Explorer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, ex.W, ex.H)
    cv2.setMouseCallback(win, ex.on_mouse)
    print(__doc__)
    while True:
        cv2.imshow(win, ex.render())
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        if k != 255:
            ex.key(k)
    cv2.destroyAllWindows()


def _shots(folder):
    os.makedirs(folder, exist_ok=True)
    ex = Explorer()
    for i, m in enumerate(ex.machines):
        ex.set_machine(i)
        ex.t = 1.3
        for tag, e, sel in (("assembled", 0.0, -1), ("exploded", 1.0, 3)):
            ex.explode = ex.explode_target = e
            ex.sel = sel
            ex.tgt, ex.rad = m.c0 + (m.c1 - m.c0) * e, m.r0 + (m.r1 - m.r0) * e
            ex.anim = False
            img = ex.render(ex._t_last + 0.05)
            path = os.path.join(folder, f"{i}_{tag}.png")
            cv2.imwrite(path, img)
            print("wrote", path)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--shots":
        _shots(sys.argv[2])
    else:
        _demo()