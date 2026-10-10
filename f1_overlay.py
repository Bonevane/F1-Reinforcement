"""
"What the AI sees": a live overlay for recording videos of the agent.

Draws, from the game's telemetry:
  * the model's 10 lookahead points (its actual inputs) as a ribbon on the road
  * your recorded line further ahead, faintly
  * the left/right track edges from your edge map (red when the car is beyond)
  * the AI's live steering / throttle / brake, speed, lap time
  * a top-down mini-map of the same view

It runs in its own window on a green (chroma-key) background. In OBS, put a
Window Capture of this window above your game capture and add a Chroma Key
filter: the green disappears and the drawings sit on top of the hood cam.

Usage (while training or eval is running, which forwards telemetry to 20778):
    python f1_overlay.py --track tracks\\austria.npy
Without training running (e.g. you driving), listen to the game directly:
    python f1_overlay.py --track tracks\\austria.npy --port 20777

Requires: pip install pygame

On-top mode (Windows): --ontop makes the overlay a see-through, click-through
window on top of the game (game in BORDERLESS WINDOWED). You see it live while
driving, and any recorder (OBS Display Capture, ShareX) captures it directly,
no chroma key needed. Calibrate this way: the game pauses when it loses focus.

Lining it up with the TV pod (saved to overlay_camera.json). On Windows the keys
work while the GAME has focus: hold CTRL and press:
    [ / ]  field of view        ; / '  camera height
    - / =  camera pitch         , / .  camera forward offset
    K / L  camera sideways      O / P  camera yaw
    H      hide/show help       Ctrl+C in the terminal quits
Drive a straight: tune until the ribbon sits on the road and the edge lines
sit on the white lines. Set the game's FOV first and keep it fixed.
"""
import argparse
import ctypes
import json
import math
import os
import socket
import sys
import threading
import time

import numpy as np
import pygame
from f1_2020_telemetry.packets import unpack_udp_packet

# Must match f1_rl.py so the ribbon shows exactly what the model gets.
TRACK_RES_M = 2.0
LOOKAHEAD_SPACING_M = 15.0
N_WAYPOINTS = 10
LOOK_STEP = max(1, int(LOOKAHEAD_SPACING_M / TRACK_RES_M))
TRACK_LIMIT_MARGIN_M = 1.8
WHITE_LINE_OUTSET_M = 0.9       # edge laps were driven with wheels ON the line

CAM_FILE = "overlay_camera.json"
# Fitted to an F1 2020 TV-pod screenshot on a straight, from pixel-measured
# kerb edges (vanishing point 629.9, 254.4 at 1280x720; track edge matched
# within ~1 px): camera 1 m up, pitched 6.84 deg down, effectively 0.55 m
# left of the car's centre, looking 0.66 deg right of the car's heading.
# FOV can't be measured from one image; it only changes how far apart the
# dots look along the road. Check it against a distance board if you can.
CAM_DEFAULT = {"fov_deg": 72.0, "height": 1.0, "forward": 0.0, "pitch_deg": -6.84,
               "lateral": -0.551, "yaw_deg": 0.657}
NEAR_M = 6.0                    # don't draw closer than this (it's under the car)

CHROMA = (0, 255, 0)

# Material 3 dark scheme, toned down for broadcast: near-neutral surfaces,
# one desaturated teal accent, muted amber for limits. 8 px spacing grid.
M3 = {
    "primary": (128, 210, 222), "primary_dim": (52, 92, 100),
    "tertiary": (230, 180, 128),
    "error": (240, 140, 130), "error_container": (92, 28, 26),
    "on_error_container": (255, 218, 214),
    "surface": (17, 19, 21), "surface_high": (27, 30, 33),
    "surface_highest": (44, 48, 52),
    "on_surface": (232, 234, 237), "on_surface_variant": (150, 157, 164),
    "outline": (88, 94, 100), "shadow": (8, 9, 10),
}
UI_SCALE = 1.1      # overall HUD size; raise/lower to taste
FONT_STACK = ("googlesansflex,googlesans,robotoflex,roboto,"
              "segoeuivariabledisplay,segoeui,arial")


class Telemetry:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.lock = threading.Lock()
        self.s = {}
        self.last = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            data, _ = self.sock.recvfrom(4096)
            try:
                pkt = unpack_udp_packet(data)
            except Exception:
                continue
            h = pkt.header
            i = h.playerCarIndex
            with self.lock:
                if h.packetId == 0:
                    m = pkt.carMotionData[i]
                    k = 1 / 32767.0
                    self.s.update(
                        pos=np.array([m.worldPositionX, m.worldPositionY, m.worldPositionZ]),
                        fwd=np.array([m.worldForwardDirX, m.worldForwardDirY, m.worldForwardDirZ]) * k,
                        right=np.array([m.worldRightDirX, m.worldRightDirY, m.worldRightDirZ]) * k)
                    self.last = time.time()
                elif h.packetId == 2:
                    lap = pkt.lapData[i]
                    self.s.update(lap_time=lap.currentLapTime, lap_dist=lap.lapDistance,
                                  lap_invalid=lap.currentLapInvalid)
                elif h.packetId == 6:
                    t = pkt.carTelemetryData[i]
                    self.s.update(speed=t.speed, steer=t.steer, throttle=t.throttle,
                                  brake=t.brake)

    def get(self):
        with self.lock:
            return dict(self.s), time.time() - self.last


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


class Camera:
    def __init__(self, w, h):
        self.w, self.h = w, h
        try:
            with open(CAM_FILE) as f:
                self.p = {**CAM_DEFAULT, **json.load(f)}
        except (OSError, ValueError):
            self.p = dict(CAM_DEFAULT)

    def save(self):
        with open(CAM_FILE, "w") as f:
            json.dump(self.p, f, indent=1)

    def setup(self, pos, fwd, right):
        f, r = unit(fwd), unit(right)
        up = unit(np.cross(r, f))
        if up[1] < 0:
            up = -up
        car_f, car_r = f, r
        yaw = math.radians(self.p.get("yaw_deg", 0.0))   # + = look right
        f, r = (unit(f * math.cos(yaw) + r * math.sin(yaw)),
                unit(r * math.cos(yaw) - f * math.sin(yaw)))
        pitch = math.radians(self.p["pitch_deg"])
        f, up = (unit(f * math.cos(pitch) + up * math.sin(pitch)),
                 unit(up * math.cos(pitch) - f * math.sin(pitch)))
        self.origin = (pos + up * self.p["height"] + car_f * self.p["forward"]
                       + car_r * self.p.get("lateral", 0.0))
        self.f, self.r, self.up = f, r, up
        self.focal = (self.w / 2) / math.tan(math.radians(self.p["fov_deg"]) / 2)

    def project(self, pts):
        """World points (N,3) -> screen (N,2), and a mask of points in front."""
        rel = pts - self.origin
        z = rel @ self.f
        x = rel @ self.r
        y = rel @ self.up
        ok = z > NEAR_M
        zs = np.where(ok, z, 1.0)
        sx = self.w / 2 + self.focal * x / zs
        sy = self.h / 2 - self.focal * y / zs
        return np.stack([sx, sy], 1), ok


def polyline(surf, color, pts, ok, width, dash=0):
    seg = []

    def flush():
        if len(seg) > 1:
            if dash:
                for i in range(0, len(seg) - 1, 2 * dash):
                    part = seg[i:i + dash + 1]
                    if len(part) > 1:
                        pygame.draw.lines(surf, color, False, part, width)
            else:
                pygame.draw.lines(surf, color, False, seg, width)

    for p, good in zip(pts, ok):
        if good and -2000 < p[0] < 4000 and -2000 < p[1] < 4000:
            seg.append((float(p[0]), float(p[1])))
        else:
            flush()
            seg = []
    flush()


class UI:
    """Drawing helpers on an 8 px grid, scaled to the window height."""
    PAD = 28          # card padding
    GAP = 8           # base spacing unit

    def __init__(self, screen):
        self.sc = screen
        self.k = screen.get_height() / 720.0 * UI_SCALE
        fdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

        def f(size, style):
            path = os.path.join(fdir, f"GoogleSansFlex-{style}.ttf")
            if os.path.exists(path):
                return pygame.font.Font(path, self.px(size))
            return pygame.font.SysFont(FONT_STACK, self.px(size), bold=style != "Regular")

        self.display = f(56, "Display")   # speed
        self.value = f(22, "SemiBold")    # lap time
        self.title = f(15, "SemiBold")
        self.body = f(13, "Regular")
        self.label = f(11, "SemiBold")    # small caps labels (tracked)
        ok = os.path.exists(os.path.join(fdir, "GoogleSansFlex-Display.ttf"))
        print(f"[overlay] font: {'Google Sans Flex (fonts/)' if ok else 'fonts/ folder missing, using system font'}")

    def px(self, v):
        return max(1, int(round(v * self.k)))

    def card(self, rect, color=None, radius=20):
        pygame.draw.rect(self.sc, color or M3["surface"], rect, border_radius=self.px(radius))

    def render(self, txt, font, color, tracking=0):
        if not tracking:
            return font.render(txt, True, color)
        imgs = [font.render(ch, True, color) for ch in txt]
        w = sum(i.get_width() for i in imgs) + tracking * (len(imgs) - 1)
        out = pygame.Surface((w, font.get_height()), pygame.SRCALPHA)
        x = 0
        for i in imgs:
            out.blit(i, (x, 0)); x += i.get_width() + tracking
        return out

    def text(self, txt, font, color, pos, anchor="topleft", tracking=0):
        img = self.render(txt, font, color, tracking)
        r = img.get_rect(**{anchor: pos})
        self.sc.blit(img, r)
        return r

    def caps(self, txt, pos, anchor="topleft", color=None):
        return self.text(txt.upper(), self.label, color or M3["on_surface_variant"], pos,
                         anchor, tracking=self.px(1.2))

    def chip(self, txt, pos, bg, fg, dot=None, anchor="topleft"):
        img = self.render(txt.upper(), self.label, fg, self.px(1.2))
        h, pad = self.px(32), self.px(16)
        dw = self.px(14) if dot else 0
        r = pygame.Rect(0, 0, img.get_width() + 2 * pad + dw, h)
        setattr(r, anchor, pos)
        pygame.draw.rect(self.sc, bg, r, border_radius=h // 2)
        x = r.x + pad
        if dot:
            pygame.draw.circle(self.sc, dot, (x + self.px(3), r.centery), self.px(4))
            x += dw
        self.sc.blit(img, img.get_rect(midleft=(x, r.centery)))
        return r

    def hbar(self, x, y, w, frac, color):
        """M3 Expressive linear indicator: active segment, gap, track, stop dot."""
        frac = max(0.0, min(1.0, frac))
        h = self.px(6); gap = self.px(4)
        act = int((w - gap) * frac)
        if act >= h:
            pygame.draw.rect(self.sc, color, (x, y, act, h), border_radius=h // 2)
            tx = x + act + gap
        else:
            tx = x
        if x + w - tx >= h:
            pygame.draw.rect(self.sc, M3["surface_highest"], (tx, y, x + w - tx, h), border_radius=h // 2)
            pygame.draw.circle(self.sc, color, (x + w - h // 2, y + h // 2), max(1, h // 4))

    def slider(self, x, y, w, value, color):
        """Centred slider: track, active span from centre, thin handle."""
        h = self.px(6); cy = y + h // 2
        cx = x + w // 2
        hw, hh, gap = self.px(4), self.px(20), self.px(4)
        hx = int(cx + max(-1, min(1, value)) * (w / 2 - hw))
        pygame.draw.rect(self.sc, M3["surface_highest"], (x, y, w, h), border_radius=h // 2)
        lo, hi = sorted((cx, hx))
        if hi - lo > h:
            pygame.draw.rect(self.sc, color, (lo, y, hi - lo, h), border_radius=h // 2)
        # gap around the handle, in the card colour
        pygame.draw.rect(self.sc, M3["surface"], (hx - hw // 2 - gap, cy - hh // 2, hw + 2 * gap, hh))
        if abs(hx - cx) > self.px(12):
            pygame.draw.rect(self.sc, M3["on_surface_variant"], (cx - 1, cy - self.px(5), 2, self.px(10)))
        pygame.draw.rect(self.sc, color, (hx - hw // 2, cy - hh // 2, hw, hh), border_radius=hw // 2)


CAL_STEPS = {pygame.K_LEFTBRACKET: ("fov_deg", -1), pygame.K_RIGHTBRACKET: ("fov_deg", 1),
             pygame.K_SEMICOLON: ("height", -0.05), pygame.K_QUOTE: ("height", 0.05),
             pygame.K_MINUS: ("pitch_deg", -0.5), pygame.K_EQUALS: ("pitch_deg", 0.5),
             pygame.K_COMMA: ("forward", -0.1), pygame.K_PERIOD: ("forward", 0.1),
             pygame.K_k: ("lateral", -0.05), pygame.K_l: ("lateral", 0.05),
             pygame.K_o: ("yaw_deg", -0.1), pygame.K_p: ("yaw_deg", 0.1)}

# Windows virtual-key codes for the same keys (US layout positions)
_VK = {pygame.K_LEFTBRACKET: 0xDB, pygame.K_RIGHTBRACKET: 0xDD, pygame.K_SEMICOLON: 0xBA,
       pygame.K_QUOTE: 0xDE, pygame.K_MINUS: 0xBD, pygame.K_EQUALS: 0xBB,
       pygame.K_COMMA: 0xBC, pygame.K_PERIOD: 0xBE, pygame.K_k: 0x4B, pygame.K_l: 0x4C,
       pygame.K_o: 0x4F, pygame.K_p: 0x50, pygame.K_h: 0x48}
# Key you hold while pressing calibration keys. 0x11 = either Ctrl,
# 0x10 = either Shift, 0xA3 = right Ctrl only, 0xA2 = left Ctrl only.
HOTKEY_MODIFIER = 0x11


class GlobalHotkeys:
    """Reads the keyboard even when the GAME has focus (F1 pauses if you click
    away). Only while CTRL is held, so normal typing never changes
    anything. Holding a key repeats it."""

    def __init__(self):
        self.u32 = ctypes.windll.user32
        self.down_since = {}
        self.last_fire = {}

    def _down(self, vk):
        return bool(self.u32.GetAsyncKeyState(vk) & 0x8000)

    def poll(self):
        out, now = [], time.time()
        if not self._down(HOTKEY_MODIFIER):
            self.down_since.clear()
            return out
        for k, vk in _VK.items():
            if self._down(vk):
                if k not in self.down_since:
                    self.down_since[k] = self.last_fire[k] = now
                    out.append(k)
                elif (k != pygame.K_h and now - self.down_since[k] > 0.35
                      and now - self.last_fire[k] > 0.06):
                    self.last_fire[k] = now
                    out.append(k)
            else:
                self.down_since.pop(k, None)
        return out


def make_ontop(hwnd):
    """Topmost, click-through, pure green fully transparent (colour key)."""
    u32 = ctypes.windll.user32
    GWL_EXSTYLE, WS_EX_LAYERED, WS_EX_TRANSPARENT = -20, 0x80000, 0x20
    WS_EX_TOPMOST, WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE = 0x8, 0x80, 0x08000000
    get = getattr(u32, "GetWindowLongPtrW", u32.GetWindowLongW)
    setl = getattr(u32, "SetWindowLongPtrW", u32.SetWindowLongW)
    get.restype = setl.restype = ctypes.c_ssize_t
    get.argtypes = [ctypes.c_void_p, ctypes.c_int]
    setl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
    style = get(hwnd, GWL_EXSTYLE)
    setl(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST
         | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
    u32.SetLayeredWindowAttributes(ctypes.c_void_p(hwnd), 0x0000FF00, 0, 0x1)  # LWA_COLORKEY
    u32.SetWindowPos(ctypes.c_void_p(hwnd), ctypes.c_void_p(-1), 0, 0, 0, 0,
                     0x0001 | 0x0002 | 0x0010)    # NOSIZE | NOMOVE | NOACTIVATE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--port", type=int, default=20778)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--bg", choices=["green", "black"], default="green")
    ap.add_argument("--no-help", action="store_true", help="start with help hidden")
    ap.add_argument("--ontop", action="store_true",
                    help="Windows: see-through click-through window over the game "
                         "(game in borderless windowed); size = your screen")
    ap.add_argument("--screenshot", help=argparse.SUPPRESS)
    a = ap.parse_args()

    track2d = np.load(a.track)
    M = len(track2d)
    edges_file = os.path.splitext(a.track)[0] + "_edges.npy"
    edges = np.load(edges_file) if os.path.exists(edges_file) else None
    if edges is not None and np.isnan(edges).any():
        edges = None
    # outward normal for the "positive" side, matching f1_rl.track_offset
    t = np.roll(track2d, -1, 0) - np.roll(track2d, 1, 0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    n_pos = np.stack([-t[:, 1], t[:, 0]], 1)
    # Road height along the line, learned as the car drives and SAVED, so after
    # one full lap hills and dips are right from the start of every run.
    elev_file = os.path.splitext(a.track)[0] + "_elev.npy"
    elev = np.full(M, np.nan)
    if os.path.exists(elev_file):
        try:
            e = np.load(elev_file)
            if e.shape == (M,):
                elev = e
        except Exception:
            pass
    print(f"[overlay] track heights known for {np.mean(~np.isnan(elev)) * 100:.0f}% of the lap "
          f"({elev_file}); drive a full lap to fill the rest", flush=True)
    elev_saved_at, elev_dirty = time.time(), False

    def save_elev():
        tmp = elev_file + ".tmp.npy"
        np.save(tmp, elev)
        os.replace(tmp, elev_file)

    def heights_for(car_y):
        known = ~np.isnan(elev)
        if known.sum() < 2:
            return np.full(M, car_y)
        idxs = np.arange(M)
        k = idxs[known]                          # interpolate gaps, wrapping
        return np.interp(idxs, np.concatenate([k - M, k, k + M]),
                         np.tile(elev[known], 3))

    tel = Telemetry(a.port)
    win = sys.platform == "win32"
    if win:
        try:                                   # real pixels, not DPI-scaled
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
    if a.ontop:
        if not win:
            sys.exit("--ontop only works on Windows")
        u32 = ctypes.windll.user32
        a.width, a.height = u32.GetSystemMetrics(0), u32.GetSystemMetrics(1)
        os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
        a.bg = "green"
    pygame.init()
    screen = pygame.display.set_mode((a.width, a.height), pygame.NOFRAME if a.ontop else 0)
    pygame.display.set_caption("AI vision overlay (chroma key green in OBS)")
    if a.ontop:
        make_ontop(pygame.display.get_wm_info()["window"])
        print(f"[overlay] on-top mode {a.width}x{a.height}: green is see-through, "
              "clicks go to the game. Put the game in BORDERLESS WINDOWED.", flush=True)
    hotkeys = GlobalHotkeys() if win else None
    if hotkeys:
        print("[overlay] calibration keys work while the game has focus: "
              "hold CTRL + [ ] ; ' - = , . K L O P H", flush=True)
    ui = UI(screen)
    smooth = {"steer": 0.0, "throttle": 0.0, "brake": 0.0}   # displayed values
    SMOOTH_TAU_S = 0.06       # input animation time constant: quick, not laggy
    cam = Camera(a.width, a.height)
    bg = CHROMA if a.bg == "green" else (0, 0, 0)
    show_help = not a.no_help
    clock = pygame.time.Clock()
    dt = 0.0

    def to3d(xz, y):
        return np.stack([xz[:, 0], y, xz[:, 1]], 1)

    try:
        while True:
            pressed = []
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return
                if e.type == pygame.KEYDOWN and not hotkeys:
                    if e.key == pygame.K_ESCAPE:
                        return
                    pressed.append(e.key)
            if hotkeys:
                pressed += hotkeys.poll()
            for k in pressed:
                if k == pygame.K_h:
                    show_help = not show_help
                if k in CAL_STEPS:
                    key, d = CAL_STEPS[k]
                    p = cam.p
                    p[key] = round(p.get(key, 0.0) + d, 3)
                    cam.save()
                    print(f"[overlay] {key} = {p[key]}", flush=True)

            s, age = tel.get()
            screen.fill(bg)
            W, H, P = a.width, a.height, ui.px
            PAD, G = P(UI.PAD), P(UI.GAP)
            if "pos" not in s or age > 1.0:
                msg = "Waiting for telemetry" if "pos" not in s else "Telemetry paused"
                sub = (f"Listening on UDP port {a.port}" if "pos" not in s
                       else f"Last packet {age:.0f} s ago. Game paused?")
                r = pygame.Rect(0, 0, P(400), PAD * 2 + P(16) + G + P(28) + G // 2 + P(18))
                r.center = (W // 2, H // 2)
                ui.card(r, M3["surface"])
                y = ui.caps("AI vision overlay", (r.x + PAD, r.y + PAD)).bottom + G
                y = ui.text(msg, ui.value, M3["on_surface"], (r.x + PAD, y)).bottom + G // 2
                ui.text(sub, ui.body, M3["on_surface_variant"], (r.x + PAD, y))
                pygame.display.flip()
                clock.tick(30)
                continue

            pos3, fwd, right = s["pos"], s["fwd"], s["right"]
            pos2 = np.array([pos3[0], pos3[2]])
            idx = int(np.argmin(np.sum((track2d - pos2) ** 2, axis=1)))
            elev[idx] = pos3[1] if np.isnan(elev[idx]) else 0.9 * elev[idx] + 0.1 * pos3[1]
            elev_dirty = True
            if time.time() - elev_saved_at > 5.0:
                save_elev()
                elev_saved_at, elev_dirty = time.time(), False
            heights = heights_for(pos3[1])
            cam.setup(pos3, fwd, right)

            # ---------------- AR layer (no anti-aliasing: keeps the chroma key clean)
            ahead = (idx + np.arange(0, 110)) % M          # ~220 m of road
            line_pts, ok = cam.project(to3d(track2d[ahead], heights[ahead]))
            polyline(screen, M3["on_surface_variant"], line_pts, ok, P(2), dash=2)

            beyond = False
            if edges is not None:
                d = track2d[idx] - pos2
                off = float(t[idx, 0] * -d[1] - t[idx, 1] * -d[0])
                limit = edges[idx, 0 if off > 0 else 1] + TRACK_LIMIT_MARGIN_M
                beyond = abs(off) > limit
                for side, sign in ((0, 1.0), (1, -1.0)):
                    w = edges[ahead, side] + WHITE_LINE_OUTSET_M
                    xz = track2d[ahead] + sign * n_pos[ahead] * w[:, None]
                    ep, eok = cam.project(to3d(xz, heights[ahead]))
                    polyline(screen, M3["error"] if beyond else M3["tertiary"], ep, eok, P(3))

            look = (idx + LOOK_STEP * np.arange(1, N_WAYPOINTS + 1)) % M
            lp, lok = cam.project(to3d(track2d[look], heights[look]))
            polyline(screen, M3["primary"], lp, lok, P(4))
            for k in range(N_WAYPOINTS - 1, -1, -1):        # far first, near on top
                if lok[k]:
                    c = (int(lp[k][0]), int(lp[k][1]))
                    r = max(P(3), int(P(9) * (1 - 0.07 * k)))
                    pygame.draw.circle(screen, M3["surface"], c, r + P(2))
                    pygame.draw.circle(screen, M3["primary"], c, r)

            # ---------------- telemetry card, bottom-left
            alpha = 1.0 - math.exp(-dt / SMOOTH_TAU_S) if dt > 0 else 1.0
            for key in smooth:
                smooth[key] += alpha * (float(s.get(key, 0)) - smooth[key])
            steer, thr, brk = smooth["steer"], smooth["throttle"], smooth["brake"]
            spd = float(s.get("speed", 0))
            cw = P(344)
            lab_h = ui.label.get_height()
            glyph = ui.display.render("8", True, (255, 255, 255)).get_bounding_rect()
            cap_top, cap_h = glyph.y, glyph.height        # visible digit box inside the font line
            row_h = lab_h + G + P(6)                       # label, gap, bar
            ch = PAD + cap_h + 3 * G + row_h + 2 * G + row_h + PAD
            card = pygame.Rect(P(24), H - P(24) - ch, cw, ch)
            ui.card(card, M3["surface"])
            L, R = card.x + PAD, card.right - PAD
            y = card.y + PAD                               # top of the digits sits exactly at PAD
            # optical alignment: digits' left side bearing pulled back to the padding line
            nimg = ui.display.render(f"{spd:.0f}", True, M3["on_surface"])
            nx = L - nimg.get_bounding_rect().x
            screen.blit(nimg, (nx, y - cap_top))
            base = y - cap_top + ui.display.get_ascent()   # shared baseline
            ux = L + ui.display.size("888")[0] + G         # fixed so km/h doesn't jitter
            ui.text("km/h", ui.body, M3["on_surface_variant"], (ux, base - ui.body.get_ascent()))
            ui.caps("Lap", (R, y), "topright")
            ui.text(f"{s.get('lap_time', 0):.1f}s", ui.value, M3["on_surface"],
                    (R, base - ui.value.get_ascent()), "topright")
            y += cap_h + 3 * G
            ui.caps("Steering", (L, y))
            ui.text(f"{steer:+.2f}", ui.label, M3["on_surface"], (R, y), "topright")
            ui.slider(L, y + lab_h + G, R - L, steer, M3["primary"])
            y += row_h + 2 * G
            half = (R - L - 2 * G) // 2
            for i, (name, v, col) in enumerate((("Throttle", thr, M3["primary"]), ("Brake", brk, M3["error"]))):
                x = L + i * (half + 2 * G)
                ui.caps(name, (x, y))
                ui.text(f"{v * 100:.0f}%", ui.label, M3["on_surface"], (x + half, y), "topright")
                ui.hbar(x, y + lab_h + G, half, v, col)

            # ---------------- status chip, top-centre
            if beyond:
                ui.chip("Track limits", (W // 2, P(24)), M3["error_container"],
                        M3["on_error_container"], dot=M3["error"], anchor="midtop")
            elif s.get("lap_invalid"):
                ui.chip("Lap invalidated", (W // 2, P(24)), M3["surface"],
                        M3["on_surface"], dot=M3["tertiary"], anchor="midtop")

            # ---------------- lookahead map card, top-left: car frame, forward = up
            mw, mh = P(232), P(300)
            mcard = pygame.Rect(P(24), P(24), mw, mh)
            ui.card(mcard, M3["surface"])
            y = ui.text("POV: you're the agent", ui.title, M3["on_surface"],
                        (mcard.centerx, mcard.y + PAD), "midtop").bottom + 2 * G
            inner = pygame.Rect(mcard.x + G, y, mw - 2 * G, mcard.bottom - G - y)
            ui.card(inner, M3["surface_high"], 14)
            f2 = unit(np.array([fwd[0], fwd[2]]))
            r2 = unit(np.array([right[0], right[2]]))
            cx, cy = inner.centerx, inner.bottom - P(24)
            scale = (cy - inner.y - P(16)) / 160.0          # metres -> px

            def to_map(xz):
                rel = xz - pos2
                return np.stack([cx + (rel @ r2) * scale, cy - (rel @ f2) * scale], 1)

            screen.set_clip(inner.inflate(-P(4), -P(4)))
            span = (idx + np.arange(-10, 100)) % M
            if edges is not None:
                for side, sign in ((0, 1.0), (1, -1.0)):
                    w = edges[span, side] + WHITE_LINE_OUTSET_M
                    pts = to_map(track2d[span] + sign * n_pos[span] * w[:, None])
                    pygame.draw.lines(screen, M3["error"] if beyond else M3["tertiary"], False,
                                      pts.tolist(), P(2))
            pygame.draw.lines(screen, M3["outline"], False, to_map(track2d[span]).tolist(), 1)
            mp = to_map(track2d[look])
            pygame.draw.lines(screen, M3["primary_dim"], False, [(cx, cy)] + mp.tolist(), P(2))
            for q in mp:
                pygame.draw.circle(screen, M3["primary"], (int(q[0]), int(q[1])), P(3))
            pygame.draw.polygon(screen, M3["on_surface"],
                                [(cx, cy - P(8)), (cx - P(6), cy + P(6)), (cx, cy + P(3)), (cx + P(6), cy + P(6))])
            screen.set_clip(None)

            # ---------------- calibration card, top-right
            if show_help:
                p = cam.p
                rows = [("FOV", f"{p['fov_deg']:.0f}", "[  ]"), ("Height", f"{p['height']:.2f}", ";  '"),
                        ("Pitch", f"{p['pitch_deg']:.1f}", "-  ="), ("Forward", f"{p['forward']:.1f}", ",  ."),
                        ("Sideways", f"{p.get('lateral', 0):.2f}", "K  L"), ("Yaw", f"{p.get('yaw_deg', 0):.1f}", "O  P")]
                rh = P(24)
                hr = pygame.Rect(W - P(24) - P(272), P(24), P(272),
                                 PAD + ui.title.get_height() + 2 * G + rh * len(rows) + G + lab_h + PAD)
                ui.card(hr, M3["surface"])
                L, R = hr.x + PAD, hr.right - PAD
                y = ui.text("Camera calibration", ui.title, M3["on_surface"], (L, hr.y + PAD)).bottom + 2 * G
                for name, val, keys in rows:
                    ui.text(name, ui.body, M3["on_surface_variant"], (L, y + rh // 2), "midleft")
                    ui.text(val, ui.body, M3["on_surface"], (L + P(128), y + rh // 2), "midright")
                    ui.text(keys, ui.label, M3["primary"], (R, y + rh // 2), "midright")
                    y += rh
                ui.caps("H hide  \u00b7  Esc quit", (L, y + G), color=M3["outline"])

            pygame.display.flip()
            if a.screenshot:
                pygame.image.save(screen, a.screenshot)
                return
            dt = clock.tick(60) / 1000.0

    except KeyboardInterrupt:
        pass
    finally:
        if not a.screenshot:
            save_elev()
            print(f"[overlay] saved track heights ({np.mean(~np.isnan(elev)) * 100:.0f}% of lap)",
                  flush=True)

if __name__ == "__main__":
    main()
