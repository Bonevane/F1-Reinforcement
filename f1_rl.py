"""
F1 2020 + Gymnasium + SAC (Stable-Baselines3): an unattended training rig.

Built for a Windows PC with 8 GB RAM and unreliable power:
  * checkpoints are written atomically (a power cut never leaves a half file
    that looks complete) and the newest loadable one is resumed automatically
  * the replay buffer is saved with every checkpoint, so resuming loses nothing
  * training crashes on purpose if telemetry stops (game crashed, menu,
    power blip) so the watchdog can restart everything cleanly
  * you can record your own laps with a wheel and feed them to the agent

Modes:
  python f1_rl.py calibrate                        check virtual pad -> game mapping
  python f1_rl.py track --track tracks/austria.npy record the reference line (drive 1 lap)
  python f1_rl.py demo  --track tracks/austria.npy record your laps as training data
  python f1_rl.py train --track tracks/austria.npy [--demos demos] [--pretrain-steps N]
  python f1_rl.py eval  --track tracks/austria.npy [--model checkpoints/step_000100000.zip] [--stochastic]

Requirements (Windows):
  pip install gymnasium "stable-baselines3[extra]" vgamepad f1-2020-telemetry numpy
  pip install pyvjoy pydirectinput   (for the default vJoy virtual wheel)
  (vgamepad installs the ViGEmBus driver; reboot if prompted)

In-game: Settings > Telemetry: UDP On, Broadcast Off, IP 127.0.0.1,
Port 20777, Send Rate 60 Hz, UDP Format 2020. Mode: Time Trial.
"""
import argparse
import csv
import glob
import math
import os
import platform
import socket
import sys
import threading
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from f1_2020_telemetry.packets import unpack_udp_packet

# ----------------------------------------------------------------- config ---
UDP_PORT = 20777
FORWARD_PORT = 20778            # every packet is re-sent here for f1_overlay.py
                                # (the game can only send to one port)
CONTROL_HZ = 20                 # agent decisions per second
N_WAYPOINTS = 10                # lookahead points fed to the agent
LOOKAHEAD_SPACING_M = 15.0
TRACK_RES_M = 2.0
MAX_EPISODE_S = 600             # long enough for several laps without a restart
OFFTRACK_LIMIT_S = 0.5          # terminate after this long off track
STUCK_SPEED_MS = 2.0
STUCK_LIMIT_S = 3.0
ON_TRACK_SURFACES = {0, 1, 2}   # tarmac, rumble strip, concrete
SURFACE_NAMES = {0: "tarmac", 1: "rumble", 2: "concrete", 3: "rock", 4: "gravel",
                 5: "mud", 6: "sand", 7: "grass", 8: "water", 9: "cobblestone",
                 10: "metal", 11: "ridged"}
TELEMETRY_MAX_AGE_S = 0.5       # data older than this counts as stale
TELEMETRY_TIMEOUT_S = 30.0      # no fresh data this long (while the game
                                # should be running) = game is gone

# After "Restart Lap" the game drives the car itself for a few seconds, then
# hands control over. We detect the handover from the game's own input
# telemetry: busy (AI steering/throttle) -> neutral (our idle pad).
HANDOVER_TIMEOUT_S = 40.0       # give up waiting and start anyway
AI_START_GRACE_S = 3.0          # no AI driving seen by then = no auto-drive
NEUTRAL_EPS = 0.03              # inputs below this count as "hands off"
NEUTRAL_HOLD_S = 0.5            # must stay neutral this long
TELEPORT_M = 30.0               # jump beyond what speed explains = a restart

# Reward shaping that keeps the car on your line and off the walls.
# Distances are from the reference line you recorded with `track`.
LINE_FREE_M = 3.0               # this much deviation from your line is free
LINE_PENALTY_PER_M = 0.05       # per step, per metre beyond that
MAX_LINE_DEV_M = 10.0           # further than this = episode over (run-off, pits).
                                # Austria's run-off is tarmac, which the game
                                # reports as track, so this is what catches it.
IMPACT_KPH_PER_STEP = 15.0      # speed lost in one step beyond braking = a hit
IMPACT_PENALTY = 3.0
INVALID_PENALTY = 10.0          # lap invalidated (track limits / corner cut)
END_ON_INVALID = True           # strictest track-limits teaching
REWARD_VERSION = 2              # demos recorded before this get shaping added

CHECKPOINT_DIR = "checkpoints"
SAVE_EVERY_STEPS = 5_000        # ~4 min of driving at 20 Hz (+ on Ctrl+C)
# When to pause and learn: at the end of every episode (crash, reset) AND
# every time the car crosses the start/finish line, which is on a straight,
# so the pause never lands mid-corner. The cap is a safety net for a car that
# drives for ages without reaching the line.
MAX_STEPS_BETWEEN_UPDATES = 3_000   # ~2.5 min

# Track limits from a recorded map of the track edges (`edge` mode, see the
# guide). With it, "off track" means the car's centre is beyond the edge you
# drove along by more than this margin: driving with your outside wheels on
# the white line puts the centre ~0.9 m inside it, and all four wheels are
# over the line once the centre is ~0.9 m outside, so ~1.8 m in total.
TRACK_LIMIT_MARGIN_M = 1.8
EDGE_MODE_MAX_DEV_M = 30.0      # with an edge map, line distance only catches pits
PEDAL_SNAP = 0.9                # pedal beyond this = 100% (sampled actions
                                # rarely hit exactly 1.0, so flat-out needs help)
THROTTLE_SNAP = 0.7             # throttle above this is sent as 100%
PEDAL_CHANGE_PENALTY = 0.02     # per step, discourages throttle flapping
TARGET_ENTROPY = -4.0           # SAC exploration noise; lower = calmer inputs
                                # (library default is -2 for 2 action dims)
KEEP_CHECKPOINTS = 3
BUFFER_SIZE = 1_000_000         # ~250 MB with this observation size
LAP_LOG = "laps.csv"
DEMO_FLUSH_EVERY = 2_000        # write demo files often: power can go anytime

# "gamepad" = virtual Xbox pad (Windows). "keyboard" = macOS/CrossOver fallback.
# "gamepad"  = virtual Xbox pad (Windows). Default: with PAD_STEER_DEADZONE
#              compensation it measured 1:1 with no lag (lagtest: 0.297 of
#              0.300 within 0.06 s).
# "vjoy"     = virtual wheel/joystick (Windows): steering applied 1:1, like a
#              real wheel. Alternative if the pad ever misbehaves.
# "keyboard" = macOS/CrossOver fallback.
CONTROL_BACKEND = "keyboard" if platform.system() == "Darwin" else "gamepad"
VJOY_DEVICE = 1                 # vJoy device number (Configure vJoy app)
# F1 2020 ignores the first ~10% of pad stick travel (measured with
# `calibrate`: 0.25 -> 0.17, 0.50 -> 0.44, 1.0 -> 1.0, i.e. linear after a
# 0.096 deadzone). The gamepad backend pre-compensates so that the agent's
# steering value is the steering the car actually gets. Set to 0 to disable.
PAD_STEER_DEADZONE = 0.096
PWM_HZ = 50

# Pause menu navigation. CHECK THESE IN YOUR GAME (see the guide).
PAUSE_SEQUENCE = ["pause"]
RESUME_SEQUENCE = ["pause"]
RESTART_SEQUENCE = ["pause", "select"]
MENU_GAP_S = 0.8                # pause between menu presses; raise if presses get lost
RESTART_ATTEMPTS = 3


_LINE_CROSSED = False           # set when the car crosses the start/finish line


class TelemetryLost(RuntimeError):
    pass


# ------------------------------------------------------------ controllers ---
class GamepadController:
    """Virtual Xbox 360 pad via vgamepad (Windows)."""

    def __init__(self):
        import vgamepad as vg
        self.pad = vg.VX360Gamepad()
        B = vg.XUSB_BUTTON
        self.buttons = {"pause": B.XUSB_GAMEPAD_START, "back": B.XUSB_GAMEPAD_B,
                        "select": B.XUSB_GAMEPAD_A, "up": B.XUSB_GAMEPAD_DPAD_UP,
                        "down": B.XUSB_GAMEPAD_DPAD_DOWN,
                        "left": B.XUSB_GAMEPAD_DPAD_LEFT,
                        "right": B.XUSB_GAMEPAD_DPAD_RIGHT}

    def apply(self, steer, pedal):
        d = PAD_STEER_DEADZONE
        if abs(steer) > 1e-4 and d > 0:       # skip the game's dead zone
            steer = math.copysign(min(1.0, d + abs(steer) * (1.0 - d)), steer)
        self.pad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
        self.pad.right_trigger_float(value_float=max(pedal, 0.0))
        self.pad.left_trigger_float(value_float=max(-pedal, 0.0))
        self.pad.update()

    def tap(self, name, hold=0.1):
        self.pad.press_button(button=self.buttons[name]); self.pad.update()
        time.sleep(hold)
        self.pad.release_button(button=self.buttons[name]); self.pad.update()


class KeyboardController:
    """macOS/CrossOver fallback: arrow keys with PWM for partial inputs."""

    def __init__(self):
        from pynput.keyboard import Controller, Key
        self.kb = Controller()
        self.axis_keys = {"left": Key.left, "right": Key.right,
                          "throttle": Key.up, "brake": Key.down}
        self.menu_keys = {"pause": Key.esc, "back": Key.esc, "select": Key.enter,
                          "up": Key.up, "down": Key.down,
                          "left": Key.left, "right": Key.right}
        self.duty = {k: 0.0 for k in self.axis_keys}
        self.held = {k: False for k in self.axis_keys}
        self.active = False
        self.lock = threading.Lock()
        threading.Thread(target=self._pwm_loop, daemon=True).start()

    def _set(self, k, on):
        if on != self.held[k]:
            (self.kb.press if on else self.kb.release)(self.axis_keys[k])
            self.held[k] = on

    def _pwm_loop(self):
        period = 1.0 / PWM_HZ
        while True:
            t0 = time.perf_counter()
            with self.lock:
                active, duty = self.active, dict(self.duty)
            if not active:
                for k in self.axis_keys:
                    self._set(k, False)
                time.sleep(period)
                continue
            on = {k: d for k, d in duty.items() if d > 0.02}
            for k in self.axis_keys:
                self._set(k, k in on)
            for k, d in sorted(on.items(), key=lambda kv: kv[1]):
                if d >= 0.98:
                    continue
                wait = t0 + d * period - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                self._set(k, False)
            rest = t0 + period - time.perf_counter()
            if rest > 0:
                time.sleep(rest)

    def apply(self, steer, pedal):
        with self.lock:
            self.duty.update(left=max(-steer, 0.0), right=max(steer, 0.0),
                             throttle=max(pedal, 0.0), brake=max(-pedal, 0.0))
            self.active = True

    def tap(self, name, hold=0.1):
        with self.lock:
            self.active = False
        time.sleep(2.0 / PWM_HZ)
        key = self.menu_keys[name]
        self.kb.press(key); time.sleep(hold); self.kb.release(key)


class VJoyController:
    """Virtual joystick via vJoy: X = steering, Y = throttle, Z = brake.

    The game treats it like a wheel: inputs apply 1:1 with no pad smoothing.
    Menus (pause, restart) are driven with keyboard keys, which F1 2020's
    menus accept, so no menu buttons need binding.

    Requires: the vJoy driver, then  pip install pyvjoy pydirectinput
    Bind the axes in-game with:  python f1_rl.py vjoybind
    """

    AXIS_MAX = 0x8000

    def __init__(self):
        import pyvjoy
        import pydirectinput
        pydirectinput.PAUSE = 0          # no built-in delay after each key
        self.pyvjoy = pyvjoy
        self.keys = pydirectinput
        self.dev = pyvjoy.VJoyDevice(VJOY_DEVICE)
        self.axes = {"steer": pyvjoy.HID_USAGE_X, "throttle": pyvjoy.HID_USAGE_Y,
                     "brake": pyvjoy.HID_USAGE_Z}
        self.menu_keys = {"pause": "esc", "back": "esc", "select": "enter",
                          "up": "up", "down": "down", "left": "left",
                          "right": "right"}
        self.apply(0.0, 0.0)

    def _axis(self, name, value01):
        v = int(max(0.0, min(1.0, value01)) * (self.AXIS_MAX - 1)) + 1
        self.dev.set_axis(self.axes[name], v)

    def apply(self, steer, pedal):
        self._axis("steer", (steer + 1.0) / 2.0)     # centre = half range
        self._axis("throttle", max(pedal, 0.0))
        self._axis("brake", max(-pedal, 0.0))

    def tap(self, name, hold=0.1):
        key = self.menu_keys[name]
        self.keys.keyDown(key)
        time.sleep(hold)
        self.keys.keyUp(key)


def make_controller():
    if CONTROL_BACKEND == "keyboard":
        return KeyboardController()
    if CONTROL_BACKEND == "vjoy":
        return VJoyController()
    return GamepadController()


def vjoybind(args):
    """Move one vJoy axis at a time so F1 2020's binding screen can detect it."""
    ctrl = VJoyController()
    steps = [
        ("Steer Left", lambda t: ctrl._axis("steer", 0.5 - 0.5 * t)),
        ("Steer Right", lambda t: ctrl._axis("steer", 0.5 + 0.5 * t)),
        ("Accelerate", lambda t: ctrl._axis("throttle", t)),
        ("Brake", lambda t: ctrl._axis("brake", t)),
    ]
    print("In F1 2020: Settings > Controls, pick the vJoy device, edit its "
          "mapping. For each action below: press Enter HERE, then within 5 s "
          "click the matching action in the game so it waits for input.")
    for name, move in steps:
        input(f"\n[{name}] press Enter, then click '{name}' in the game...")
        ctrl.apply(0.0, 0.0)
        for i in range(5, 0, -1):
            print(f"  moving in {i}...", end="\r", flush=True)
            time.sleep(1)
        for k in range(21):                  # sweep over ~1 s, hold at the end
            move(k / 20)
            time.sleep(0.05)
        time.sleep(1.5)
        ctrl.apply(0.0, 0.0)
        print(f"  done: {name}            ")
    print("\nAll four sent. Set deadzones to 0, linearity 0, saturation 100 "
          "for this device, then run:  python f1_rl.py calibrate  and  lagtest")


# -------------------------------------------------------------- telemetry ---
class Telemetry:
    """Background UDP listener; keeps the latest player-car state.

    Motion packets are only sent while the player is in control of the car,
    so their age tells us whether the game is actually running a session.
    """

    def __init__(self, port=UDP_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.lock = threading.Lock()
        self.state = {}
        self.last_motion = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        fwd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        while True:
            data, _ = self.sock.recvfrom(4096)
            try:
                fwd.sendto(data, ("127.0.0.1", FORWARD_PORT))
            except OSError:
                pass
            try:
                pkt = unpack_udp_packet(data)
            except Exception:
                continue
            h = pkt.header
            i = h.playerCarIndex
            with self.lock:
                if h.packetId == 0:        # motion
                    m = pkt.carMotionData[i]
                    self.state.update(
                        x=m.worldPositionX, z=m.worldPositionZ,
                        vx=m.worldVelocityX, vz=m.worldVelocityZ,
                        fx=m.worldForwardDirX / 32767.0, fz=m.worldForwardDirZ / 32767.0,
                        rx=m.worldRightDirX / 32767.0, rz=m.worldRightDirZ / 32767.0,
                        g_lat=m.gForceLateral, g_lon=m.gForceLongitudinal,
                    )
                    self.last_motion = time.time()
                elif h.packetId == 2:      # lap data
                    lap = pkt.lapData[i]
                    self.state.update(lap_num=lap.currentLapNum,
                                      lap_time=lap.currentLapTime,
                                      last_lap_time=lap.lastLapTime,
                                      lap_dist=lap.lapDistance,
                                      pit_status=lap.pitStatus,
                                      lap_invalid=lap.currentLapInvalid)
                elif h.packetId == 6:      # car telemetry
                    t = pkt.carTelemetryData[i]
                    self.state.update(speed=t.speed / 3.6,
                                      steer_in=t.steer, throttle_in=t.throttle,
                                      brake_in=t.brake,
                                      surfaces=list(t.surfaceType))

    def snapshot(self, max_age=TELEMETRY_MAX_AGE_S, timeout=None):
        """Latest state; waits for fresh motion data. max_age=None: any age."""
        timeout = TELEMETRY_TIMEOUT_S if timeout is None else timeout
        needed = {"x", "speed", "surfaces", "lap_num"}
        t0 = time.time()
        while True:
            with self.lock:
                s = dict(self.state)
                age = time.time() - self.last_motion
            if needed <= s.keys() and (max_age is None or age <= max_age):
                return s
            if time.time() - t0 > timeout:
                raise TelemetryLost(
                    "No fresh telemetry. Game closed, in a menu, or UDP off?")
            time.sleep(0.02)


def inputs_busy(s):
    return (abs(s.get("steer_in", 0.0)) > NEUTRAL_EPS
            or s.get("throttle_in", 0.0) > NEUTRAL_EPS
            or s.get("brake_in", 0.0) > NEUTRAL_EPS)


def wait_for_handover(tel, log=print):
    """Return the first fresh state after the game hands back control.

    The game's input telemetry shows the AI's inputs while it drives. We keep
    our pad neutral (in demo mode: hands off the wheel and pedals), so the
    handover shows up as inputs going from busy to neutral and staying there.
    """
    t0 = time.time()
    saw_ai, neutral_since = False, None
    while time.time() - t0 < HANDOVER_TIMEOUT_S:
        s = tel.snapshot(max_age=None)   # motion may pause during auto-drive
        now = time.time()
        if inputs_busy(s):
            saw_ai, neutral_since = True, None
        elif saw_ai:
            neutral_since = neutral_since or now
            if now - neutral_since >= NEUTRAL_HOLD_S:
                return tel.snapshot()
        elif now - t0 > AI_START_GRACE_S:
            return tel.snapshot()        # the game never drove: already ours
        time.sleep(0.02)
    log("Handover not detected; starting anyway")
    return tel.snapshot()


# ------------------------------------------------------------ track line ---
class LineWatcher:
    """Detects start/finish line crossings.

    The lap number alone misses one: after "Restart Lap" in Time Trial the
    car is on an out-lap, and crossing the line into the first timed lap
    doesn't change it. So also watch the lap distance going from negative
    (before the line) to positive, and the lap timer resetting.
    """

    MIN_GAP_S = 20.0    # no lap is shorter; ignores the double signal at the
                        # out-lap line (distance flips first, timer resets later)

    def __init__(self, s, just_crossed=False):
        self.prev = s
        self.last_cross = time.time() if just_crossed else 0.0

    def update(self, s):
        crossed = self._crossed(s)
        if crossed and time.time() - self.last_cross < self.MIN_GAP_S:
            return False
        if crossed:
            self.last_cross = time.time()
        return crossed

    def _crossed(self, s):
        p, self.prev = self.prev, s
        if s["lap_num"] != p["lap_num"]:
            return True
        pd, d = p.get("lap_dist"), s.get("lap_dist")
        if pd is not None and d is not None:
            if pd < 0 <= d:                  # out-lap -> first timed lap
                return True
            if pd > 200 and 0 <= d < 100:    # end of a lap: distance wraps to 0
                return True
        pt, t = p.get("lap_time"), s.get("lap_time")
        if pt is not None and t is not None and pt > 2.0 and t < pt - 1.0:
            return True
        return False


def status_line(tel, s, extra=""):
    """One-line live readout of what the telemetry says right now."""
    age = time.time() - getattr(tel, "last_motion", time.time())
    fresh = "LIVE" if age < 0.5 else f"STALE {age:.0f}s (paused/menu?)"
    print(f"\r  [{fresh}] lap {s.get('lap_num')} | lap dist "
          f"{s.get('lap_dist', 0):7.0f} m | lap time {s.get('lap_time', 0):6.1f} s | "
          f"{s.get('speed', 0) * 3.6:5.0f} km/h {extra}      ", end="", flush=True)


def wait_for_line(tel, message):
    """Block until the car crosses the start/finish line, showing live status."""
    print(message)
    print("  (Lap dist counts up from 0 at the line; it is negative on an "
          "out-lap before the line.)")
    s = tel.snapshot(max_age=None, timeout=600)
    line = LineWatcher(s)
    last = 0.0
    while True:
        s = tel.snapshot(max_age=2.0, timeout=600)
        if line.update(s):
            status_line(tel, s, "<- LINE CROSSED")
            print()
            return
        if time.time() - last > 0.5:
            status_line(tel, s)
            last = time.time()
        time.sleep(0.02)


def record_track(path, tel):
    """Drive one full lap yourself; saves an evenly spaced (x, z) loop."""
    print("Waiting for session... make sure you are on track.")
    wait_for_line(tel, "Drive to and CROSS the start/finish line to begin recording...")
    print(">>> Crossed line! Recording. Complete this lap cleanly! <<<")
    line = LineWatcher(tel.snapshot(), just_crossed=True)

    pts = []
    last_print = time.time()
    while True:
        s = tel.snapshot(max_age=2.0, timeout=120)
        if line.update(s):
            print(">>> Finish line crossed! Finalizing track line... <<<")
            break

        p = (s["x"], s["z"])
        if not pts or math.dist(p, pts[-1]) > 0.5:
            pts.append(p)

        if time.time() - last_print > 2.0:
            print(f"  Recording... {len(pts)} points logged so far", end="\r")
            last_print = time.time()

        time.sleep(0.02)

    pts = np.array(pts)
    if len(pts) < 50:
        print("\nError: Too few points recorded. Did the lap register properly?")
        return

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    grid = np.arange(0, cum[-1], TRACK_RES_M)
    line = np.stack([np.interp(grid, cum, pts[:, 0]),
                     np.interp(grid, cum, pts[:, 1])], axis=1)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, line)
    print(f"\nSaved {len(line)} points ({cum[-1]:.0f} m) to {path}")


def shaping(obs, next_obs):
    """Line-keeping and wall-hit penalties, computed only from observations so
    the same numbers can be added to demos recorded with an older reward.

    obs[0] = forward speed / 80 (m/s), obs[4] = offset from your line / 10 (m).
    Works on single vectors or on arrays of them. Returns (penalty, far_off_line).
    """
    obs, next_obs = np.asarray(obs), np.asarray(next_obs)
    dev = np.abs(next_obs[..., 4] * 10.0)
    penalty = LINE_PENALTY_PER_M * np.maximum(0.0, dev - LINE_FREE_M)
    drop_kph = (obs[..., 0] - next_obs[..., 0]) * 80.0 * 3.6
    penalty = penalty + IMPACT_PENALTY * (drop_kph > IMPACT_KPH_PER_STEP)
    return penalty, dev > MAX_LINE_DEV_M


def edges_path(track_path):
    root, _ = os.path.splitext(track_path)
    return root + "_edges.npy"


def track_offset(track, idx, pos):
    """Signed sideways distance (m) of pos from the reference line at idx.
    Uses the line's own direction, so the sign is the same side of the track
    whichever way the car points. Same formula for recording and checking."""
    M = len(track)
    t = track[(idx + 1) % M] - track[(idx - 1) % M]
    t = t / (np.linalg.norm(t) + 1e-9)
    d = np.asarray(pos) - track[idx]
    return float(t[0] * d[1] - t[1] * d[0])


def record_edge(track_path, tel):
    """Drive one lap hugging ONE track edge (outside wheels on the white line).
    Run it twice, once per edge; the side is worked out automatically."""
    track = np.load(track_path)
    M = len(track)
    print("Drive one full lap with your outside wheels ON the white line of "
          "one edge (left or right, your choice). Steady speed is fine.")
    wait_for_line(tel, "Cross the start/finish line to begin "
                       "(don't restart the lap while recording)...")
    print(">>> Recording this edge. Cross the line again to finish. <<<")
    line = LineWatcher(tel.snapshot(), just_crossed=True)
    samples = [[] for _ in range(M)]
    last = 0.0
    while True:
        s = tel.snapshot(max_age=2.0, timeout=120)
        if line.update(s):
            status_line(tel, s, "<- LINE CROSSED")
            print()
            break
        if time.time() - last > 0.5:
            covered = sum(1 for v in samples if v) / M * 100
            status_line(tel, s, f"| edge covered {covered:3.0f}%")
            last = time.time()
        pos = np.array([s["x"], s["z"]])
        idx = int(np.argmin(np.sum((track - pos) ** 2, axis=1)))
        samples[idx].append(track_offset(track, idx, pos))
        time.sleep(0.02)
    vals = np.array([np.median(v) if v else np.nan for v in samples])
    side = 0 if np.nanmedian(vals) > 0 else 1       # 0 = positive side
    widths = np.abs(vals)
    # fill gaps by circular interpolation, then smooth out wobbles
    ok = ~np.isnan(widths)
    if ok.sum() < M * 0.5:
        sys.exit("Too few points recorded along the edge; try again.")
    idxs = np.arange(M)
    widths = np.interp(idxs, idxs[ok], widths[ok], period=M)
    k = 5
    padded = np.concatenate([widths[-k:], widths, widths[:k]])
    widths = np.array([np.median(padded[i:i + 2 * k + 1]) for i in range(M)])

    path = edges_path(track_path)
    edges = np.load(path) if os.path.exists(path) else np.full((M, 2), np.nan)
    edges[:, side] = widths
    np.save(path, edges)
    print(f"\nSaved this edge to {path}: distance from your line "
          f"min {widths.min():.1f} m, typical {np.median(widths):.1f} m, "
          f"max {widths.max():.1f} m")
    if np.isnan(edges).any():
        print("Now drive the OTHER edge with the same command.")
    else:
        total = edges.sum(axis=1)
        print(f"Both edges recorded. Track width between your two edge laps: "
              f"typical {np.median(total):.1f} m. Track limits are now active.")


# ------------------------------------------------------------ environment ---
class F1Env(gym.Env):
    """control=False: observe only (used to record your own driving)."""
    metadata = {"render_modes": []}

    def __init__(self, track_path, tel=None, control=True):
        self.tel = tel or Telemetry()
        self.track = np.load(track_path)
        self.M = len(self.track)
        self.edges = None
        ep = edges_path(track_path)
        if os.path.exists(ep):
            e = np.load(ep)
            if not np.isnan(e).any():
                self.edges = e
                print(f"Track limits: using edge map {ep}", flush=True)
            else:
                print(f"Edge map {ep} has only one side; drive the other edge "
                      f"to activate track limits", flush=True)
        self.ctrl = make_controller() if control else None
        self.dt = 1.0 / CONTROL_HZ
        self.look_step = max(1, int(LOOKAHEAD_SPACING_M / TRACK_RES_M))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        obs_dim = 5 + 2 + 2 * N_WAYPOINTS
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,),
                                            dtype=np.float32)
        self.paused = False
        self.prev_action = np.zeros(2, dtype=np.float32)

    # --- controller helpers
    def _apply(self, a):
        if self.ctrl:
            pedal = float(a[1])
            if pedal >= THROTTLE_SNAP:               # throttle: >70% = flat out
                pedal = 1.0
            elif pedal <= -PEDAL_SNAP:               # brake: near-full = full
                pedal = -1.0
            self.ctrl.apply(float(a[0]), pedal)

    def _tap_sequence(self, seq, hold=0.1, gap=0.4):
        for name in seq:
            self.ctrl.tap(name, hold=hold)
            time.sleep(gap)

    def _motion_fresh(self, within=1.5):
        """Is the car being simulated (not paused, not in a menu)?"""
        end = time.time() + within
        while time.time() < end:
            if time.time() - getattr(self.tel, "last_motion", time.time()) < 0.3:
                return True
            time.sleep(0.05)
        return False

    def _unpause(self):
        """Resume, and make sure it took: a missed press would leave the game
        paused and the next step would crash for lack of telemetry. The check
        waits generously, since pressing again too early would re-pause."""
        for _ in range(3):
            self._tap_sequence(RESUME_SEQUENCE, gap=MENU_GAP_S)
            if self._motion_fresh(within=4.0):
                break
            print("Resume press didn't register; retrying", flush=True)
        self.paused = False

    def _escape_menus(self):
        """Back out of whatever menu or dialog we landed in, then resume."""
        for _ in range(3):
            if self._motion_fresh(within=1.0):
                return
            self._tap_sequence(["back"], gap=MENU_GAP_S)
        if not self._motion_fresh():
            self._tap_sequence(RESUME_SEQUENCE, gap=MENU_GAP_S)

    # --- geometry
    def _nearest(self, pos):
        return int(np.argmin(np.sum((self.track - pos) ** 2, axis=1)))

    def _obs(self, s):
        pos = np.array([s["x"], s["z"]])
        f = np.array([s["fx"], s["fz"]]); f /= np.linalg.norm(f) + 1e-8
        r = np.array([s["rx"], s["rz"]]); r /= np.linalg.norm(r) + 1e-8
        v = np.array([s["vx"], s["vz"]])
        idx = self._nearest(pos)
        ahead = self.track[(idx + self.look_step * np.arange(1, N_WAYPOINTS + 1)) % self.M] - pos
        local = np.stack([ahead @ r, ahead @ f], axis=1) / 100.0
        lat_offset = (self.track[idx] - pos) @ r
        feats = [v @ f / 80.0, v @ r / 20.0, s["g_lat"] / 5.0, s["g_lon"] / 5.0,
                 lat_offset / 10.0, *self.prev_action]
        return np.concatenate([feats, local.ravel()]).astype(np.float32), idx

    # --- shared bookkeeping (agent and human recording use the same rules)
    def begin_episode(self, s):
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.steps = self.off_steps = self.stuck_steps = 0
        self.prev_lap = s["lap_num"]
        self.line = LineWatcher(s)
        self.lap_invalid = bool(s.get("lap_invalid", 0))
        # The first line crossing of an episode ends an out-lap or a partial
        # lap, never a full timed lap, so it is not reported as a lap time.
        self.crossed_line = False
        self.prev_pos = (s["x"], s["z"])
        self.prev_t = time.time()
        # Per-episode stats for the one-line report printed at the end.
        self.ep_progress_m = 0.0
        self.ep_max_kph = 0.0
        self.ep_return = 0.0
        self.ep_off_total = 0
        self.ep_invalid_steps = 0
        self.ep_beyond_edge = 0
        self.ep_start_dist = s.get("lap_dist", 0.0)
        self.in_pits = False
        obs, self.prev_idx = self._obs(s)
        self.prev_obs = obs
        return obs

    def transition(self, action, s):
        """Reward, termination and info for having taken `action` -> state s."""
        obs, idx = self._obs(s)
        self.steps += 1
        info = {"speed": s["speed"]}

        pos = (s["x"], s["z"])
        now = time.time()
        # At 300+ km/h a short hiccup can move the car tens of metres, so only
        # a jump well beyond what the speed explains counts as a restart.
        possible = max(s["speed"], 10.0) * (now - self.prev_t) * 1.5
        jumped = math.dist(pos, self.prev_pos) > possible + TELEPORT_M
        self.prev_pos, self.prev_t = pos, now
        if jumped:                                   # restart/teleport: not a
            info["teleport"] = True                  # real transition; end here
            self.prev_action = np.asarray(action, dtype=np.float32)
            return obs, 0.0, False, True, info

        d = (idx - self.prev_idx) % self.M          # progress, wraparound-safe
        if d > self.M // 2:
            d -= self.M
        reward = 0.1 * d * TRACK_RES_M
        self.prev_idx = idx

        off = any(x not in ON_TRACK_SURFACES for x in s["surfaces"])
        beyond_edge = False
        if self.edges is not None:
            o = track_offset(self.track, idx, (s["x"], s["z"]))
            limit = self.edges[idx, 0 if o > 0 else 1] + TRACK_LIMIT_MARGIN_M
            beyond_edge = abs(o) > limit
            off = off or beyond_edge
        self.ep_beyond_edge += int(beyond_edge)
        if off and self.off_steps == 0:              # log the moment it goes off
            names = [SURFACE_NAMES.get(x, str(x)) for x in s["surfaces"]]
            edge_txt = (f", centre {abs(o):.1f} m from line vs limit {limit:.1f} m"
                        if self.edges is not None else "")
            print(f"  OFF at lap dist {s.get('lap_dist', 0):.0f} m | wheels RL RR FL FR: "
                  f"{' '.join(names)}{edge_txt}"
                  f"{' -> BEYOND EDGE' if beyond_edge else ''}", flush=True)
        self.off_steps = self.off_steps + 1 if off else 0
        if off:
            reward -= 1.0
        reward -= 0.05 * abs(action[0] - self.prev_action[0])
        reward -= PEDAL_CHANGE_PENALTY * abs(action[1] - self.prev_action[1])

        invalidated = False
        if s.get("lap_invalid", 0):
            self.ep_invalid_steps += 1
        if s.get("lap_invalid", 0) and not self.lap_invalid:
            reward -= INVALID_PENALTY                # track limits / corner cut
            self.lap_invalid = invalidated = True
            print(f"  lap invalidated at lap dist {s.get('lap_dist', 0):.0f} m",
                  flush=True)

        line_pen, far_off_line = shaping(self.prev_obs, obs)
        line_pen, far_off_line = float(line_pen), bool(far_off_line)
        if self.edges is not None:       # the edge map judges track limits now
            far_off_line = abs(float(obs[4]) * 10.0) > EDGE_MODE_MAX_DEV_M
        reward -= line_pen
        self.in_pits = s.get("pit_status", 0) != 0

        if self.line.update(s):                      # crossed the line
            global _LINE_CROSSED
            _LINE_CROSSED = True                     # good moment to learn
            if self.crossed_line and s.get("last_lap_time", 0) > 0:
                info["lap_time"] = s["last_lap_time"]
                info["lap_valid"] = not self.lap_invalid
            self.crossed_line = True
            self.prev_lap = s["lap_num"]
            self.lap_invalid = False

        if s["speed"] < STUCK_SPEED_MS and self.steps > STUCK_LIMIT_S * CONTROL_HZ:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        off_end = self.off_steps > OFFTRACK_LIMIT_S * CONTROL_HZ
        stuck_end = self.stuck_steps > STUCK_LIMIT_S * CONTROL_HZ
        invalid_end = END_ON_INVALID and invalidated
        terminated = (off_end or stuck_end or far_off_line or self.in_pits
                      or invalid_end)
        if terminated:
            reward -= 10.0
        truncated = self.steps >= MAX_EPISODE_S * CONTROL_HZ

        self.ep_progress_m += d * TRACK_RES_M
        self.ep_max_kph = max(self.ep_max_kph, s["speed"] * 3.6)
        self.ep_return += reward
        self.ep_off_total += int(off)
        if terminated or truncated:
            info["end_reason"] = ("pit lane" if self.in_pits else
                                  "track limits (beyond edge)" if off_end and beyond_edge else
                                  "off track" if off_end else
                                  "too far off your line" if far_off_line else
                                  "lap invalidated" if invalid_end else
                                  "stuck/too slow" if stuck_end else "time limit")
            info["lap_dist"] = s.get("lap_dist", 0.0)

        self.prev_action = np.asarray(action, dtype=np.float32)
        obs[5:7] = self.prev_action                  # keep obs consistent
        self.prev_obs = obs
        return obs, float(reward), terminated, truncated, info

    # --- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._apply([0.0, 0.0])
        if self.paused:
            self._unpause()
        if getattr(self, "in_pits", False):
            # In the pit lane the game shows its own prompt; Enter twice
            # clears it (found by hand), then the normal restart works.
            print("Car is in the pits: clearing the pit prompt", flush=True)
            time.sleep(1.0)
            self._tap_sequence(["select", "select"], gap=MENU_GAP_S)
            time.sleep(1.0)
            self.in_pits = False
        for attempt in range(1, RESTART_ATTEMPTS + 1):
            self._tap_sequence(RESTART_SEQUENCE, gap=MENU_GAP_S)
            time.sleep(1.0)                      # let the restart settle
            # The game drives the first stretch itself. Those frames are not
            # the agent's doing, so the episode starts once control is handed over.
            try:
                s = wait_for_handover(self.tel)
                break
            except TelemetryLost:
                pass
            # Maybe a confirmation dialog ("restart lap?") is waiting.
            print(f"Restart attempt {attempt}: game didn't come back; "
                  f"pressing select (Enter/A) in case a confirmation is showing", flush=True)
            self._tap_sequence(["select"], gap=MENU_GAP_S)
            time.sleep(1.0)
            try:
                s = wait_for_handover(self.tel)
                print("  That worked. If this happens often, add one more "
                      "\"select\" to RESTART_SEQUENCE.", flush=True)
                break
            except TelemetryLost:
                print("  Still stuck; backing out of the menus and retrying",
                      flush=True)
                self._escape_menus()
        else:
            raise TelemetryLost(
                "Could not restart the lap. Watch the pause menu: does "
                "RESTART_SEQUENCE in f1_rl.py match the presses needed?")
        obs = self.begin_episode(s)
        # Freeze the game while SB3 trains on the finished episode.
        self._tap_sequence(PAUSE_SEQUENCE)
        self.paused = True
        return obs, {}

    def step(self, action):
        action = np.clip(action, -1, 1)
        if self.paused:
            self._unpause()
            self.t_next = time.perf_counter()

        self._apply(action)
        self.t_next += self.dt
        delay = self.t_next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self.t_next = time.perf_counter()

        s = self.tel.snapshot()                  # raises TelemetryLost if gone
        out = self.transition(action, s)
        if out[2] or out[3]:
            self._apply([0.0, 0.0])
            self._report(out[4])
        return out

    def _report(self, info):
        """One line per episode: how long, how far, how fast, and why it ended."""
        self.episodes_done = getattr(self, "episodes_done", 0) + 1
        reason = info.get("end_reason", "restart detected")
        print(f"ep {self.episodes_done:4d} | {self.steps / CONTROL_HZ:5.1f} s | "
              f"progress {self.ep_progress_m:6.0f} m | max {self.ep_max_kph:3.0f} km/h | "
              f"off-track {self.ep_off_total / CONTROL_HZ:4.1f} s | "
              f"lap-invalid {self.ep_invalid_steps / CONTROL_HZ:4.1f} s | "
              f"return {self.ep_return:7.1f} | ended: {reason} "
              f"(lap dist {self.ep_start_dist:.0f} -> {info.get('lap_dist', 0):.0f} m)",
              flush=True)

    def close(self):
        self._apply([0.0, 0.0])


# -------------------------------------------------------- calibration ---
def lagtest(args):
    """Measure how the game applies a sudden pad steering input while moving:
    the delay before it responds, and whether it reaches the full value."""
    tel = Telemetry()
    ctrl = make_controller()
    print("Put the car at the start of a straight, unpaused, game window "
          "focused, wheel unplugged. Starting in 5 s...")
    tel.snapshot(timeout=120)
    time.sleep(5)
    ctrl.apply(0.0, 0.7)                     # build speed
    time.sleep(3.0)
    target = 0.3
    t0 = time.time()
    ctrl.apply(target, 0.3)                  # sudden steering step
    rows = []
    while time.time() - t0 < 1.5:
        s = tel.snapshot()
        rows.append((time.time() - t0, s["steer_in"], s["speed"] * 3.6))
        time.sleep(0.05)
    ctrl.apply(0.0, -0.6)                    # straighten and brake
    time.sleep(2.0)
    ctrl.apply(0.0, 0.0)
    print(f"\nsent steer {target:+.2f} at t=0")
    print(f"{'t (s)':>6} {'game steer':>11} {'km/h':>6}")
    for t, st, v in rows:
        print(f"{t:6.2f} {st:11.3f} {v:6.0f}")
    reach = next((t for t, st, _ in rows if st >= 0.9 * target), None)
    final = rows[-1][1]
    print(f"\nreached 90% after: {reach:.2f} s" if reach is not None
          else "\nnever reached 90% of the sent value")
    print(f"value after 1.5 s: {final:.3f} of {target:.3f} sent")
    print("A wheel is instant. Over ~0.15 s to reach 90%, or a final value "
          "well below what was sent, means the game filters pad steering.")


def calibrate(args):
    """Send known inputs through the virtual pad, read back what the game saw."""
    tel = Telemetry()
    ctrl = make_controller()
    print("Sit in Time Trial on track, unpaused, game window focused.")
    tel.snapshot(timeout=120)
    print(f"{'sent':>12} | {'game steer':>10} {'throttle':>9} {'brake':>7}")
    tests = [("steer", v) for v in (-1, -0.5, -0.25, 0, 0.25, 0.5, 1)] + \
            [("pedal", v) for v in (0.25, 0.5, 1, -0.5, -1, 0)]
    for kind, v in tests:
        ctrl.apply(v if kind == "steer" else 0.0, v if kind == "pedal" else 0.0)
        time.sleep(1.2)
        s = tel.snapshot()
        print(f"{kind} {v:+.2f} | {s['steer_in']:>10.2f} "
              f"{s['throttle_in']:>9.2f} {s['brake_in']:>7.2f}")
    ctrl.apply(0.0, 0.0)
    print("Good: game values match what was sent (within ~0.05). "
          "If not, set deadzones to 0 and linearity to 0 for the controller.")


# --------------------------------------------------- recording your laps ---
def record_demos(args):
    """Record your own driving (wheel) as (obs, action, reward, next_obs, done)."""
    env = F1Env(args.track, control=False)
    tel = env.tel
    os.makedirs(args.out, exist_ok=True)
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    buf = {k: [] for k in ("obs", "actions", "rewards", "next_obs", "dones")}
    part, total, laps = 0, 0, 0

    def flush():
        nonlocal part
        if not buf["obs"]:
            return
        path = os.path.join(args.out, f"demo_{session}_part{part:03d}.npz")
        tmp = path + ".tmp.npz"
        np.savez(tmp, reward_version=np.array([REWARD_VERSION]),
                 **{k: np.asarray(v, dtype=np.float32) for k, v in buf.items()})
        os.replace(tmp, path)
        print(f"\n  [Disk Flush] Saved {len(buf['obs'])} transitions -> {path}")
        part += 1
        for v in buf.values():
            v.clear()

    def human_action(s):
        return np.array([s["steer_in"], s["throttle_in"] - s["brake_in"]],
                        dtype=np.float32)

    def restarted():
        # No waiting for a handover here: a real wheel never reads exactly
        # neutral, so that check stalls. The few seconds the game drives are
        # recorded too; they are competent driving and harmless as data.
        print("\nRestart detected: new episode, still recording")
        return tel.snapshot(max_age=2.0, timeout=60.0)

    print("Drive. Ctrl+C to stop. Recording starts when you are on track.")
    print("Restarting the lap is fine; recording carries on by itself.")
    s = tel.snapshot(timeout=600)
    obs = env.begin_episode(s)
    t_next = time.perf_counter()
    last_ui_update = time.time()

    try:
        while True:
            a = human_action(s)
            t_next += env.dt
            delay = t_next - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                t_next = time.perf_counter()

            try:
                s = tel.snapshot(max_age=2.0, timeout=60.0)
            except TelemetryLost:
                print("\nPaused or in menu; waiting...")
                flush()
                s = tel.snapshot(timeout=3600)
                if math.dist((s["x"], s["z"]), env.prev_pos) > TELEPORT_M:
                    print("\nRestart detected: new episode, still recording")
                obs = env.begin_episode(s)
                t_next = time.perf_counter()
                continue

            nobs, r, term, trunc, info = env.transition(a, s)
            if info.get("teleport"):                 # lap restarted: skip the
                s = restarted()                      # jump and the auto-drive
                obs = env.begin_episode(s)
                t_next = time.perf_counter()
                continue
            if env.in_pits:                          # pit lane isn't driving
                obs = env.begin_episode(s)           # data; skip it
                continue
            for k, v in zip(buf, (obs, a, r, nobs, float(term))):
                buf[k].append(v)
            total += 1

            # Live single-line status ticker (updates every ~0.5s)
            if time.time() - last_ui_update > 0.5:
                speed_kph = s["speed"] * 3.6
                current_lap_t = s.get("lap_time", 0.0)
                print(
                    f"\r  [Recording] Transitions: {total:<6} | Lap {s['lap_num']} ({current_lap_t:05.1f}s) | Speed: {speed_kph:5.1f} km/h | Reward: {r:+5.2f}",
                    end="",
                    flush=True,
                )
                last_ui_update = time.time()

            if "lap_time" in info:
                laps += 1
                valid_str = "VALID" if info["lap_valid"] else "INVALID"
                print(f"\n>>> Lap {laps} Complete: {info['lap_time']:.3f} s ({valid_str}) | Total Samples: {total} <<<")

            if term or trunc:
                obs = env.begin_episode(s)
            else:
                obs = nobs

            if len(buf["obs"]) >= DEMO_FLUSH_EVERY:
                flush()

    except KeyboardInterrupt:
        pass

    flush()
    print(f"\nDone: {total} transitions recorded across {laps} completed laps.")


def load_demos(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    files = [f for f in files if not f.endswith(".tmp.npz")]
    if not files:
        return None
    keys = ("obs", "actions", "rewards", "next_obs", "dones")
    parts, upgraded = [], 0
    for f in files:
        d = np.load(f)
        part = {k: d[k] for k in keys}
        version = int(d["reward_version"][0]) if "reward_version" in d.files else 1
        if version < REWARD_VERSION:
            # Older recordings: drop restart jumps (huge rewards), then add the
            # line/wall shaping so demos and agent share one reward function.
            ok = np.abs(part["rewards"]) < 20
            part = {k: v[ok] for k, v in part.items()}
            pen, _ = shaping(part["obs"], part["next_obs"])
            pen = pen + PEDAL_CHANGE_PENALTY * np.abs(
                part["next_obs"][:, 6] - part["obs"][:, 6])  # [5:7] = last action
            part["rewards"] = (part["rewards"] - pen).astype(np.float32)
            upgraded += 1
        parts.append(part)
    if upgraded:
        print(f"Added line/wall shaping to {upgraded} older demo files")
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    # Safety net: real per-step rewards stay well inside this range.
    ok = np.abs(out["rewards"]) < 40
    if not ok.all():
        print(f"Dropping {int((~ok).sum())} restart-jump transitions from demos")
        out = {k: v[ok] for k, v in out.items()}
    return out


# ------------------------------------------------------------ checkpoints ---
def _fsync(path):
    with open(path, "rb+") as f:
        os.fsync(f.fileno())


def save_checkpoint(model):
    """Atomic: write to temp names, flush to disk, then rename into place.
    The buffer is written first; a .zip only exists if its buffer does."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    stem = os.path.join(CHECKPOINT_DIR, f"step_{model.num_timesteps:09d}")
    if os.path.exists(stem + ".zip"):
        return                                   # already saved at this step
    model.save_replay_buffer(stem + "_buffer.tmp.pkl")
    _fsync(stem + "_buffer.tmp.pkl")
    os.replace(stem + "_buffer.tmp.pkl", stem + "_buffer.pkl")
    model.save(stem + ".tmp.zip")
    _fsync(stem + ".tmp.zip")
    os.replace(stem + ".tmp.zip", stem + ".zip")
    for old in list_checkpoints()[KEEP_CHECKPOINTS:]:
        for p in (old, old.replace(".zip", "_buffer.pkl")):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
    print(f"[checkpoint] {stem}.zip", flush=True)


def list_checkpoints():
    cks = [p for p in glob.glob(os.path.join(CHECKPOINT_DIR, "step_*.zip"))
           if not p.endswith(".tmp.zip")]
    return sorted(cks, key=os.path.getmtime, reverse=True)


def resume_latest(env):
    """Newest checkpoint that actually loads; damaged ones are skipped."""
    from stable_baselines3 import SAC
    for ck in list_checkpoints():
        try:
            model = SAC.load(ck, env=env)
            model.load_replay_buffer(ck.replace(".zip", "_buffer.pkl"))
            print(f"Resumed from {ck} at step {model.num_timesteps}", flush=True)
            return model
        except Exception as e:                   # half-written, corrupted...
            print(f"Skipping unreadable checkpoint {ck}: {e!r}", flush=True)
            os.replace(ck, ck + ".bad")
            try:
                os.remove(ck.replace(".zip", "_buffer.pkl"))
            except FileNotFoundError:
                pass
    return None


# ---------------------------------------------------------------- training ---
def make_callbacks():
    from stable_baselines3.common.callbacks import BaseCallback

    class RigCallback(BaseCallback):
        """Periodic atomic checkpoints + a CSV row for every completed lap."""

        def __init__(self):
            super().__init__()
            self.last_save = None

        def _on_training_start(self):
            self.last_save = self.num_timesteps
            new = not os.path.exists(LAP_LOG)
            self.lap_file = open(LAP_LOG, "a", newline="")
            self.lap_csv = csv.writer(self.lap_file)
            if new:
                self.lap_csv.writerow(["time", "step", "lap_time_s", "valid"])

        def _on_step(self):
            for info in self.locals.get("infos", []):
                if info.get("lap_time"):
                    self.lap_csv.writerow([datetime.now().isoformat(timespec="seconds"),
                                           self.num_timesteps,
                                           f"{info['lap_time']:.3f}",
                                           int(info["lap_valid"])])
                    self.lap_file.flush()
            return True

        def _on_rollout_end(self):
            # Called right before each learning phase. At a line crossing the
            # game is still running: pause it so the car doesn't drive blind
            # while the networks update. The next step() unpauses it.
            env = self.training_env.envs[0].unwrapped
            if not env.paused:
                env._apply([0.0, 0.0])
                env._tap_sequence(PAUSE_SEQUENCE, gap=MENU_GAP_S)
                env.paused = True
            if self.num_timesteps - self.last_save >= SAVE_EVERY_STEPS:
                save_checkpoint(self.model)
                self.last_save = self.num_timesteps

        def _on_training_end(self):
            self.lap_file.close()

    return RigCallback()


def _has_tensorboard():
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False


def train(args):
    from stable_baselines3 import SAC
    from stable_baselines3.common.logger import configure

    env = F1Env(args.track)
    model = resume_latest(env)
    if model is None:
        print("No checkpoint found: starting a new run.", flush=True)
        demos = load_demos(args.demos) if args.demos else None
        model = SAC(
            "MlpPolicy", env,
            learning_rate=3e-4,
            buffer_size=BUFFER_SIZE,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=(1, "episode"),   # overridden below: also at the line
            gradient_steps=-1,           # one update per env step collected
            learning_starts=500 if demos is not None else 2_000,
            ent_coef="auto",
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log="runs" if _has_tensorboard() else None,
            seed=args.seed,
            verbose=1,
        )
        if demos is not None:
            n = len(demos["obs"])
            for i in range(n):
                model.replay_buffer.add(
                    demos["obs"][i:i + 1], demos["next_obs"][i:i + 1],
                    demos["actions"][i:i + 1], demos["rewards"][i:i + 1],
                    demos["dones"][i:i + 1], [{}])
            print(f"Loaded {n} demo transitions into the replay buffer.", flush=True)
            if args.pretrain_steps:
                print(f"Pretraining on demos for {args.pretrain_steps} updates...",
                      flush=True)
                model.set_logger(configure(None, ["stdout"]))
                model.train(gradient_steps=args.pretrain_steps,
                            batch_size=model.batch_size)
                model._custom_logger = False     # let learn() set up its own
        save_checkpoint(model)                   # step 0: resumable from now

    # Learn at episode ends and at every start/finish line crossing (see
    # MAX_STEPS_BETWEEN_UPDATES). Stable-Baselines3 has no "event" trigger, so
    # replace the function it asks "keep driving or stop and learn?".
    import stable_baselines3.common.off_policy_algorithm as opa

    def keep_collecting(train_freq, num_steps, num_episodes):
        global _LINE_CROSSED
        if num_episodes > 0 or num_steps >= MAX_STEPS_BETWEEN_UPDATES:
            _LINE_CROSSED = False
            return False
        if _LINE_CROSSED and num_steps > 0:
            _LINE_CROSSED = False
            return False
        return True

    opa.should_collect_more_steps = keep_collecting
    model.target_entropy = TARGET_ENTROPY      # also applies to resumed runs

    remaining = args.steps - model.num_timesteps
    if remaining <= 0:
        print("Step budget already reached.")
        return
    try:
        model.learn(total_timesteps=remaining, callback=make_callbacks(),
                    reset_num_timesteps=False, tb_log_name="sac")
        save_checkpoint(model)
    except KeyboardInterrupt:
        # Ctrl+C: keep everything learned so far. Press Ctrl+C only once and
        # wait for "[checkpoint]": a second press interrupts the save.
        env.close()
        print("\nStopping: saving a checkpoint, please wait...", flush=True)
        save_checkpoint(model)
        print("Saved. Run the same command again to continue.", flush=True)
    except Exception:
        # Any crash (telemetry lost, game closed...): keep the progress, then
        # show the error. Rerunning the command continues from here.
        print("\nCrashed: saving a checkpoint before exiting...", flush=True)
        try:
            save_checkpoint(model)
        except Exception as e:
            print(f"Could not save: {e!r}", flush=True)
        raise
    finally:
        env.close()


def evaluate(args):
    """Free drive: the trained model just drives. No restarts, no pausing, no
    learning. Going off track, invalid laps etc. are only logged; the run never
    ends on its own. If YOU restart the lap or pause, it waits and carries on.
    Ctrl+C to stop (inputs go back to neutral)."""
    from stable_baselines3 import SAC
    model_path = args.model or (list_checkpoints() or [None])[0]
    if model_path is None:
        sys.exit("No model given and no checkpoint found.")
    model = SAC.load(model_path)
    env = F1Env(args.track)
    print(f"[eval] model: {model_path} "
          f"({'stochastic, like training' if args.stochastic else 'deterministic'})")
    print("[eval] free drive: no restarts, no pauses. Ctrl+C to stop.")
    print("[eval] waiting for telemetry (be in the car, unpaused)...", flush=True)
    s = env.tel.snapshot(timeout=3600)
    obs = env.begin_episode(s)
    env.crossed_line = False
    laps, last_status, waiting = 0, 0.0, False
    t_next = time.perf_counter()
    try:
        while True:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            action = np.clip(action, -1, 1)
            env._apply(action)
            t_next += env.dt
            delay = t_next - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                t_next = time.perf_counter()
            try:
                s = env.tel.snapshot(timeout=0.3)
            except TelemetryLost:
                env._apply([0.0, 0.0])
                if not waiting:
                    print("[eval] telemetry stopped (paused/menu?) - waiting, inputs neutral",
                          flush=True)
                    waiting = True
                time.sleep(0.2)
                t_next = time.perf_counter()
                continue
            if waiting:
                print("[eval] telemetry back - driving", flush=True)
                waiting = False
                obs = env.begin_episode(s)
                continue
            obs, _, _, _, info = env.transition(action, s)
            if info.get("teleport"):                  # you restarted the lap
                print("[eval] car moved (lap restart?) - carrying on from here", flush=True)
                obs = env.begin_episode(s)
                continue
            if "lap_time" in info:
                laps += 1
                print(f"[eval] LAP {laps}: {info['lap_time']:.3f} s "
                      f"({'valid' if info['lap_valid'] else 'INVALID'})", flush=True)
            if time.time() - last_status > 2.0:
                last_status = time.time()
                print(f"[eval] lap dist {s.get('lap_dist', 0):6.0f} m | "
                      f"{s['speed'] * 3.6:5.0f} km/h | steer {action[0]:+.2f} "
                      f"pedal {action[1]:+.2f}", flush=True)
    except KeyboardInterrupt:
        print("\n[eval] stopped", flush=True)
    finally:
        env._apply([0.0, 0.0])
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["calibrate", "lagtest", "vjoybind", "track", "edge", "demo", "train", "eval"])
    p.add_argument("--track", help="reference line .npy (all modes but calibrate)")
    p.add_argument("--steps", type=int, default=2_000_000,
                   help="total env-step budget for the run (20 Hz: 72k = 1 hour)")
    p.add_argument("--demos", help="folder of recorded demo .npz files")
    p.add_argument("--pretrain-steps", type=int, default=0,
                   help="gradient updates on demos before driving (new runs only)")
    p.add_argument("--backend", choices=["vjoy", "gamepad", "keyboard"],
                   help="override CONTROL_BACKEND (e.g. lagtest the Xbox pad)")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed for a new run (use 1, 2, 3 for repeats)")
    p.add_argument("--out", default="demos", help="demo output folder")
    p.add_argument("--model")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--stochastic", action="store_true",
                   help="eval: sample actions like in training instead of the mean")
    a = p.parse_args()
    if a.backend:
        CONTROL_BACKEND = a.backend
    if a.mode not in ("calibrate", "lagtest", "vjoybind") and not a.track:
        p.error("--track is required for this mode")
    if a.mode == "calibrate":
        calibrate(a)
    elif a.mode == "lagtest":
        lagtest(a)
    elif a.mode == "vjoybind":
        vjoybind(a)
    elif a.mode == "edge":
        record_edge(a.track, Telemetry())
    elif a.mode == "track":
        record_track(a.track, Telemetry())
    elif a.mode == "demo":
        record_demos(a)
    elif a.mode == "train":
        train(a)
    else:
        evaluate(a)
