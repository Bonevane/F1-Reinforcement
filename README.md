# F1 2020 RL Driver

A Soft Actor-Critic (SAC) agent that learns to drive **F1 2020** (Time Trial, Red Bull Ring) from the game's UDP telemetry, controlling the car through a virtual Xbox pad. It is bootstrapped from laps recorded on a racing wheel, then improves by driving on its own.

It also includes a **live overlay** that draws what the agent "sees" on top of the TV-pod camera, for recording videos.

## How it works

| Part | What it does |
|---|---|
| **Observation (27 values)** | forward/sideways speed, lateral & longitudinal g, offset from the reference line, previous action, and 10 lookahead points of the reference line (every 14 m, ~140 m ahead) in the car's frame |
| **Action (2 values)** | steering and a combined throttle/brake pedal, sent at 20 Hz through a virtual Xbox pad (with dead-zone compensation) |
| **Reward** | +0.1 per metre of progress along the track; penalties for leaving the track, going beyond the recorded track edges, invalidating the lap, big speed losses (impacts), jerky steering/pedal changes, and ending an episode |
| **Learning** | SAC (Stable-Baselines3), 1M-transition replay buffer pre-filled with wheel demos, optional offline pretraining (SAC updates on the demos before driving) |
| **Training loop** | the game is paused while the network updates, at episode ends and at the start/finish line (a straight, so the pause never lands mid-corner); laps are restarted automatically through the pause menu |
| **Robustness** | atomic checkpoints every 5,000 steps (+ on Ctrl+C / crash), automatic resume from the newest loadable checkpoint, built for power cuts |

## Files

```
f1_rl.py            training rig: recording tools, Gymnasium env, SAC training, eval
f1_overlay.py       "AI vision" overlay (pygame), composited in OBS or drawn on top of the game
tracks/
  austria.npy         reference line (driven once, 2 m resolution)
  austria_edges.npy   track edges (one lap hugging each white line)
  austria_elev.npy    road height along the line (learned by the overlay)
fonts/              Google Sans Flex (SIL OFL), used by the overlay
overlay_camera.json overlay calibration for the TV-pod camera
```

## Setup (Windows)

```bash
pip install -r requirements.txt
```

In F1 2020: *Settings → Telemetry*: UDP On, Broadcast Off, IP `127.0.0.1`, Port `20777`, Send Rate 60 Hz, Format 2020. Mode: Time Trial.

## Usage

```bash
# check the virtual pad reaches the game correctly
python f1_rl.py calibrate --backend gamepad

# record the track (one lap each)
python f1_rl.py track --track tracks/austria.npy        # reference line
python f1_rl.py edge  --track tracks/austria.npy        # run twice, once per edge

# record your own laps (wheel or pad) as demonstrations
python f1_rl.py demo --track tracks/austria.npy --out demos/wheel

# train (resumes automatically from checkpoints/)
python f1_rl.py train --track tracks/austria.npy --demos demos/wheel --pretrain-steps 20000 --backend gamepad

# let the trained agent drive (no learning, no restarts)
python f1_rl.py eval --track tracks/austria.npy --backend gamepad
```

### Overlay

```bash
# with train/eval running (they forward telemetry to port 20778)
python f1_overlay.py --track tracks/austria.npy --no-help

# see-through window on top of the game (borderless windowed) - works with any screen recorder
python f1_overlay.py --track tracks/austria.npy --no-help --ontop
```

Without `--ontop`, capture the window in OBS and add a *Chroma Key* (green) filter. Calibrate the camera by holding **Ctrl** and pressing `[ ]` FOV, `; '` height, `- =` pitch, `, .` forward, `K L` sideways, `O P` yaw (works while the game has focus).

## Trained model

The trained policy is attached to the GitHub Release (`step_*.zip`). Put it in `checkpoints/` or pass it with `--model`.
