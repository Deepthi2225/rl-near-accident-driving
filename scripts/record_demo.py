"""
Record a GIF of a policy driving one scenario, from a chase camera behind the ego car.

First searches the unseen evaluation scenarios (headless, fast) for a close call: the hazard
happens and the policy still reaches the goal, as close to the hazard as possible (for the
cut-in, the first episode where the car cuts in and brakes hard, since the cars are closest
while side by side before the cut-in). Then
replays that scenario with rendering on and records it.

    python scripts/record_demo.py --scenario cut_in --policy switcher:checkpoints/cut_in/ppo_switcher_s0.pt

Writes assets/demo_<scenario>.gif.
"""
import argparse
import os
import queue
import sys

import carla
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from evaluate import EVAL_SEED_OFFSET, SwitcherPolicy, make_policy  # noqa: E402
from whew.experts import COMMAND_NAMES  # noqa: E402
from whew.scenarios import SCENARIOS  # noqa: E402

HAZARD_KEY = {"pedestrian": "ped_crossed", "cut_in": "cutter_braked", "red_light": "runner_launched",
              "left_turn": "oncoming_launched"}
SCENARIO_NUMBER = {"pedestrian": 1, "cut_in": 2, "red_light": 3, "left_turn": 4}
POLICY_LABEL = {"ppo_from_coil": "PPO fine-tuned from imitation", "ppo_scratch": "PPO from scratch",
                "ppo_switcher": "PPO mode switcher", "expert": "rule-based expert", "coil": "imitation (CoIL)"}


def label_for(spec):
    kind, arg = spec.split(":", 1)
    if kind in ("expert", "coil"):
        return f"{arg} {POLICY_LABEL[kind]}"
    base = os.path.basename(arg).rsplit("_s", 1)[0]
    return POLICY_LABEL.get(base, base)


def font(size):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def run(env, policy, seed, camera=None):
    obs, _ = env.reset(seed=seed)
    frames, q = [], None
    if camera is not None:
        q = queue.Queue()
        cam = env.world.spawn_actor(camera["bp"], camera["tf"], attach_to=env.ego)
        env._actors.append(cam)  # destroyed with the scenario on the next reset / close
        cam.listen(q.put)
    done, step = False, 0
    while not done:
        obs, _, terminated, truncated, info = env.step([policy(obs, env)])
        done = terminated or truncated
        if q is not None:
            img = q.get(timeout=10.0)
            while not q.empty():
                img = q.get_nowait()
            if step % 2 == 0 or done:  # 10 frames per second
                frames.append((np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)[:, :, 2::-1].copy(),
                               env._speed(env.ego) * 3.6, getattr(policy, "mode", None), env.t))
        step += 1
    return info, frames


def annotate(frames, title, subtitle, outcome, width):
    big, small = font(max(14, width // 30)), font(max(12, width // 40))
    out = []
    for i, (rgb, kmh, mode, t) in enumerate(frames):
        im = Image.fromarray(rgb)
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
        d = ImageDraw.Draw(im, "RGBA")
        d.rectangle([0, 0, width, small.size * 2 + big.size + 14], fill=(16, 14, 12, 170))
        d.text((10, 6), title, font=big, fill=(255, 255, 255))
        d.text((10, 10 + big.size), subtitle, font=small, fill=(220, 220, 220))
        hud = f"{kmh:4.0f} km/h   t = {t:4.1f} s"
        if mode is not None:
            hud += f"   mode: {COMMAND_NAMES[mode]}"
        d.text((10, 14 + big.size + small.size), hud, font=small, fill=(255, 214, 102))
        if i == len(frames) - 1:
            w = d.textlength(outcome, font=big)
            y = im.height - 2.2 * big.size  # bottom edge, clear of the cars
            d.rectangle([(width - w) / 2 - 12, y - 6, (width + w) / 2 + 12, y + big.size + 8], fill=(16, 14, 12, 200))
            d.text(((width - w) / 2, y), outcome, font=big, fill=(120, 230, 140))
        out.append(im)
    return out


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scenario", choices=list(SCENARIOS), required=True)
    p.add_argument("--policy", required=True, help="e.g. ppo:checkpoints/ppo_from_coil_s0.pt")
    p.add_argument("--search", type=int, default=40, help="evaluation scenarios to search for a close call")
    p.add_argument("--seed", type=int, default=None, help="record this evaluation scenario (skips the search)")
    p.add_argument("--width", type=int, default=360)
    p.add_argument("--colors", type=int, default=48, help="GIF palette size (smaller file)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    sc = SCENARIOS[args.scenario]
    out = args.out or f"assets/demo_{args.scenario}.gif"
    policy = make_policy(args.policy, sc["coil"], sc["expert"])

    if args.seed is None:  # headless search
        env = sc["env"](town=sc["town"], no_rendering=True)
        best = None
        try:
            for i in range(args.search):
                info, _ = run(env, policy, EVAL_SEED_OFFSET + i)
                if info["outcome"] == "success" and info.get(HAZARD_KEY[args.scenario]):
                    if best is None or info["min_dist"] < best[1]:
                        best = (EVAL_SEED_OFFSET + i, info["min_dist"])
                    if args.scenario == "cut_in":
                        break
        finally:
            env.close()
        if best is None:
            raise SystemExit("No episode with the hazard and a success found; try a larger --search")
        args.seed = best[0]
        print(f"Close call: scenario seed {best[0]} (closest distance {best[1]:.1f} m)", flush=True)

    env = sc["env"](town=sc["town"], no_rendering=False)
    try:
        bp = env.blueprints.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(2 * args.width))
        bp.set_attribute("image_size_y", str(2 * args.width * 9 // 16))
        bp.set_attribute("fov", "90")
        camera = {"bp": bp, "tf": carla.Transform(carla.Location(x=-8.0, z=4.5), carla.Rotation(pitch=-16.0))}
        info, frames = run(env, policy, args.seed, camera)
    finally:
        env.close()
    print(f"Recorded {len(frames)} frames: {info['outcome']} in {info['time']:.1f} s", flush=True)

    outcome = {"success": "Reached the goal safely", "collision": "Collision",
               "timeout": "Ran out of time", "off_road": "Lost control"}[info["outcome"]]
    title = f"WHEW!  Scenario {SCENARIO_NUMBER[args.scenario]}: {sc['title']}"
    images = annotate(frames, title, label_for(args.policy), outcome, args.width)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pal = [im.convert("P", palette=Image.ADAPTIVE, colors=args.colors) for im in images]
    durations = [100] * (len(pal) - 1) + [2000]  # hold the last frame for 2 s
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
    print(f"Saved {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
