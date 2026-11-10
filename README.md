<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/whew-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/whew-logo-light.svg">
    <img src="assets/whew-logo-light.svg" alt="WHEW! - reinforcement learning for near-accident driving" width="600">
  </picture>
</p>

<p align="center"><i>Bold when it can be. Careful when it must be.</i></p>

---

**WHEW!** is a reinforcement learning project for near-accident driving in the CARLA simulator.
It learns how to drive through the seconds before a potential crash, where a
purely cautious driver is safe but slow and a purely aggressive driver is fast but
crashes. The goal is a policy that is bold when it can be and careful when it must be.

> Status: v0.2. Scenario 1 works end to end: expert data, conditional imitation learning,
> PPO fine-tuning and a PPO-from-scratch baseline. More scenarios are next.

## Approach

1. **Near-accident scenarios** built on the CARLA simulator.
2. **Expert demonstrations** from two rule-based drivers: *aggressive* and *timid*.
3. **Conditional imitation learning (CoIL)**: one network with a branch per driving style.
4. **PPO fine-tuning**, starting from the imitation policy, to learn when to be bold and
   when to be careful, and to beat both experts.

## Scenarios

| # | Scenario | Status |
|---|----------|--------|
| 1 | **Pedestrian dart-out**: a pedestrian steps into the road from behind a parked car | Working |
| 2 | Sudden cut-in from the adjacent lane | Planned |
| 3 | Red-light runner at an intersection | Planned |
| 4 | Unprotected left turn with a fast oncoming car | Planned |

### Scenario 1: pedestrian dart-out
The ego car drives down a straight road. A parked car hides a pedestrian, who steps
out (80% of episodes) when the ego car gets close, at a random distance and walking speed.
The pedestrian can only be observed once they leave the blind spot.

* **Action**: one value in [-1, 1] (throttle if positive, brake if negative); a built-in
  controller keeps the car in its lane.
* **Observation** (7 values): ego speed, distance to goal, distance to the parked car,
  pedestrian visible flag, pedestrian longitudinal / lateral position and lateral speed.
* **Reward**: +1 per metre of progress, -4 per second of driving, -150 for a collision,
  +20 for reaching the goal. Being fast pays, but never at the cost of a crash.

## Results (scenario 1: pedestrian dart-out)

50 unseen scenarios per policy (same scenario seeds for every policy, disjoint from the
training data). PPO policies use their mean action.

| Policy | Success | Collision | Time to goal | Closest to pedestrian | Return |
|---|---|---|---|---|---|
| Aggressive expert | 72% | 28% | 7.5 s | 3.2 m | 12.9 |
| Timid expert | 96% | 2% | 14.0 s | 8.0 m | 34.2 |
| CoIL, aggressive branch | 72% | 28% | 7.4 s | 3.1 m | 13.1 |
| CoIL, timid branch | 96% | 2% | 13.9 s | 8.0 m | 34.5 |
| **PPO from CoIL (timid)** | **96%** | 4% | **11.1 s** | 5.8 m | **42.8** |
| PPO from scratch | 72% | 2% | 11.0 s | 7.2 m | 22.7 |

<p align="center"><img src="results/learning_curves.png" alt="PPO learning curves" width="100%"></p>

* **Imitation works**: each CoIL branch reproduces its expert almost exactly.
* **PPO from CoIL** keeps the timid driver's success rate while reaching the goal about
  **20% faster** (14.0 s to 11.1 s), and has the highest return of all policies.
* **PPO from scratch** learns to avoid the pedestrian but often stops and times out
  (26% of episodes); starting from imitation avoids this and is safe from the first step.
* Caveats: 200k training steps and one seed per method; 4% vs 2% collisions is 2 vs 1
  episode out of 50. More seeds and episodes are needed before strong claims.

---

## Setup

Requirements: Windows or Linux, Python 3.10, an NVIDIA GPU (tested on an RTX 4060 Laptop, 8 GB).

1. Download **CARLA 0.9.15** from the [CARLA releases](https://github.com/carla-simulator/carla/releases)
   and extract it **outside** this repository.
2. Create a virtual environment and install the dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
   The `carla` client comes from PyPI (`carla==0.9.15`); the wheel bundled with CARLA
   only supports Python 3.7.
3. Start the simulator (low quality is enough and saves GPU memory):

   ```bash
   CarlaUE4.exe -quality-level=Low
   ```

## Usage

```bash
# Watch the two experts drive (spectator camera follows the car)
python scripts/collect_expert.py --episodes 3 --render --no-save

# 1. Collect demonstrations: 100 episodes per expert -> data/pedestrian_crossing_expert.npz
python scripts/collect_expert.py --episodes 100

# 2. Train conditional imitation learning -> checkpoints/coil.pt
python scripts/train_coil.py

# 3. PPO fine-tuning from the CoIL timid branch, and the from-scratch baseline
python scripts/train_ppo.py --init coil --command timid --name ppo_from_coil
python scripts/train_ppo.py --init scratch --name ppo_scratch

# 4. Evaluate everything on the same unseen scenarios, then plot the learning curves
python scripts/evaluate.py --policies expert:aggressive expert:timid coil:aggressive coil:timid     ppo:checkpoints/ppo_from_coil.pt ppo:checkpoints/ppo_scratch.pt
python scripts/plot_results.py

# Watch a policy drive (spectator camera follows the car)
python scripts/evaluate.py --policies ppo:checkpoints/ppo_from_coil.pt --episodes 5 --render
```

Each PPO run takes about 5-10 minutes (200k steps) on an RTX 4060 laptop with CARLA headless.

## Project structure

```
whew/
  envs/pedestrian_crossing.py   # Scenario 1 as a gymnasium environment
  experts.py                    # aggressive and timid rule-based drivers
  coil.py                       # conditional imitation learning network (one branch per style)
  ppo.py                        # PPO (GAE, Gaussian policy) with a CoIL-shaped actor
scripts/
  collect_expert.py             # expert data collection + expert evaluation
  train_coil.py                 # imitation learning
  train_ppo.py                  # PPO fine-tuning / from-scratch baseline
  evaluate.py                   # compare policies on the same unseen scenarios
  plot_results.py               # learning curves
results/                        # evaluation JSON, learning-curve CSVs and plot
```

## Roadmap

- [x] Scenario 1 environment and expert drivers
- [x] Conditional imitation learning on the expert data
- [x] PPO fine-tuning from the imitation policy, plus a PPO-from-scratch baseline
- [x] Evaluation: success, collisions, time to goal, closest distance; learning curves
- [ ] Several seeds per method and more evaluation episodes
- [ ] High-level PPO that switches between the aggressive and timid CoIL branches (as in H-ReIL)
- [ ] Scenarios 2-4, then full steering control and camera input

## Background and credits

* WHEW! extends an earlier course project on conditional imitation learning + PPO in the
  2D CARLO simulator, done jointly with Samrudhi Rao.
* The aggressive / timid formulation follows Cao et al., *Reinforcement Learning based
  Control of Imitative Policies for Near-Accident Driving*, RSS 2020.
* Conditional imitation learning: Codevilla et al., *End-to-end Driving via Conditional
  Imitation Learning*, ICRA 2018.
* Simulator: [CARLA](https://carla.org/) (Dosovitskiy et al., 2017).

## Author

Deepthi M (PES1UG23AM092)
