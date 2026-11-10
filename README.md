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

> Status: v0.3. Scenario 1 works end to end: expert data, conditional imitation learning,
> PPO fine-tuning and a PPO-from-scratch baseline, evaluated over 3 seeds. More scenarios are next.

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
out (80% of episodes) when the ego car is within 12-30 m **or** 1-2 s away, whichever comes
first, at a random walking speed. The time-based trigger means speeding cannot outrun the
pedestrian. The pedestrian can only be observed once they leave the blind spot.

* **Action**: one value in [-1, 1] (throttle if positive, brake if negative); a built-in
  controller keeps the car in its lane.
* **Observation** (7 values): ego speed, distance to goal, distance to the parked car,
  pedestrian visible flag, pedestrian longitudinal / lateral position and lateral speed.
* **Reward**: +1 per metre of progress, -4 per second of driving, -2 per second for each
  m/s above the 50 km/h speed limit, -150 for a collision, +20 for reaching the goal.
  Being fast pays, but never at the cost of a crash or a speeding violation.

## Results (scenario 1: pedestrian dart-out)

100 unseen scenarios per policy (the same scenario seeds for every policy, disjoint from the
training data). PPO: 3 training seeds per method, 200k steps each, mean ± std over seeds;
policies use their mean action.

| Policy | Success | Collision | Time to goal | Closest to pedestrian | Top speed | Return |
|---|---|---|---|---|---|---|
| Aggressive expert | 66% | 34% | 7.3 s | 2.9 m | 45 km/h | 2.0 |
| Timid expert | 98% | 1% | 14.1 s | 7.4 m | 34 km/h | 36.2 |
| CoIL, aggressive branch | 66% | 34% | 7.3 s | 2.9 m | 44 km/h | 2.0 |
| CoIL, timid branch | 98% | 1% | 14.0 s | 7.4 m | 35 km/h | 36.5 |
| **PPO from CoIL (timid)** | **97% ± 1%** | 3% ± 1% | **10.8 ± 0.1 s** | 5.1 ± 0.1 m | 45 ± 2 km/h | **46.6 ± 2.1** |
| PPO from scratch | 86% ± 9% | 14% ± 8% | 8.6 ± 2.7 s | 3.4 ± 0.7 m | 63 ± 16 km/h | 22.7 ± 16.1 |

<p align="center"><img src="results/learning_curves.png" alt="PPO learning curves over 3 seeds" width="100%"></p>

* **Imitation works**: each CoIL branch reproduces its expert almost exactly.
* **PPO from CoIL** keeps the timid driver's safety (97% success) while reaching the goal
  **23% faster** (14.1 s to 10.8 s), stays under the speed limit, and has the highest return.
  All three seeds end up in almost the same place.
* **PPO from scratch** is far less reliable: results vary strongly between seeds, it crashes
  more (14%), and two of three seeds learned to speed (64-82 km/h) despite the penalty.
  Starting from imitation is safe from the first training step.
* **Scenario fix (v0.3)**: in v0.2 one from-scratch run learned to drive at ~108 km/h so the
  pedestrian stepped out after the car had passed. The time-based trigger and the speeding
  penalty remove that loophole; all numbers above are from the fixed scenario.
* Caveats: one scenario, 200k training steps, 3 seeds. The collision difference between PPO
  from CoIL (3%) and the timid expert (1%) is small at 100 episodes.

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
for s in 0 1 2; do   # 3 seeds per method
  python scripts/train_ppo.py --init coil --command timid --seed $s --name ppo_from_coil_s$s
  python scripts/train_ppo.py --init scratch --seed $s --name ppo_scratch_s$s
done

# 4. Evaluate everything on the same unseen scenarios, then plot the learning curves
python scripts/evaluate.py --episodes 100 --policies expert:aggressive expert:timid coil:aggressive coil:timid     ppo:checkpoints/ppo_from_coil_s{0,1,2}.pt ppo:checkpoints/ppo_scratch_s{0,1,2}.pt
python scripts/plot_results.py   # results/learning_curves.png and results/summary.md

# Watch a policy drive (spectator camera follows the car)
python scripts/evaluate.py --policies ppo:checkpoints/ppo_from_coil_s0.pt --episodes 5 --render
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
- [x] Several seeds per method and more evaluation episodes
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
