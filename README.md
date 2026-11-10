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

> Status: early development (v0.1). The first scenario and the expert drivers are working;
> imitation learning and PPO training are next.

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
* **Reward**: metres of progress, -150 for a collision, +20 for reaching the goal.

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

# Collect demonstrations: 100 episodes per expert -> data/pedestrian_crossing_expert.npz
python scripts/collect_expert.py --episodes 100
```

The script prints each expert's success rate, collision rate, time to goal and closest
distance to the pedestrian.

## Project structure

```
whew/
  envs/pedestrian_crossing.py   # Scenario 1 as a gymnasium environment
  experts.py                    # aggressive and timid rule-based drivers
scripts/
  collect_expert.py             # expert data collection + expert evaluation
```

## Roadmap

- [x] Scenario 1 environment and expert drivers
- [ ] Conditional imitation learning on the expert data
- [ ] PPO fine-tuning from the imitation policy, plus a PPO-from-scratch baseline
- [ ] Evaluation: success, collisions, time to goal, closest distance; learning curves over several seeds
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
