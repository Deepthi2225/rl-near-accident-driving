"""
Near-accident scenario 2: sudden cut-in from the adjacent lane.

The ego car drives in the right lane of a two-lane (same direction) road. Another
car (the "cutter") drives in the left lane and keeps pace 7-13 m ahead of the ego car
(up to 58 km/h). At a random moment 1.5-3.5 s into the episode it swerves into the ego
lane (85% of episodes) once there are at least 2 m of clear road behind it; after merging it brakes hard for 1-2 s in most of those episodes
(a "brake check") and then drives on. Speeding up or slowing down cannot avoid the
cut-in; only keeping a safe distance can. Braking early is safe but slow, ignoring the
cutter is fast but risky.

Action: a in [-1, 1] (throttle if positive, brake if negative); lane keeping is automatic.
Actions take effect after a 0.3 s reaction / actuation delay, like a real driver and brakes.

Observation (7 floats):
    0 ego_speed       m/s
    1 dist_to_goal    m (along the road)
    2 gap             m bumper-to-bumper gap to the cutter along the road (+ = cutter ahead)
    3 cutter_lat      m lateral offset of the cutter from the ego lane centre (- = left)
    4 rel_speed       m/s cutter longitudinal speed minus ego speed
    5 cutter_vlat     m/s lateral cutter velocity (+ = towards the right, i.e. into the ego lane)
    6 in_lane         1 if the cutter overlaps the ego lane, else 0

Requires a running CARLA 0.9.15 server.
"""
import carla
import numpy as np

from .base import CarlaScenarioEnv

CAR_LENGTH = 4.7  # approximate bumper-to-bumper correction between two car centres
CUTTER_MODELS = ["vehicle.lincoln.mkz_2020", "vehicle.audi.tt", "vehicle.toyota.prius",
                 "vehicle.mercedes.coupe_2020", "vehicle.nissan.patrol"]


class CutInEnv(CarlaScenarioEnv):
    need_left_lane = True

    def __init__(self, cut_prob=0.85, brake_prob=0.7, action_delay=6, **kwargs):
        self.cut_prob, self.brake_prob = cut_prob, brake_prob
        super().__init__(action_delay=action_delay, **kwargs)
        self.cutter_bps = [self.blueprints.find(m) for m in CUTTER_MODELS
                           if self.blueprints.filter(m)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()
        rng = self.np_random

        for _ in range(20):
            self._set_route(self.routes[rng.integers(len(self.routes))])
            rot = carla.Rotation(yaw=self.yaw)
            s_cutter = float(rng.uniform(2.0, 8.0))
            ego = self.world.try_spawn_actor(self.ego_bp, carla.Transform(self._lane_point(0.0, z=0.4), rot))
            if ego is None:
                continue
            self._actors.append(ego)
            bp = self.cutter_bps[rng.integers(len(self.cutter_bps))]
            cutter = self.world.try_spawn_actor(bp, carla.Transform(
                self._lane_point(s_cutter, -self.lane_width, z=0.4), rot))
            if cutter is None:
                self._destroy()
                continue
            self._actors.append(cutter)
            break
        else:
            raise RuntimeError("Could not spawn the scenario actors (blocked spawn points?)")

        self.ego, self.cutter = ego, cutter
        self._attach_collision_sensor()

        # scenario randomisation
        self.will_cut = bool(rng.random() < self.cut_prob)
        self.will_brake = bool(rng.random() < self.brake_prob)
        self.trigger_gap = float(rng.uniform(7.0, 13.0))      # centre-to-centre gap the cutter keeps
        self.cut_time = float(rng.uniform(1.5, 3.5))          # s into the episode when it swerves in
        self.brake_strength = float(rng.uniform(0.7, 1.0))
        self.brake_to = float(rng.uniform(0.0, 3.0))          # m/s the cutter slows down to
        self.brake_duration = float(rng.uniform(1.0, 2.0))    # s of hard braking before driving on
        v0 = float(rng.uniform(8.0, 11.0))
        self.cutter_speed = min(v0 + float(rng.uniform(2.0, 4.0)), 14.0)
        self.triggered = self.merged = self.braking = False
        self.brake_start = None

        self._settle([ego, cutter])
        self._launch(ego, v0)
        self._launch(cutter, self.cutter_speed)
        self.world.tick()
        self._start_episode()
        return self._obs(), {}

    def _before_tick(self, s_ego):
        s_c, l_c = self._frame(self.cutter.get_location())
        if (self.will_cut and not self.triggered and self.t >= self.cut_time
                and s_c - s_ego - CAR_LENGTH >= 2.0):  # only cut in with 2 m of room
            self.triggered = True
            self.cutter_speed = min(max(self._speed(self.cutter), 6.0), 16.0)  # keep its speed after merging
        target_lat = 0.0 if self.triggered else -self.lane_width
        if self.triggered and abs(l_c) < 0.6:
            self.merged = True
        if self.merged and self.will_brake and self.brake_start is None:
            self.braking, self.brake_start = True, self.t
        if self.braking and self.t - self.brake_start > self.brake_duration:
            self.braking = False  # brake check over: speed up again

        v = self._speed(self.cutter)
        if self.braking and v > self.brake_to:
            throttle, brake = 0.0, self.brake_strength
        else:
            if self.braking:
                target_v = self.brake_to
            elif self.will_cut and not self.triggered:  # pace the ego car, trigger_gap ahead
                target_v = float(np.clip(self._speed(self.ego) + 0.8 * (self.trigger_gap - (s_c - s_ego)), 0.0, 16.0))
            else:
                target_v = self.cutter_speed
            u = 0.4 * (target_v - v)
            throttle, brake = float(np.clip(u, 0.0, 1.0)), float(np.clip(-u, 0.0, 1.0))
        steer = self._steer_actor(self.cutter, s_c, target_lat, lookahead=8.0)
        self.cutter.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))

    def _after_tick(self, ego_loc):
        self.min_dist = min(self.min_dist, ego_loc.distance(self.cutter.get_location()))

    def _extra_info(self):
        return {"cut_in": self.will_cut, "cutter_braked": self.will_cut and self.will_brake}

    def _obs(self):
        speed = self._speed(self.ego)
        s_ego, _ = self._frame(self.ego.get_location())
        s_c, l_c = self._frame(self.cutter.get_location())
        v_long, v_lat = self._long_lat_velocity(self.cutter)
        ego_long, _ = self._long_lat_velocity(self.ego)
        in_lane = abs(l_c) < self.lane_width / 2 + 0.9
        return np.array([speed, self.s_goal - s_ego, s_c - s_ego - CAR_LENGTH, l_c, v_long - ego_long,
                         v_lat, float(in_lane)], dtype=np.float32)
