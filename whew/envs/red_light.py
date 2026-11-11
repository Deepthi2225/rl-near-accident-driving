"""
Near-accident scenario 3: red-light runner at an intersection.

The ego car drives straight through a four-way intersection on a green light. In 80% of
episodes a car on the crossing road runs its red light and drives straight across the
ego car's path. Buildings hide the side street, so the runner can only be seen once it
is 15-25 m from the crossing point (varies per episode). It is timed to reach the
crossing point at about the same moment as the ego car (it adapts while hidden), so a
driver who ignores the intersection is often hit; slowing down for every intersection is
safe but slow.

Action: a in [-1, 1] (throttle if positive, brake if negative); lane keeping is automatic.
Actions take effect after a 0.3 s reaction / actuation delay (as in scenario 2).

Observation (7 floats):
    0 ego_speed          m/s
    1 dist_to_goal       m (along the road)
    2 dist_to_conflict   m from the ego car to the crossing point (negative once past it)
    3 runner_visible     1 if the crossing car can be seen, else 0
    4 runner_dist        m from the crossing car to the crossing point (0 if not visible)
    5 runner_speed       m/s (0 if not visible)
    6 runner_tta         s until the crossing car reaches the crossing point (10 if not visible)

Requires a running CARLA 0.9.15 server; uses Town05 (a grid city with many intersections).
"""
import math

import carla
import numpy as np

from .base import CarlaScenarioEnv

RUNNER_MODELS = ["vehicle.lincoln.mkz_2020", "vehicle.audi.tt", "vehicle.toyota.prius",
                 "vehicle.mercedes.coupe_2020", "vehicle.nissan.patrol", "vehicle.dodge.charger_2020"]
MAX_TTA = 10.0


def _yaw_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


class RedLightRunnerEnv(CarlaScenarioEnv):
    def __init__(self, runner_prob=0.8, action_delay=6, town="Town05", **kwargs):
        self.runner_prob = runner_prob
        super().__init__(action_delay=action_delay, town=town, **kwargs)
        self.runner_bps = [self.blueprints.find(m) for m in RUNNER_MODELS if self.blueprints.filter(m)]

    # ------------------------------------------------------------------ routes
    def _find_routes(self, step=2.0):
        """Straight routes in any driving lane that cross exactly one four-way intersection,
        entering it 30-44 m after the start, with a straight crossing path through it."""
        n = int(self.route_length / step)
        routes, seen = [], set()
        for wp in self.map.generate_waypoints(step):
            if wp.is_junction or wp.lane_type != carla.LaneType.Driving:
                continue
            yaw0 = wp.transform.rotation.yaw
            route, j_in, j_out, ok = [wp], None, None, True
            for i in range(1, n + 1):
                nxt = route[-1].next(step)
                if not nxt:
                    ok = False
                    break
                best = min(nxt, key=lambda w: _yaw_diff(w.transform.rotation.yaw, yaw0))
                if _yaw_diff(best.transform.rotation.yaw, yaw0) > 4.0:
                    ok = False
                    break
                route.append(best)
                if best.is_junction and j_in is None:
                    j_in = i
                elif j_in is not None and j_out is None and not best.is_junction:
                    j_out = i
                elif j_out is not None and best.is_junction:
                    ok = False  # a second junction
                    break
            if not ok or j_in is None or j_out is None or not 30 <= step * j_in <= 44:
                continue
            if step * j_out > self.route_length - 12.0:
                continue
            junction = route[j_in].get_junction()
            key = (wp.road_id, wp.lane_id, junction.id)
            if key in seen:
                continue
            crossings = []
            for a, b in junction.get_waypoints(carla.LaneType.Driving):
                ya = a.transform.rotation.yaw
                if 70 < _yaw_diff(ya, yaw0) < 110 and _yaw_diff(ya, b.transform.rotation.yaw) < 10:
                    back = a.previous(30.0)
                    if back and not back[0].is_junction and _yaw_diff(back[0].transform.rotation.yaw, ya) < 10:
                        crossings.append(a)
            if not crossings:
                continue
            seen.add(key)
            routes.append((route, crossings, junction.id))
        return routes

    # ------------------------------------------------------------------ episode
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()
        rng = self.np_random

        for _ in range(30):
            route, crossings, junction_id = self.routes[rng.integers(len(self.routes))]
            self._set_route(route)
            entry = crossings[rng.integers(len(crossings))]
            ego = self.world.try_spawn_actor(self.ego_bp, carla.Transform(
                self._lane_point(0.0, z=0.4), carla.Rotation(yaw=self.yaw)))
            if ego is None:
                continue
            self._actors.append(ego)

            # crossing line: runner position r(u) = entry + u * dir; it meets our lane centre at u_c
            self.r_origin = entry.transform.location
            self.r_dir = entry.transform.get_forward_vector()
            s_a, l_a = self._frame(self.r_origin)
            ds = self.r_dir.x * self.fwd.x + self.r_dir.y * self.fwd.y
            dl = self.r_dir.x * self.right.x + self.r_dir.y * self.right.y
            u_c = -l_a / dl
            self.s_conflict = s_a + ds * u_c
            self.u_conflict = u_c
            start_back = float(rng.uniform(30.0, 40.0))
            back = entry.previous(start_back)
            if not back:
                self._destroy()
                continue
            tr = back[0].transform
            bp = self.runner_bps[rng.integers(len(self.runner_bps))]
            runner = self.world.try_spawn_actor(bp, carla.Transform(
                tr.location + carla.Location(z=0.4), tr.rotation))
            if runner is None:
                self._destroy()
                continue
            self._actors.append(runner)
            break
        else:
            raise RuntimeError("Could not spawn the scenario actors (blocked spawn points?)")

        self.ego, self.runner = ego, runner
        self._attach_collision_sensor()
        self._set_lights(junction_id)

        # scenario randomisation
        self.will_run = bool(rng.random() < self.runner_prob)
        self.runner_speed = float(rng.uniform(9.0, 14.0))      # initial speed; adapts while hidden
        self.timing = float(rng.uniform(-0.6, 0.4))            # s; runner arrives this much after the ego would
        self.vis_range = float(rng.uniform(15.0, 25.0))        # m from the crossing point where it becomes visible
        v0 = float(rng.uniform(8.0, 11.0))
        self.launched = self.will_run  # the runner is already moving when the episode starts
        self.committed = False

        self._settle([ego, runner])
        self._launch(ego, v0)
        if self.will_run:
            self.runner.set_target_velocity(carla.Vector3D(
                self.r_dir.x * self.runner_speed, self.r_dir.y * self.runner_speed, 0.0))
        self.world.tick()
        self._start_episode()
        return self._obs(), {}

    def _set_lights(self, junction_id):
        """Cosmetic: green for the ego car's road, red for the crossing road."""
        try:
            lights = self.world.get_traffic_lights_in_junction(junction_id)
            for tl in lights:
                wps = tl.get_affected_lane_waypoints()
                ours = any(_yaw_diff(w.transform.rotation.yaw, self.yaw) < 30 for w in wps)
                tl.set_state(carla.TrafficLightState.Green if ours else carla.TrafficLightState.Red)
                tl.freeze(True)
        except RuntimeError:
            pass

    # ------------------------------------------------------------------ runner
    def _runner_u(self):
        d = self.runner.get_location() - self.r_origin
        return d.x * self.r_dir.x + d.y * self.r_dir.y

    def _before_tick(self, s_ego):
        if not self.will_run:
            self.runner.apply_control(carla.VehicleControl(brake=1.0))
            return
        r_dist = self.u_conflict - self._runner_u()
        if not self.committed:
            # While hidden, match the ego car's arrival time (plus the episode's timing offset),
            # so neither speeding up nor slowing down far from the junction dodges the runner.
            tta_ego = (self.s_conflict - s_ego) / max(self._speed(self.ego), 0.5)
            self.runner_speed = float(np.clip(r_dist / max(tta_ego + self.timing, 0.5), 7.0, 15.0))
            if r_dist <= self.vis_range:
                self.committed = True  # visible from here on: constant speed, never brakes
        # drive straight along the crossing line at constant speed (it never brakes)
        tr = self.runner.get_transform()
        target = self.r_origin + self.r_dir * (self._runner_u() + 8.0)
        v = target - tr.location
        f = tr.get_forward_vector()
        angle = math.atan2(f.x * v.y - f.y * v.x, f.x * v.x + f.y * v.y)
        u = 0.5 * (self.runner_speed - self._speed(self.runner))
        self.runner.apply_control(carla.VehicleControl(
            throttle=float(np.clip(u, 0.0, 1.0)), brake=float(np.clip(-u, 0.0, 1.0)),
            steer=float(np.clip(1.5 * angle, -0.5, 0.5))))

    def _after_tick(self, ego_loc):
        self.min_dist = min(self.min_dist, ego_loc.distance(self.runner.get_location()))

    def _extra_info(self):
        return {"red_light_runner": self.will_run, "runner_launched": self.launched}

    def _obs(self):
        speed = self._speed(self.ego)
        s_ego, _ = self._frame(self.ego.get_location())
        r_dist = self.u_conflict - self._runner_u()
        r_speed = self._speed(self.runner)
        visible = self.launched and r_dist <= self.vis_range
        if visible:
            tta = float(np.clip(r_dist / max(r_speed, 0.5), -MAX_TTA, MAX_TTA))
            obs_r = (1.0, r_dist, r_speed, tta)
        else:
            obs_r = (0.0, 0.0, 0.0, MAX_TTA)
        return np.array([speed, self.s_goal - s_ego, self.s_conflict - s_ego, *obs_r], dtype=np.float32)
