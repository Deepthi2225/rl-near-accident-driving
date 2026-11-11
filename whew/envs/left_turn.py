"""
Near-accident scenario 4: unprotected left turn across a fast oncoming car.

The ego car approaches a four-way intersection and turns left on a green light with no
turn arrow. In 85% of episodes a fast oncoming car comes straight through the
intersection. It appears about 32-44 m before the junction, timed around the ego car's
arrival, so the driver has to judge the gap: go first, or wait and let it pass.

Action: a in [-1, 1] (throttle if positive, brake if negative); steering along the turn is
automatic. Actions take effect after a 0.3 s reaction / actuation delay (as in scenario 2).
Cornering: sideways acceleration above 4 m/s^2 is penalised and above 11 m/s^2 counts as
losing control (CARLA's tyres would otherwise allow a 90-degree turn at 50 km/h).

Observation (7 floats):
    0 ego_speed          m/s
    1 dist_to_goal       m (along the route)
    2 dist_to_conflict   m along the route to the point where the paths cross (negative once past)
    3 oncoming_visible   1 if the oncoming car can be seen, else 0
    4 oncoming_dist      m from the oncoming car to the crossing point (0 if not visible)
    5 oncoming_speed     m/s (0 if not visible)
    6 oncoming_tta       s until the oncoming car reaches the crossing point (10 if not visible)

Requires a running CARLA 0.9.15 server; uses Town05.
"""
import math

import carla
import numpy as np

from .base import CarlaScenarioEnv
from .red_light import MAX_TTA, RUNNER_MODELS, _yaw_diff

APPROACH_STEPS = 20  # 40 m of straight road before the junction
EXIT_STEPS = 10      # 20 m of straight road after it


class UnprotectedLeftEnv(CarlaScenarioEnv):
    ego_lookahead = 5.0  # shorter than on straight roads, so the car follows the curve

    def __init__(self, oncoming_prob=0.85, action_delay=6, town="Town05", max_time=25.0,
                 lat_acc_limit=4.0, lat_acc_penalty=3.0, lat_acc_max=11.0, **kwargs):
        self.oncoming_prob = oncoming_prob
        super().__init__(action_delay=action_delay, town=town, max_time=max_time, lat_acc_limit=lat_acc_limit,
                         lat_acc_penalty=lat_acc_penalty, lat_acc_max=lat_acc_max, **kwargs)
        self.car_bps = [self.blueprints.find(m) for m in RUNNER_MODELS if self.blueprints.filter(m)]

    # ------------------------------------------------------------------ routes
    @staticmethod
    def _conflict_on(points, origin, direction):
        """First crossing of the polyline with the line origin + u * direction: (s, u), or None."""
        ox, oy, dx, dy = origin.x, origin.y, direction.x, direction.y
        side = (points[:, 0] - ox) * dy - (points[:, 1] - oy) * dx
        for i in np.nonzero(side[:-1] * side[1:] < 0)[0]:
            t = side[i] / (side[i] - side[i + 1])
            p = points[i] + t * (points[i + 1] - points[i])
            u = (p[0] - ox) * dx + (p[1] - oy) * dy
            s = float(np.sum(np.linalg.norm(np.diff(points[:i + 1], axis=0), axis=1))
                      + t * np.linalg.norm(points[i + 1] - points[i]))
            if u > -2.0:
                return s, u
        return None

    def _find_routes(self, step=2.0):
        """Left-turn routes: 40 m straight approach, the turn, then 20 m of straight road."""
        routes, seen = [], set()
        for wp in self.map.generate_waypoints(4.0):
            if not wp.is_junction or wp.get_junction().id in seen:
                continue
            junction = wp.get_junction()
            seen.add(junction.id)
            pairs = junction.get_waypoints(carla.LaneType.Driving)
            for a, b in pairs:
                ya, yb = a.transform.rotation.yaw, b.transform.rotation.yaw
                if not -110 < (yb - ya + 180) % 360 - 180 < -70:
                    continue
                route = self._build_route(a, b, step)
                if route is None:
                    continue
                pts = np.array([[w.transform.location.x, w.transform.location.y] for w in route])
                oncoming = []
                for a2, b2 in pairs:
                    y2 = a2.transform.rotation.yaw
                    if _yaw_diff(y2, b2.transform.rotation.yaw) > 10 or _yaw_diff(y2, ya + 180.0) > 12:
                        continue
                    back = a2.previous(44.0)
                    if not back or back[0].is_junction or _yaw_diff(back[0].transform.rotation.yaw, y2) > 10:
                        continue
                    hit = self._conflict_on(pts, a2.transform.location, a2.transform.get_forward_vector())
                    if hit is not None:
                        oncoming.append((a2, hit[0], hit[1]))
                if oncoming:
                    routes.append((route, oncoming, junction.id))
        return routes

    @staticmethod
    def _build_route(a, b, step):
        ya, yb = a.transform.rotation.yaw, b.transform.rotation.yaw
        approach, cur = [], a
        for _ in range(APPROACH_STEPS):
            prev = cur.previous(step)
            if not prev or prev[0].is_junction or _yaw_diff(prev[0].transform.rotation.yaw, ya) > 4.0:
                return None
            cur = prev[0]
            approach.append(cur)
        turn, cur = [a], a
        for _ in range(40):
            nxt = cur.next(step)
            if not nxt:
                return None
            cur = min(nxt, key=lambda w: w.transform.location.distance(b.transform.location))
            turn.append(cur)
            if not cur.is_junction:
                break
        else:
            return None
        if cur.transform.location.distance(b.transform.location) > 6.0:
            return None
        tail = []
        for _ in range(EXIT_STEPS):
            nxt = cur.next(step)
            if not nxt or nxt[0].is_junction or _yaw_diff(nxt[0].transform.rotation.yaw, yb) > 6.0:
                return None
            cur = nxt[0]
            tail.append(cur)
        return approach[::-1] + turn + tail

    def _set_route(self, route):
        super()._set_route(route)
        self.points = np.array([[w.transform.location.x, w.transform.location.y] for w in route])
        self.s_goal = 2.0 * (len(route) - 1) - 5.0

    def _frame(self, loc):
        """(s, l) along the curved route: arc length and lateral offset (+ right)."""
        p = np.array([loc.x, loc.y])
        a, seg = self.points[:-1], np.diff(self.points, axis=0)
        lengths2 = np.maximum(np.sum(seg ** 2, axis=1), 1e-6)
        t = np.clip(np.sum((p - a) * seg, axis=1) / lengths2, 0.0, 1.0)
        d = p - (a + seg * t[:, None])
        i = int(np.argmin(np.sum(d ** 2, axis=1)))
        length = math.sqrt(lengths2[i])
        lateral = (-d[i, 0] * seg[i, 1] + d[i, 1] * seg[i, 0]) / length
        return 2.0 * i + float(t[i]) * length, float(lateral)

    # ------------------------------------------------------------------ episode
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()
        rng = self.np_random

        for _ in range(30):
            route, oncoming, junction_id = self.routes[rng.integers(len(self.routes))]
            self._set_route(route)
            entry, self.s_conflict, self.u_conflict = oncoming[rng.integers(len(oncoming))]
            ego = self.world.try_spawn_actor(self.ego_bp, carla.Transform(
                self._lane_point(0.0, z=0.4), carla.Rotation(yaw=self.yaw)))
            if ego is None:
                continue
            self._actors.append(ego)
            self.r_origin = entry.transform.location
            self.r_dir = entry.transform.get_forward_vector()
            tr = entry.previous(float(rng.uniform(32.0, 44.0)))[0].transform
            bp = self.car_bps[rng.integers(len(self.car_bps))]
            car = self.world.try_spawn_actor(bp, carla.Transform(tr.location + carla.Location(z=0.4), tr.rotation))
            if car is None:
                self._destroy()
                continue
            self._actors.append(car)
            break
        else:
            raise RuntimeError("Could not spawn the scenario actors (blocked spawn points?)")

        self.ego, self.car = ego, car
        self._attach_collision_sensor()
        self._set_lights(junction_id)

        self.will_come = bool(rng.random() < self.oncoming_prob)
        self.car_speed = float(rng.uniform(11.0, 16.0))     # 40-58 km/h
        self.timing = float(rng.uniform(-3.0, 1.2))         # s it arrives before the ego would (negative: after)
        v0 = float(rng.uniform(8.0, 11.0))
        self.launched = False

        self._settle([ego, car])
        self._launch(ego, v0)
        self.world.tick()
        self._start_episode()
        return self._obs(), {}

    def _set_lights(self, junction_id):
        """Cosmetic: green for both directions of the ego car's road (an unprotected turn)."""
        try:
            for tl in self.world.get_traffic_lights_in_junction(junction_id):
                aligned = any(_yaw_diff(w.transform.rotation.yaw, self.yaw) < 30
                              or _yaw_diff(w.transform.rotation.yaw, self.yaw + 180.0) < 30
                              for w in tl.get_affected_lane_waypoints())
                tl.set_state(carla.TrafficLightState.Green if aligned else carla.TrafficLightState.Red)
                tl.freeze(True)
        except RuntimeError:
            pass

    # ------------------------------------------------------------------ oncoming car
    def _car_u(self):
        d = self.car.get_location() - self.r_origin
        return d.x * self.r_dir.x + d.y * self.r_dir.y

    def _before_tick(self, s_ego):
        if not self.will_come:
            self.car.apply_control(carla.VehicleControl(brake=1.0))
            return
        if not self.launched:
            dist_ego = self.s_conflict - s_ego
            tta_ego = dist_ego / max(self._speed(self.ego), 0.5)
            dist_car = self.u_conflict - self._car_u()
            if tta_ego > dist_car / self.car_speed + self.timing and dist_ego > 14.0:
                self.car.apply_control(carla.VehicleControl(brake=1.0))
                return
            self.launched = True
            v = self.car_speed
            self.car.set_target_velocity(carla.Vector3D(self.r_dir.x * v, self.r_dir.y * v, 0.0))
        tr = self.car.get_transform()
        v = self.r_origin + self.r_dir * (self._car_u() + 8.0) - tr.location
        f = tr.get_forward_vector()
        angle = math.atan2(f.x * v.y - f.y * v.x, f.x * v.x + f.y * v.y)
        u = 0.5 * (self.car_speed - self._speed(self.car))
        self.car.apply_control(carla.VehicleControl(
            throttle=float(np.clip(u, 0.0, 1.0)), brake=float(np.clip(-u, 0.0, 1.0)),
            steer=float(np.clip(1.5 * angle, -0.5, 0.5))))

    def _after_tick(self, ego_loc):
        self.min_dist = min(self.min_dist, ego_loc.distance(self.car.get_location()))

    def _extra_info(self):
        return {"oncoming": self.will_come, "oncoming_launched": self.launched}

    def _obs(self):
        speed = self._speed(self.ego)
        s_ego, _ = self._frame(self.ego.get_location())
        if self.launched:
            r_dist = self.u_conflict - self._car_u()
            r_speed = self._speed(self.car)
            tta = float(np.clip(r_dist / max(r_speed, 0.5), -MAX_TTA, MAX_TTA))
            car_obs = (1.0, r_dist, r_speed, tta)
        else:
            car_obs = (0.0, 0.0, 0.0, MAX_TTA)
        return np.array([speed, self.s_goal - s_ego, self.s_conflict - s_ego, *car_obs], dtype=np.float32)
