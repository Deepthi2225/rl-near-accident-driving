"""
Near-accident scenario 1: pedestrian dart-out from behind a parked car.

The ego car drives along a straight road. A parked car on the right hides a
pedestrian, who may step into the road when the ego car gets close. The
pedestrian is only observable once they step out from behind the parked car,
so the agent has to decide how fast to pass the blind spot.

Action  (1-D, like the CARLO project): a in [-1, 1]
        a > 0 -> throttle = a,  a < 0 -> brake = -a.
        Steering is done by a built-in lane-keeping controller.

Observation (7 floats):
    0 ego_speed       m/s
    1 dist_to_goal    m (along the road)
    2 occluder_long   m from ego to the parked car (along the road, + = ahead)
    3 ped_visible     1 if the pedestrian is visible, else 0
    4 ped_long        m from ego to pedestrian along the road (60 if not visible)
    5 ped_lat         m from the ego lane centre (+ = right; 10 if not visible)
    6 ped_vlat        m/s lateral pedestrian velocity (+ = to the right; 0 if not visible)

Requires a running CARLA 0.9.15 server.
"""
import math

import carla
import gymnasium as gym
import numpy as np
from gymnasium import spaces

HIDDEN_LONG, HIDDEN_LAT = 60.0, 10.0


class PedestrianCrossingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, host="localhost", port=2000, town=None, dt=0.05, max_time=20.0,
                 route_length=80.0, cross_prob=0.8, render_mode=None, no_rendering=False,
                 timeout=30.0, time_penalty=4.0):
        super().__init__()
        self.dt, self.max_time, self.route_length = dt, max_time, route_length
        self.cross_prob, self.render_mode = cross_prob, render_mode
        self.time_penalty = time_penalty  # reward lost per second of driving

        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        if town and not self.client.get_world().get_map().name.endswith(town):
            self.client.load_world(town)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        self._orig_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = dt
        settings.no_rendering_mode = no_rendering
        self.world.apply_settings(settings)

        bps = self.world.get_blueprint_library()
        self.ego_bp = bps.find("vehicle.tesla.model3")
        self.ego_bp.set_attribute("role_name", "hero")
        self.parked_bp = bps.find("vehicle.audi.a2")
        self.walker_bps = list(bps.filter("walker.pedestrian.*"))
        self.collision_bp = bps.find("sensor.other.collision")

        self.routes = self._find_straight_routes()
        if not self.routes:
            raise RuntimeError(f"No straight {route_length:.0f} m right-most lane found in {self.map.name}; "
                               "try town='Town04' or a shorter route_length.")

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._actors = []

    # ------------------------------------------------------------------ setup
    def _find_straight_routes(self, step=2.0):
        n = int(self.route_length / step)
        routes, seen = [], set()
        for sp in self.map.get_spawn_points():
            wp = self.map.get_waypoint(sp.location)
            key = (wp.road_id, wp.lane_id)
            if key in seen or wp.is_junction:
                continue
            right = wp.get_right_lane()
            if right is not None and right.lane_type == carla.LaneType.Driving:
                continue  # need the right-most driving lane (pedestrian comes from the kerb)
            route, ok = [wp], True
            for _ in range(n):
                nxt = route[-1].next(step)
                if not nxt or nxt[0].is_junction:
                    ok = False
                    break
                route.append(nxt[0])
            if not ok:
                continue
            yaw0 = wp.transform.rotation.yaw
            if any(abs((w.transform.rotation.yaw - yaw0 + 180) % 360 - 180) > 3 for w in route):
                continue
            seen.add(key)
            routes.append(route)
        return routes

    def _frame(self, loc):
        """(s, l): distance along the route and lateral offset (+ right) from the lane centre."""
        d = loc - self.origin
        return d.x * self.fwd.x + d.y * self.fwd.y, d.x * self.right.x + d.y * self.right.y

    def _lane_point(self, s, lateral=0.0, z=0.0):
        wp = self.route[int(np.clip(round(s / 2.0), 0, len(self.route) - 1))]
        loc = wp.transform.location + self.right * lateral
        return carla.Location(loc.x, loc.y, loc.z + z)

    def _destroy(self):
        # Destroy by id through a server-side batch: after collisions the client can wrongly
        # report actors as dead, and actor.destroy() then leaves them in the world.
        for a in self._actors:
            if isinstance(a, carla.Sensor):
                try:
                    a.stop()
                except RuntimeError:
                    pass
        # Async batch + one tick: apply_batch_sync can occasionally hang in synchronous mode.
        if self._actors:
            self.client.apply_batch([carla.command.DestroyActor(a.id) for a in reversed(self._actors)])
            if self.world.get_settings().synchronous_mode:
                self.world.tick()
        self._actors = []

    # -------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()
        rng = self.np_random

        for _ in range(20):  # retry if a spawn location is blocked
            self.route = self.routes[rng.integers(len(self.routes))]
            start = self.route[0].transform
            self.origin, self.fwd, self.right = start.location, start.get_forward_vector(), start.get_right_vector()
            self.lane_width = self.route[0].lane_width
            self.s_goal = self.route_length - 5.0
            self.s_park = float(rng.uniform(35.0, 55.0))
            rot = carla.Rotation(yaw=start.rotation.yaw)

            ego = self.world.try_spawn_actor(
                self.ego_bp, carla.Transform(self._lane_point(0.0, z=0.4), rot))
            if ego is None:
                continue
            self._actors.append(ego)
            parked = self.world.try_spawn_actor(self.parked_bp, carla.Transform(
                self._lane_point(self.s_park, self.lane_width / 2 + 0.9, z=0.3), rot))
            walker_bp = self.walker_bps[rng.integers(len(self.walker_bps))]
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")
            walker = self.world.try_spawn_actor(walker_bp, carla.Transform(
                self._lane_point(self.s_park + 3.5, self.lane_width / 2 + 1.8, z=1.0),
                carla.Rotation(yaw=start.rotation.yaw - 90)))
            if parked is None or walker is None:
                self._actors += [a for a in (parked, walker) if a is not None]
                self._destroy()
                continue
            self._actors += [parked, walker]
            break
        else:
            raise RuntimeError("Could not spawn the scenario actors (blocked spawn points?)")

        parked.set_simulate_physics(False)
        self.ego, self.walker = ego, walker
        self.collision = None
        sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=ego)
        sensor.listen(lambda e: setattr(self, "collision", e.other_actor.type_id))
        self._actors.append(sensor)

        # scenario randomisation
        self.will_cross = bool(rng.random() < self.cross_prob)
        self.trigger_dist = float(rng.uniform(12.0, 30.0))
        self.ped_speed = float(rng.uniform(1.2, 2.5))
        self.triggered = False

        # let everything settle, then give the ego car its initial speed
        ego.apply_control(carla.VehicleControl(brake=1.0))
        for _ in range(10):
            self.world.tick()
        v0 = float(rng.uniform(7.0, 11.0))
        ego.set_target_velocity(carla.Vector3D(self.fwd.x * v0, self.fwd.y * v0, 0.0))
        self.world.tick()

        self.t = 0.0
        self.s_prev = self._frame(ego.get_location())[0]
        self.min_dist = float("inf")
        self._follow_camera()
        return self._obs(), {}

    def _steer(self, s_ego):
        tr = self.ego.get_transform()
        target = self._lane_point(s_ego + 8.0)
        v = target - tr.location
        f = tr.get_forward_vector()
        angle = math.atan2(f.x * v.y - f.y * v.x, f.x * v.x + f.y * v.y)
        return float(np.clip(1.5 * angle, -0.5, 0.5))

    def step(self, action):
        a = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
        s_ego = self._frame(self.ego.get_location())[0]
        self.ego.apply_control(carla.VehicleControl(
            throttle=max(a, 0.0), brake=max(-a, 0.0), steer=self._steer(s_ego)))

        s_ped = self._frame(self.walker.get_location())[0]
        if self.will_cross and not self.triggered and s_ped - s_ego < self.trigger_dist:
            self.triggered = True
        if self.triggered:
            self.walker.apply_control(carla.WalkerControl(
                direction=carla.Vector3D(-self.right.x, -self.right.y, 0.0), speed=self.ped_speed))

        self.world.tick()
        self.t += self.dt

        ego_loc = self.ego.get_location()
        s_ego, l_ego = self._frame(ego_loc)
        self.min_dist = min(self.min_dist, ego_loc.distance(self.walker.get_location()))

        reward = (s_ego - self.s_prev) - self.time_penalty * self.dt
        self.s_prev = s_ego
        outcome = None
        if self.collision is not None:
            reward -= 150.0
            outcome = "collision"
        elif s_ego >= self.s_goal:
            reward += 20.0
            outcome = "success"
        elif abs(l_ego) > 2.5:
            reward -= 50.0
            outcome = "off_road"
        terminated = outcome is not None
        truncated = not terminated and self.t >= self.max_time
        if truncated:
            outcome = "timeout"

        self._follow_camera()
        info = {"outcome": outcome, "time": self.t, "min_dist_to_ped": self.min_dist,
                "collided_with": self.collision, "ped_crossed": self.will_cross}
        return self._obs(), float(reward), terminated, truncated, info

    def _obs(self):
        v = self.ego.get_velocity()
        speed = math.hypot(v.x, v.y)
        s_ego, _ = self._frame(self.ego.get_location())
        s_ped, l_ped = self._frame(self.walker.get_location())
        wv = self.walker.get_velocity()
        # occlusion: visible once the pedestrian is out from behind the parked car
        # (parked car spans lateral ~[w/2 - 0.1, w/2 + 1.9]) or the ego is already past it
        visible = l_ped < self.lane_width / 2 + 0.2 or s_ego > self.s_park + 2.0
        if visible:
            ped = [s_ped - s_ego, l_ped, wv.x * self.right.x + wv.y * self.right.y]
        else:
            ped = [HIDDEN_LONG, HIDDEN_LAT, 0.0]
        return np.array([speed, self.s_goal - s_ego, self.s_park - s_ego, float(visible), *ped],
                        dtype=np.float32)

    def _follow_camera(self):
        if self.render_mode != "human":
            return
        tr = self.ego.get_transform()
        f = tr.get_forward_vector()
        loc = tr.location - f * 10.0 + carla.Location(z=6.0)
        self.world.get_spectator().set_transform(
            carla.Transform(loc, carla.Rotation(pitch=-20, yaw=tr.rotation.yaw)))

    def render(self):
        self._follow_camera()

    def close(self):
        self._destroy()
        try:
            self.world.apply_settings(self._orig_settings)
        except RuntimeError:
            pass
