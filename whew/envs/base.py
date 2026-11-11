"""
Shared CARLA plumbing for WHEW! scenarios.

A scenario subclasses CarlaScenarioEnv and implements:
    reset()                       spawn its actors (use the helpers below)
    _before_tick(s_ego)           control the other actors for this step
    _after_tick(ego_loc)          update scenario statistics (e.g. min distance)
    _obs()                        7-float observation
    _extra_info()                 scenario-specific info entries

The ego car is controlled longitudinally by the agent (a in [-1, 1]: throttle if
positive, brake if negative); a lane-keeping controller steers it.
Reward (shared): +1 per metre of progress, -time_penalty per second, -speeding_penalty
per second for each m/s above the speed limit, -150 for a collision, +20 at the goal,
-50 for leaving the road.
"""
import math
import time
from collections import deque

import carla
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CarlaScenarioEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    need_left_lane = False  # scenarios with a second same-direction lane on the left set this

    def __init__(self, host="localhost", port=2000, town=None, dt=0.05, max_time=20.0,
                 route_length=80.0, render_mode=None, no_rendering=False, timeout=30.0,
                 time_penalty=4.0, speed_limit=13.9, speeding_penalty=2.0, action_delay=0):
        super().__init__()
        self.action_delay = action_delay  # simulation steps between choosing and applying an action
        self.dt, self.max_time, self.route_length = dt, max_time, route_length
        self.render_mode = render_mode
        self.time_penalty = time_penalty  # reward lost per second of driving
        self.speed_limit = speed_limit  # m/s (13.9 m/s = 50 km/h, a city speed limit)
        self.speeding_penalty = speeding_penalty  # per (m/s over the limit) per second

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

        self.blueprints = self.world.get_blueprint_library()
        self.ego_bp = self.blueprints.find("vehicle.tesla.model3")
        self.ego_bp.set_attribute("role_name", "hero")
        self.collision_bp = self.blueprints.find("sensor.other.collision")

        self.routes = self._find_routes()
        if not self.routes:
            raise RuntimeError(f"No suitable straight {route_length:.0f} m route found in {self.map.name}; "
                               "try town='Town04' or a shorter route_length.")

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._actors = []

    # ------------------------------------------------------------------ routes / geometry
    def _find_routes(self):
        return self._find_straight_routes()

    def _find_straight_routes(self, step=2.0):
        """Straight, junction-free routes in the right-most driving lane."""
        n = int(self.route_length / step)
        routes, seen = [], set()
        for sp in self.map.get_spawn_points():
            wp = self.map.get_waypoint(sp.location)
            key = (wp.road_id, wp.lane_id)
            if key in seen or wp.is_junction:
                continue
            right = wp.get_right_lane()
            if right is not None and right.lane_type == carla.LaneType.Driving:
                continue  # need the right-most driving lane
            if self.need_left_lane:
                left = wp.get_left_lane()
                if left is None or left.lane_type != carla.LaneType.Driving or left.lane_id * wp.lane_id <= 0:
                    continue  # need a same-direction lane on the left
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

    def _set_route(self, route):
        self.route = route
        start = route[0].transform
        self.origin, self.fwd, self.right = start.location, start.get_forward_vector(), start.get_right_vector()
        self.yaw = start.rotation.yaw
        self.lane_width = route[0].lane_width
        self.s_goal = self.route_length - 5.0

    def _frame(self, loc):
        """(s, l): distance along the route and lateral offset (+ right) from the lane centre."""
        d = loc - self.origin
        return d.x * self.fwd.x + d.y * self.fwd.y, d.x * self.right.x + d.y * self.right.y

    def _lane_point(self, s, lateral=0.0, z=0.0):
        wp = self.route[int(np.clip(round(s / 2.0), 0, len(self.route) - 1))]
        loc = wp.transform.location + self.right * lateral
        return carla.Location(loc.x, loc.y, loc.z + z)

    def _speed(self, actor):
        v = actor.get_velocity()
        return math.hypot(v.x, v.y)

    def _long_lat_velocity(self, actor):
        v = actor.get_velocity()
        return v.x * self.fwd.x + v.y * self.fwd.y, v.x * self.right.x + v.y * self.right.y

    def _steer_actor(self, actor, s, lateral=0.0, lookahead=8.0, gain=1.5, limit=0.5):
        tr = actor.get_transform()
        v = self._lane_point(s + lookahead, lateral) - tr.location
        f = tr.get_forward_vector()
        angle = math.atan2(f.x * v.y - f.y * v.x, f.x * v.x + f.y * v.y)
        return float(np.clip(gain * angle, -limit, limit))

    # ------------------------------------------------------------------ actors
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

    def _attach_collision_sensor(self):
        self.collision = None
        sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
        sensor.listen(lambda e: setattr(self, "collision", e.other_actor.type_id))
        self._actors.append(sensor)

    def _settle(self, vehicles):
        for veh in vehicles:
            veh.apply_control(carla.VehicleControl(brake=1.0))
        for _ in range(10):
            self.world.tick()

    def _launch(self, actor, speed):
        actor.set_target_velocity(carla.Vector3D(self.fwd.x * speed, self.fwd.y * speed, 0.0))

    def _start_episode(self):
        self.t = 0.0
        self.s_prev = self._frame(self.ego.get_location())[0]
        self.min_dist = float("inf")
        self.max_speed = 0.0
        self._pending = deque([0.0] * self.action_delay)
        self._follow_camera()

    # ------------------------------------------------------------------ gym API
    def step(self, action):
        a = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
        if self.action_delay:  # reaction / actuation delay: apply the action chosen action_delay steps ago
            self._pending.append(a)
            a = self._pending.popleft()
        s_ego = self._frame(self.ego.get_location())[0]
        self.ego.apply_control(carla.VehicleControl(
            throttle=max(a, 0.0), brake=max(-a, 0.0), steer=self._steer_actor(self.ego, s_ego)))
        self._before_tick(s_ego)

        if self.render_mode == "human":  # play back in real time so it can be watched
            wait = self.dt - (time.perf_counter() - getattr(self, "_last_wall", 0.0))
            if wait > 0:
                time.sleep(wait)
            self._last_wall = time.perf_counter()
        self.world.tick()
        self.t += self.dt

        ego_loc = self.ego.get_location()
        s_ego, l_ego = self._frame(ego_loc)
        self._after_tick(ego_loc)

        speed = self._speed(self.ego)
        self.max_speed = max(self.max_speed, speed)
        speeding = max(0.0, speed - self.speed_limit)
        reward = (s_ego - self.s_prev) - self.time_penalty * self.dt - self.speeding_penalty * speeding * self.dt
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
        info = {"outcome": outcome, "time": self.t, "min_dist": self.min_dist,
                "collided_with": self.collision, "speeding": speeding > 0, "max_speed": self.max_speed,
                **self._extra_info()}
        return self._obs(), float(reward), terminated, truncated, info

    def _before_tick(self, s_ego):
        pass

    def _after_tick(self, ego_loc):
        pass

    def _extra_info(self):
        return {}

    def _obs(self):
        raise NotImplementedError

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
