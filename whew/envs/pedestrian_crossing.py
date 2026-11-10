"""
Near-accident scenario 1: pedestrian dart-out from behind a parked car.

The ego car drives along a straight road. A parked car on the right hides a
pedestrian, who may step into the road when the ego car is within 12-30 m or
1-2 s away (whichever comes first, so speeding cannot outrun them).
Driving above the 50 km/h speed limit is penalised. The
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
import carla
import numpy as np

from .base import CarlaScenarioEnv

HIDDEN_LONG, HIDDEN_LAT = 60.0, 10.0


class PedestrianCrossingEnv(CarlaScenarioEnv):
    def __init__(self, cross_prob=0.8, **kwargs):
        self.cross_prob = cross_prob
        super().__init__(**kwargs)
        self.parked_bp = self.blueprints.find("vehicle.audi.a2")
        self.walker_bps = list(self.blueprints.filter("walker.pedestrian.*"))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()
        rng = self.np_random

        for _ in range(20):  # retry if a spawn location is blocked
            self._set_route(self.routes[rng.integers(len(self.routes))])
            self.s_park = float(rng.uniform(35.0, 55.0))
            rot = carla.Rotation(yaw=self.yaw)

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
                carla.Rotation(yaw=self.yaw - 90)))
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
        self._attach_collision_sensor()

        # scenario randomisation
        self.will_cross = bool(rng.random() < self.cross_prob)
        # The pedestrian steps out when the car is within 12-30 m OR 1-2 s away, whichever
        # comes first. The time-based part means speeding cannot outrun the pedestrian.
        self.trigger_dist = float(rng.uniform(12.0, 30.0))
        self.trigger_tta = float(rng.uniform(1.0, 2.0))
        self.ped_speed = float(rng.uniform(1.2, 2.5))
        self.triggered = False

        # let everything settle, then give the ego car its initial speed
        self._settle([ego])
        self._launch(ego, float(rng.uniform(7.0, 11.0)))
        self.world.tick()
        self._start_episode()
        return self._obs(), {}

    def _before_tick(self, s_ego):
        s_ped = self._frame(self.walker.get_location())[0]
        if self.will_cross and not self.triggered:
            gap = s_ped - s_ego
            tta = gap / max(self._speed(self.ego), 0.1)
            if gap < self.trigger_dist or tta < self.trigger_tta:
                self.triggered = True
        if self.triggered:
            self.walker.apply_control(carla.WalkerControl(
                direction=carla.Vector3D(-self.right.x, -self.right.y, 0.0), speed=self.ped_speed))

    def _after_tick(self, ego_loc):
        self.min_dist = min(self.min_dist, ego_loc.distance(self.walker.get_location()))

    def _extra_info(self):
        return {"min_dist_to_ped": self.min_dist, "ped_crossed": self.will_cross}

    def _obs(self):
        speed = self._speed(self.ego)
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
