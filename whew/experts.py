"""
Rule-based expert drivers for the pedestrian dart-out scenario.

Two styles, as in the CARLO project (and H-ReIL, Cao et al. 2020):
  AGGRESSIVE (0): drives fast, brakes only for a pedestrian it is sure to hit.
  TIMID      (1): slows down near the blind spot and brakes with a large safety margin.
Neither is optimal on its own; the RL agent should learn when to be which.
"""
import numpy as np

AGGRESSIVE, TIMID = 0, 1
COMMAND_NAMES = {AGGRESSIVE: "aggressive", TIMID: "timid"}

_STYLE = {
    #            cruise  blind-spot speed  time margin (s)
    AGGRESSIVE: (13.0,   13.0,             0.2),
    TIMID:      (10.0,    4.5,             1.5),
}


def expert_action(obs, command, lane_width=3.5):
    """Returns a in [-1, 1] (throttle > 0, brake < 0) for the given observation."""
    v, _, occ_long, visible, p_long, p_lat, p_vlat = [float(x) for x in obs]
    cruise, blind_speed, margin = _STYLE[command]

    target = cruise
    if -3.0 < occ_long < 25.0:
        target = blind_speed

    if visible and p_long > 0.0:
        half = lane_width / 2 + 1.0  # lane half-width + safety buffer
        t_ego = p_long / max(v, 0.5)
        if -half <= p_lat <= half:
            conflict = True  # pedestrian is in (or at the edge of) our lane
        elif p_lat > half and p_vlat < -0.1:   # approaching from the right
            conflict = t_ego > (p_lat - half) / -p_vlat - margin
        elif p_lat < -half and p_vlat > 0.1:   # approaching from the left
            conflict = t_ego > (-half - p_lat) / p_vlat - margin
        else:
            conflict = False
        if conflict:
            return -1.0

    return float(np.clip(0.5 * (target - v), -1.0, 1.0))
