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


# --------------------------------------------------------------------------- cut-in
#            cruise  headway (s)  brake if time-to-collision below (s)  react to drifting cutter
_CUT_IN_STYLE = {
    AGGRESSIVE: (13.0,   0.5,          0.8,                                 False),
    TIMID:      (10.0,   1.8,          3.5,                                 True),
}


def expert_action_cut_in(obs, command, lane_width=3.5):
    """Rule-based driver for the cut-in scenario. Returns a in [-1, 1]."""
    v, _, gap, lat, rel_v, vlat, in_lane = [float(x) for x in obs]
    cruise, headway, ttc_brake, react_early = _CUT_IN_STYLE[command]
    target = cruise

    # the cutter matters if it is ahead and in our lane, or (timid only) drifting towards it
    relevant = gap > -2.0 and (in_lane > 0.5 or (react_early and lat < 0 and vlat > 0.3 and gap < 25.0))
    if relevant:
        closing = -rel_v  # > 0 when we are catching up
        ttc = gap / closing if closing > 0.1 else float("inf")
        if gap < 1.0 or ttc < ttc_brake:
            return -1.0
        # follow the cutter at the preferred time headway
        target = min(cruise, max(0.0, v + rel_v + 0.5 * (gap - headway * v)))

    return float(np.clip(0.5 * (target - v), -1.0, 1.0))
