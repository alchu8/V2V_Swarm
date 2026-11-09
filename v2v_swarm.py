"""
V2V Swarm -- Basic Model (Python implementation)

NumPy-vectorized simulator with two complementary modes:

  1. run(config, seed) -- headless, returns metrics dict; use for parameter
     sweeps and statistical analysis in Jupyter or scripts.

  2. run_animated(config, seed) -- interactive PyGame viewer with pause,
     speed, V2V-toggle, and reset controls. Use for visual debugging and
     intuition building. Requires 'pip install pygame'.

Direct invocation ('python v2v_swarm.py') runs the animated viewer if
PyGame is available, and falls back to a headless run + matplotlib plot
otherwise.

Parameter-sweep idiom (headless):
    import pandas as pd
    rows = []
    for alpha in [0.0024, 0.0030, 0.0040]:
        for v2v in [False, True]:
            for seed in range(20):
                cfg = Config(alpha=alpha, v2v_enabled=v2v, num_robots=20)
                r = run(cfg, seed=seed)
                rows.append({'alpha': alpha, 'v2v': v2v, 'seed': seed,
                             'tasks_done': r['tasks_done'],
                             'peak_reach': r['peak_reach']})
    df = pd.DataFrame(rows)
    print(df.groupby(['alpha', 'v2v'])['tasks_done'].agg(['mean', 'std']))

Omitted vs the full simulator: no multi-hop chain reachability (the safety filter
here is single-hop -- a robot stays in the set from which it can reach the depot
either directly or via ONE peer), and no refueler role. Bare bones to extend.
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import csv
from collections import defaultdict
from scipy.spatial import ConvexHull


@dataclass
class Config:
    num_robots: int = 12
    num_refuelers: int = 1
    refueler_radius: float = 0.5 # fraction of solo reach
    alpha: float = 0.0024 # battery depletion rate per unit distance traveled
    max_battery = 1.0
    task_rate: float = 0.06
    task_energy_cost: float = 0.18
    low_soc_threshold: float = 0.25 # TODO dynamic threshold
    transfer_rate: float = 0.15
    v2v_enabled: bool = True
    immobilize_on_empty: bool = True

    world_size: float = 500.0
    depot_radius: float = 3.0
    move_step: float = 0.5
    recharge_rate: float = 0.004

    max_steps: int = 2000 # TODO new termination condition
    task_ttl: int = 200
    max_tasks: int = 30
    task_engage_radius: float = 1.5
    revive_radius: float = 3.0
    comm_range: float = 10.0


class SimState:
    """Mutable simulation state. Shared between run() and run_animated()
    so the step logic lives in exactly one place (step_sim below)."""
    def __init__(self, config: Config, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        N = config.num_robots
        L = config.world_size
        self.current_station = [None] * N
        self.depot = np.array([L / 2, L / 2])
        solo_reach = config.max_battery / (2.0 * config.alpha)
        # Initial positions: uniform in a disk around the depot of radius
        # min(135, 0.9*solo_reach), matching the JS widget's init geometry.
        init_r_max = min(135.0, solo_reach * 0.9)
        ang = self.rng.uniform(0, 2 * np.pi, size=N)
        r = init_r_max * np.sqrt(self.rng.uniform(0, 1, size=N))
        self.pos = self.depot + np.column_stack([r * np.cos(ang), r * np.sin(ang)])
        self.soc = config.max_battery - (np.linalg.norm(self.pos - self.depot, axis=1) / (2*solo_reach))
        # Wander targets: same disk so robots roam within reach by default.
        r = solo_reach * 0.9 * np.sqrt(self.rng.uniform(0, 1, size=N))
        self.target = self.depot + np.column_stack([r * np.cos(ang), r * np.sin(ang)])
        # 0 wander, 1 agent returning to depot, 2 docked and recharging, 3 immobilized, 
        # 4 refueler heading to station, 5 refueler returning to depot, 6 refueler docked and recharging,
        # 7 returning agent heading to refueler, 8 returning agent recharging at refueler
        self.mode = np.zeros(N, dtype=int)
        self.prev_a = self.rng.uniform(0, 2 * np.pi, size=N)
        N_stationary = config.num_refuelers
        self.charging_loc = np.empty((N_stationary, 2))
        if N_stationary > 0:
            self.mode[:N_stationary] = 4
            angle = 0
            angle_step = 2 * np.pi / N_stationary
            for i in range(N_stationary):
                # TODO parameterize angle
                angle += angle_step % (2 * np.pi)
                radius = config.refueler_radius * solo_reach
                self.charging_loc[i] = self.depot + np.array([
                    radius * np.cos(angle),
                    radius * np.sin(angle)
                ])
                self.pos[i] = self.depot
                self.target[i] = self.charging_loc[i].copy()
        self.task_pos = np.empty((0, 2))
        self.task_ttl = np.empty(0, dtype=int)
        self.spatial_grid = SpatialGrid(world_size=config.world_size, cell_size=config.comm_range)

class SpatialGrid:
    def __init__(self, world_size, cell_size):
        self.cell_size = cell_size
        self.inv_cell_size = 1.0 / cell_size
        self.grid = {}

    def cell_index(self, position):
        return tuple(
            np.floor(position * self.inv_cell_size).astype(int)
        )

    def rebuild(self, positions):
        """Rebuild grid from current robot positions."""
        self.grid.clear()

        for i, p in enumerate(positions):
            cell = self.cell_index(p)

            if cell not in self.grid:
                self.grid[cell] = []

            self.grid[cell].append(i)

    def neighbors(self, position, radius):
        """Return candidate robot indices near position."""
        center = self.cell_index(position)

        # Since cell_size = communication radius,
        # only the 3x3 neighboring cells are required.
        candidates = []

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = (center[0] + dx, center[1] + dy)

                if cell in self.grid:
                    candidates.extend(self.grid[cell])

        return candidates

def is_safe_state(state, config, i, pos, batt, v2v_enabled):
    """CBF-style safety check: from position pos with battery batt, can robot i still
    get home? Either directly (batt covers the trip to the depot), or -- when V2V is
    on -- by reaching ONE peer that can spare enough to top it up for the rest of the
    way. This is the reference's safety filter; it is the single-hop case of the
    widget's filter (no multi-hop chaining). With V2V off it reduces to plain direct
    reachability, which is exactly why conservative solo motion never strands a
    robot: the filter below never lets one step outside the still-returnable set.
    """
    depot = state.depot
    if batt >= config.alpha * np.linalg.norm(pos - depot):
        return True
    if not v2v_enabled:
        return False
    for j in range(config.num_robots):
        if j == i or state.mode[j] not in [0, 4]:  # allow stationary (mode 4) as helpers
            continue
        d_jD = np.linalg.norm(state.pos[j] - depot)
        d_ij = np.linalg.norm(pos - state.pos[j])
        if batt < config.alpha * d_ij:            # i cannot even reach peer j
            continue
        x_max = state.soc[j] - config.alpha * d_jD  # most j can give and still get itself home
        if x_max <= 0:
            continue
        b_post = batt - config.alpha * d_ij + x_max  # i's battery after reaching j and topping up
        if b_post >= config.alpha * d_jD:         # ...enough to reach the depot from j
            return True
    return False

def softmin(x, T=1.0):
    # temperature if needed for separating
    T = max(float(T), 1e-8)
    # Subtracting the min stabilizes the calculation numerically
    shifted_inputs = x - np.min(x)
    exponentiated = np.exp(np.divide(-shifted_inputs, T))
    return exponentiated / np.sum(exponentiated)

def step_sim(state: SimState, config: Config, v2v_enabled: bool):
    """Advance the simulation by one step. Mutates state in place.
    Returns (tasks_done_delta, tasks_expired_delta) for this step."""
    rng = state.rng
    L = config.world_size
    N = config.num_robots
    done_delta = 0
    expired_delta = 0

    # Spawn task: annulus around the depot, matching the JS widget.
    # if rng.random() < config.task_rate and len(state.task_pos) < config.max_tasks:
    #     solo_reach = 1.0 / (2.0 * config.alpha)
    #     r_min, r_max = 30.0, min(solo_reach * 0.9, L / 2.0 - 45.0)
    #     ang = rng.uniform(0, 2 * np.pi)
    #     r = np.sqrt(rng.uniform(r_min ** 2, r_max ** 2))
    #     task = state.depot + r * np.array([np.cos(ang), np.sin(ang)])
    #     state.task_pos = np.vstack([state.task_pos, task])
    #     state.task_ttl = np.append(state.task_ttl, config.task_ttl)

    # Expire tasks
    # if len(state.task_ttl) > 0:
    #     state.task_ttl -= 1
    #     expired = state.task_ttl <= 0
    #     if expired.any():
    #         expired_delta = int(expired.sum())
    #         state.task_pos = state.task_pos[~expired]
    #         state.task_ttl = state.task_ttl[~expired]

    # Robot updates
    for i in range(N):
        # Immobilized (depleted): frozen in place. Cannot move or engage tasks;
        # only the revival pass below can bring it back.
        if state.mode[i] == 3:
            continue
        # Docked: recharge in place
        if state.mode[i] == 2:
            state.soc[i] += config.recharge_rate
            if state.soc[i] >= 1.0:
                state.soc[i] = 1.0
                state.mode[i] = 0
                stationary_indices = np.where(state.mode == 4)[0]
                if len(stationary_indices) > 0:
                    dists = np.linalg.norm(state.pos[stationary_indices] - state.pos[i], axis=1)
                    softmin_weights = softmin(dists, T=1)
                    nearest_idx = np.random.choice(stationary_indices, size=1, p=softmin_weights)[0]
                    nearest_pos = state.pos[nearest_idx]
                    dir_vec = nearest_pos - state.pos[i]
                    dir_angle = np.arctan2(dir_vec[1], dir_vec[0])
                    _a = dir_angle + 0.5 * rng.uniform(-np.pi/2, np.pi/2)
                    state.prev_a[i] = _a
                _r = (1.0 / (2.0 * config.alpha)) * 0.9 * np.sqrt(rng.uniform(0, 1))
                # dangling carrot to drive ballistic motion
                state.target[i] = state.pos[i] + _r * np.array([np.cos(state.prev_a[i]), np.sin(state.prev_a[i])])
            continue

        if state.mode[i] == 6: # refueler docked at depot and ready to recharge
            state.soc[i] += config.recharge_rate
            if state.soc[i] >= 1.0:
                state.soc[i] = 1.0
                state.mode[i] = 4 # refueler back to charging position
                state.target[i] = state.charging_loc[i].copy()
            continue

        # Pick fresh wander target if reached
        d_target = np.linalg.norm(state.target[i] - state.pos[i])
        if state.mode[i] == 0: #and d_target < 1.0:
            # Find nearest stationary agent
            # stationary_indices = np.where(state.mode == 4)[0]
            # if len(stationary_indices) > 0:
            #     dists = np.linalg.norm(state.pos[stationary_indices] - state.pos[i], axis=1)
            #     nearest_idx = stationary_indices[np.argmin(dists)]
            #     nearest_pos = state.pos[nearest_idx]
            #     dir_vec = nearest_pos - state.pos[i]
            #     dir_angle = np.arctan2(dir_vec[1], dir_vec[0])
            #     #bias_factor = 0.8
            #     #_a = (1 - bias_factor) * rng.uniform(0, 2 * np.pi) + bias_factor * dir_angle
            #     #_a = dir_angle + bias_factor * rng.uniform(-np.pi, np.pi)
            #     _a = _a % (2 * np.pi)
            # else:
            #     _a = rng.uniform(0, 2 * np.pi)
            # Go along same direction as last step, with some random perturbation
            _a = state.prev_a[i] + 0.5 * rng.uniform(-np.pi/2, np.pi/2)
            _r = (1.0 / (2.0 * config.alpha)) * 0.9 * np.sqrt(rng.uniform(0, 1))
            # dangling carrot to drive ballistic motion
            state.target[i] = state.pos[i] + _r * np.array([np.cos(_a), np.sin(_a)])
            d_target = np.linalg.norm(state.target[i] - state.pos[i])

        # Seek-charge trigger: decide to return once SoC drops below the threshold.
        if (state.mode[i] == 0) and state.soc[i] < config.low_soc_threshold:
            state.mode[i] = 1
            state.target[i] = state.depot.copy()
            d_target = np.linalg.norm(state.target[i] - state.pos[i])
        
        # Refueler trigger: return once SoC drops below the threshold
        if (state.mode[i] == 4) and state.soc[i] < config.low_soc_threshold:
            state.mode[i] = 5 # refueler returning to depot
            state.target[i] = state.depot.copy()
            d_target = np.linalg.norm(state.target[i] - state.pos[i])

        # While returning, if we cannot reach the depot alone, pull just enough
        # from a peer to get home -- never a fixed target SoC. Metered by
        # transfer_rate, so it spans several steps; the full recharge happens at
        # the depot. Peer V2V is purely an anti-stranding safety net.
        safety_padding = 0.02 # extra SoC to avoid running dry on the depot's pad
        if state.mode[i] == 1 and v2v_enabled:
            dist_home = np.linalg.norm(state.pos[i] - state.depot)
            home_need = config.alpha * dist_home + safety_padding - state.soc[i]
            if home_need > 0:
                others = (np.arange(N) != i) & (state.mode == 4) # TODO scenario 2, charge from wandering peers no refuelers
                if others.any():
                    idx = np.where(others)[0]
                    dist_stations_home = np.linalg.norm(state.pos[idx] - state.depot, axis=1)
                    surplus = state.soc[idx] - config.alpha * dist_stations_home - safety_padding
                    eligible = surplus > 0
                    if eligible.any():
                        highest_surplus_station = idx[eligible][np.argmax(surplus[eligible])]
                        state.current_station[i] = highest_surplus_station
                        # can_give = (state.soc[highest_surplus_station]
                        #             - config.alpha * np.linalg.norm(state.pos[highest_surplus_station] - state.depot)
                        #             - safety_padding)
                        state.target[i] = state.pos[highest_surplus_station]
                        state.mode[i] = 7

        if state.mode[i] == 7 and np.linalg.norm(state.pos[i] - state.target[i]) < config.depot_radius:
            state.mode[i] = 8 # returning agent recharging at refueler station
            state.pos[i] = state.target[i].copy()

        if state.mode[i] == 8: # returning agent recharging at refueler station
            dist_home = np.linalg.norm(state.pos[i] - state.depot)
            home_need = config.alpha * dist_home + safety_padding - state.soc[i]
            can_give = (state.soc[state.current_station[i]]
                    - config.alpha * np.linalg.norm(state.pos[state.current_station[i]] - state.depot)
                    - safety_padding)
            donation = min(can_give, home_need)
            donation_per_step = min(donation, config.transfer_rate)
            state.soc[i] += donation_per_step
            state.soc[state.current_station[i]] -= donation_per_step
            if donation <= 0.000001:
                state.mode[i] = 0

        # Move toward target. In wander mode the CBF-style safety filter is active:
        # never take a step that would leave the safe (still-returnable) set. If the
        # natural step is unsafe, bend it toward the depot until it is safe; if even a
        # full step toward the depot is unsafe (the robot is already past its safety
        # boundary), commit toward the depot anyway and let the return logic recover.
        # Home/immobilized modes are not filtered -- a returning robot commits.
        if d_target > 0:
            s = min(config.move_step, d_target)
            step_dir = (state.target[i] - state.pos[i]) / d_target
            if state.mode[i] == 0:
                cand = state.pos[i] + step_dir * s
                cand_batt = state.soc[i] - config.alpha * s
                if not is_safe_state(state, config, i, cand, cand_batt, v2v_enabled):
                    to_depot = state.depot - state.pos[i]
                    dd = np.linalg.norm(to_depot)
                    if dd > 1e-9:
                        depot_dir = to_depot / dd
                        resolved = False
                        for w in (0.25, 0.5, 0.75, 1.0):
                            bd = (1.0 - w) * step_dir + w * depot_dir
                            nm = np.linalg.norm(bd)
                            if nm < 1e-9:
                                continue
                            bd = bd / nm
                            if is_safe_state(state, config, i, state.pos[i] + bd * s,
                                             state.soc[i] - config.alpha * s, v2v_enabled):
                                step_dir = bd
                                resolved = True
                                break
                        if not resolved:
                            step_dir = depot_dir
            state.pos[i] += step_dir * s
            state.soc[i] -= config.alpha * s

        # Depletion: out of charge. If it ran dry on the depot's charging pad,
        # it docks and charges rather than dying -- no robot is lost at the depot.
        # Otherwise it is immobilized in the field (counts as lost). A robot with 0
        # SoC cannot move, so it engages no task this step.
        if config.immobilize_on_empty and state.soc[i] <= 0.0:
            state.soc[i] = 0.0
            if np.linalg.norm(state.pos[i] - state.depot) < config.depot_radius:
                state.mode[i] = 2
                state.pos[i] = state.depot.copy()
            else:
                state.mode[i] = 3
            continue

        # Task engagement (only in wander mode, with safety check)
        # if state.mode[i] == 0 and len(state.task_pos) > 0:
        #     dists = np.linalg.norm(state.task_pos - state.pos[i], axis=1)
        #     near = int(np.argmin(dists))
        #     if dists[near] < config.task_engage_radius:
        #         home_d = np.linalg.norm(state.pos[i] - state.depot)
        #         if state.soc[i] - config.task_energy_cost > config.alpha * home_d:
        #             state.soc[i] -= config.task_energy_cost
        #             done_delta += 1
        #             state.task_pos = np.delete(state.task_pos, near, axis=0)
        #             state.task_ttl = np.delete(state.task_ttl, near)

        # Dock when close enough
        if state.mode[i] == 1 and np.linalg.norm(state.pos[i] - state.depot) < config.depot_radius:
            state.mode[i] = 2
            state.pos[i] = state.depot.copy()
        if state.mode[i] == 5 and np.linalg.norm(state.pos[i] - state.depot) < config.depot_radius:
            state.mode[i] = 6 # refueler docked at depot and ready to recharge
            state.pos[i] = state.depot.copy()

    # Contact revival: a depleted robot keeps just enough residual power to run a
    # help beacon and accept an inductive top-up. A passing robot revives it only
    # if it can spare enough to cover the dead robot's full trip home; a partial
    # hand-off that left it crawling at near-zero would be worse than honest. So a
    # death near a well-charged neighbor is recoverable; an isolated one is
    # permanent (counts as a loss). No dispatch. Revival is itself a V2V transfer,
    # so it happens only when V2V is enabled -- turn V2V off and lost robots stay
    # lost, exactly as in the pure-solo regime.
    if config.immobilize_on_empty and v2v_enabled:
        for d in np.where(state.mode == 3)[0]:
            need = config.alpha * np.linalg.norm(state.pos[d] - state.depot) + 0.05
            for j in range(N):
                if j == d or state.mode[j] == 3:
                    continue
                if np.linalg.norm(state.pos[j] - state.pos[d]) > config.revive_radius:
                    continue
                give = (state.soc[j]
                        - config.alpha * np.linalg.norm(state.pos[j] - state.depot)
                        - 0.02)
                if give < need:
                    continue
                state.soc[j] -= need # full amount in one time step?
                state.soc[d] += need
                state.mode[d] = 1
                state.target[d] = state.depot.copy()
                break

    return done_delta, expired_delta


def run(config: Config = None, seed: int = 0):
    """Run a single simulation headless. Returns a dict of metrics + final state."""
    if config is None:
        config = Config()
    state = SimState(config, seed)
    solo_reach = 1.0 / (2.0 * config.alpha)
    metrics = {
        'tasks_done': 0, 'tasks_expired': 0, 'mean_peak_reach': 0.0, 'mean_area': 0.0,
        'peak_reach': 0.0, 'peak_area': 0.0,
        'time_at_outer80': 0, 'time_past_solo': 0, 'total_robot_time': 0,
        'peak_lost': 0,
        'soc_trace': [], 'reach_trace': [],
    }
    #prev_mean_peak_reach = np.inf

    norm_reach_sum = 0.0
    norm_area_sum = 0.0
    for step in range(config.max_steps):
    #step = 0
    #while np.abs(metrics['mean_peak_reach'] - prev_mean_peak_reach) > 1e-2:
        dd, ed = step_sim(state, config, config.v2v_enabled)
        metrics['tasks_done'] += dd
        metrics['tasks_expired'] += ed
        metrics['peak_lost'] = max(metrics['peak_lost'], int(np.sum(state.mode == 3)))

        d_from = compute_d_from_depot(state)
        norm_reach = float(d_from.max()) / solo_reach
        norm_reach_sum += norm_reach
        metrics['mean_peak_reach'] = norm_reach_sum / (step+1)
        # step += 1
        # prev_mean_peak_reach = metrics['mean_peak_reach']
        metrics['peak_reach'] = max(metrics['peak_reach'], norm_reach) # peak of all time
        metrics['time_past_solo'] += int(np.sum(d_from > solo_reach))
        metrics['time_at_outer80'] += int(np.sum(d_from > 0.8 * solo_reach))
        idx = np.where(state.mode != 3)[0]
        if len(idx) < 3:
            hull = None
        else:
            points = state.pos[idx]
            # Check collinearity using first 3 points
            p0, p1, p2 = points[:3]
            area = 0.5 * np.abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
            if area < 1e-6:  # Collinear points
                hull = None
            else:
                hull = ConvexHull(state.pos[idx])
                norm_area = hull.volume / (config.world_size * config.world_size)
                metrics['peak_area'] = max(metrics['peak_area'], norm_area) # peak of all time
                norm_area_sum += norm_area
        metrics['mean_area'] = norm_area_sum / (step+1)
        metrics['total_robot_time'] += config.num_robots

        if step % 50 == 0:
            metrics['soc_trace'].append((step, state.soc.copy()))
            metrics['reach_trace'].append((step, norm_reach))

    metrics['final_pos'] = state.pos.copy()
    metrics['final_soc'] = state.soc.copy()
    metrics['final_mode'] = state.mode.copy()
    metrics['lost'] = int(np.sum(state.mode == 3))
    metrics['fraction_past_solo'] = metrics['time_past_solo'] / max(1, metrics['total_robot_time'])
    metrics['fraction_at_outer80'] = metrics['time_at_outer80'] / max(1, metrics['total_robot_time'])
    metrics['solo_reach'] = solo_reach
    metrics['config'] = config
    metrics['seed'] = seed
    return metrics

def compute_d_from_depot(state):
    idx = np.where(state.mode != 3)[0] # indices of active robots (not immobilized)
    d_from = np.linalg.norm(state.pos[idx] - state.depot, axis=1)
    return d_from

def plot_run(result):
    """Static post-mortem plot of a single run."""
    config = result['config']
    L = config.world_size
    depot = np.array([L / 2, L / 2])
    solo_reach = result['solo_reach']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(depot[0] + solo_reach * np.cos(theta),
            depot[1] + solo_reach * np.sin(theta),
            'k--', alpha=0.3, label=f'Solo-reach (1/2alpha = {solo_reach:.0f})')
    ax.scatter(*depot, s=200, c='darkgreen', marker='o', label='Depot', zorder=10)

    pos = result['final_pos']
    mode = result['final_mode']
    for m, (c, lab) in enumerate(zip(['steelblue', 'firebrick', 'forestgreen', 'dimgray'],
                                     ['Wandering', 'Heading home', 'Docked', 'Lost (immobilized)'])):
        mask = mode == m
        if mask.any():
            ax.scatter(pos[mask, 0], pos[mask, 1], c=c, s=60, alpha=0.7, label=lab)
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('Final positions')
    ax.grid(alpha=0.3)

    ax = axes[1]
    if result['soc_trace']:
        steps = np.array([s[0] for s in result['soc_trace']])
        socs = np.array([s[1] for s in result['soc_trace']])
        for i in range(socs.shape[1]):
            ax.plot(steps, socs[:, i], alpha=0.3, linewidth=0.8)
        ax.plot(steps, socs.mean(axis=1), 'k-', linewidth=2, label='Mean SoC')
    ax.axhline(config.low_soc_threshold, color='red', linestyle='--', alpha=0.5, label='Seek-charge threshold')
    ax.set_xlabel('Simulation step')
    ax.set_ylabel('State of charge')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.set_title('SoC traces (gray = individuals; black = mean)')
    ax.grid(alpha=0.3)

    fig.suptitle(
        f'N={config.num_robots}  alpha={config.alpha:.4f}  '
        f'V2V={"on" if config.v2v_enabled else "off"}  |  '
        f'Tasks done={result["tasks_done"]}  '
        f'Peak reach={result["peak_reach"]:.2f}x  '
        f'Time @ outer 80%={result["fraction_at_outer80"]*100:.1f}%',
        fontsize=11,
    )
    plt.tight_layout()
    return fig


def run_animated(config: Config = None, seed: int = 0):
    """Interactive PyGame viewer.

    Keyboard controls:
        SPACE   pause / resume
        + / -   speed up / slow down (1 to 20 sim-steps per frame)
        v       toggle V2V on/off mid-run
        r       reset
        q/ESC   quit

    Requires 'pip install pygame'. The window stays open until you quit;
    the function returns after pygame.quit() is called.
    """
    try:
        import pygame
    except ImportError as e:
        raise ImportError("pygame is required for animated mode. Install with: pip install pygame") from e

    # pygame.font has a circular-import bug under Python 3.14 + pygame 2.6
    # (pygame.font tries to import 'Font' from itself during partial init,
    # raising NotImplementedError when SysFont is called). pygame.freetype
    # is the modern replacement and is unaffected. Use it via a thin wrapper
    # that matches the .render(text, color) and .line_height surface we use
    # below, so the rest of the function reads the same.
    import pygame.freetype

    if config is None:
        config = Config()

    # Window geometry
    WIN_W, WIN_H = 940, 640
    SIM_PAD = 30
    SIM_SIZE = WIN_H - 2 * SIM_PAD
    HUD_X = SIM_PAD + SIM_SIZE + 30
    L = config.world_size

    pygame.init()
    pygame.freetype.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("V2V Swarm Explorer (Python)")
    clock = pygame.time.Clock()

    class _Font:
        """Thin adapter so freetype.Font.render() returns just a Surface
        (matching pygame.font.Font's signature) and so we can attach a
        line_height for the HUD layout loop."""
        def __init__(self, size, bold=False):
            self.font = pygame.freetype.SysFont("monospace", size, bold=bold)
            self.line_height = int(size * 1.4)
        def render(self, text, color):
            surf, _rect = self.font.render(text, color)
            return surf

    font = _Font(13)
    big_font = _Font(18, bold=True)

    # Brand-matching colors (match the JS widget where possible)
    PAPER = (245, 237, 225)
    INK = (31, 58, 46)
    INK_LIGHT = (140, 168, 156)
    ACCENT = (217, 126, 59)
    GREEN = (74, 128, 96)
    RED = (196, 74, 59)
    BLUE = (90, 122, 150)
    GOLD = (200, 152, 60)
    BLACK = (0, 0, 0)

    state = SimState(config, seed)
    solo_reach = 1.0 / (2.0 * config.alpha)

    # World is sized so solo-reach fits inside. Simple world->screen mapping;
    # no depot-centered zoom, no inner-field rect. Matches the JS widget
    # where the canvas IS the world.
    def w2s(x, y):
        sx = SIM_PAD + x / L * SIM_SIZE
        sy = SIM_PAD + (L - y) / L * SIM_SIZE
        return int(sx), int(sy)

    def w2s_dist(d):
        return int(d / L * SIM_SIZE)

    paused = False
    speed = 1
    step_counter = 0
    tasks_done = 0
    tasks_expired = 0
    v2v_enabled = config.v2v_enabled
    reach_sum = 0.0
    reach_samples = 0
    mean_peak_reach = 0.0
    area_sum = 0.0
    mean_area = 0.0
    hull = None
    idx = []

    running = True
    try:
        while running:
            # -- Input --
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        speed = min(speed + 1, 20)
                    elif event.key == pygame.K_MINUS:
                        speed = max(speed - 1, 1)
                    elif event.key == pygame.K_v:
                        v2v_enabled = not v2v_enabled
                    elif event.key == pygame.K_r:
                        state = SimState(config, seed)
                        step_counter = tasks_done = tasks_expired = 0

            # -- Simulate --
            if not paused:
                for _ in range(speed):
                    dd, ed = step_sim(state, config, v2v_enabled)
                    tasks_done += dd
                    tasks_expired += ed
                    step_counter += 1
                    d_from = compute_d_from_depot(state)
                    peak_now = float(d_from.max()) / solo_reach
                    reach_sum += peak_now
                    reach_samples += 1
                    mean_peak_reach = reach_sum / reach_samples
                    idx = np.where(state.mode != 3)[0]
                    hull = ConvexHull(state.pos[idx])
                    area_sum += hull.volume / (L * L)
                    mean_area = area_sum / reach_samples

            # -- Draw --
            screen.fill(PAPER)

            # Sim area border (the canvas edge)
            pygame.draw.rect(screen, INK_LIGHT, (SIM_PAD, SIM_PAD, SIM_SIZE, SIM_SIZE), 1)

            # Solo-reach circle (dashed, matching the JS widget's aesthetic).
            # pygame.draw doesn't have a native dashed-circle primitive, so we
            # approximate with N short chord segments around the circumference.
            dx, dy = w2s(state.depot[0], state.depot[1])
            sr_px = w2s_dist(solo_reach)
            if sr_px > 0:
                n_dashes = 64
                dash_frac = 0.55
                arc = 2 * np.pi / n_dashes
                for i in range(n_dashes):
                    a1 = i * arc
                    a2 = a1 + arc * dash_frac
                    pygame.draw.line(
                        screen, INK_LIGHT,
                        (dx + sr_px * np.cos(a1), dy + sr_px * np.sin(a1)),
                        (dx + sr_px * np.cos(a2), dy + sr_px * np.sin(a2)),
                        1,
                    )

            # Depot
            pygame.draw.circle(screen, GREEN, (dx, dy), max(4, w2s_dist(config.depot_radius)))
            pygame.draw.circle(screen, INK, (dx, dy), max(4, w2s_dist(config.depot_radius)), 1)

            # Tasks
            for tp in state.task_pos:
                tx, ty = w2s(tp[0], tp[1])
                pygame.draw.polygon(screen, ACCENT,
                                    [(tx, ty - 5), (tx + 5, ty), (tx, ty + 5), (tx - 5, ty)])

            # Robots + SoC bars
            for i in range(config.num_robots):
                rx, ry = w2s(state.pos[i, 0], state.pos[i, 1])
                if state.mode[i] == 3:
                    # Lost: depleted and immobilized. Draw a gray X, no SoC bar.
                    g = (110, 110, 110)
                    pygame.draw.line(screen, g, (rx - 5, ry - 5), (rx + 5, ry + 5), 2)
                    pygame.draw.line(screen, g, (rx - 5, ry + 5), (rx + 5, ry - 5), 2)
                    continue
                if state.mode[i] == 2:
                    c = GREEN
                elif state.mode[i] == 4 or state.mode[i] == 5 or state.mode[i] == 6:
                    c = BLACK         # refueler
                elif state.mode[i] == 1:
                    c = RED          # returning home
                elif state.soc[i] < 0.3:
                    c = GOLD         # battery low (still foraging)
                else:
                    c = BLUE
                pygame.draw.circle(screen, c, (rx, ry), 6)
                pygame.draw.circle(screen, INK, (rx, ry), 6, 1)
                bw, bh = 16, 3
                bx = rx - bw // 2
                by = ry - 13
                pygame.draw.rect(screen, INK_LIGHT, (bx, by, bw, bh), 1)
                fill = max(0, min(bw - 2, int((bw - 2) * state.soc[i])))
                pygame.draw.rect(screen, c, (bx + 1, by + 1, fill, bh - 2))

            # draw the hull outline
            if hull and len(idx) >= 3:
                pygame.draw.polygon(screen, RED, [w2s(state.pos[idx[i], 0], state.pos[idx[i], 1]) for i in hull.vertices], 1)

            # HUD
            d_from = compute_d_from_depot(state)
            peak_now = float(d_from.max()) / solo_reach
            num_dead = np.sum(state.mode == 3)
            num_alive = config.num_robots - num_dead

            hud = [
                ("V2V Swarm (Python)", big_font, INK),
                ("", font, INK),
            ]
            if paused:
                hud.append(([" PAUSED "], big_font, ACCENT))
                hud.append(("", font, INK))
            hud += [
                (f"Step:    {step_counter}", font, INK),
                (f"Speed:   {speed}x", font, INK),
                (f"V2V:     {'ON' if v2v_enabled else 'OFF'}", font, GREEN if v2v_enabled else RED),
                ("", font, INK),
                (f"Mean SoC:      {state.soc.mean():.2f}", font, INK),
                (f"Current reach: {peak_now:.2f}x", font, INK),
                (f"Mean peak reach:    {mean_peak_reach:.2f}x", font, INK),
                (f"Mean area:    {mean_area:.2f}x", font, INK),
                (f"Robots alive: {num_alive}", font, INK),
                ("", font, INK),
                ("CONTROLS", big_font, INK),
                ("SPACE  pause/run", font, INK),
                ("+/-    speed up/down", font, INK),
                ("V      toggle V2V", font, INK),
                ("R      reset", font, INK),
                ("Q/ESC  quit", font, INK),
                #(f"DEBUG MSGS: {stationary_count}", font, INK),
            ]
            y = SIM_PAD
            for text, f, color in hud:
                surf = f.render(text, color)
                screen.blit(surf, (HUD_X, y))
                y += f.line_height

            pygame.display.flip()
            clock.tick(30)

    except KeyboardInterrupt:
        # Ctrl+C in console: exit gracefully without a traceback.
        pass
    finally:
        pygame.quit()


def main():
    """Entry point. Runs the animated PyGame viewer. If PyGame is not
    installed, prints clear install instructions and exits -- does NOT
    silently fall back to a static plot, because then it's not obvious
    the animated viewer was missed.

    For headless work (parameter sweeps, statistical analysis), import
    this module and call run() and plot_run() directly; those have no
    PyGame dependency.
    """
    # Probe PyGame availability before doing any work.
    try:
        import pygame  # noqa: F401
    except ImportError:
        bar = '=' * 70
        print()
        print(bar)
        print('  This script needs PyGame for the interactive simulation viewer.')
        print()
        print('  Install it with:')
        print()
        print('      pip install pygame')
        print()
        print('  Then re-run this script to get the live animated window with')
        print('  pause / speed / V2V-toggle / reset controls.')
        print()
        print('  (For headless parameter sweeps that do not need PyGame, import')
        print('   this module and call run(config, seed) directly.)')
        print(bar)
        print()
        raise SystemExit(1)

    num_robots_list = [10, 50, 100, 150]
    num_refuelers_list = [1, 3, 5]
    radius_list = [0.25, 0.5, 0.75]
    rng = np.random.default_rng()
    seeds_list = rng.integers(0, 1000000, size=10)
    v2v_list = [False, True]
    # figs = []
    with open("output.csv", "w", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["V2V", "Seed", "Num Refuelers", "Num Robots", "Refueler Radius", "Peak Reach", "Time Past Solo per capita", "Num Stranded", "Peak Area"])
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for v2v in v2v_list:
                # tasks_done = []
                # tasks_expired = []
                for nr in num_robots_list:
                    for nrf in num_refuelers_list:
                        for radius in radius_list:
                            for seed in seeds_list:
                                cfg = Config(world_size=500.0, num_robots=nr, num_refuelers=nrf, refueler_radius=radius, alpha=0.0030, v2v_enabled=v2v)
                                futures.append(executor.submit(run, cfg, seed)) # sequential
            peak_reach_list = []
            time_past_solo = [] # per capita
            num_stranded_list = []
            peak_area_list = []
            for future in concurrent.futures.as_completed(futures):
                metrics = future.result()
                writer.writerow([metrics['config'].v2v_enabled, metrics['seed'], metrics['config'].num_refuelers, metrics['config'].num_robots, metrics['config'].refueler_radius, f"{metrics['peak_reach']:.4f}", f"{metrics['time_past_solo']/metrics['config'].num_robots:.4f}", metrics['lost'], f"{metrics['peak_area']:.4f}"])
                # include all the seed runs for each parameter configuration
                peak_reach_list.append((metrics['config'].v2v_enabled, metrics['config'].num_robots, metrics['config'].num_refuelers, metrics['config'].refueler_radius, metrics['peak_reach']))
                time_past_solo.append((metrics['config'].v2v_enabled, metrics['config'].num_robots, metrics['config'].num_refuelers, metrics['config'].refueler_radius, metrics['time_past_solo']/metrics['config'].num_robots))
                num_stranded_list.append((metrics['config'].v2v_enabled, metrics['config'].num_robots, metrics['config'].num_refuelers, metrics['config'].refueler_radius, metrics['lost']))
                peak_area_list.append((metrics['config'].v2v_enabled, metrics['config'].num_robots, metrics['config'].num_refuelers, metrics['config'].refueler_radius, metrics['peak_area']))
            # Group values by their key (parameter configuration), a dictionary mapping tuples to lists
            grouped_peak_reach_data = defaultdict(list)
            grouped_time_data = defaultdict(list)
            grouped_stranded_data = defaultdict(list)
            grouped_peak_area_data = defaultdict(list)
            for i in range(len(peak_reach_list)):
                key = (peak_reach_list[i][0], peak_reach_list[i][1], peak_reach_list[i][2], peak_reach_list[i][3])
                grouped_peak_reach_data[key].append(peak_reach_list[i][4])
                grouped_time_data[key].append(time_past_solo[i][4])
                grouped_stranded_data[key].append(num_stranded_list[i][4])
                grouped_peak_area_data[key].append(peak_area_list[i][4])
            # Calculate the average for each key = averaged over all the seed runs for each parameter configuration
            averages_peak_reach = {key: sum(values) / len(values) for key, values in grouped_peak_reach_data.items()}
            averages_time = {key: sum(values) / len(values) for key, values in grouped_time_data.items()}
            averages_stranded = {key: sum(values) / len(values) for key, values in grouped_stranded_data.items()}
            averages_peak_area = {key: sum(values) / len(values) for key, values in grouped_peak_area_data.items()}
            for key, _ in averages_peak_reach.items():
                print('V2V: ' + str(key[0]) + ', Num Robots: ' + str(key[1]) + ', Num Refuelers: ' + str(key[2]) + ', Radius: ' + str(key[3]))
                print('Peak Reach: ' + "{:.4f}".format(averages_peak_reach[key]) + ', Time Past Solo per capita: ' + "{:.4f}".format(averages_time[key]) + ', Num Stranded: ' + str(averages_stranded[key]) + ', Peak Area: ' + "{:.4f}".format(averages_peak_area[key]))
            for v2v in v2v_list:
                fig, axes = plt.subplots(len(radius_list), 2, figsize=(14, 6), subplot_kw={'projection': '3d'})
                fig.suptitle('v2v: ' + str(v2v))
                for i in range(len(radius_list)):
                    radius = radius_list[i]
                    # Create 3D plots with constant v2v and radius
                    filtered_peak_reach_averages = {k: averages_peak_reach[k] for k,_ in averages_peak_reach.items() if k[3] == radius and k[0] == v2v}
                    filtered_peak_area_averages = {k: averages_peak_area[k] for k,_ in averages_peak_area.items() if k[3] == radius and k[0] == v2v}
                    filtered_time_averages = {k: averages_time[k] for k,_ in averages_time.items() if k[3] == radius and k[0] == v2v}
                    x_peak_reach = []
                    y_peak_reach = []
                    z_peak_reach = []
                    x_time = []
                    y_time = []
                    z_time = []
                    x_peak_area = []
                    y_peak_area = []
                    z_peak_area = []
                    for key, value in filtered_peak_reach_averages.items():
                        x_peak_reach.append(key[1]) # num_robots
                        y_peak_reach.append(key[2]) # num_refuelers
                        z_peak_reach.append(value)
                    for key, value in filtered_time_averages.items():
                        x_time.append(key[1])
                        y_time.append(key[2])
                        z_time.append(value)
                    for key, value in filtered_peak_area_averages.items():
                        x_peak_area.append(key[1])
                        y_peak_area.append(key[2])
                        z_peak_area.append(value)
                    axes[i, 0].scatter(x_peak_reach, y_peak_reach, z_peak_reach, c='blue', label='Peak Reach')
                    axes[i, 0].scatter(x_peak_area, y_peak_area, z_peak_area, c='green', label='Peak Area')
                    axes[i, 0].legend()
                    axes[i, 0].set_xlabel('Num Robots')
                    axes[i, 0].set_ylabel('Num Refuelers')
                    axes[i, 0].set_zlabel('x solo reach or world')
                    axes[i, 0].set_title(f'radius={radius}')
                    #fig.colorbar(plot1, ax=axes[i, 0], shrink=0.5, aspect=10, pad=0.1)
                    plot2 = axes[i, 1].scatter(x_time, y_time, z_time, c=z_time, cmap='viridis', s=40)
                    axes[i, 1].set_xlabel('Num Robots')
                    axes[i, 1].set_ylabel('Num Refuelers')
                    axes[i, 1].set_zlabel('Time Past Solo per capita (s)')
                    axes[i, 1].set_title(f'radius={radius}')
                    fig.colorbar(plot2, ax=axes[i, 1], shrink=0.5, aspect=10, pad=0.1)
                plt.tight_layout()
                plt.show()

                # fig, axes = plt.subplots(2, 2, figsize=(11, 8))
                # fig.suptitle('v2v: ' + str(v2v))
                # ax_mpr = axes[0,0]
                # ax_tps = axes[0,1]
                # ax_td = axes[1,0]
                # ax_te = axes[1,1]
                # ax_mpr.plot(num_robots_list, mean_peak_reach)
                # ax_mpr.set_xlabel('fleet size')
                # ax_mpr.set_ylabel('mean peak reach')
                # ax_mpr.set_title('mean peak reach')
                # ax_tps.plot(num_robots_list, time_past_solo)
                # ax_tps.set_xlabel('fleet size')
                # ax_tps.set_ylabel('time past solo')
                # ax_tps.set_title('time past solo')
                # ax_td.plot(num_robots_list, tasks_done)
                # ax_td.set_xlabel('fleet size')
                # ax_td.set_ylabel('tasks done')
                # ax_td.set_title('tasks done')
                # ax_te.plot(num_robots_list, tasks_expired)
                # ax_te.set_xlabel('fleet size')
                # ax_te.set_ylabel('tasks expired')
                # ax_te.set_title('tasks expired')
                # figs.append((fig, axes))
                # plt.tight_layout()
                # plt.show()
                # fig = plt.figure(figsize=(14, 6))
                # ax1 = fig.add_subplot(1, 2, 1, projection='3d')
                # scatter = ax1.scatter(x, y, z, c=z, cmap='viridis', s=40)

        # plt.tight_layout()
        # plt.show()

def main_anim():
    """Entry point. Runs the animated PyGame viewer. If PyGame is not
    installed, prints clear install instructions and exits -- does NOT
    silently fall back to a static plot, because then it's not obvious
    the animated viewer was missed.

    For headless work (parameter sweeps, statistical analysis), import
    this module and call run() and plot_run() directly; those have no
    PyGame dependency.
    """
    # Probe PyGame availability before doing any work.
    try:
        import pygame  # noqa: F401
    except ImportError:
        bar = '=' * 70
        print()
        print(bar)
        print('  This script needs PyGame for the interactive simulation viewer.')
        print()
        print('  Install it with:')
        print()
        print('      pip install pygame')
        print()
        print('  Then re-run this script to get the live animated window with')
        print('  pause / speed / V2V-toggle / reset controls.')
        print()
        print('  (For headless parameter sweeps that do not need PyGame, import')
        print('   this module and call run(config, seed) directly.)')
        print(bar)
        print()
        raise SystemExit(1)

    cfg = Config(world_size=500.0, num_robots=100, num_refuelers=5, refueler_radius=0.3, alpha=0.0030, v2v_enabled=True)
    run_animated(cfg, seed=632145)


if __name__ == '__main__':
    main()

