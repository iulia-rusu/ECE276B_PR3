from dataclasses import dataclass
import numpy as np
import utils


@dataclass
class GpiConfig:
    traj: callable
    obstacles: np.ndarray
    ex_space: np.ndarray
    ey_space: np.ndarray
    eth_space: np.ndarray
    v_space: np.ndarray
    w_space: np.ndarray
    Q: np.ndarray
    q: float
    R: np.ndarray
    gamma: float
    num_evals: int
    collision_margin: float


class GPI:
    def __init__(self, config: GpiConfig):
        self.config = config
        self.nt = utils.T
        self.nx = len(config.ex_space)
        self.ny = len(config.ey_space)
        self.nth = len(config.eth_space)
        self.nv = len(config.v_space)
        self.nw = len(config.w_space)
        self.dt = utils.time_step

        # will later be filed in by compute_policy fxn
        self.V = None             # value function
        self.policy = None        # best action 
        self.stage_costs = None 
        self.next_state_x = None  
        self.next_state_y = None
        self.next_state_th = None
        self.trans_probs = None   # transition probabilities

    def __call__(self, t, cur_state, cur_ref_state):
        # compute error state between car and reference
        error = cur_state - np.array(cur_ref_state)
        error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))

        # find nearest grid point and look up precomputed policy
        t_idx, ix, iy, ith = self.state_metric_to_index(
            np.array([t % self.nt, error[0], error[1], error[2]])
        )
        iv = self.policy[t_idx, ix, iy, ith, 0]
        iw = self.policy[t_idx, ix, iy, ith, 1]
        return np.array([self.config.v_space[iv], self.config.w_space[iw]])

    def state_metric_to_index(self, metric_state):
        # convert continuous (t, ex, ey, eth) to nearest grid indices
        t_idx = int(metric_state[0]) % self.nt
        ix = int(np.argmin(np.abs(self.config.ex_space - metric_state[1])))
        iy = int(np.argmin(np.abs(self.config.ey_space - metric_state[2])))
        ith = int(np.argmin(np.abs(self.config.eth_space - metric_state[3])))
        return t_idx, ix, iy, ith

    def state_index_to_metric(self, state_index):
        t_idx, ix, iy, ith = state_index
        return np.array([
            float(t_idx),
            self.config.ex_space[ix],
            self.config.ey_space[iy],
            self.config.eth_space[ith],
        ])

    def control_metric_to_index(self, control_metric):
        iv = np.digitize(control_metric[0], self.config.v_space, right=True)
        iw = np.digitize(control_metric[1], self.config.w_space, right=True)
        return iv, iw

    def control_index_to_metric(self, iv, iw):
        return self.config.v_space[iv], self.config.w_space[iw]

    def compute_stage_costs(self):
        # precompute L[t, ix, iy, ith, iv, iw] for all state-action combinations
        cfg = self.config
        nt, nx, ny, nth, nv, nw = self.nt, self.nx, self.ny, self.nth, self.nv, self.nw

        # reference positions over the trajectory period
        refs = []
        for t in range(nt):
            refs.append(cfg.traj(t))
        refs = np.array(refs)
        ref_x = refs[:, 0].reshape(nt, 1, 1, 1)
        ref_y = refs[:, 1].reshape(nt, 1, 1, 1)

        # broadcast error grids to (1, nx, 1, 1), (1, 1, ny, 1), (1, 1, 1, nth)
        ex_grid = cfg.ex_space[None, :, None, None]
        ey_grid = cfg.ey_space[None, None, :, None]
        eth_grid = cfg.eth_space[None, None, None, :]

        # tracking cost: p_tilde^T Q p_tilde + q(1 - cos(theta_tilde))
        Q = cfg.Q
        tracking_cost = (
            Q[0, 0] * ex_grid**2
            + 2 * Q[0, 1] * ex_grid * ey_grid
            + Q[1, 1] * ey_grid**2
            + cfg.q * (1 - np.cos(eth_grid))
        )

        # collision penalty: actual position = error + reference position
        actual_x = ex_grid + ref_x
        actual_y = ey_grid + ref_y
        robot_r = 0.3
        in_collision = np.zeros((nt, nx, ny, nth), dtype=bool)
        for obs in cfg.obstacles:
            obs_x, obs_y, obs_r = obs
            dist_sq = (actual_x - obs_x)**2 + (actual_y - obs_y)**2
            in_collision = in_collision | (dist_sq < (obs_r + robot_r)**2)
        in_collision = in_collision | (np.abs(actual_x) > 3.0) | (np.abs(actual_y) > 3.0)

        collision_penalty = in_collision.astype(np.float32) * cfg.collision_margin

        # control cost: u^T R u
        R = cfg.R
        v_grid, w_grid = np.meshgrid(cfg.v_space, cfg.w_space, indexing='ij')
        control_cost = R[0, 0]*v_grid**2 + 2*R[0, 1]*v_grid*w_grid + R[1, 1]*w_grid**2

        # combine into full (nt, nx, ny, nth, nv, nw) array
        state_part = (tracking_cost + collision_penalty).astype(np.float32).reshape(nt, nx, ny, nth, 1, 1)
        ctrl_part = control_cost.astype(np.float32).reshape(1, 1, 1, 1, nv, nw)
        self.stage_costs = state_part + ctrl_part

    def compute_transition_matrix(self):
        # for each (t, state, control), find which grid cell the car lands in
        cfg = self.config
        nt, nx, ny, nth, nv, nw = self.nt, self.nx, self.ny, self.nth, self.nv, self.nw
        dt = self.dt

        self.next_state_x = np.zeros((nt, nx, ny, nth, nv, nw, 8), dtype=np.uint8)
        self.next_state_y = np.zeros((nt, nx, ny, nth, nv, nw, 8), dtype=np.uint8)
        self.next_state_th = np.zeros((nt, nx, ny, nth, nv, nw, 8), dtype=np.uint8)
        self.trans_probs = np.zeros((nt, nx, ny, nth, nv, nw, 8), dtype=np.float32)

        refs = []
        refs_next = []
        for t in range(nt):
            refs.append(cfg.traj(t))
            refs_next.append(cfg.traj(t + 1))
        refs = np.array(refs)
        refs_next = np.array(refs_next)

        v_grid, w_grid = np.meshgrid(cfg.v_space, cfg.w_space, indexing='ij')

        for t_idx in range(nt):
            if t_idx % 10 == 0:
                print(f"transition matrix: t={t_idx}/{nt}", flush=True)

            ref_heading = refs[t_idx, 2]
            ref_x_shift = refs[t_idx, 0] - refs_next[t_idx, 0]
            ref_y_shift = refs[t_idx, 1] - refs_next[t_idx, 1]
            ref_th_shift = refs[t_idx, 2] - refs_next[t_idx, 2]

            # broadcast state and control grids to (nx, ny, nth, nv, nw)
            ex_grid = cfg.ex_space[:, None, None, None, None]
            ey_grid = cfg.ey_space[None, :, None, None, None]
            eth_grid = cfg.eth_space[None, None, :, None, None]
            v_vals = v_grid[None, None, None, :, :]
            w_vals = w_grid[None, None, None, :, :]

            # integration error dynamics (eq 2), no noise
            half_angle = w_vals * dt / 2
            sinc_val = np.where(np.abs(half_angle) < 1e-6,
                                1.0 - half_angle**2 / 6.0,
                                np.sin(half_angle) / half_angle)
            direction = eth_grid + ref_heading + half_angle
            ex_next = ex_grid + dt * sinc_val * np.cos(direction) * v_vals + ref_x_shift
            ey_next = ey_grid + dt * sinc_val * np.sin(direction) * v_vals + ref_y_shift
            eth_next = eth_grid + dt * w_vals + ref_th_shift
            eth_next = np.arctan2(np.sin(eth_next), np.cos(eth_next))

            full_shape = (nx, ny, nth, nv, nw)
            ex_next = np.broadcast_to(ex_next, full_shape).copy()
            ey_next = np.broadcast_to(ey_next, full_shape).copy()
            eth_next = np.broadcast_to(eth_next, full_shape).copy()

            # find nearest grid point for each next state
            # (deterministic transition avoids collision-state value bleeding)
            ix_lower = np.clip(np.searchsorted(cfg.ex_space, ex_next.ravel(), side='right').reshape(full_shape) - 1, 0, nx-2)
            dist_lo  = np.abs(ex_next - cfg.ex_space[ix_lower])
            dist_hi  = np.abs(ex_next - cfg.ex_space[ix_lower + 1])
            ix_nearest = np.where(dist_lo <= dist_hi, ix_lower, ix_lower + 1).astype(np.uint8)

            iy_lower = np.clip(np.searchsorted(cfg.ey_space, ey_next.ravel(), side='right').reshape(full_shape) - 1, 0, ny-2)
            dist_lo  = np.abs(ey_next - cfg.ey_space[iy_lower])
            dist_hi  = np.abs(ey_next - cfg.ey_space[iy_lower + 1])
            iy_nearest = np.where(dist_lo <= dist_hi, iy_lower, iy_lower + 1).astype(np.uint8)

            ith_lower = np.clip(np.searchsorted(cfg.eth_space, eth_next.ravel(), side='right').reshape(full_shape) - 1, 0, nth-2)
            dist_lo   = np.abs(eth_next - cfg.eth_space[ith_lower])
            dist_hi   = np.abs(eth_next - cfg.eth_space[(ith_lower + 1) % nth])
            ith_nearest = np.where(dist_lo <= dist_hi, ith_lower, (ith_lower + 1) % nth).astype(np.uint8)

            # slot 0 gets all the probability, slots 1-7 stay zero
            near_x = np.zeros(full_shape + (8,), dtype=np.uint8)
            near_y = np.zeros(full_shape + (8,), dtype=np.uint8)
            near_th = np.zeros(full_shape + (8,), dtype=np.uint8)
            probs = np.zeros(full_shape + (8,), dtype=np.float32)
            near_x[:, :, :, :, :, 0] = ix_nearest
            near_y[:, :, :, :, :, 0] = iy_nearest
            near_th[:, :, :, :, :, 0] = ith_nearest
            probs[:, :, :, :, :, 0] = 1.0

            self.next_state_x[t_idx] = near_x
            self.next_state_y[t_idx] = near_y
            self.next_state_th[t_idx] = near_th
            self.trans_probs[t_idx] = probs

    def init_value_function(self):
        # start with zero value everywhere and a neutral default policy
        self.V = np.zeros((self.nt, self.nx, self.ny, self.nth), dtype=np.float32)
        self.policy = np.zeros((self.nt, self.nx, self.ny, self.nth, 2), dtype=np.int16)
        self.policy[:, :, :, :, 0] = self.nv // 2
        self.policy[:, :, :, :, 1] = int(np.argmin(np.abs(self.config.w_space)))

    @utils.timer
    def policy_evaluation(self):
        # Bellman backup: V(t,e) = L(t,e,pi) + gamma * V(t+1, e')
        cfg = self.config
        nt, nx, ny, nth = self.nt, self.nx, self.ny, self.nth

        iv_sel = self.policy[:, :, :, :, 0]
        iw_sel = self.policy[:, :, :, :, 1]

        t_idx = np.arange(nt)[:, None, None, None]
        x_idx = np.arange(nx)[None, :, None, None]
        y_idx = np.arange(ny)[None, None, :, None]
        th_idx = np.arange(nth)[None, None, None, :]

        # look up stage cost and transitions for the current policy
        cost_pi = self.stage_costs[t_idx, x_idx, y_idx, th_idx, iv_sel, iw_sel]
        next_ix = self.next_state_x[t_idx, x_idx, y_idx, th_idx, iv_sel, iw_sel, :]
        next_iy = self.next_state_y[t_idx, x_idx, y_idx, th_idx, iv_sel, iw_sel, :]
        next_ith = self.next_state_th[t_idx, x_idx, y_idx, th_idx, iv_sel, iw_sel, :]
        next_p = self.trans_probs[t_idx, x_idx, y_idx, th_idx, iv_sel, iw_sel, :]

        t_next = ((np.arange(nt) + 1) % nt)[:, None, None, None, None]
        V_future = self.V[t_next, next_ix, next_iy, next_ith]
        exp_future = np.sum(next_p * V_future, axis=-1)
        self.V[:] = (cost_pi + cfg.gamma * exp_future).astype(np.float32)

    @utils.timer
    def policy_improvement(self):
        """policy improvment: pick the control that minimizes Q = L + gamma * V(next state)
        """
        
        cfg = self.config
        nt, nx, ny, nth, nv, nw = self.nt, self.nx, self.ny, self.nth, self.nv, self.nw

        t_next = ((np.arange(nt) + 1) % nt)[:, None, None, None, None, None, None]
        V_future = self.V[t_next, self.next_state_x, self.next_state_y, self.next_state_th]
        exp_future = np.sum(self.trans_probs * V_future, axis=-1)
        Q_values = self.stage_costs + cfg.gamma * exp_future

        Q_flat = Q_values.reshape(nt, nx, ny, nth, nv * nw)
        best_action = np.argmin(Q_flat, axis=-1)
        self.policy[:, :, :, :, 0] = (best_action // nw).astype(np.int16)
        self.policy[:, :, :, :, 1] = (best_action % nw).astype(np.int16)

    def compute_policy(self, num_iters):
        print("Initialising value function...")
        self.init_value_function()

        print("Computing stage costs...")
        self.compute_stage_costs()

        print("Computing transition matrix...")
        self.compute_transition_matrix()

        print(f"Running GPI for {num_iters} iterations...")
        for i in range(num_iters):
            print(f"  Iteration {i + 1}/{num_iters}")
            for i in range(self.config.num_evals):
                self.policy_evaluation()
            self.policy_improvement()
        print("Policy computation complete.")
