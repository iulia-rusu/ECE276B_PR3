from dataclasses import dataclass
import numpy as np
import utils


@dataclass
class GpiGaussianConfig:
    obstacles: np.ndarray
    traj: callable
    Q: np.ndarray
    R: np.ndarray
    q: float
    gamma: float
    collision_margin: float
    num_evals: int
    ex_space: np.ndarray
    ey_space: np.ndarray
    eth_space: np.ndarray
    v_space: np.ndarray
    w_space: np.ndarray


class GPIGaussian:
    # noise std devs from eq (1) of the PDF: sigma = [0.04, 0.04, 0.004]
    SIGMA = np.array([0.04, 0.04, 0.004])

    def __init__(self, config: GpiGaussianConfig):
        self.config = config
        self.nt = utils.T
        self.nx = len(config.ex_space)
        self.ny = len(config.ey_space)
        self.nth = len(config.eth_space)
        self.nv = len(config.v_space)
        self.nw = len(config.w_space)
        self.dt = utils.time_step

        self.value_function = None
        self.policy = None
        self.stage_costs = None
        self.next_state_x = None
        self.next_state_y = None
        self.next_state_th = None
        self.transition_probs = None

    def __call__(self, t, cur_state, cur_ref_state):
        assert self.policy is not None, "Policy has not been computed yet. Call compute_policy first."
        error = cur_state - np.array(cur_ref_state)
        error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
        t_idx, ix, iy, ith = self.state_metric_to_index(
            np.array([t % self.nt, error[0], error[1], error[2]])
        )
        iv = self.policy[t_idx, ix, iy, ith, 0]
        iw = self.policy[t_idx, ix, iy, ith, 1]
        return np.array([self.config.v_space[iv], self.config.w_space[iw]])

    def state_metric_to_index(self, metric_state):
        t_idx = int(metric_state[0]) % self.nt
        ix = int(np.argmin(np.abs(self.config.ex_space - metric_state[1])))
        iy = int(np.argmin(np.abs(self.config.ey_space - metric_state[2])))
        ith = int(np.argmin(np.abs(self.config.eth_space - metric_state[3])))
        return t_idx, ix, iy, ith

    def compute_stage_costs(self):
        config = self.config
        nt, nx, ny, nth = self.nt, self.nx, self.ny, self.nth
        nv, nw = self.nv, self.nw
        robot_r = 0.3

        # error grids over state space (nx, ny, nth)
        ex_grid = config.ex_space[:, None, None]
        ey_grid = config.ey_space[None, :, None]
        eth_grid = config.eth_space[None, None, :]

        # tracking cost depends only on error, not time
        Q = config.Q
        tracking_cost = (
            Q[0, 0] * ex_grid**2
            + 2 * Q[0, 1] * ex_grid * ey_grid
            + Q[1, 1] * ey_grid**2
            + config.q * (1 - np.cos(eth_grid))
        )

        # control cost depends only on action, not state or time
        R = config.R
        v_grid, w_grid = np.meshgrid(config.v_space, config.w_space, indexing='ij')
        control_cost = R[0, 0]*v_grid**2 + 2*R[0, 1]*v_grid*w_grid + R[1, 1]*w_grid**2

        self.stage_costs = np.zeros((nt, nx, ny, nth, nv, nw), dtype=np.float64)

        for t in range(nt):
            ref = config.traj(t)
            actual_x = ex_grid + ref[0]
            actual_y = ey_grid + ref[1]

            # collision: robot overlaps obstacle or leaves free space
            in_collision = np.zeros((nx, ny, nth), dtype=bool)
            for obs in config.obstacles:
                obs_x, obs_y, obs_r = obs
                dist_sq = (actual_x - obs_x)**2 + (actual_y - obs_y)**2
                in_collision |= dist_sq < (obs_r + robot_r)**2
            in_collision |= (np.abs(actual_x) > 3.0) | (np.abs(actual_y) > 3.0)

            collision_penalty = in_collision.astype(np.float64) * config.collision_margin
            state_cost = tracking_cost + collision_penalty  # (nx, ny, nth)

            # combine state and control costs into full (nx, ny, nth, nv, nw) array
            self.stage_costs[t] = (
                state_cost[:, :, :, None, None] + control_cost[None, None, None, :, :]
            )

    def compute_transition_matrix(self):
        config = self.config
        nt, nx, ny, nth = self.nt, self.nx, self.ny, self.nth
        nv, nw = self.nv, self.nw
        dt = self.dt
        sigma = self.SIGMA

        # 8 neighbor offsets: corners of the grid cell containing the mean next state
        neighbors_8 = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ])

        self.next_state_x = np.zeros((nt, nx, ny, nth, nv, nw, 8), dtype=np.uint8)
        self.next_state_y = np.zeros((nt, nx, ny, nth, nv, nw, 8), dtype=np.uint8)
        self.next_state_th = np.zeros((nt, nx, ny, nth, nv, nw, 8), dtype=np.uint8)
        self.transition_probs = np.zeros((nt, nx, ny, nth, nv, nw, 8), dtype=np.float32)

        refs = np.array([config.traj(t) for t in range(nt)])
        refs_next = np.array([config.traj(t + 1) for t in range(nt)])

        v_grid, w_grid = np.meshgrid(config.v_space, config.w_space, indexing='ij')

        for t_idx in range(nt):
            # if t_idx % 10 == 0:
            #     print(f"Computing transition matrix for time step {t_idx} out of {nt}", flush=True)

            ref_heading = refs[t_idx, 2]
            ref_x_shift = refs[t_idx, 0] - refs_next[t_idx, 0]
            ref_y_shift = refs[t_idx, 1] - refs_next[t_idx, 1]
            ref_th_shift = refs[t_idx, 2] - refs_next[t_idx, 2]

            ex_g = config.ex_space[:, None, None, None, None]
            ey_g = config.ey_space[None, :, None, None, None]
            eth_g = config.eth_space[None, None, :, None, None]
            v_vals = v_grid[None, None, None, :, :]
            w_vals = w_grid[None, None, None, :, :]

            # mean next error state with zero noise (eq 2 of PDF)
            half_angle = w_vals * dt / 2
            sinc_val = np.where(
                np.abs(half_angle) < 1e-6,
                1.0 - half_angle**2 / 6.0,
                np.sin(half_angle) / half_angle,
            )
            direction = eth_g + ref_heading + half_angle
            mean_ex_next = ex_g + dt * sinc_val * np.cos(direction) * v_vals + ref_x_shift
            mean_ey_next = ey_g + dt * sinc_val * np.sin(direction) * v_vals + ref_y_shift
            mean_eth_next = eth_g + dt * w_vals + ref_th_shift
            mean_eth_next = np.arctan2(np.sin(mean_eth_next), np.cos(mean_eth_next))

            full_shape = (nx, ny, nth, nv, nw)
            mean_ex_next = np.broadcast_to(mean_ex_next, full_shape).copy()
            mean_ey_next = np.broadcast_to(mean_ey_next, full_shape).copy()
            mean_eth_next = np.broadcast_to(mean_eth_next, full_shape).copy()

            # lower-bound indices for each predicted next state
            ix_lo = np.clip(
                np.searchsorted(config.ex_space, mean_ex_next.ravel(), side='right').reshape(full_shape) - 1,
                0, nx - 2,
            ) #finds where mean_ex_next would be instered into config.ex_space
            iy_lo = np.clip(
                np.searchsorted(config.ey_space, mean_ey_next.ravel(), side='right').reshape(full_shape) - 1,
                0, ny - 2,
            )
            ith_lo = np.clip(
                np.searchsorted(config.eth_space, mean_eth_next.ravel(), side='right').reshape(full_shape) - 1,
                0, nth - 2,
            )

            near_x = np.empty(full_shape + (8,), dtype=np.uint8)
            near_y = np.empty(full_shape + (8,), dtype=np.uint8)
            near_th = np.empty(full_shape + (8,), dtype=np.uint8)
            log_w = np.empty(full_shape + (8,), dtype=np.float64)

            for k, (di, dj, dk) in enumerate(neighbors_8):
                ni_x = np.clip(ix_lo + di, 0, nx - 1).astype(np.uint8)
                ni_y = np.clip(iy_lo + dj, 0, ny - 1).astype(np.uint8)
                ni_th = ((ith_lo + dk) % nth).astype(np.uint8)

                near_x[:, :, :, :, :, k] = ni_x
                near_y[:, :, :, :, :, k] = ni_y
                near_th[:, :, :, :, :, k] = ni_th

                # displacement from mean to this grid point
                dx = config.ex_space[ni_x] - mean_ex_next
                dy = config.ey_space[ni_y] - mean_ey_next
                dth = config.eth_space[ni_th] - mean_eth_next
                dth = np.arctan2(np.sin(dth), np.cos(dth))

                # computing log-likelihood of each neighbor
                log_w[:, :, :, :, :, k] = -(
                    (dx / sigma[0])**2
                    + (dy / sigma[1])**2
                    + (dth / sigma[2])**2
                ) / 2.0

            # normalize
            log_w -= log_w.max(axis=-1, keepdims=True)
            weights = np.exp(log_w)
            weights /= weights.sum(axis=-1, keepdims=True)

            self.next_state_x[t_idx] = near_x
            self.next_state_y[t_idx] = near_y
            self.next_state_th[t_idx] = near_th
            self.transition_probs[t_idx] = weights.astype(np.float32)

    def init_value_function(self):
        self.value_function = np.zeros((self.nt, self.nx, self.ny, self.nth), dtype=np.float64)
        self.policy = np.zeros((self.nt, self.nx, self.ny, self.nth, 2), dtype=np.int16)
        # start with a moderate forward speed and no turning
        self.policy[:, :, :, :, 0] = int(np.argmin(np.abs(self.config.v_space - 0.5)))
        self.policy[:, :, :, :, 1] = int(np.argmin(np.abs(self.config.w_space)))

    @utils.timer
    def policy_evaluation(self):
        # implements the Bellman backup 
        config = self.config
        nt, nx, ny, nth = self.nt, self.nx, self.ny, self.nth

        x_idx = np.arange(nx)[:, None, None]
        y_idx = np.arange(ny)[None, :, None]
        th_idx = np.arange(nth)[None, None, :]

        for t in range(nt):
            t_next = (t + 1) % nt
            iv_sel = self.policy[t, :, :, :, 0]
            iw_sel = self.policy[t, :, :, :, 1]

            cost_pi = self.stage_costs[t, x_idx, y_idx, th_idx, iv_sel, iw_sel]
            next_ix= self.next_state_x[t, x_idx, y_idx, th_idx, iv_sel, iw_sel, :]
            next_iy = self.next_state_y[t, x_idx, y_idx, th_idx, iv_sel, iw_sel, :]
            next_ith = self.next_state_th[t, x_idx, y_idx, th_idx, iv_sel, iw_sel, :]
            next_p = self.transition_probs[t, x_idx, y_idx, th_idx, iv_sel, iw_sel, :]

            V_future = self.value_function[t_next, next_ix, next_iy, next_ith]
            exp_future = np.sum(next_p * V_future, axis=-1)
            self.value_function[t] = cost_pi + config.gamma * exp_future

    @utils.timer
    def policy_improvement(self):
        # pick the action that minimizes Q(t, e, u) = L(t, e, u) + gamma * E[V(t+1, e')]
        config = self.config
        nt, nx, ny, nth = self.nt, self.nx, self.ny, self.nth
        nv, nw = self.nv, self.nw

        for t in range(nt):
            t_next = (t + 1) % nt
            V_future = self.value_function[t_next, self.next_state_x[t], self.next_state_y[t], self.next_state_th[t]]
            exp_future = np.sum(self.transition_probs[t] * V_future, axis=-1)
            Q_values = self.stage_costs[t] + config.gamma * exp_future  # (nx, ny, nth, nv, nw)

            best_action = np.argmin(Q_values.reshape(nx, ny, nth, nv * nw), axis=-1)
            self.policy[t, :, :, :, 0] = (best_action // nw).astype(np.int16)
            self.policy[t, :, :, :, 1] = (best_action % nw).astype(np.int16)

    def compute_policy(self, num_iters):
        print("Computing stage costs for all states and actions.")
        self.compute_stage_costs()

        print("Computing transition matrix using Gaussian noise model.")
        self.compute_transition_matrix()

        print("Initialising value function and policy.")
        self.init_value_function()

        print(f"Running generalized policy iteration for {num_iters} iterations.")
        for i in range(num_iters):
            print(f"Iteration {i + 1} out of {num_iters}")
            for _ in range(self.config.num_evals):
                self.policy_evaluation()
            self.policy_improvement()
        print("Policy computation is complete.")
