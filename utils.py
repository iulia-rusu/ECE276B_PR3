import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from matplotlib import animation
from time import time
from tqdm import tqdm

v_max = 1
v_min = 0.1
w_max = 1
w_min = -1
time_step = 0.5  # time between steps in seconds
T = 100
sim_time = 120  # simulation time
sigma = np.array([0.04, 0.04, 0.004])

# Reference trajectory parameters
A = 2.0
B = 2.0


# This function returns the reference point at time step k
def lemniscate(k):
    """Reference trajectory: x = A sin(at), y = B sin(2at), periodic with period T."""
    k = k % T
    t = k * time_step
    a = 2 * np.pi / (T * time_step)
    xref = A * np.sin(a * t)
    yref = B * np.sin(2 * a * t)
    vx = A * a * np.cos(a * t)
    vy = 2 * B * a * np.cos(2 * a * t)
    thetaref = np.arctan2(vy, vx)
    return [xref, yref, thetaref]


# Initial state on the reference trajectory at k=2
_x0, _y0, _th0 = lemniscate(2)
x_init = _x0
y_init = _y0
theta_init = _th0


# This function implements a simple P controller
def simple_controller(cur_state, ref_state):
    k_v = 0.55
    k_w = 1.0
    v = k_v * np.sqrt((cur_state[0] - ref_state[0]) ** 2 + (cur_state[1] - ref_state[1]) ** 2)
    v = np.clip(v, v_min, v_max)
    angle_diff = ref_state[2] - cur_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    w = k_w * angle_diff
    w = np.clip(w, w_min, w_max)
    return [v, w]


# This function implement the car dynamics
def car_next_state(time_step, cur_state, control, noise=True):
    theta = cur_state[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    f = rot_3d_z @ control
    w_xy = np.random.normal(0, sigma[0], 2)
    w_theta = np.random.normal(0, sigma[2], 1)
    w = np.concatenate((w_xy, w_theta))
    if noise:
        return cur_state + time_step * f.flatten() + w
    else:
        return cur_state + time_step * f.flatten()


def visualize(car_states, ref_traj, obstacles, t, time_step, save=False, name="result"):
    init_state = car_states[0, :]

    def create_triangle(state=[0, 0, 0], h=0.5, w=0.25, update=False):
        x, y, th = state
        triangle = np.array([[h, 0], [0, w / 2], [0, -w / 2], [h, 0]]).T
        rotation_matrix = np.array([[cos(th), -sin(th)], [sin(th), cos(th)]])

        coords = np.array([[x, y]]) + (rotation_matrix @ triangle).T
        if update == True:
            return coords
        else:
            return coords[:3, :]

    def init():
        return (
            path,
            current_state,
            target_state,
        )

    def animate(i):
        # get variables
        x = car_states[i, 0]
        y = car_states[i, 1]
        th = car_states[i, 2]

        # update path
        if i == 0:
            path.set_data(np.array([]), np.array([]))
        x_new = np.hstack((path.get_xdata(), x))
        y_new = np.hstack((path.get_ydata(), y))
        path.set_data(x_new, y_new)

        # update horizon
        # x_new = car_states[0, :, i]
        # y_new = car_states[1, :, i]
        # horizon.set_data(x_new, y_new)

        # update current_state
        current_state.set_xy(create_triangle([x, y, th], update=True))

        # update current_target
        x_ref = ref_traj[i, 0]
        y_ref = ref_traj[i, 1]
        th_ref = ref_traj[i, 2]
        target_state.set_xy(create_triangle([x_ref, y_ref, th_ref], update=True))

        # update target_state
        # xy = target_state.get_xy()
        # target_state.set_xy(xy)

        return (
            path,
            current_state,
            target_state,
        )

    circles = []
    for obs in obstacles:
        circles.append(plt.Circle((obs[0], obs[1]), obs[2], color="r", alpha=0.5))
    # create figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))
    min_scale_x = min(init_state[0], np.min(ref_traj[:, 0])) - 1.5
    max_scale_x = max(init_state[0], np.max(ref_traj[:, 0])) + 1.5
    min_scale_y = min(init_state[1], np.min(ref_traj[:, 1])) - 1.5
    max_scale_y = max(init_state[1], np.max(ref_traj[:, 1])) + 1.5
    ax.set_xlim(left=min_scale_x, right=max_scale_x)
    ax.set_ylim(bottom=min_scale_y, top=max_scale_y)
    for circle in circles:
        ax.add_patch(circle)
    # create lines:
    #   path
    (path,) = ax.plot([], [], "k", linewidth=2)

    #   current_state
    current_triangle = create_triangle(init_state[:3])
    current_state = ax.fill(current_triangle[:, 0], current_triangle[:, 1], color="r")
    current_state = current_state[0]
    #   target_state
    target_triangle = create_triangle(ref_traj[0, 0:3])
    target_state = ax.fill(target_triangle[:, 0], target_triangle[:, 1], color="b")
    target_state = target_state[0]

    #   reference trajectory
    ax.scatter(ref_traj[:, 0], ref_traj[:, 1], marker="x")

    sim = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=len(t),
        interval=time_step * 100,
        blit=True,
        repeat=True,
    )
    plt.show()

    if save == True:
        sim.save("./fig/" + name + "_result.gif", writer="ffmpeg", fps=15)

    return


def timer(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        tqdm.write(f"{func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func

