from time import time
import numpy as np
import utils
from mujoco_car import MujocoCarSim
from cec import CEC
from gpi import GPI, GpiConfig
from gpi_gaussian import GPIGaussian, GpiGaussianConfig


use_mujoco   = False
control_alg  = "gpi_gaussian"   # "cec",  or "gpi_gaussian"


def main():
    # Obstacles in the environment (x, y, radius)
    obstacles = np.array([
        [2.35,  0.95, 0.5],
        [-2.35, -0.95, 0.5],
        [1.0,   0.0,  0.5],
        [-1.0,  0.0,  0.5],
    ])

    if control_alg == "cec":
        controller = CEC(
            horizon=10,
            obstacles=obstacles,
            Q=np.diag([1.0, 1.0]),
            R=np.diag([0.1, 0.1]),
            q=1.0,
        )
        control_fn = controller.get_control

    elif control_alg == "gpi_gaussian":
        # adaptive grid with finer spacing near the reference and coarser spacing further away
        fine_pos = np.linspace(-0.6, 0.6, 13)
        mid_pos = np.array([-1.8, -1.5, -1.2, -0.9, 0.9, 1.2, 1.5, 1.8])
        edge_pos = np.array([-3.0, -2.5, -2.0, 2.0, 2.5, 3.0])
        ex_space = np.unique(np.concatenate([edge_pos, mid_pos, fine_pos]))
        ey_space = ex_space.copy()
        eth_space = np.linspace(-np.pi, np.pi, 21, endpoint=False)
        v_space = np.array([0.1, 0.3, 0.55, 0.775, 1.0])
        w_space = np.linspace(utils.w_min, utils.w_max, 9)

        config = GpiGaussianConfig(
            obstacles=obstacles,
            traj=utils.lemniscate,
            Q=np.diag([10.0, 10.0]),
            R=np.diag([0.01, 0.01]),
            q=5.0,
            gamma=0.99,
            collision_margin=1000,
            num_evals=10,
            ex_space=ex_space,
            ey_space=ey_space,
            eth_space=eth_space,
            v_space=v_space,
            w_space=w_space,
        )
        controller = GPIGaussian(config)
        controller.compute_policy(num_iters=100)
        control_fn = controller


    # Params
    traj = utils.lemniscate
    ref_traj = []
    error_trans = 0.0
    error_rot = 0.0
    car_states = []
    times = []

    # Start main loop
    main_loop = time()  # return time in sec

    # Initialize state
    cur_state = np.array([utils.x_init, utils.y_init, utils.theta_init])
    cur_iter = 0

    # Initialize MuJoCo simulation environment
    mujoco_sim = None
    if use_mujoco:
        mujoco_sim = MujocoCarSim()

    # Main loop
    while cur_iter * utils.time_step < utils.sim_time:
        t1 = time()
        # Get reference state
        cur_time = cur_iter * utils.time_step
        cur_ref = traj(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        # TODO: Replace this simple controller with your own controller
        # control = utils.simple_controller(cur_state, cur_ref) #cec?
        control = control_fn(cur_iter, cur_state, cur_ref)
        print("[v,w]", control)
        ################################################################

        # Apply control input
        if use_mujoco:
            next_state = mujoco_sim.car_next_state(control)
        else:
            next_state = utils.car_next_state(utils.time_step, cur_state, control, noise=True)

        # Update current state
        cur_state = next_state
        # Loop time
        t2 = utils.time()
        print(cur_iter)
        print(t2 - t1)
        times.append(t2 - t1)
        cur_err = cur_state - cur_ref
        cur_err[2] = np.arctan2(np.sin(cur_err[2]), np.cos(cur_err[2]))
        error_trans = error_trans + np.linalg.norm(cur_err[:2])
        error_rot = error_rot + np.abs(cur_err[2])
        print(cur_err, error_trans, error_rot)
        print("======================")
        cur_iter = cur_iter + 1

    main_loop_time = time()
    print("\n\n")
    print("Total time: ", main_loop_time - main_loop)
    print("Average iteration time: ", np.array(times).mean() * 1000, "ms")
    print("Final error_trains: ", error_trans)
    print("Final error_rot: ", error_rot)

    # Proper shut down of MuJoCo
    if use_mujoco:
        mujoco_sim.viewer_handle.close()

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    #can save or not save the gif
    utils.visualize(car_states, ref_traj, obstacles, times, utils.time_step, save=False, name=control_alg)


if __name__ == "__main__":
    main()

