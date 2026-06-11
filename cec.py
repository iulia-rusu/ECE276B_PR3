import casadi
import numpy as np
from utils import lemniscate, time_step, v_min, v_max, w_min, w_max #need reference trajectory



class CEC:
    def __init__(self, horizon, obstacles, Q, R, q):
        #define all the variables
       
        #cost function should be in casadi
        self.Q = casadi.DM(Q)
        self.R = casadi.DM(R)
        self.q = q
        self.dt = time_step
        self.horizon = horizon
        self.obstacles = obstacles
        self.robot_radius = 0.3

    def get_control(self, t, cur_state, cur_ref_state):
        """Gets called at every time step, trajectory repeats every 100 time steps.
        inputs is current state which is the x, y position and the theta (robots heading angle),
        and the refrence state, what the robot state should be.
        Return: the optimzied control for the next time step
        computes error in heading and postion
        """
        T = self.horizon # set manually, is currently usually 10 steps
        dt = self.dt #tau
        # error between current state and reference
        e0 = cur_state - np.array(cur_ref_state) #where car is relative to where it should be
        e0[2] = np.arctan2(np.sin(e0[2]), np.cos(e0[2]))

        # optimization variables error states and controls over horizon
        num_state_vars = 3 * T
        num_control_vars = 2 * T # because v and w 
        total_vars = num_state_vars + num_control_vars
        optims = casadi.MX.sym('optims', total_vars) #empty vector but casadi

        total_cost = casadi.MX(0) #casadi varibale, will acum the stage costs at each t
        # constraints are dictated by the motion model in utils car_next_state
        constraints = [] # you can only pick values of state_k that are physically reachable from the previous state via the motion model
        constraint_lb = []  #lower bound per constraint: 0 for dynamics equality, min_dist_sq for obstacle inequality
        constraint_ub = []  #upper bound per constraint: 0 for dynamics equality, inf for obstacle inequality
        # error state from the previous time step
        e_prev = casadi.DM(e0)

        for k in range(T):
            # grab state and control for this step
            state_k = optims[3*k : 3*k + 3] # state it k+ 1 since we compute error once the action happens
            control_k = optims[num_state_vars + 2*k : num_state_vars + 2*k + 2] # controls happens at the time step of the input
            #for clearer naming of each control component
            linear_vel = control_k[0] 
            angular_vel = control_k[1]

            # get reference at this step and next
            ref_current = lemniscate(t + k)
            ref_forward = lemniscate(t + k + 1)
            ref_heading = ref_current[2]

            # motion model in error coordinates
            direction = e_prev[2] + ref_heading

            ex_next = e_prev[0] + dt * casadi.cos(direction) * linear_vel + (ref_current[0] - ref_forward[0])
            ey_next = e_prev[1] + dt * casadi.sin(direction) * linear_vel + (ref_current[1] - ref_forward[1])
            ehead_next = e_prev[2] + dt * angular_vel + (ref_current[2] - ref_forward[2])
            # use casadi to define what the constraints of next state should be
            predicted_next = casadi.vertcat(ex_next, ey_next, ehead_next)

            # dynamics constraint predicted must match decision variable
            constraints.append(state_k - predicted_next)
            constraint_lb += [0.0, 0.0, 0.0]
            constraint_ub += [0.0, 0.0, 0.0]

            # computing  stage cost
            position_error = state_k[:2]
            heading_error = state_k[2]
            position_cost = casadi.mtimes(casadi.mtimes(position_error.T, self.Q), position_error)
            heading_cost = self.q * (1 - casadi.cos(heading_error))
            control_cost = casadi.mtimes(casadi.mtimes(control_k.T, self.R), control_k)
            total_cost += position_cost + heading_cost + control_cost

            # obstacle avoidance
            actual_x = state_k[0] + ref_forward[0]
            actual_y = state_k[1] + ref_forward[1]
            for obs in self.obstacles:
                obs_x = float(obs[0])
                obs_y = float(obs[1])
                obs_r = float(obs[2])
                min_dist_sq = (obs_r + self.robot_radius) ** 2
                dist_sq = (actual_x - obs_x)**2 + (actual_y - obs_y)**2
                constraints.append(dist_sq)
                constraint_lb.append(min_dist_sq)
                constraint_ub.append(float('inf'))
            #make current state equal to previous error 
            e_prev = state_k
        g_vec = casadi.vertcat(*constraints)

        # variable bounds states unconstrained, controls within limits of utils
        lower_x = [-float('inf')] * num_state_vars
        upper_x = [ float('inf')] * num_state_vars
        for _ in range(T):
            lower_x += [v_min, w_min]
            upper_x += [v_max, w_max]

        # initial guess for x params 
        x0 = [0.0] * num_state_vars
        for _ in range(T):
            x0 += [0.5, 0.0]
        #build dictionary for nlp
        nlp = {'x': optims, 'f': total_cost, 'g': g_vec}
        solver_opts = {'ipopt': {'print_level': 0, 'max_iter': 200}, 'print_time': 0}
        solver = casadi.nlpsol('solver', 'ipopt', nlp, solver_opts)

        solution = solver(x0=x0, lbx=lower_x, ubx=upper_x, lbg=constraint_lb, ubg=constraint_ub)
        x_opt = np.array(solution['x']).flatten()

        # return the first control input
        u_opt = x_opt[num_state_vars : num_state_vars + 2]
        return u_opt
