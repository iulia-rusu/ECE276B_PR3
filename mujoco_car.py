from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
import dm_control.utils.transformations as tr
import mujoco.viewer
import numpy as np
import utils


class CarObservables(composer.Observables):

    @property
    def body(self):
        return self._entity._mjcf_root.find('body', 'buddy')

    def get_sensor_mjcf(self, sensor_name):
        return self._entity._mjcf_root.find('sensor', sensor_name)

    @composer.observable
    def body_position(self):
        return observable.MJCFFeature('xpos', self._entity._mjcf_root.find('body', 'buddy'))

    @composer.observable
    def wheel_speeds(self):
        def get_wheel_speeds(physics):
            return np.concatenate([
                physics.bind(self.get_sensor_mjcf(f'buddy_wheel_{wheel}_vel')).sensordata
                for wheel in ['fl', 'fr', 'bl', 'br']
            ])

        return observable.Generic(get_wheel_speeds)

    @composer.observable
    def body_pose_2d(self):
        def get_pose_2d(physics):
            pos = physics.bind(self.body).xpos[:2]
            yaw = tr.quat_to_euler(physics.bind(self.body).xquat)[2]
            return np.append(pos, yaw)

        return observable.Generic(get_pose_2d)

    @composer.observable
    def body_vel_2d(self):
        def get_vel_2d(physics):
            quat = physics.bind(self.body).xquat
            velocity_local = physics.bind(self.get_sensor_mjcf('velocimeter')).sensordata
            return tr.quat_rotate(quat, velocity_local)[:2]

        return observable.Generic(get_vel_2d)

    @composer.observable
    def steering_pos(self):
        return observable.MJCFFeature('sensordata', self.get_sensor_mjcf('buddy_steering_pos'))

    @property
    def all_observables(self):
        return [self.body_pose_2d]


class Car(composer.Robot):
    def _build(self, name='buddy'):
        model_path = "mujoco_assets/env.xml"
        self._mjcf_root = mjcf.from_path(f'{model_path}')
        if name:
            self._mjcf_root.model = name

        self._actuators = self.mjcf_model.find_all('actuator')

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def actuators(self):
        return self._actuators

    def apply_action(self, physics, action):
        """Apply action to car's actuators.
        `action` is expected to be a numpy array with [linear_velocity, angular_velocity]
        """
        L = 0.2938
        v, omega = action[0], action[1]
        if v >= 0.05:
            phi = np.arctan2(omega * L, v)
        else:
            phi = 0.0

        physics.bind(self.mjcf_model.find('actuator', 'buddy_steering_pos')).ctrl = phi
        physics.bind(self.mjcf_model.find('actuator', 'buddy_throttle_velocity')).ctrl = v

    def build_observables(self):
        return CarObservables(self)


class MujocoCarSim:
    """
    This class implements an Ackermann drive car in Mujoco simulation environment
    """
    def __init__(self):
        # Initialize MuJoCo car simulation
        self.car_model = Car()
        self.observation = self.car_model.build_observables()
        self.physics = mjcf.Physics.from_mjcf_model(self.car_model.mjcf_model)
        self.mujoco_dt = self.physics.model.opt.timestep
        self.viewer_handle = mujoco.viewer.launch_passive(self.physics.model.ptr, self.physics.data.ptr)
        self.viewer_handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.viewer_handle.cam.fixedcamid = self.physics.model.camera('overhead_track').id

        # Get simulation time step in our control calculation
        self.sim_dt = utils.time_step

    def get_car_pose(self) -> np.ndarray:
        """
        This function returns the current car pose
        """
        return self.observation.body_pose_2d(self.physics)

    def car_next_state(self, control: np.ndarray) -> np.ndarray:
        """
        This function steps the car to next time step in MuJoCo and returns the next car pose
        """
        # Initialize MuJoCo time per step
        mujoco_time = 0

        # Apply control for a specific duration
        while mujoco_time < self.sim_dt:
            # ---- Apply control input ----
            self.car_model.apply_action(self.physics, control)

            # ---- Step the simulation ----
            self.physics.step()

            # ---- Update timestep ----
            mujoco_time += self.mujoco_dt

            # ---- Synchronization  ----
            self.viewer_handle.sync()

        return self.get_car_pose()

