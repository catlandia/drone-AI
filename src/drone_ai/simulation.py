"""
Drone Physics Simulation

A realistic quadcopter physics simulation that models:
- Rigid body dynamics (position, velocity, orientation, angular velocity)
- Motor dynamics with thrust and torque
- Aerodynamic drag
- Gravity effects
- Ground collision detection

This simulation is designed for sim-to-real transfer, using realistic
parameters that can be tuned to match real drone hardware.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional
import math


@dataclass
class DroneConfig:
    """Configuration parameters for the drone simulation.

    These parameters can be tuned to match specific real-world drones
    for better sim-to-real transfer.
    """
    # Physical properties
    mass: float = 0.027  # kg (Crazyflie 2.1 mass)
    arm_length: float = 0.0397  # meters (distance from center to motor)

    # Inertia tensor (diagonal elements for simplified model)
    ixx: float = 1.4e-5  # kg*m^2
    iyy: float = 1.4e-5  # kg*m^2
    izz: float = 2.17e-5  # kg*m^2

    # Motor properties
    max_rpm: float = 21000  # Maximum motor RPM
    min_rpm: float = 0  # Minimum motor RPM
    motor_constant: float = 1.28192e-8  # Thrust coefficient (N/(rad/s)^2)
    moment_constant: float = 5.964552e-3  # Torque coefficient ratio

    # Aerodynamic properties
    drag_coeff_xy: float = 0.1  # Drag coefficient in x-y plane
    drag_coeff_z: float = 0.2  # Drag coefficient in z direction

    # Motor response (first-order dynamics)
    motor_time_constant: float = 0.02  # seconds

    # Environment
    gravity: float = 9.81  # m/s^2
    air_density: float = 1.225  # kg/m^3

    # Simulation
    dt: float = 0.01  # Simulation timestep (seconds)


@dataclass
class DroneState:
    """Complete state of the drone at a given instant."""
    # Position (meters) - world frame
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Velocity (m/s) - world frame
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Orientation (quaternion: w, x, y, z)
    orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    # Angular velocity (rad/s) - body frame
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Motor speeds (rad/s)
    motor_speeds: np.ndarray = field(default_factory=lambda: np.zeros(4))

    def copy(self) -> 'DroneState':
        """Create a deep copy of the state."""
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            motor_speeds=self.motor_speeds.copy()
        )

    def get_euler_angles(self) -> np.ndarray:
        """Convert quaternion orientation to Euler angles (roll, pitch, yaw)."""
        return quaternion_to_euler(self.orientation)

    def get_rotation_matrix(self) -> np.ndarray:
        """Get the rotation matrix from body to world frame."""
        return quaternion_to_rotation_matrix(self.orientation)


class DroneSimulation:
    """
    Physics-based quadcopter simulation.

    The drone uses a standard X-configuration with motors numbered:
        1 (front-left)    2 (front-right)
              \\          /
                \\      /
                  [  ]
                /      \\
              /          \\
        3 (rear-left)     4 (rear-right)

    Motors 1 and 4 spin clockwise, motors 2 and 3 spin counter-clockwise.
    """

    def __init__(self, config: Optional[DroneConfig] = None):
        """Initialize the simulation with given configuration."""
        self.config = config or DroneConfig()
        self.state = DroneState()
        self._compute_allocation_matrix()

    def _compute_allocation_matrix(self):
        """Compute the control allocation matrix for motor mixing."""
        L = self.config.arm_length
        k = self.config.motor_constant
        b = self.config.moment_constant * k

        # Allocation matrix: [thrust, roll_torque, pitch_torque, yaw_torque]^T = A * [f1, f2, f3, f4]^T
        # Motor positions in X-config at 45 degrees
        angle = np.pi / 4

        self.allocation_matrix = np.array([
            [1, 1, 1, 1],  # Total thrust
            [L * np.sin(angle), -L * np.sin(angle), L * np.sin(angle), -L * np.sin(angle)],  # Roll
            [L * np.cos(angle), L * np.cos(angle), -L * np.cos(angle), -L * np.cos(angle)],  # Pitch
            [-b/k, b/k, b/k, -b/k]  # Yaw (reaction torques)
        ])

        # Inverse for computing motor forces from desired thrust/torques
        self.allocation_matrix_inv = np.linalg.pinv(self.allocation_matrix)

    def reset(self,
              position: Optional[np.ndarray] = None,
              velocity: Optional[np.ndarray] = None,
              orientation: Optional[np.ndarray] = None) -> DroneState:
        """Reset the simulation to initial conditions."""
        self.state = DroneState()

        if position is not None:
            self.state.position = np.array(position, dtype=np.float64)
        else:
            # Start 1 meter above ground by default
            self.state.position = np.array([0.0, 0.0, 1.0])

        if velocity is not None:
            self.state.velocity = np.array(velocity, dtype=np.float64)

        if orientation is not None:
            self.state.orientation = np.array(orientation, dtype=np.float64)
            self.state.orientation /= np.linalg.norm(self.state.orientation)

        # Initialize motors to hover thrust
        hover_thrust = self.config.mass * self.config.gravity
        hover_force_per_motor = hover_thrust / 4
        hover_rpm = np.sqrt(hover_force_per_motor / self.config.motor_constant)
        self.state.motor_speeds = np.full(4, hover_rpm)

        return self.state.copy()

    def step(self, action: np.ndarray) -> DroneState:
        """
        Advance the simulation by one timestep.

        Args:
            action: Motor speed commands as normalized values [0, 1] for each motor,
                   or as [thrust, roll, pitch, yaw] commands depending on control mode.

        Returns:
            Updated drone state
        """
        # Convert normalized action [0, 1] to motor speeds (rad/s)
        action = np.clip(action, 0, 1)
        target_speeds = action * (self.config.max_rpm * 2 * np.pi / 60)

        # Motor dynamics (first-order response)
        alpha = self.config.dt / (self.config.motor_time_constant + self.config.dt)
        self.state.motor_speeds = (1 - alpha) * self.state.motor_speeds + alpha * target_speeds

        # Compute forces and torques
        forces, torques = self._compute_forces_and_torques()

        # Update state using semi-implicit Euler integration
        self._integrate(forces, torques)

        return self.state.copy()

    def _compute_forces_and_torques(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute total forces and torques on the drone."""
        # Motor thrusts
        motor_thrusts = self.config.motor_constant * self.state.motor_speeds ** 2

        # Total thrust and torques from motors (body frame)
        wrench = self.allocation_matrix @ motor_thrusts
        thrust_body = np.array([0, 0, wrench[0]])
        torques_body = wrench[1:4]

        # Rotation matrix (body to world)
        R = self.state.get_rotation_matrix()

        # Transform thrust to world frame
        thrust_world = R @ thrust_body

        # Gravity (world frame)
        gravity = np.array([0, 0, -self.config.mass * self.config.gravity])

        # Aerodynamic drag (world frame, simplified)
        drag = -np.array([
            self.config.drag_coeff_xy * self.state.velocity[0],
            self.config.drag_coeff_xy * self.state.velocity[1],
            self.config.drag_coeff_z * self.state.velocity[2]
        ]) * np.abs(self.state.velocity)

        # Total forces (world frame)
        total_forces = thrust_world + gravity + drag

        return total_forces, torques_body

    def _integrate(self, forces: np.ndarray, torques: np.ndarray):
        """Integrate equations of motion using semi-implicit Euler."""
        dt = self.config.dt

        # Linear dynamics (world frame)
        acceleration = forces / self.config.mass
        self.state.velocity += acceleration * dt
        self.state.position += self.state.velocity * dt

        # Angular dynamics (body frame)
        I = np.diag([self.config.ixx, self.config.iyy, self.config.izz])
        I_inv = np.diag([1/self.config.ixx, 1/self.config.iyy, 1/self.config.izz])

        # Euler's equation for rigid body rotation
        omega = self.state.angular_velocity
        gyroscopic = np.cross(omega, I @ omega)
        angular_acceleration = I_inv @ (torques - gyroscopic)

        self.state.angular_velocity += angular_acceleration * dt

        # Update orientation quaternion
        omega_quat = np.array([0, *self.state.angular_velocity])
        q = self.state.orientation
        q_dot = 0.5 * quaternion_multiply(q, omega_quat)
        self.state.orientation += q_dot * dt
        self.state.orientation /= np.linalg.norm(self.state.orientation)

        # Ground collision
        if self.state.position[2] < 0:
            self.state.position[2] = 0
            self.state.velocity[2] = max(0, self.state.velocity[2])

    def get_state(self) -> DroneState:
        """Get a copy of the current drone state."""
        return self.state.copy()

    def is_crashed(self) -> bool:
        """Check if the drone has crashed or is in an unrecoverable state."""
        # Check if on ground with significant tilt
        euler = self.state.get_euler_angles()
        on_ground = self.state.position[2] < 0.05
        tilted = abs(euler[0]) > np.pi/4 or abs(euler[1]) > np.pi/4

        # Check for extreme velocities
        high_velocity = np.linalg.norm(self.state.velocity) > 20
        high_angular = np.linalg.norm(self.state.angular_velocity) > 30

        return (on_ground and tilted) or high_velocity or high_angular

    def compute_hover_action(self) -> np.ndarray:
        """Compute the action required for steady hover."""
        hover_thrust = self.config.mass * self.config.gravity
        hover_force_per_motor = hover_thrust / 4
        hover_speed = np.sqrt(hover_force_per_motor / self.config.motor_constant)
        max_speed = self.config.max_rpm * 2 * np.pi / 60
        return np.full(4, hover_speed / max_speed)


# Quaternion utilities
def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (Hamilton product)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to Euler angles (roll, pitch, yaw)."""
    w, x, y, z = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1, 1)
    pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles (roll, pitch, yaw) to quaternion."""
    roll, pitch, yaw = euler

    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)

    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    ])
