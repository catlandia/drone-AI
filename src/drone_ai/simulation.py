"""
Drone Physics Simulation

A realistic quadcopter physics simulation that models:
- Rigid body dynamics (position, velocity, orientation, angular velocity)
- Motor dynamics with thrust and torque
- Aerodynamic drag
- Gravity effects
- Ground collision detection
- Package carrying and dropping mechanics

This simulation is designed for sim-to-real transfer, using realistic
parameters that can be tuned to match real drone hardware.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional
from enum import Enum
import math


class PackageStatus(Enum):
    """Status of the package in delivery missions."""
    WAITING = "waiting"      # At pickup location, not yet grabbed
    ATTACHED = "attached"    # Attached to drone
    DROPPING = "dropping"    # In free fall after release
    DELIVERED = "delivered"  # Successfully landed in drop zone
    MISSED = "missed"        # Landed outside drop zone


@dataclass
class EnvironmentConfig:
    """Configuration for environmental conditions (sim-to-real transfer).

    Randomize these parameters during training to make the agent
    robust to real-world variations.
    """
    # Wind disturbances
    wind_speed: float = 0.0           # m/s base wind speed
    wind_direction: float = 0.0       # radians (0 = +x direction)
    wind_turbulence: float = 0.0      # Random wind variation intensity
    wind_gust_probability: float = 0.0  # Chance of sudden gust per step

    # Sensor noise (simulates real sensor imperfections)
    position_noise: float = 0.0       # meters std dev
    velocity_noise: float = 0.0       # m/s std dev
    orientation_noise: float = 0.0    # radians std dev
    motor_noise: float = 0.0          # Motor command noise (0-1)

    # Ground effects (thrust increases near ground)
    ground_effect_height: float = 0.3  # Height where ground effect starts
    ground_effect_strength: float = 0.1  # Max thrust multiplier at ground

    # Temperature effects (affects air density)
    temperature_offset: float = 0.0   # Celsius offset from standard (20C)

    # Battery simulation
    battery_voltage_drop: float = 0.0  # Simulates voltage sag under load


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
class PackageConfig:
    """Configuration for delivery package."""
    mass: float = 0.010  # 10 grams
    size: float = 0.05   # 5cm cube (for collision)
    drag_coeff: float = 0.5  # Drag during fall
    pickup_radius: float = 0.15  # How close drone must be to pickup
    drop_zone_radius: float = 0.3  # Target zone radius for successful delivery


@dataclass
class PackageState:
    """State of the delivery package."""
    status: PackageStatus = PackageStatus.WAITING
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pickup_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dropzone_position: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> 'PackageState':
        return PackageState(
            status=self.status,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            pickup_position=self.pickup_position.copy(),
            dropzone_position=self.dropzone_position.copy()
        )


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

    Supports package delivery missions with pickup and drop mechanics.
    """

    def __init__(
        self,
        config: Optional[DroneConfig] = None,
        package_config: Optional[PackageConfig] = None,
        env_config: Optional[EnvironmentConfig] = None
    ):
        """Initialize the simulation with given configuration."""
        self.config = config or DroneConfig()
        self.package_config = package_config or PackageConfig()
        self.env_config = env_config or EnvironmentConfig()
        self.state = DroneState()
        self.package: Optional[PackageState] = None
        self._current_wind = np.zeros(3)  # Current wind velocity
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

    def reset(
        self,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        package_pickup: Optional[np.ndarray] = None,
        package_dropzone: Optional[np.ndarray] = None
    ) -> DroneState:
        """Reset the simulation to initial conditions.

        Args:
            position: Initial drone position
            velocity: Initial drone velocity
            orientation: Initial drone orientation (quaternion)
            package_pickup: If provided, sets up a delivery mission with pickup at this location
            package_dropzone: Drop zone location for delivery mission
        """
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

        # Set up package for delivery mission
        if package_pickup is not None:
            self.package = PackageState(
                status=PackageStatus.WAITING,
                position=np.array(package_pickup, dtype=np.float64),
                velocity=np.zeros(3),
                pickup_position=np.array(package_pickup, dtype=np.float64),
                dropzone_position=np.array(package_dropzone if package_dropzone is not None
                                          else [0, 0, 0], dtype=np.float64)
            )
        else:
            self.package = None

        return self.state.copy()

    def step(self, action: np.ndarray, drop_package: bool = False) -> DroneState:
        """
        Advance the simulation by one timestep.

        Args:
            action: Motor speed commands as normalized values [0, 1] for each motor,
                   or as [thrust, roll, pitch, yaw] commands depending on control mode.
            drop_package: If True and package is attached, release the package.

        Returns:
            Updated drone state
        """
        # Convert normalized action [0, 1] to motor speeds (rad/s)
        action = np.clip(action, 0, 1)
        target_speeds = action * (self.config.max_rpm * 2 * np.pi / 60)

        # Motor dynamics (first-order response)
        alpha = self.config.dt / (self.config.motor_time_constant + self.config.dt)
        self.state.motor_speeds = (1 - alpha) * self.state.motor_speeds + alpha * target_speeds

        # Compute forces and torques (accounting for package mass if attached)
        forces, torques = self._compute_forces_and_torques()

        # Update state using semi-implicit Euler integration
        self._integrate(forces, torques)

        # Handle package mechanics
        if self.package is not None:
            self._update_package(drop_package)

        return self.state.copy()

    def _compute_forces_and_torques(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute total forces and torques on the drone."""
        env = self.env_config

        # Motor thrusts (with optional noise)
        motor_speeds = self.state.motor_speeds
        if env.motor_noise > 0:
            noise = np.random.normal(0, env.motor_noise, 4) * motor_speeds
            motor_speeds = np.clip(motor_speeds + noise, 0, None)

        motor_thrusts = self.config.motor_constant * motor_speeds ** 2

        # Ground effect - thrust increases when close to ground
        if self.state.position[2] < env.ground_effect_height:
            ground_factor = 1.0 + env.ground_effect_strength * (
                1.0 - self.state.position[2] / env.ground_effect_height
            )
            motor_thrusts *= ground_factor

        # Battery voltage drop effect (reduces thrust under load)
        if env.battery_voltage_drop > 0:
            load_factor = np.mean(motor_speeds) / (self.config.max_rpm * 2 * np.pi / 60)
            voltage_factor = 1.0 - env.battery_voltage_drop * load_factor
            motor_thrusts *= voltage_factor

        # Total thrust and torques from motors (body frame)
        wrench = self.allocation_matrix @ motor_thrusts
        thrust_body = np.array([0, 0, wrench[0]])
        torques_body = wrench[1:4]

        # Rotation matrix (body to world)
        R = self.state.get_rotation_matrix()

        # Transform thrust to world frame
        thrust_world = R @ thrust_body

        # Calculate total mass (drone + package if attached)
        total_mass = self.config.mass
        if self.package is not None and self.package.status == PackageStatus.ATTACHED:
            total_mass += self.package_config.mass

        # Gravity (world frame)
        gravity = np.array([0, 0, -total_mass * self.config.gravity])

        # === WIND FORCES ===
        self._update_wind()
        # Effective velocity relative to air
        air_velocity = self.state.velocity - self._current_wind

        # Aerodynamic drag (relative to air, not ground)
        # Temperature affects air density
        temp_factor = 1.0 - env.temperature_offset * 0.003  # ~0.3% per degree
        effective_density = self.config.air_density * temp_factor

        drag = -effective_density * np.array([
            self.config.drag_coeff_xy * air_velocity[0],
            self.config.drag_coeff_xy * air_velocity[1],
            self.config.drag_coeff_z * air_velocity[2]
        ]) * np.abs(air_velocity)

        # Total forces (world frame)
        total_forces = thrust_world + gravity + drag

        return total_forces, torques_body

    def _update_wind(self):
        """Update wind conditions with turbulence and gusts."""
        env = self.env_config

        if env.wind_speed == 0 and env.wind_turbulence == 0:
            self._current_wind = np.zeros(3)
            return

        # Base wind vector
        base_wind = np.array([
            env.wind_speed * np.cos(env.wind_direction),
            env.wind_speed * np.sin(env.wind_direction),
            0.0
        ])

        # Add turbulence (random fluctuation)
        if env.wind_turbulence > 0:
            turbulence = np.random.normal(0, env.wind_turbulence, 3)
            turbulence[2] *= 0.5  # Less vertical turbulence
            base_wind += turbulence

        # Random gusts
        if env.wind_gust_probability > 0:
            if np.random.random() < env.wind_gust_probability:
                gust_direction = np.random.uniform(0, 2 * np.pi)
                gust_strength = np.random.uniform(1, 3) * env.wind_speed
                base_wind[0] += gust_strength * np.cos(gust_direction)
                base_wind[1] += gust_strength * np.sin(gust_direction)

        self._current_wind = base_wind

    def get_noisy_state(self) -> DroneState:
        """Get state with sensor noise added (for realistic observations)."""
        env = self.env_config
        noisy_state = self.state.copy()

        if env.position_noise > 0:
            noisy_state.position += np.random.normal(0, env.position_noise, 3)

        if env.velocity_noise > 0:
            noisy_state.velocity += np.random.normal(0, env.velocity_noise, 3)

        if env.orientation_noise > 0:
            euler = noisy_state.get_euler_angles()
            euler += np.random.normal(0, env.orientation_noise, 3)
            noisy_state.orientation = euler_to_quaternion(euler)

        return noisy_state

    def _update_package(self, drop_command: bool):
        """Update package state based on drone position and drop command."""
        if self.package is None:
            return

        pkg = self.package
        dt = self.config.dt

        if pkg.status == PackageStatus.WAITING:
            # Check if drone is close enough to pick up
            dist_to_pickup = np.linalg.norm(
                self.state.position - pkg.pickup_position
            )
            # Must be close horizontally and low altitude for pickup
            horizontal_dist = np.linalg.norm(
                self.state.position[:2] - pkg.pickup_position[:2]
            )
            altitude = self.state.position[2]

            if horizontal_dist < self.package_config.pickup_radius and altitude < 0.3:
                pkg.status = PackageStatus.ATTACHED
                pkg.position = self.state.position.copy()
                pkg.velocity = self.state.velocity.copy()

        elif pkg.status == PackageStatus.ATTACHED:
            # Package follows drone
            pkg.position = self.state.position.copy()
            pkg.velocity = self.state.velocity.copy()

            # Check for drop command
            if drop_command:
                pkg.status = PackageStatus.DROPPING
                # Package inherits drone velocity at release
                pkg.velocity = self.state.velocity.copy()

        elif pkg.status == PackageStatus.DROPPING:
            # Package in free fall with drag
            # Gravity
            acc = np.array([0, 0, -self.config.gravity])

            # Air drag on package
            speed = np.linalg.norm(pkg.velocity)
            if speed > 0.01:
                drag_force = -0.5 * self.config.air_density * \
                            self.package_config.drag_coeff * \
                            self.package_config.size**2 * \
                            speed * pkg.velocity
                acc += drag_force / self.package_config.mass

            # Integrate
            pkg.velocity += acc * dt
            pkg.position += pkg.velocity * dt

            # Check for ground contact
            if pkg.position[2] <= 0:
                pkg.position[2] = 0
                pkg.velocity = np.zeros(3)

                # Check if landed in drop zone
                dist_to_dropzone = np.linalg.norm(
                    pkg.position[:2] - pkg.dropzone_position[:2]
                )
                if dist_to_dropzone <= self.package_config.drop_zone_radius:
                    pkg.status = PackageStatus.DELIVERED
                else:
                    pkg.status = PackageStatus.MISSED

    def _integrate(self, forces: np.ndarray, torques: np.ndarray):
        """Integrate equations of motion using semi-implicit Euler."""
        dt = self.config.dt

        # Calculate total mass (drone + package if attached)
        total_mass = self.config.mass
        if self.package is not None and self.package.status == PackageStatus.ATTACHED:
            total_mass += self.package_config.mass

        # Linear dynamics (world frame)
        acceleration = forces / total_mass
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
        # Account for package mass if attached
        total_mass = self.config.mass
        if self.package is not None and self.package.status == PackageStatus.ATTACHED:
            total_mass += self.package_config.mass

        hover_thrust = total_mass * self.config.gravity
        hover_force_per_motor = hover_thrust / 4
        hover_speed = np.sqrt(hover_force_per_motor / self.config.motor_constant)
        max_speed = self.config.max_rpm * 2 * np.pi / 60
        return np.full(4, hover_speed / max_speed)

    def get_package_state(self) -> Optional[PackageState]:
        """Get a copy of the current package state."""
        if self.package is None:
            return None
        return self.package.copy()

    def has_package(self) -> bool:
        """Check if drone currently has a package attached."""
        return (self.package is not None and
                self.package.status == PackageStatus.ATTACHED)

    def is_package_delivered(self) -> bool:
        """Check if package was successfully delivered."""
        return (self.package is not None and
                self.package.status == PackageStatus.DELIVERED)

    def is_package_missed(self) -> bool:
        """Check if package missed the drop zone."""
        return (self.package is not None and
                self.package.status == PackageStatus.MISSED)

    def get_delivery_accuracy(self) -> Optional[float]:
        """Get distance from package landing to drop zone center.

        Returns None if package hasn't landed yet.
        """
        if self.package is None:
            return None
        if self.package.status not in [PackageStatus.DELIVERED, PackageStatus.MISSED]:
            return None
        return float(np.linalg.norm(
            self.package.position[:2] - self.package.dropzone_position[:2]
        ))


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
