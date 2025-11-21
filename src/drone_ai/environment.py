"""
Gymnasium Environment for Drone Flight Training

This module provides a Gymnasium-compatible environment for training
reinforcement learning agents to control a quadcopter drone.

The environment supports multiple task types:
- Hover: Maintain a stable position
- Waypoint: Navigate to target positions
- Trajectory: Follow a predefined path
- Velocity: Track a target velocity
- Delivery: Pick up package and drop at target location
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple, List
from enum import Enum

from drone_ai.simulation import (
    DroneSimulation, DroneConfig, DroneState, EnvironmentConfig,
    PackageConfig, PackageState, PackageStatus
)


class TaskType(Enum):
    """Available training tasks."""
    HOVER = "hover"
    WAYPOINT = "waypoint"
    TRAJECTORY = "trajectory"
    VELOCITY = "velocity"
    DELIVERY = "delivery"  # Package pickup and drop mission


class DroneEnv(gym.Env):
    """
    Gymnasium environment for drone flight control.

    Observation Space (19-26 dimensions depending on task):
        Base (19 dims):
        - Position (3): x, y, z in world frame
        - Velocity (3): vx, vy, vz in world frame
        - Orientation (3): roll, pitch, yaw angles
        - Angular velocity (3): p, q, r in body frame
        - Target position (3): relative target in world frame
        - Previous action (4): last motor commands

        Delivery task adds (7 dims):
        - Package status (1): 0=waiting, 1=attached, 2=dropping, 3=delivered, 4=missed
        - Has package (1): binary flag
        - Pickup position relative (3): relative to drone
        - Dropzone position relative (3): relative to drone

    Action Space:
        Standard tasks (4 dims): Motor commands [0, 1]
        Delivery task (5 dims): Motor commands [0, 1] + drop signal [0, 1]

    Rewards:
        - Position tracking reward
        - Velocity penalty
        - Attitude stability reward
        - Action smoothness reward
        - Crash penalty
        - Delivery task: pickup bonus, delivery accuracy bonus
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        task: TaskType = TaskType.HOVER,
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
        domain_randomization: bool = False,
        difficulty: float = 0.5,
        config: Optional[DroneConfig] = None,
        package_config: Optional[PackageConfig] = None
    ):
        """
        Initialize the drone environment.

        Args:
            task: The task type for training
            max_steps: Maximum episode length
            render_mode: Rendering mode ('human' or 'rgb_array')
            domain_randomization: Enable randomization for sim-to-real
            difficulty: Task difficulty [0, 1]
            config: Drone configuration parameters
            package_config: Package configuration for delivery task
        """
        super().__init__()

        self.task = task
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.domain_randomization = domain_randomization
        self.difficulty = np.clip(difficulty, 0, 1)

        # Initialize simulation
        self.base_config = config or DroneConfig()
        self.package_config = package_config or PackageConfig()
        self.sim = DroneSimulation(self.base_config, self.package_config)

        # Define action space based on task
        if task == TaskType.DELIVERY:
            # 4 motor commands + 1 drop signal
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(5,),
                dtype=np.float32
            )
        else:
            # 4 motor commands normalized to [0, 1]
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32
            )

        # Define observation space based on task
        # Base: [position(3), velocity(3), euler(3), angular_vel(3), target_rel(3), prev_action(4/5)]
        if task == TaskType.DELIVERY:
            # Add: package_status(1), has_package(1), pickup_rel(3), dropzone_rel(3)
            obs_dim = 19 + 1 + 8  # 28 total (prev_action is 5 for delivery)
        else:
            obs_dim = 19
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Task-specific parameters
        self.target_position = np.array([0.0, 0.0, 1.0])
        self.target_velocity = np.zeros(3)
        self.waypoints: List[np.ndarray] = []
        self.current_waypoint_idx = 0

        # Delivery task parameters
        self.pickup_position = np.zeros(3)
        self.dropzone_position = np.zeros(3)
        self.package_picked_up = False
        self.delivery_phase = "pickup"  # "pickup", "deliver", "drop", "done"

        # Episode state
        self.step_count = 0
        self.prev_action = np.zeros(5 if task == TaskType.DELIVERY else 4)
        self.episode_reward = 0.0
        self.position_history: List[np.ndarray] = []

        # Reward weights (tunable)
        # Main learning priorities: ACCURACY, CAREFULNESS, SPEED
        self.reward_weights = {
            # === CAREFULNESS (smooth, stable flight) ===
            'position': 0.5,              # Track target position
            'velocity': 0.3,              # Penalize fast/jerky movement
            'orientation': 0.5,           # Keep level (careful flight)
            'angular_velocity': 0.2,      # Smooth rotations
            'action_smoothness': 0.3,     # Smooth control inputs
            'alive': 0.05,                # Small survival bonus
            'crash': -50.0,               # Heavy crash penalty (be careful!)
            'success': 2.0,

            # === DELIVERY TASK REWARDS ===
            # Accuracy rewards (most important for delivery)
            'pickup': 5.0,                # Bonus for picking up package
            'delivery': 30.0,             # Big bonus for successful delivery
            'accuracy': 50.0,             # HUGE bonus for precise drops (scaled by distance)
            'bullseye': 20.0,             # Extra bonus for perfect center drop

            # Carefulness penalties
            'drop_penalty': -10.0,        # Penalty for dropping at wrong location
            'missed': -30.0,              # Heavy penalty for missing drop zone
            'rough_handling': -0.1,       # Penalty for jerky movements while carrying

            # Speed rewards
            'time_bonus': 0.5,            # Bonus for faster completion
            'efficiency': 5.0,            # Bonus for direct path to target
        }

        # Track timing for speed rewards
        self.pickup_step = None  # Step when package was picked up

        # Visualization
        self._renderer = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Apply domain randomization if enabled
        if self.domain_randomization:
            self._apply_domain_randomization()
        else:
            self.sim = DroneSimulation(self.base_config, self.package_config)

        # Set up task-specific targets first (needed for delivery task reset)
        self._setup_task()

        # Reset drone state with optional randomization
        init_position = self._get_initial_position()
        init_velocity = self._get_initial_velocity()
        init_orientation = self._get_initial_orientation()

        # Reset simulation with package info for delivery task
        if self.task == TaskType.DELIVERY:
            self.sim.reset(
                position=init_position,
                velocity=init_velocity,
                orientation=init_orientation,
                package_pickup=self.pickup_position,
                package_dropzone=self.dropzone_position
            )
            self.package_picked_up = False
            self.delivery_phase = "pickup"
            self.pickup_step = None  # Reset timing tracker
        else:
            self.sim.reset(
                position=init_position,
                velocity=init_velocity,
                orientation=init_orientation
            )

        # Reset episode state
        self.step_count = 0
        hover_action = self.sim.compute_hover_action()
        if self.task == TaskType.DELIVERY:
            self.prev_action = np.concatenate([hover_action, [0.0]])  # Add drop signal
        else:
            self.prev_action = hover_action
        self.episode_reward = 0.0
        self.position_history = [self.sim.state.position.copy()]

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        self.step_count += 1

        # Clip action to valid range
        action = np.clip(action, 0, 1).astype(np.float32)

        # Handle delivery task with drop signal
        if self.task == TaskType.DELIVERY:
            motor_action = action[:4]
            drop_signal = action[4] > 0.5  # Threshold for drop command
            self.sim.step(motor_action, drop_package=drop_signal)
        else:
            self.sim.step(action)

        # Record position for trajectory visualization
        self.position_history.append(self.sim.state.position.copy())

        # Compute reward
        reward = self._compute_reward(action)
        self.episode_reward += reward

        # Check termination conditions
        terminated = self._check_terminated()
        truncated = self.step_count >= self.max_steps

        # Update task state (e.g., move to next waypoint)
        self._update_task()

        # Store action for smoothness penalty
        self.prev_action = action.copy()

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector from current state."""
        state = self.sim.state

        # Position (normalized by typical operating range)
        position = state.position / 5.0  # Normalize to ~[-1, 1] for 5m range

        # Velocity (normalized)
        velocity = state.velocity / 5.0  # Normalize to ~[-1, 1] for 5m/s

        # Orientation (Euler angles)
        euler = state.get_euler_angles()

        # Angular velocity (normalized)
        angular_velocity = state.angular_velocity / 10.0  # Normalize

        # Relative target position
        target_rel = (self.target_position - state.position) / 5.0

        # Previous action
        prev_action = self.prev_action

        # Base observation
        base_obs = [
            position,
            velocity,
            euler,
            angular_velocity,
            target_rel,
            prev_action
        ]

        # Add delivery-specific observations
        if self.task == TaskType.DELIVERY:
            pkg = self.sim.get_package_state()
            if pkg is not None:
                # Package status as normalized value (0-4 -> 0-1)
                status_map = {
                    PackageStatus.WAITING: 0.0,
                    PackageStatus.ATTACHED: 0.25,
                    PackageStatus.DROPPING: 0.5,
                    PackageStatus.DELIVERED: 0.75,
                    PackageStatus.MISSED: 1.0
                }
                pkg_status = np.array([status_map.get(pkg.status, 0.0)])

                # Has package flag
                has_package = np.array([1.0 if self.sim.has_package() else 0.0])

                # Relative positions to pickup and dropzone
                pickup_rel = (pkg.pickup_position - state.position) / 5.0
                dropzone_rel = (pkg.dropzone_position - state.position) / 5.0

                base_obs.extend([pkg_status, has_package, pickup_rel, dropzone_rel])
            else:
                # No package, add zeros
                base_obs.extend([
                    np.zeros(1),  # status
                    np.zeros(1),  # has_package
                    np.zeros(3),  # pickup_rel
                    np.zeros(3),  # dropzone_rel
                ])

        observation = np.concatenate(base_obs).astype(np.float32)

        return observation

    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward for current state and action."""
        state = self.sim.state
        weights = self.reward_weights

        # Position error reward (negative quadratic)
        pos_error = np.linalg.norm(state.position - self.target_position)
        pos_reward = -weights['position'] * pos_error ** 2

        # Velocity penalty (prefer low velocities for hover)
        vel_penalty = -weights['velocity'] * np.linalg.norm(state.velocity) ** 2

        # Orientation reward (prefer level flight)
        euler = state.get_euler_angles()
        orient_penalty = -weights['orientation'] * (euler[0]**2 + euler[1]**2)

        # Angular velocity penalty
        ang_vel_penalty = -weights['angular_velocity'] * np.linalg.norm(state.angular_velocity) ** 2

        # Action smoothness (penalize jerky control)
        motor_action = action[:4] if self.task == TaskType.DELIVERY else action
        prev_motor = self.prev_action[:4] if self.task == TaskType.DELIVERY else self.prev_action
        action_diff = np.linalg.norm(motor_action - prev_motor)
        smoothness_penalty = -weights['action_smoothness'] * action_diff ** 2

        # Alive bonus
        alive_bonus = weights['alive']

        # Success bonus (close to target)
        success_bonus = 0.0
        if pos_error < 0.1 and np.linalg.norm(state.velocity) < 0.5:
            success_bonus = weights['success'] * 0.1  # Small per-step bonus

        # Crash penalty
        crash_penalty = 0.0
        if self.sim.is_crashed():
            crash_penalty = weights['crash']

        # Delivery-specific rewards
        delivery_reward = 0.0
        if self.task == TaskType.DELIVERY:
            delivery_reward = self._compute_delivery_reward(action)

        total_reward = (
            pos_reward +
            vel_penalty +
            orient_penalty +
            ang_vel_penalty +
            smoothness_penalty +
            alive_bonus +
            success_bonus +
            crash_penalty +
            delivery_reward
        )

        return float(total_reward)

    def _compute_delivery_reward(self, action: np.ndarray) -> float:
        """Compute delivery-specific rewards.

        Priorities:
        1. ACCURACY - Precise drops get massive bonuses
        2. CAREFULNESS - Smooth handling, no crashes
        3. SPEED - Faster completion = more reward
        """
        weights = self.reward_weights
        reward = 0.0

        pkg = self.sim.get_package_state()
        if pkg is None:
            return 0.0

        # === PICKUP PHASE ===
        if pkg.status == PackageStatus.ATTACHED and not self.package_picked_up:
            reward += weights['pickup']
            self.package_picked_up = True
            self.pickup_step = self.step_count  # Record pickup time for speed calc
            self.delivery_phase = "deliver"

        # === CAREFULNESS: Rough handling penalty while carrying ===
        if pkg.status == PackageStatus.ATTACHED:
            # Penalize jerky movements while carrying package
            motor_action = action[:4]
            prev_motor = self.prev_action[:4]
            action_jerk = np.linalg.norm(motor_action - prev_motor)
            if action_jerk > 0.1:  # Threshold for "rough" handling
                reward += weights['rough_handling'] * action_jerk

            # Also penalize excessive tilt while carrying
            euler = self.sim.state.get_euler_angles()
            tilt = abs(euler[0]) + abs(euler[1])
            if tilt > 0.3:  # More than ~17 degrees
                reward += weights['rough_handling'] * tilt

        # === SUCCESSFUL DELIVERY: ACCURACY + SPEED ===
        if pkg.status == PackageStatus.DELIVERED:
            if self.delivery_phase != "done":
                # Base delivery bonus
                reward += weights['delivery']

                # ACCURACY BONUS (the main prize!)
                accuracy = self.sim.get_delivery_accuracy()
                if accuracy is not None:
                    # Exponential bonus - much better reward for precise drops
                    # accuracy = 0 (perfect) -> full bonus
                    # accuracy = radius (edge) -> ~37% bonus
                    accuracy_ratio = accuracy / self.package_config.drop_zone_radius
                    accuracy_bonus = weights['accuracy'] * np.exp(-accuracy_ratio * 2)
                    reward += accuracy_bonus

                    # BULLSEYE bonus for very precise drops (within 10% of radius)
                    if accuracy < self.package_config.drop_zone_radius * 0.1:
                        reward += weights['bullseye']

                # SPEED BONUS - faster delivery = more reward
                if self.pickup_step is not None:
                    delivery_time = self.step_count - self.pickup_step
                    # Bonus inversely proportional to time taken
                    # Max bonus if delivered in ~100 steps, decreasing after
                    speed_factor = max(0, 1 - delivery_time / 500)
                    reward += weights['time_bonus'] * speed_factor * 20

                    # Efficiency bonus - compare to optimal path
                    optimal_distance = np.linalg.norm(
                        self.dropzone_position[:2] - self.pickup_position[:2]
                    )
                    # Rough estimate: optimal time = distance / avg_speed
                    optimal_steps = optimal_distance * 100  # ~1m/s average
                    if delivery_time < optimal_steps * 1.5:
                        reward += weights['efficiency']

                self.delivery_phase = "done"

        # === MISSED DELIVERY: Heavy penalty ===
        if pkg.status == PackageStatus.MISSED:
            if self.delivery_phase != "done":
                reward += weights['missed']

                # Additional penalty based on how far off
                miss_distance = self.sim.get_delivery_accuracy()
                if miss_distance is not None:
                    # Extra penalty for being way off
                    reward -= min(10, miss_distance * 2)

                self.delivery_phase = "done"

        # === PREMATURE DROP PENALTY ===
        drop_signal = action[4] > 0.5 if len(action) > 4 else False
        if drop_signal and pkg.status == PackageStatus.ATTACHED:
            dist_to_dropzone = np.linalg.norm(
                self.sim.state.position[:2] - pkg.dropzone_position[:2]
            )
            if dist_to_dropzone > self.package_config.drop_zone_radius * 1.5:
                # Heavy penalty for dropping way off target
                reward += weights['drop_penalty']
                # Scale penalty by distance - worse drops = worse penalty
                reward -= min(5, dist_to_dropzone)

        return reward

    def _check_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Crash detection
        if self.sim.is_crashed():
            return True

        # Out of bounds
        position = self.sim.state.position
        if np.any(np.abs(position[:2]) > 10) or position[2] > 20:
            return True

        # Delivery task termination
        if self.task == TaskType.DELIVERY:
            if self.sim.is_package_delivered() or self.sim.is_package_missed():
                return True

        return False

    def _setup_task(self):
        """Set up task-specific parameters."""
        if self.task == TaskType.HOVER:
            # Hover at a fixed or random position
            if self.difficulty > 0.5:
                self.target_position = self._random_target()
            else:
                self.target_position = np.array([0.0, 0.0, 1.0])

        elif self.task == TaskType.WAYPOINT:
            # Generate waypoints
            n_waypoints = int(3 + self.difficulty * 7)  # 3-10 waypoints
            self.waypoints = [self._random_target() for _ in range(n_waypoints)]
            self.current_waypoint_idx = 0
            self.target_position = self.waypoints[0]

        elif self.task == TaskType.VELOCITY:
            # Track a target velocity
            speed = self.difficulty * 3.0  # Up to 3 m/s
            angle = self.np_random.uniform(0, 2 * np.pi)
            self.target_velocity = np.array([
                speed * np.cos(angle),
                speed * np.sin(angle),
                0.0
            ])
            self.target_position = self.sim.state.position.copy()

        elif self.task == TaskType.DELIVERY:
            # Set up delivery mission
            # Pickup location (package starts here, on the ground)
            pickup_range = 1.0 + self.difficulty * 2.0  # 1-3m from start
            pickup_angle = self.np_random.uniform(0, 2 * np.pi)
            self.pickup_position = np.array([
                pickup_range * np.cos(pickup_angle),
                pickup_range * np.sin(pickup_angle),
                0.0  # On ground
            ])

            # Drop zone location (opposite side, further away)
            dropzone_range = 2.0 + self.difficulty * 3.0  # 2-5m from start
            dropzone_angle = pickup_angle + np.pi + self.np_random.uniform(-0.5, 0.5)
            self.dropzone_position = np.array([
                dropzone_range * np.cos(dropzone_angle),
                dropzone_range * np.sin(dropzone_angle),
                0.0  # On ground
            ])

            # Initial target is pickup location (fly there first)
            self.target_position = self.pickup_position.copy()
            self.target_position[2] = 0.5  # Hover above pickup point

    def _update_task(self):
        """Update task state (e.g., advance to next waypoint)."""
        if self.task == TaskType.WAYPOINT:
            # Check if reached current waypoint
            dist = np.linalg.norm(self.sim.state.position - self.target_position)
            if dist < 0.3:
                self.current_waypoint_idx += 1
                if self.current_waypoint_idx < len(self.waypoints):
                    self.target_position = self.waypoints[self.current_waypoint_idx]

        elif self.task == TaskType.VELOCITY:
            # Update target position based on target velocity
            self.target_position += self.target_velocity * self.base_config.dt

        elif self.task == TaskType.DELIVERY:
            # Update target based on delivery phase
            pkg = self.sim.get_package_state()
            if pkg is not None:
                if self.delivery_phase == "pickup":
                    # Target is above pickup point
                    self.target_position = self.pickup_position.copy()
                    self.target_position[2] = 0.3  # Low altitude for pickup

                elif self.delivery_phase == "deliver":
                    # Package picked up, target is above drop zone
                    self.target_position = self.dropzone_position.copy()
                    self.target_position[2] = 1.5  # Higher for drop

                elif self.delivery_phase == "done":
                    # Mission complete, hover at current position
                    pass

    def _random_target(self) -> np.ndarray:
        """Generate a random target position."""
        range_xy = 2.0 * self.difficulty + 0.5
        range_z = 1.0 * self.difficulty + 0.5

        return np.array([
            self.np_random.uniform(-range_xy, range_xy),
            self.np_random.uniform(-range_xy, range_xy),
            self.np_random.uniform(0.5, 0.5 + range_z)
        ])

    def _get_initial_position(self) -> np.ndarray:
        """Get initial position with optional randomization."""
        base = np.array([0.0, 0.0, 1.0])
        if self.difficulty > 0.3:
            noise = self.np_random.uniform(-0.5, 0.5, 3) * self.difficulty
            noise[2] = abs(noise[2])  # Keep z positive
            base += noise
        return base

    def _get_initial_velocity(self) -> np.ndarray:
        """Get initial velocity with optional randomization."""
        if self.difficulty > 0.5:
            return self.np_random.uniform(-0.5, 0.5, 3) * self.difficulty
        return np.zeros(3)

    def _get_initial_orientation(self) -> Optional[np.ndarray]:
        """Get initial orientation with optional randomization."""
        if self.difficulty > 0.7:
            euler = self.np_random.uniform(-0.2, 0.2, 3) * self.difficulty
            from drone_ai.simulation import euler_to_quaternion
            return euler_to_quaternion(euler)
        return None

    def _apply_domain_randomization(self):
        """Apply domain randomization for sim-to-real transfer.

        Randomizes both physical drone parameters AND environmental conditions
        to make the agent robust to real-world variations.
        """
        config = DroneConfig()

        # === DRONE PHYSICAL PARAMETERS ===
        # Randomize mass (±20%)
        config.mass = self.base_config.mass * self.np_random.uniform(0.8, 1.2)

        # Randomize inertia (±15%)
        config.ixx = self.base_config.ixx * self.np_random.uniform(0.85, 1.15)
        config.iyy = self.base_config.iyy * self.np_random.uniform(0.85, 1.15)
        config.izz = self.base_config.izz * self.np_random.uniform(0.85, 1.15)

        # Randomize motor constant (±10%)
        config.motor_constant = self.base_config.motor_constant * self.np_random.uniform(0.9, 1.1)

        # Randomize drag (±30%)
        config.drag_coeff_xy = self.base_config.drag_coeff_xy * self.np_random.uniform(0.7, 1.3)
        config.drag_coeff_z = self.base_config.drag_coeff_z * self.np_random.uniform(0.7, 1.3)

        # Randomize motor response time (±25%)
        config.motor_time_constant = self.base_config.motor_time_constant * self.np_random.uniform(0.75, 1.25)

        # === ENVIRONMENTAL CONDITIONS ===
        env_config = EnvironmentConfig()

        # Wind conditions (scales with difficulty)
        wind_intensity = self.difficulty * self.np_random.uniform(0, 1)
        env_config.wind_speed = wind_intensity * 3.0  # Up to 3 m/s
        env_config.wind_direction = self.np_random.uniform(0, 2 * np.pi)
        env_config.wind_turbulence = wind_intensity * 0.5  # Random fluctuations
        env_config.wind_gust_probability = wind_intensity * 0.01  # Occasional gusts

        # Sensor noise (simulates real sensor imperfections)
        noise_level = self.difficulty * self.np_random.uniform(0.3, 1.0)
        env_config.position_noise = noise_level * 0.02  # Up to 2cm std dev
        env_config.velocity_noise = noise_level * 0.05  # Up to 5cm/s std dev
        env_config.orientation_noise = noise_level * 0.02  # Up to ~1 degree std dev
        env_config.motor_noise = noise_level * 0.05  # Up to 5% motor variation

        # Ground effects (always present, vary strength)
        env_config.ground_effect_height = self.np_random.uniform(0.2, 0.4)
        env_config.ground_effect_strength = self.np_random.uniform(0.05, 0.15)

        # Temperature effects (affects air density)
        env_config.temperature_offset = self.np_random.uniform(-15, 25)  # -15C to +25C from standard

        # Battery simulation (voltage sag under load)
        env_config.battery_voltage_drop = self.np_random.uniform(0, 0.1)  # Up to 10% drop

        # Store for observation noise
        self._env_config = env_config

        self.sim = DroneSimulation(config, self.package_config, env_config)

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about current state."""
        state = self.sim.state
        info = {
            'position': state.position.copy(),
            'velocity': state.velocity.copy(),
            'euler_angles': state.get_euler_angles(),
            'angular_velocity': state.angular_velocity.copy(),
            'target_position': self.target_position.copy(),
            'position_error': np.linalg.norm(state.position - self.target_position),
            'step': self.step_count,
            'episode_reward': self.episode_reward,
            'crashed': self.sim.is_crashed(),
        }

        # Add delivery-specific info
        if self.task == TaskType.DELIVERY:
            pkg = self.sim.get_package_state()
            info['delivery_phase'] = self.delivery_phase
            info['pickup_position'] = self.pickup_position.copy()
            info['dropzone_position'] = self.dropzone_position.copy()

            if pkg is not None:
                info['package_status'] = pkg.status.value
                info['package_position'] = pkg.position.copy()
                info['has_package'] = self.sim.has_package()
                info['package_delivered'] = self.sim.is_package_delivered()
                info['package_missed'] = self.sim.is_package_missed()
                info['delivery_accuracy'] = self.sim.get_delivery_accuracy()

        return info

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

    def _render_human(self):
        """Render using pygame."""
        # Lazy import and initialization
        if self._renderer is None:
            from drone_ai.visualization import DroneRenderer
            self._renderer = DroneRenderer()

        self._renderer.render(
            self.sim.state,
            self.target_position,
            self.position_history
        )

    def _render_rgb_array(self) -> np.ndarray:
        """Render to RGB array."""
        if self._renderer is None:
            from drone_ai.visualization import DroneRenderer
            self._renderer = DroneRenderer(headless=True)

        return self._renderer.render_to_array(
            self.sim.state,
            self.target_position,
            self.position_history
        )

    def close(self):
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# Register the environment with Gymnasium
def register_envs():
    """Register drone environments with Gymnasium."""
    gym.register(
        id="DroneHover-v0",
        entry_point="drone_ai.environment:DroneEnv",
        kwargs={"task": TaskType.HOVER, "difficulty": 0.3},
        max_episode_steps=1000
    )

    gym.register(
        id="DroneHover-v1",
        entry_point="drone_ai.environment:DroneEnv",
        kwargs={"task": TaskType.HOVER, "difficulty": 0.7, "domain_randomization": True},
        max_episode_steps=1000
    )

    gym.register(
        id="DroneWaypoint-v0",
        entry_point="drone_ai.environment:DroneEnv",
        kwargs={"task": TaskType.WAYPOINT, "difficulty": 0.5},
        max_episode_steps=2000
    )

    gym.register(
        id="DroneDelivery-v0",
        entry_point="drone_ai.environment:DroneEnv",
        kwargs={"task": TaskType.DELIVERY, "difficulty": 0.3},
        max_episode_steps=2000
    )

    gym.register(
        id="DroneDelivery-v1",
        entry_point="drone_ai.environment:DroneEnv",
        kwargs={"task": TaskType.DELIVERY, "difficulty": 0.7, "domain_randomization": True},
        max_episode_steps=3000
    )
