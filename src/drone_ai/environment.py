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
    PackageConfig, PackageState, PackageStatus, Obstacle
)


class TaskType(Enum):
    """Available training tasks."""
    HOVER = "hover"
    WAYPOINT = "waypoint"
    TRAJECTORY = "trajectory"
    VELOCITY = "velocity"
    DELIVERY = "delivery"  # Package pickup and drop mission
    DELIVERY_ROUTE = "delivery_route"  # Long-range round-trip delivery mission


class DroneEnv(gym.Env):
    """
    Gymnasium environment for drone flight control.

    Observation Space (always 31 dimensions for consistent network architecture):
        Base (20 dims):
        - Position (3): x, y, z in world frame (normalized)
        - Velocity (3): vx, vy, vz in world frame (normalized)
        - Orientation (3): roll, pitch, yaw angles
        - Angular velocity (3): p, q, r in body frame (normalized)
        - Target position (3): relative target in world frame (normalized)
        - Previous action (5): last motor commands + drop signal

        Extended (11 dims, filled based on task):
        - Package status (1): 0=waiting, 0.25=attached, 0.5=dropping, 0.75=delivered, 1=missed
        - Has package (1): binary flag
        - Location 1 relative (3): pickup/base position relative to drone
        - Location 2 relative (3): dropzone position relative to drone
        - Deliveries completed (1): normalized count
        - Drop prediction (1): predicted accuracy if dropped now
        - Obstacle proximity (1): distance to nearest obstacle

    Action Space:
        Standard tasks (4 dims): Motor commands [0, 1] (internally padded to 5)
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
        if task in [TaskType.DELIVERY, TaskType.DELIVERY_ROUTE]:
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

        # Define observation space - ALWAYS 31 dims for consistent network architecture
        # This allows progressive curriculum learning across all task types.
        # Base: [position(3), velocity(3), euler(3), angular_vel(3), target_rel(3), prev_action(5)]
        # Extended: [pkg_status(1), has_package(1), location1_rel(3), location2_rel(3),
        #            deliveries_completed(1), drop_prediction(1), obstacle_proximity(1)]
        # Total: 20 (base) + 11 (extended) = 31
        obs_dim = 31  # Fixed size for all tasks
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

        # Delivery route parameters (long-range round-trip delivery)
        self.base_position = np.zeros(3)  # Starting/reload position
        self.route_distance = 100.0  # Distance to dropzone in meters (scales with difficulty)
        self.drop_accuracy_radius = 5.0  # Must drop within 5m of target
        self.deliveries_completed = 0
        self.deliveries_successful = 0
        self.route_phase = "outbound"  # "outbound", "dropping", "return", "reload"
        self.route_score = 0.0  # Cumulative score for the route

        # Waypoints for varied routes (not just straight lines)
        self.route_waypoints: List[np.ndarray] = []
        self.current_waypoint_idx = 0

        # Obstacles (trees, buildings)
        self.obstacles: List[Obstacle] = []

        # Episode state
        self.step_count = 0
        self.prev_action = np.zeros(5)  # Always 5 dims for consistent observation space
        self.episode_reward = 0.0
        self.position_history: List[np.ndarray] = []

        # Reward weights (tunable)
        # REBALANCED: Rewards should be mostly positive to encourage learning
        # Total reward per step should be around +0.5 to +2.0 for good behavior
        self.reward_weights = {
            # === CORE REWARDS (positive shaping) ===
            'alive': 0.5,                 # Guaranteed positive per step (survival is good!)
            'in_zone': 1.0,               # Bonus for being within hover zone (0.5m)
            'centered': 0.5,              # Extra bonus for being well-centered in zone
            'stable': 0.3,                # Bonus for low velocity (stable hover)
            'level': 0.2,                 # Bonus for level orientation

            # === PENALTIES (smaller, linear, capped) ===
            'position_outside': 0.3,      # Penalty per meter outside zone (linear, not squared)
            'velocity': 0.1,              # Small velocity penalty (capped)
            'orientation': 0.1,           # Small tilt penalty (capped)
            'angular_velocity': 0.05,     # Small rotation penalty (capped)
            'action_smoothness': 0.1,     # Small jerk penalty (capped)

            # === TERMINAL PENALTIES ===
            'crash': -10.0,               # Crash penalty (reduced from -50)
            'upside_down': -2.0,          # Upside down penalty (reduced from -15)
            'extreme_velocity': -3.0,     # Penalty for going too fast (>20 m/s)
            'extreme_spin': -3.0,         # Penalty for spinning too fast (>50 rad/s)
            'success': 1.0,               # Extra bonus for perfect hover

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

            # === DELIVERY ROUTE REWARDS (long-range) ===
            'route_delivery': 100.0,      # Big reward for successful delivery in route
            'route_accuracy': 50.0,       # Bonus for accuracy (within 5m)
            'route_missed': -80.0,        # Heavy penalty for missing (outside 5m)
            'route_reload': 20.0,         # Bonus for successful return and reload
            'route_progress': 0.01,       # Small reward for progress toward target

            # Smart dropping (drop while moving)
            'smart_drop_bonus': 30.0,     # Bonus for accurate drop while moving fast
            'speed_delivery_bonus': 0.5,  # Bonus multiplier for faster deliveries

            # Obstacle avoidance
            'obstacle_proximity': -0.5,   # Penalty for being too close to obstacles
            'obstacle_collision': -100.0, # Heavy penalty for hitting obstacles
            'waypoint_reached': 5.0,      # Bonus for reaching waypoints

            # Out of bounds penalty (prevents exploit of flying away)
            'out_of_bounds': -5.0,        # Heavy per-step penalty for being out of bounds
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
        elif self.task == TaskType.DELIVERY_ROUTE:
            # Start at base with package already attached
            self.sim.reset(
                position=self.base_position.copy(),
                velocity=init_velocity,
                orientation=init_orientation,
                package_pickup=self.base_position,  # Pickup at base
                package_dropzone=self.dropzone_position
            )
            # Add obstacles to simulation
            self.sim.obstacles = self.obstacles
            self.sim._obstacle_collision = False

            # Immediately attach package (start with package)
            if self.sim.package is not None:
                self.sim.package.status = PackageStatus.ATTACHED
                self.sim.package.position = self.base_position.copy()
            self.route_phase = "outbound"
            self.deliveries_completed = 0
            self.deliveries_successful = 0
            self.route_score = 0.0
            self.pickup_step = 0  # Start timing from beginning
            self.current_waypoint_idx = 0
        else:
            self.sim.reset(
                position=init_position,
                velocity=init_velocity,
                orientation=init_orientation
            )

        # Reset episode state
        self.step_count = 0
        hover_action = self.sim.compute_hover_action()
        # Always use 5-dim action for consistent observation space
        self.prev_action = np.concatenate([hover_action, [0.0]])  # 4 motors + drop signal
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

        # Ensure action is always 5 dimensions for consistent observation space
        if len(action) == 4:
            action = np.concatenate([action, [0.0]])  # Add zero drop signal

        # Handle delivery task with drop signal
        if self.task in [TaskType.DELIVERY, TaskType.DELIVERY_ROUTE]:
            motor_action = action[:4]
            drop_signal = action[4] > 0.5  # Threshold for drop command
            self.sim.step(motor_action, drop_package=drop_signal)
        else:
            self.sim.step(action[:4])  # Only use motor commands

        # NO CRASH RESETS - drone cannot die or reset, must fly properly
        # Bad behaviors get continuous penalties in _compute_reward()

        # Record position for trajectory visualization
        self.position_history.append(self.sim.state.position.copy())

        # Compute reward
        reward = self._compute_reward(action)
        self.episode_reward += reward

        # Check termination conditions (crashes no longer terminate!)
        terminated = self._check_terminated()
        truncated = self.step_count >= self.max_steps

        # Update task state (e.g., move to next waypoint)
        self._update_task()

        # Store action for smoothness penalty (always 5 dims)
        self.prev_action = action.copy()

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector from current state.

        Always returns 31 dimensions for consistent network architecture:
        - Base (20): position(3), velocity(3), euler(3), angular_vel(3), target_rel(3), prev_action(5)
        - Extended (11): pkg_status(1), has_package(1), location1_rel(3), location2_rel(3),
                         deliveries_completed(1), drop_prediction(1), obstacle_proximity(1)
        """
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

        # Previous action (always 5 dims)
        prev_action = self.prev_action

        # Base observation (20 dims)
        base_obs = [
            position,           # 3
            velocity,           # 3
            euler,              # 3
            angular_velocity,   # 3
            target_rel,         # 3
            prev_action         # 5
        ]

        # Extended observations (11 dims) - filled based on task type
        # Default values for non-delivery tasks
        pkg_status = np.array([0.0])
        has_package = np.array([0.0])
        location1_rel = np.zeros(3)  # pickup/base position relative
        location2_rel = np.zeros(3)  # dropzone position relative
        deliveries_norm = np.array([0.0])
        drop_prediction = np.array([2.0])  # No package/not applicable
        obstacle_proximity = np.array([1.0])  # No obstacles

        # Fill in delivery-specific observations
        if self.task == TaskType.DELIVERY:
            pkg = self.sim.get_package_state()
            if pkg is not None:
                status_map = {
                    PackageStatus.WAITING: 0.0,
                    PackageStatus.ATTACHED: 0.25,
                    PackageStatus.DROPPING: 0.5,
                    PackageStatus.DELIVERED: 0.75,
                    PackageStatus.MISSED: 1.0
                }
                pkg_status = np.array([status_map.get(pkg.status, 0.0)])
                has_package = np.array([1.0 if self.sim.has_package() else 0.0])
                location1_rel = (pkg.pickup_position - state.position) / 5.0
                location2_rel = (pkg.dropzone_position - state.position) / 5.0

        elif self.task == TaskType.DELIVERY_ROUTE:
            pkg = self.sim.get_package_state()
            status_map = {
                PackageStatus.WAITING: 0.0,
                PackageStatus.ATTACHED: 0.25,
                PackageStatus.DROPPING: 0.5,
                PackageStatus.DELIVERED: 0.75,
                PackageStatus.MISSED: 1.0
            }
            if pkg is not None:
                pkg_status = np.array([status_map.get(pkg.status, 0.0)])
                has_package = np.array([1.0 if self.sim.has_package() else 0.0])

            # Relative positions to base and dropzone (normalized for longer distances)
            location1_rel = (self.base_position - state.position) / self.route_distance
            location2_rel = (self.dropzone_position - state.position) / self.route_distance

            # Number of deliveries completed (normalized)
            deliveries_norm = np.array([self.deliveries_completed / 10.0])

            # PREDICTIVE DROP: How accurate would a drop be right now?
            if self.sim.has_package():
                pred_accuracy = self.sim.get_drop_accuracy_prediction(self.dropzone_position)
                drop_prediction = np.array([min(2.0, pred_accuracy / self.drop_accuracy_radius)])

            # Nearest obstacle distance (normalized)
            if self.obstacles:
                nearest_obs = self.sim.get_nearest_obstacle_distance()
                obstacle_proximity = np.array([min(1.0, nearest_obs / 20.0)])

        # Combine all observations (always 31 dims)
        base_obs.extend([
            pkg_status,         # 1
            has_package,        # 1
            location1_rel,      # 3
            location2_rel,      # 3
            deliveries_norm,    # 1
            drop_prediction,    # 1
            obstacle_proximity  # 1
        ])

        observation = np.concatenate(base_obs).astype(np.float32)

        return observation

    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward for current state and action.

        REBALANCED reward system:
        - Mostly positive rewards for good behavior
        - Small, capped penalties for mistakes
        - Linear penalties (not squared) for better gradients
        - Target: +0.5 to +2.0 per step for good behavior
        """
        state = self.sim.state
        weights = self.reward_weights
        reward = 0.0

        # === GUARANTEED POSITIVE: Alive bonus ===
        reward += weights['alive']  # +0.5 just for being alive

        # === POSITION REWARD (hover zone) ===
        pos_error = np.linalg.norm(state.position - self.target_position)
        hover_zone_radius = 0.5  # Must stay within 0.5m radius

        if pos_error <= hover_zone_radius:
            # Inside hover zone - BIG positive reward!
            reward += weights['in_zone']  # +1.0 for being in zone

            # Extra reward for being centered (closer to center = more reward)
            # Scales from 0.5 (at edge) to 0.5 (at center)
            centered_ratio = 1.0 - (pos_error / hover_zone_radius)
            reward += weights['centered'] * centered_ratio  # Up to +0.5 for center
        else:
            # Outside zone - LINEAR penalty (not squared!)
            # Capped at 3m outside to prevent explosion
            excess_error = min(pos_error - hover_zone_radius, 3.0)
            reward -= weights['position_outside'] * excess_error  # Max -0.9

        # === STABILITY REWARDS ===
        velocity = np.linalg.norm(state.velocity)
        euler = state.get_euler_angles()
        tilt = abs(euler[0]) + abs(euler[1])  # Roll + pitch

        # Reward for low velocity (stable)
        if velocity < 0.5:
            reward += weights['stable']  # +0.3 for very stable
        elif velocity < 1.0:
            reward += weights['stable'] * 0.5  # +0.15 for somewhat stable

        # Reward for level orientation
        if tilt < 0.2:  # ~11 degrees
            reward += weights['level']  # +0.2 for level flight
        elif tilt < 0.4:  # ~23 degrees
            reward += weights['level'] * 0.5  # +0.1 for somewhat level

        # === SMALL PENALTIES (linear, capped) ===
        # Velocity penalty (only if moving fast)
        if velocity > 1.0:
            vel_penalty = min(velocity - 1.0, 2.0) * weights['velocity']
            reward -= vel_penalty  # Max -0.2

        # Orientation penalty (only if tilted significantly)
        if tilt > 0.3:  # ~17 degrees
            tilt_penalty = min(tilt - 0.3, 1.0) * weights['orientation']
            reward -= tilt_penalty  # Max -0.1

        # Angular velocity penalty (only if spinning fast)
        ang_vel = np.linalg.norm(state.angular_velocity)
        if ang_vel > 1.0:
            ang_penalty = min(ang_vel - 1.0, 2.0) * weights['angular_velocity']
            reward -= ang_penalty  # Max -0.1

        # Action smoothness (only penalize large changes)
        motor_action = action[:4]
        prev_motor = self.prev_action[:4]
        action_diff = np.linalg.norm(motor_action - prev_motor)
        if action_diff > 0.2:
            smooth_penalty = min(action_diff - 0.2, 0.5) * weights['action_smoothness']
            reward -= smooth_penalty  # Max -0.05

        # === SUCCESS BONUS ===
        # Perfect hover: in zone, stable, and level
        if pos_error < hover_zone_radius * 0.5 and velocity < 0.3 and tilt < 0.1:
            reward += weights['success']  # +1.0 for perfect hover

        # === COLLISION PENALTIES (no death, just penalty) ===
        # Ground collision penalty (except at safe zones like dropzone/pickup)
        if state.position[2] < 0.05:
            # Check if at safe zone (purple dropzone pad or pickup zone)
            at_safe_zone = False
            if self.task in [TaskType.DELIVERY, TaskType.DELIVERY_ROUTE]:
                # Dropzone is safe - purple pad
                dist_to_dropzone = np.linalg.norm(state.position[:2] - self.dropzone_position[:2])
                if dist_to_dropzone < 1.0:  # Within 1m of dropzone center
                    at_safe_zone = True
                # Pickup zone is also safe
                dist_to_pickup = np.linalg.norm(state.position[:2] - self.pickup_position[:2])
                if dist_to_pickup < 1.0:  # Within 1m of pickup center
                    at_safe_zone = True

            if not at_safe_zone:
                reward += weights['crash']  # -10.0 for hitting ground outside safe zones

        # Obstacle collision penalty
        if self.sim.check_obstacle_collision():
            reward += weights['obstacle_collision']  # -100.0 for hitting obstacle

        # Upside-down penalty (severe tilt > 60 degrees)
        tilt_threshold = np.pi / 3  # 60 degrees
        if abs(euler[0]) > tilt_threshold or abs(euler[1]) > tilt_threshold:
            reward += weights['upside_down']  # -2.0 for being upside down

        # Extreme velocity penalty (prevents exploit - penalize but don't terminate)
        speed = np.linalg.norm(state.velocity)
        if speed > 20.0:  # Going faster than 20 m/s
            reward += weights['extreme_velocity']  # -3.0 per step

        # Extreme spin penalty (prevents exploit - penalize but don't terminate)
        spin = np.linalg.norm(state.angular_velocity)
        if spin > 50.0:  # Spinning faster than 50 rad/s
            reward += weights['extreme_spin']  # -3.0 per step

        # Out of bounds penalty (prevents exploit of flying away to lock in rewards)
        position = state.position
        if self.task == TaskType.DELIVERY_ROUTE:
            max_dist = self.route_distance * 1.5
            if np.any(np.abs(position[:2]) > max_dist) or position[2] > 50 or position[2] < -0.5:
                reward += weights['out_of_bounds']  # -5.0 per step out of bounds
        else:
            if np.any(np.abs(position[:2]) > 10) or position[2] > 20 or position[2] < -0.5:
                reward += weights['out_of_bounds']  # -5.0 per step out of bounds

        # === DELIVERY-SPECIFIC REWARDS ===
        if self.task == TaskType.DELIVERY:
            reward += self._compute_delivery_reward(action)
        elif self.task == TaskType.DELIVERY_ROUTE:
            reward += self._compute_route_reward(action)

        return float(reward)

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

    def _compute_route_reward(self, action: np.ndarray) -> float:
        """Compute delivery route rewards (long-range round-trip).

        Priorities:
        1. ACCURACY - Drop within 5m of target for reward, outside for penalty
        2. CAREFULNESS - Smooth handling during flight
        3. SPEED - Faster round trips = more reward
        """
        weights = self.reward_weights
        reward = 0.0

        pkg = self.sim.get_package_state()
        state = self.sim.state

        # === PROGRESS REWARD (small continuous reward for moving toward target) ===
        if self.route_phase == "outbound" and pkg is not None:
            dist_to_dropzone = np.linalg.norm(state.position[:2] - self.dropzone_position[:2])
            # Reward for getting closer to dropzone
            progress = self.route_distance - dist_to_dropzone
            reward += weights['route_progress'] * max(0, progress)

        elif self.route_phase == "return":
            dist_to_base = np.linalg.norm(state.position[:2] - self.base_position[:2])
            # Reward for getting closer to base
            progress = self.route_distance - dist_to_base
            reward += weights['route_progress'] * max(0, progress)

        # === WAYPOINT NAVIGATION ===
        if self.route_phase == "outbound" and self.route_waypoints:
            if self.current_waypoint_idx < len(self.route_waypoints):
                current_wp = self.route_waypoints[self.current_waypoint_idx]
                dist_to_wp = np.linalg.norm(state.position - current_wp)

                if dist_to_wp < 5.0:  # Reached waypoint (within 5m)
                    reward += weights['waypoint_reached']
                    self.current_waypoint_idx += 1
                    # Update target to next waypoint or dropzone
                    if self.current_waypoint_idx < len(self.route_waypoints):
                        self.target_position = self.route_waypoints[self.current_waypoint_idx].copy()
                    else:
                        self.target_position = self.dropzone_position.copy()
                        self.target_position[2] = 10.0  # Approach altitude

        # === OBSTACLE AVOIDANCE ===
        if self.obstacles:
            nearest_dist = self.sim.get_nearest_obstacle_distance()
            if nearest_dist < 5.0:  # Within 5m of obstacle
                proximity_penalty = weights['obstacle_proximity'] * (5.0 - nearest_dist)
                reward += proximity_penalty

        # === CAREFULNESS: Rough handling penalty while carrying ===
        if pkg is not None and pkg.status == PackageStatus.ATTACHED:
            motor_action = action[:4]
            prev_motor = self.prev_action[:4]
            action_jerk = np.linalg.norm(motor_action - prev_motor)
            if action_jerk > 0.1:
                reward += weights['rough_handling'] * action_jerk

            euler = state.get_euler_angles()
            tilt = abs(euler[0]) + abs(euler[1])
            if tilt > 0.3:
                reward += weights['rough_handling'] * tilt

        # === DELIVERY RESULT ===
        if pkg is not None and pkg.status in [PackageStatus.DELIVERED, PackageStatus.MISSED]:
            if self.route_phase == "outbound":
                accuracy = self.sim.get_delivery_accuracy()
                if accuracy is not None:
                    if accuracy <= self.drop_accuracy_radius:
                        # SUCCESS: Within 5m - big reward!
                        reward += weights['route_delivery']
                        # Accuracy bonus: closer = more reward
                        accuracy_bonus = weights['route_accuracy'] * (1 - accuracy / self.drop_accuracy_radius)
                        reward += accuracy_bonus

                        # SMART DROP BONUS: Extra reward for dropping while moving fast
                        # This encourages the drone to calculate drop timing, not hover
                        speed = np.linalg.norm(state.velocity[:2])  # Horizontal speed
                        if speed > 2.0:  # Moving faster than 2 m/s
                            smart_drop_bonus = weights['smart_drop_bonus'] * min(1.0, speed / 5.0)
                            reward += smart_drop_bonus
                            self.route_score += smart_drop_bonus

                        self.deliveries_successful += 1
                        self.route_score += 100 + accuracy_bonus
                    else:
                        # MISSED: Outside 5m - heavy penalty for wrong drop
                        reward += weights['route_missed']  # -80 base penalty
                        # Extra penalty based on how far outside - scales harshly
                        overshoot = accuracy - self.drop_accuracy_radius
                        # Quadratic penalty for very wrong drops
                        wrong_drop_penalty = min(100, overshoot * overshoot)
                        reward -= wrong_drop_penalty
                        self.route_score -= 50 + wrong_drop_penalty

                self.deliveries_completed += 1
                self.route_phase = "return"
                # Reset waypoint index for return trip (go straight back)
                self.current_waypoint_idx = 0
                # Update target to base for return trip
                self.target_position = self.base_position.copy()
                self.target_position[2] = 1.0  # Fly at 1m altitude

        # === RELOAD AT BASE ===
        if self.route_phase == "return":
            dist_to_base = np.linalg.norm(state.position[:2] - self.base_position[:2])
            altitude = state.position[2]

            # Check if back at base and low enough to reload
            if dist_to_base < 1.0 and altitude < 0.5:
                reward += weights['route_reload']
                self.route_score += 20

                # Reset package for next delivery
                self.route_phase = "reload"

        # === RELOAD COMPLETE - Start next delivery ===
        if self.route_phase == "reload":
            # Respawn package at base, attached to drone
            if self.sim.package is not None:
                self.sim.package.status = PackageStatus.ATTACHED
                self.sim.package.position = state.position.copy()
                self.sim.package.pickup_position = self.base_position.copy()
            self.route_phase = "outbound"
            self.target_position = self.dropzone_position.copy()
            self.target_position[2] = 1.5  # Higher altitude for dropping

        return reward

    def _check_terminated(self) -> bool:
        """Check if episode should terminate.

        IMPORTANT: Crashes NO LONGER terminate! This prevents AI from exploiting
        death to lock in rewards. Crashes just reset position with penalty.
        Only legitimate task completion ends the episode.
        """
        # CRASHES DO NOT TERMINATE - handled in step() with position reset

        # Delivery task - terminate only when package is delivered or missed
        if self.task == TaskType.DELIVERY:
            if self.sim.is_package_delivered() or self.sim.is_package_missed():
                return True

        # All other tasks run until max_steps (truncation)
        return False

    def _setup_task(self):
        """Set up task-specific parameters."""
        if self.task == TaskType.HOVER:
            # Hover at a fixed or random position
            if self.difficulty > 0.5:
                self.target_position = self._random_target()
            else:
                self.target_position = np.array([0.0, 0.0, 2.0])  # 2m hover height

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

        elif self.task == TaskType.DELIVERY_ROUTE:
            # Long-range delivery route setup
            # Base position is at origin (starting point)
            self.base_position = np.array([0.0, 0.0, 1.0])  # Start hovering at 1m

            # Route distance scales with difficulty: 50m to 1000m (1km)
            # Lower difficulty = shorter routes for learning
            # difficulty 0.0 -> 50m, difficulty 1.0 -> 1000m
            self.route_distance = 50.0 + self.difficulty * 950.0

            # Drop zone is route_distance away in a random direction
            dropzone_angle = self.np_random.uniform(0, 2 * np.pi)
            self.dropzone_position = np.array([
                self.route_distance * np.cos(dropzone_angle),
                self.route_distance * np.sin(dropzone_angle),
                0.0  # On ground
            ])

            # Accuracy radius is 5m (as specified)
            self.drop_accuracy_radius = 5.0

            # Generate waypoints for varied routes (not straight lines)
            self._generate_route_waypoints()

            # Generate obstacles along the route
            self._generate_obstacles()

            # Initial target is first waypoint (or dropzone if no waypoints)
            if self.route_waypoints:
                self.current_waypoint_idx = 0
                self.target_position = self.route_waypoints[0].copy()
            else:
                self.target_position = self.dropzone_position.copy()
                self.target_position[2] = 1.5  # Higher altitude for dropping

            # Reset route state
            self.route_phase = "outbound"
            self.deliveries_completed = 0
            self.deliveries_successful = 0
            self.route_score = 0.0

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

    def _generate_route_waypoints(self):
        """Generate waypoints for varied route (not straight line)."""
        self.route_waypoints = []

        # Number of waypoints scales with difficulty and distance
        num_waypoints = int(1 + self.difficulty * 3)  # 1-4 waypoints

        # Direction from base to dropzone
        direction = self.dropzone_position[:2] - self.base_position[:2]
        total_dist = np.linalg.norm(direction)
        if total_dist < 1:
            return

        direction_norm = direction / total_dist

        # Perpendicular direction for offsets
        perp = np.array([-direction_norm[1], direction_norm[0]])

        for i in range(num_waypoints):
            # Position along the route
            t = (i + 1) / (num_waypoints + 1)
            base_pos = self.base_position[:2] + direction * t

            # Random lateral offset (makes route curved/varied)
            max_offset = self.route_distance * 0.2 * self.difficulty  # Up to 20% of distance
            offset = self.np_random.uniform(-max_offset, max_offset)

            waypoint = np.array([
                base_pos[0] + perp[0] * offset,
                base_pos[1] + perp[1] * offset,
                self.np_random.uniform(5, 15)  # Vary altitude 5-15m
            ])
            self.route_waypoints.append(waypoint)

        # Add final approach to dropzone
        final_approach = self.dropzone_position.copy()
        final_approach[2] = 10.0  # Approach at 10m altitude
        self.route_waypoints.append(final_approach)

    def _generate_obstacles(self):
        """Generate obstacles (trees, buildings) along the route."""
        self.obstacles = []

        # Number of obstacles scales with difficulty
        num_obstacles = int(self.difficulty * 15)  # 0-15 obstacles

        if num_obstacles == 0:
            return

        # Direction from base to dropzone
        direction = self.dropzone_position[:2] - self.base_position[:2]
        total_dist = np.linalg.norm(direction)
        if total_dist < 10:
            return

        direction_norm = direction / total_dist
        perp = np.array([-direction_norm[1], direction_norm[0]])

        for _ in range(num_obstacles):
            # Position along route corridor (not too close to endpoints)
            t = self.np_random.uniform(0.1, 0.9)
            base_pos = self.base_position[:2] + direction * t

            # Lateral offset (within corridor)
            corridor_width = min(50, self.route_distance * 0.1)
            lateral_offset = self.np_random.uniform(-corridor_width, corridor_width)

            pos = np.array([
                base_pos[0] + perp[0] * lateral_offset,
                base_pos[1] + perp[1] * lateral_offset,
                0.0
            ])

            # Random obstacle type
            obs_type = self.np_random.choice(["tree", "building", "pole"])
            if obs_type == "tree":
                radius = self.np_random.uniform(1, 3)
                height = self.np_random.uniform(5, 15)
            elif obs_type == "building":
                radius = self.np_random.uniform(5, 15)
                height = self.np_random.uniform(8, 25)
            else:  # pole
                radius = self.np_random.uniform(0.5, 1)
                height = self.np_random.uniform(10, 30)

            # Don't place obstacle too close to waypoints or dropzone
            too_close = False
            for wp in self.route_waypoints:
                if np.linalg.norm(pos[:2] - wp[:2]) < radius + 10:
                    too_close = True
                    break
            if np.linalg.norm(pos[:2] - self.dropzone_position[:2]) < radius + 10:
                too_close = True
            if np.linalg.norm(pos[:2] - self.base_position[:2]) < radius + 10:
                too_close = True

            if not too_close:
                self.obstacles.append(Obstacle(
                    position=pos,
                    radius=radius,
                    height=height,
                    obstacle_type=obs_type
                ))

    def _get_initial_position(self) -> np.ndarray:
        """Get initial position with optional randomization."""
        # For DELIVERY task, spawn at the pickup zone (purple zone) ON THE GROUND
        if self.task == TaskType.DELIVERY:
            base = self.pickup_position.copy()
            base[2] = 0.1  # Start on ground (just slightly above to avoid collision)
        elif self.task == TaskType.DELIVERY_ROUTE:
            base = self.base_position.copy()
            base[2] = 0.1  # Start on ground at base
        else:
            base = np.array([0.0, 0.0, 2.0])  # Hover task starts at 2m height

        if self.difficulty > 0.3:
            noise = self.np_random.uniform(-0.3, 0.3, 3) * self.difficulty
            noise[2] = abs(noise[2]) * 0.5  # Smaller vertical noise for ground starts
            base += noise
            base[2] = max(0.1, base[2])  # Never go below ground
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

        # Add delivery route info (long-range)
        elif self.task == TaskType.DELIVERY_ROUTE:
            pkg = self.sim.get_package_state()
            info['route_phase'] = self.route_phase
            info['base_position'] = self.base_position.copy()
            info['dropzone_position'] = self.dropzone_position.copy()
            info['route_distance'] = self.route_distance
            info['drop_accuracy_radius'] = self.drop_accuracy_radius
            info['deliveries_completed'] = self.deliveries_completed
            info['deliveries_successful'] = self.deliveries_successful
            info['route_score'] = self.route_score

            if pkg is not None:
                info['package_status'] = pkg.status.value
                info['package_position'] = pkg.position.copy()
                info['has_package'] = self.sim.has_package()
                info['delivery_accuracy'] = self.sim.get_delivery_accuracy()

            # Distance to current target
            if self.route_phase == "outbound":
                info['distance_to_dropzone'] = np.linalg.norm(
                    state.position[:2] - self.dropzone_position[:2]
                )
            else:
                info['distance_to_base'] = np.linalg.norm(
                    state.position[:2] - self.base_position[:2]
                )

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

    # Long-range delivery route environments
    gym.register(
        id="DroneDeliveryRoute-v0",
        entry_point="drone_ai.environment:DroneEnv",
        kwargs={"task": TaskType.DELIVERY_ROUTE, "difficulty": 0.3},
        max_episode_steps=10000  # Longer episodes for long-range flight
    )

    gym.register(
        id="DroneDeliveryRoute-v1",
        entry_point="drone_ai.environment:DroneEnv",
        kwargs={"task": TaskType.DELIVERY_ROUTE, "difficulty": 0.7, "domain_randomization": True},
        max_episode_steps=20000  # Even longer for multiple deliveries
    )

    gym.register(
        id="DroneDeliveryRoute-v2",
        entry_point="drone_ai.environment:DroneEnv",
        kwargs={"task": TaskType.DELIVERY_ROUTE, "difficulty": 1.0, "domain_randomization": True},
        max_episode_steps=50000  # Full 1km routes with domain randomization
    )
