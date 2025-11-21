"""
Gymnasium Environment for Drone Flight Training

This module provides a Gymnasium-compatible environment for training
reinforcement learning agents to control a quadcopter drone.

The environment supports multiple task types:
- Hover: Maintain a stable position
- Waypoint: Navigate to target positions
- Trajectory: Follow a predefined path
- Acrobatic: Perform specific maneuvers
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple, List
from enum import Enum

from drone_ai.simulation import DroneSimulation, DroneConfig, DroneState


class TaskType(Enum):
    """Available training tasks."""
    HOVER = "hover"
    WAYPOINT = "waypoint"
    TRAJECTORY = "trajectory"
    VELOCITY = "velocity"


class DroneEnv(gym.Env):
    """
    Gymnasium environment for drone flight control.

    Observation Space (18 dimensions):
        - Position (3): x, y, z in world frame
        - Velocity (3): vx, vy, vz in world frame
        - Orientation (3): roll, pitch, yaw angles
        - Angular velocity (3): p, q, r in body frame
        - Target position (3): relative target in world frame
        - Previous action (3): last commanded attitude + thrust

    Action Space (4 dimensions):
        - Motor commands: 4 normalized motor speeds [0, 1]
        OR
        - Attitude control: [thrust, roll_cmd, pitch_cmd, yaw_rate_cmd]

    Rewards:
        - Position tracking reward
        - Velocity penalty
        - Attitude stability reward
        - Action smoothness reward
        - Crash penalty
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        task: TaskType = TaskType.HOVER,
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
        domain_randomization: bool = False,
        difficulty: float = 0.5,
        config: Optional[DroneConfig] = None
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
        """
        super().__init__()

        self.task = task
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.domain_randomization = domain_randomization
        self.difficulty = np.clip(difficulty, 0, 1)

        # Initialize simulation
        self.base_config = config or DroneConfig()
        self.sim = DroneSimulation(self.base_config)

        # Define action space: 4 motor commands normalized to [0, 1]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # Define observation space
        # [position(3), velocity(3), euler(3), angular_vel(3), target_rel(3), prev_action(4)]
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

        # Episode state
        self.step_count = 0
        self.prev_action = np.zeros(4)
        self.episode_reward = 0.0
        self.position_history: List[np.ndarray] = []

        # Reward weights (tunable)
        self.reward_weights = {
            'position': 1.0,
            'velocity': 0.1,
            'orientation': 0.2,
            'angular_velocity': 0.05,
            'action_smoothness': 0.1,
            'alive': 0.1,
            'crash': -10.0,
            'success': 5.0
        }

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
            self.sim = DroneSimulation(self.base_config)

        # Reset drone state with optional randomization
        init_position = self._get_initial_position()
        init_velocity = self._get_initial_velocity()
        init_orientation = self._get_initial_orientation()

        self.sim.reset(
            position=init_position,
            velocity=init_velocity,
            orientation=init_orientation
        )

        # Set up task-specific targets
        self._setup_task()

        # Reset episode state
        self.step_count = 0
        self.prev_action = self.sim.compute_hover_action()
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

        # Step the simulation
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

        observation = np.concatenate([
            position,
            velocity,
            euler,
            angular_velocity,
            target_rel,
            prev_action
        ]).astype(np.float32)

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
        action_diff = np.linalg.norm(action - self.prev_action)
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

        total_reward = (
            pos_reward +
            vel_penalty +
            orient_penalty +
            ang_vel_penalty +
            smoothness_penalty +
            alive_bonus +
            success_bonus +
            crash_penalty
        )

        return float(total_reward)

    def _check_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Crash detection
        if self.sim.is_crashed():
            return True

        # Out of bounds
        position = self.sim.state.position
        if np.any(np.abs(position[:2]) > 10) or position[2] > 20:
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
        """Apply domain randomization for sim-to-real transfer."""
        config = DroneConfig()

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

        self.sim = DroneSimulation(config)

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about current state."""
        state = self.sim.state
        return {
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
