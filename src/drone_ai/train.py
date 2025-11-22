"""
Training Script for Drone AI

This script provides a complete training pipeline for the drone flight controller:
- Training with PPO
- Multi-environment parallel training (multiple drones)
- GPU acceleration
- Logging to TensorBoard
- Checkpointing
- Curriculum learning support
- Domain randomization scheduling
"""

import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from drone_ai.environment import DroneEnv, TaskType
from drone_ai.agent import PPOAgent, PPOConfig


def get_device_info() -> str:
    """Get detailed device information."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"GPU: {gpu_name} ({gpu_memory:.1f} GB)"
    else:
        return "CPU only (no GPU detected - training will be slower)"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Drone AI")

    # Environment settings
    parser.add_argument("--task", type=str, default="hover",
                       choices=["hover", "waypoint", "trajectory", "velocity", "delivery", "delivery_route"],
                       help="Training task")
    parser.add_argument("--difficulty", type=float, default=0.3,
                       help="Task difficulty [0, 1]")
    parser.add_argument("--domain-randomization", action="store_true",
                       help="Enable domain randomization for sim-to-real")

    # Parallel environments
    parser.add_argument("--num-envs", type=int, default=1,
                       help="Number of parallel environments (drones) for faster training")

    # Training settings
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                       help="Total training timesteps")
    parser.add_argument("--n-steps", type=int, default=2048,
                       help="Steps per environment per update")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10,
                       help="PPO epochs per update")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")

    # Curriculum learning
    parser.add_argument("--curriculum", action="store_true",
                       help="Use curriculum learning (gradually increase difficulty)")

    # Logging and saving
    parser.add_argument("--log-dir", type=str, default="logs",
                       help="Directory for logs")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                       help="Directory for checkpoints")
    parser.add_argument("--save-freq", type=int, default=50000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--log-freq", type=int, default=1000,
                       help="Log metrics every N steps")
    parser.add_argument("--eval-freq", type=int, default=10000,
                       help="Evaluate every N steps")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Number of evaluation episodes")

    # Visualization
    parser.add_argument("--render", action="store_true",
                       help="Enable live visualization during training")
    parser.add_argument("--render-freq", type=int, default=1,
                       help="Render every N steps (1=every step, higher=faster training)")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cpu, cuda)")
    parser.add_argument("--name", type=str, default=None,
                       help="Experiment name")

    return parser.parse_args()


class Trainer:
    """Training manager for drone AI."""

    def __init__(self, args):
        self.args = args
        self.num_envs = args.num_envs

        # Set up experiment name
        if args.name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.name = f"drone_{args.task}_{timestamp}"

        # Create directories
        self.log_dir = Path(args.log_dir) / args.name
        self.save_dir = Path(args.save_dir) / args.name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = self.save_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)

        # Set random seeds
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

        # Create environment(s)
        self.task = TaskType(args.task)
        self.difficulty = args.difficulty

        # Create parallel environments if num_envs > 1
        if self.num_envs > 1:
            self.envs = [self._create_env(self.difficulty, seed=args.seed + i)
                        for i in range(self.num_envs)]
            self.env = self.envs[0]  # Reference for dimensions
        else:
            self.env = self._create_env(self.difficulty)
            self.envs = [self.env]

        # Create agent with adjusted buffer size for parallel envs
        ppo_config = PPOConfig(
            learning_rate=args.lr,
            n_steps=args.n_steps * self.num_envs,  # More data per update
            batch_size=args.batch_size,
            n_epochs=args.n_epochs
        )

        self.agent = PPOAgent(
            obs_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0],
            config=ppo_config,
            device=args.device
        )

        # TensorBoard writer
        self.writer = SummaryWriter(str(self.log_dir))

        # Training state
        self.total_steps = 0
        self.episodes = 0
        self.best_reward = float('-inf')

        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = 0.0

        # Per-environment tracking
        self.env_episode_rewards = [0.0] * self.num_envs
        self.env_episode_lengths = [0] * self.num_envs

        # Visualization (only for first environment)
        self.renderer = None
        if args.render:
            from drone_ai.visualization import DroneRenderer
            self.renderer = DroneRenderer(width=1024, height=768)
            print("Live visualization enabled (showing drone 1)")

    def _create_env(self, difficulty: float, seed: int = None) -> DroneEnv:
        """Create environment with given difficulty."""
        env = DroneEnv(
            task=self.task,
            difficulty=difficulty,
            domain_randomization=self.args.domain_randomization
        )
        return env

    def _render_frame(self, episode_reward: float):
        """Render the current frame with training metrics."""
        # Get current state
        state = self.env.sim.state
        target = self.env.target_position

        # Build training metrics dict
        mean_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
        training_metrics = {
            'episodes': self.episodes,
            'episode_reward': episode_reward,
            'mean_reward': mean_reward,
            'total_steps': self.total_steps,
        }

        # Add task-specific metrics
        if self.task == TaskType.DELIVERY_ROUTE:
            training_metrics['deliveries_successful'] = self.env.deliveries_successful
            training_metrics['deliveries_completed'] = self.env.deliveries_completed

        # Get optional elements based on task
        package = None
        obstacles = None
        waypoints = None
        current_waypoint_idx = 0
        dropzone_radius = 0.3

        if self.task in [TaskType.DELIVERY, TaskType.DELIVERY_ROUTE]:
            package = self.env.sim.get_package_state()
            dropzone_radius = self.env.drop_accuracy_radius if hasattr(self.env, 'drop_accuracy_radius') else 0.3

        if self.task == TaskType.DELIVERY_ROUTE:
            obstacles = self.env.obstacles
            waypoints = self.env.route_waypoints
            current_waypoint_idx = self.env.current_waypoint_idx

        # Render
        self.renderer.render(
            state=state,
            target=target,
            trajectory=self.env.position_history,
            package=package,
            dropzone_radius=dropzone_radius,
            obstacles=obstacles,
            waypoints=waypoints,
            current_waypoint_idx=current_waypoint_idx,
            training_metrics=training_metrics
        )

    def train(self):
        """Main training loop with parallel environment support."""
        print(f"\nStarting training: {self.args.name}")
        print(f"Device: {get_device_info()}")
        print(f"Task: {self.args.task}, Difficulty: {self.difficulty}")
        print(f"Parallel drones: {self.num_envs}")
        print(f"Total timesteps: {self.args.total_timesteps:,}")
        if self.num_envs > 1:
            print(f"Effective steps per update: {self.args.n_steps} x {self.num_envs} = {self.args.n_steps * self.num_envs}")
        print("-" * 50)

        pbar = tqdm(total=self.args.total_timesteps, desc="Training")

        # Initialize all environments
        observations = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=self.args.seed + i)
            observations.append(obs)

        while self.total_steps < self.args.total_timesteps:
            # Collect rollout from all environments
            for step in range(self.args.n_steps):
                # Process each environment
                for env_idx, env in enumerate(self.envs):
                    obs = observations[env_idx]

                    # Select action
                    action, action_info = self.agent.select_action(obs)

                    # Step environment
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    # Store transition
                    self.agent.store_transition(
                        obs, action, reward,
                        action_info['value'],
                        action_info['log_prob'],
                        done
                    )

                    self.env_episode_rewards[env_idx] += reward
                    self.env_episode_lengths[env_idx] += 1
                    self.total_steps += 1

                    # Live visualization (only first environment)
                    if env_idx == 0 and self.renderer is not None and self.total_steps % self.args.render_freq == 0:
                        self._render_frame(self.env_episode_rewards[0])

                    if done:
                        # Record episode stats
                        self.episode_rewards.append(self.env_episode_rewards[env_idx])
                        self.episode_lengths.append(self.env_episode_lengths[env_idx])
                        self.episodes += 1

                        # Log to TensorBoard
                        self.writer.add_scalar('episode/reward', self.env_episode_rewards[env_idx], self.total_steps)
                        self.writer.add_scalar('episode/length', self.env_episode_lengths[env_idx], self.total_steps)
                        self.writer.add_scalar('episode/position_error',
                                             info.get('position_error', 0), self.total_steps)

                        # Reset this environment
                        observations[env_idx], _ = env.reset()
                        self.env_episode_rewards[env_idx] = 0.0
                        self.env_episode_lengths[env_idx] = 0
                    else:
                        observations[env_idx] = next_obs

                # Update progress bar
                pbar.update(self.num_envs)

            # PPO update (using last observation from first env for value bootstrap)
            update_info = self.agent.update(observations[0])

            # Log update info
            self.writer.add_scalar('train/loss', update_info['loss'], self.total_steps)
            self.writer.add_scalar('train/policy_loss', update_info['policy_loss'], self.total_steps)
            self.writer.add_scalar('train/value_loss', update_info['value_loss'], self.total_steps)
            self.writer.add_scalar('train/entropy', update_info['entropy'], self.total_steps)
            self.writer.add_scalar('train/action_std', update_info['action_std'], self.total_steps)

            # Curriculum learning: increase difficulty based on performance
            if self.args.curriculum and len(self.episode_rewards) >= 10:
                recent_rewards = self.episode_rewards[-10:]
                mean_reward = np.mean(recent_rewards)

                # Increase difficulty if performing well
                if mean_reward > -5 and self.difficulty < 1.0:
                    self.difficulty = min(1.0, self.difficulty + 0.05)
                    # Recreate all environments with new difficulty
                    for i in range(self.num_envs):
                        self.envs[i] = self._create_env(self.difficulty)
                        observations[i], _ = self.envs[i].reset(seed=self.args.seed + i)
                    self.env = self.envs[0]
                    self.writer.add_scalar('curriculum/difficulty', self.difficulty, self.total_steps)
                    tqdm.write(f"Difficulty increased to {self.difficulty:.2f}")

            # Periodic evaluation
            if self.total_steps % self.args.eval_freq == 0:
                eval_stats = self.evaluate()
                self.writer.add_scalar('eval/mean_reward', eval_stats['mean_reward'], self.total_steps)
                self.writer.add_scalar('eval/success_rate', eval_stats['success_rate'], self.total_steps)

                # Save best model
                if eval_stats['mean_reward'] > self.best_reward:
                    self.best_reward = eval_stats['mean_reward']
                    self.save_checkpoint('best')
                    tqdm.write(f"New best reward: {self.best_reward:.2f}")

            # Periodic saving
            if self.total_steps % self.args.save_freq == 0:
                self.save_checkpoint(f'step_{self.total_steps}')

            # Update progress bar description
            if len(self.episode_rewards) > 0:
                recent = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
                pbar.set_postfix({
                    'reward': f'{np.mean(recent):.1f}',
                    'episodes': self.episodes,
                    'envs': self.num_envs,
                    'diff': f'{self.difficulty:.2f}'
                })

        pbar.close()

        # Final save
        self.save_checkpoint('final')

        print("\nTraining complete!")
        print(f"Final episodes: {self.episodes}")
        print(f"Best evaluation reward: {self.best_reward:.2f}")

    def evaluate(self, n_episodes: Optional[int] = None) -> Dict[str, float]:
        """Evaluate the agent."""
        if n_episodes is None:
            n_episodes = self.args.eval_episodes

        eval_env = DroneEnv(
            task=self.task,
            difficulty=self.difficulty,
            domain_randomization=False  # No randomization for eval
        )

        rewards = []
        successes = []

        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            done = False
            success = False

            while not done:
                action, _ = self.agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated

                # Check for success (close to target)
                if info.get('position_error', 1) < 0.2:
                    success = True

            rewards.append(episode_reward)
            successes.append(success)

        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'success_rate': np.mean(successes),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }

    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        path = self.save_dir / f'{name}.pt'
        self.agent.save(str(path))

        # Save training state
        state = {
            'total_steps': self.total_steps,
            'episodes': self.episodes,
            'difficulty': self.difficulty,
            'best_reward': self.best_reward,
            'episode_rewards': self.episode_rewards[-100:],  # Last 100
        }
        state_path = self.save_dir / f'{name}_state.json'
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)


def main():
    """Main entry point."""
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
