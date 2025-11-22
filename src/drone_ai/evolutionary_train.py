"""
Evolutionary Training for Drone AI

This module implements a genetic/evolutionary algorithm approach to training:
- Each drone in the population has its own neural network
- After each "age" (generation), drones are evaluated by fitness (reward)
- Best performers are selected and mutated to create the next generation
- This leads to "survival of the fittest" - the best drone emerges

Key concepts:
- Population: Multiple drones training simultaneously
- Age/Generation: Fixed number of steps before selection
- Fitness: Total reward accumulated during an age
- Selection: Top performers are chosen to reproduce
- Mutation: Small random changes to neural network weights
- Elitism: Best drone is kept unchanged (no mutation)
"""

import argparse
import time
import copy
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from drone_ai.environment import DroneEnv, TaskType
from drone_ai.agent import PPOAgent, PPOConfig, ActorCritic


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
    parser = argparse.ArgumentParser(description="Evolutionary Drone AI Training")

    # Environment settings
    parser.add_argument("--task", type=str, default="hover",
                       choices=["hover", "waypoint", "trajectory", "velocity", "delivery", "delivery_route"],
                       help="Training task")
    parser.add_argument("--difficulty", type=float, default=0.3,
                       help="Task difficulty [0, 1]")
    parser.add_argument("--domain-randomization", action="store_true",
                       help="Enable domain randomization for sim-to-real")

    # Evolutionary algorithm settings
    parser.add_argument("--population-size", type=int, default=8,
                       help="Number of drones in the population")
    parser.add_argument("--num-ages", type=int, default=100,
                       help="Number of ages (generations) to train")
    parser.add_argument("--steps-per-age", type=int, default=10000,
                       help="Training steps per age before selection")
    parser.add_argument("--elite-count", type=int, default=2,
                       help="Number of top performers to keep unchanged")
    parser.add_argument("--mutation-rate", type=float, default=0.1,
                       help="Probability of mutating each weight")
    parser.add_argument("--mutation-strength", type=float, default=0.1,
                       help="Standard deviation of mutation noise")

    # Training settings
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate for PPO updates within age")

    # Logging and saving
    parser.add_argument("--log-dir", type=str, default="logs",
                       help="Directory for logs")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                       help="Directory for checkpoints")
    parser.add_argument("--save-freq", type=int, default=10,
                       help="Save checkpoint every N ages")

    # Visualization
    parser.add_argument("--render", action="store_true",
                       help="Enable live visualization during training")
    parser.add_argument("--render-freq", type=int, default=5,
                       help="Render every N steps")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cpu, cuda)")
    parser.add_argument("--name", type=str, default=None,
                       help="Experiment name")

    return parser.parse_args()


class EvolutionaryTrainer:
    """
    Evolutionary training manager for drone AI.

    Uses genetic algorithm approach:
    1. Create population of drones with different neural networks
    2. Run each drone for an "age" (fixed steps)
    3. Evaluate fitness (total reward)
    4. Select best performers
    5. Create next generation through mutation
    6. Repeat until convergence
    """

    def __init__(self, args):
        self.args = args
        self.population_size = args.population_size

        # Set up experiment name
        if args.name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.name = f"evo_drone_{args.task}_{timestamp}"

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

        # Set device
        if args.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(args.device)

        # Create environments (one per drone)
        self.task = TaskType(args.task)
        self.difficulty = args.difficulty
        self.envs = [self._create_env(seed=args.seed + i) for i in range(self.population_size)]

        # Get dimensions from first environment
        obs_dim = self.envs[0].observation_space.shape[0]
        action_dim = self.envs[0].action_space.shape[0]

        # Create population of agents
        self.population: List[PPOAgent] = []
        ppo_config = PPOConfig(learning_rate=args.lr)

        for i in range(self.population_size):
            agent = PPOAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                config=ppo_config,
                device=args.device
            )
            self.population.append(agent)

        # Evolution tracking
        self.current_age = 0
        self.best_fitness_ever = float('-inf')
        self.best_agent_ever = None

        # Fitness history for each drone
        self.fitness_history: List[List[float]] = [[] for _ in range(self.population_size)]

        # TensorBoard writer
        self.writer = SummaryWriter(str(self.log_dir))

        # Visualization
        self.renderer = None
        if args.render:
            from drone_ai.visualization import DroneRenderer
            self.renderer = DroneRenderer(width=1024, height=768)
            print(f"Live visualization enabled (showing all {self.population_size} drones)")
            print("Note: First drone (green) is primary, others are semi-transparent")

    def _create_env(self, seed: int = None) -> DroneEnv:
        """Create environment with given seed."""
        env = DroneEnv(
            task=self.task,
            difficulty=self.difficulty,
            domain_randomization=self.args.domain_randomization
        )
        return env

    def _evaluate_fitness(self, agent_idx: int, steps: int) -> float:
        """
        Evaluate a single agent's fitness over given steps.
        Returns total accumulated reward.
        """
        env = self.envs[agent_idx]
        agent = self.population[agent_idx]

        obs, _ = env.reset(seed=self.args.seed + agent_idx + self.current_age * 1000)
        total_reward = 0.0
        episode_rewards = []
        current_episode_reward = 0.0

        for step in range(steps):
            # Select action
            action, _ = agent.select_action(obs)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            current_episode_reward += reward

            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs

        # Add final partial episode
        if current_episode_reward != 0:
            episode_rewards.append(current_episode_reward)

        return total_reward, episode_rewards

    def _mutate_agent(self, agent: PPOAgent, mutation_rate: float, mutation_strength: float) -> PPOAgent:
        """
        Create a mutated copy of an agent.
        Applies random noise to neural network weights.
        """
        # Create a new agent with same config
        new_agent = PPOAgent(
            obs_dim=agent.obs_dim,
            action_dim=agent.action_dim,
            config=agent.config,
            device=str(self.device)
        )

        # Copy weights from parent
        new_agent.policy.load_state_dict(copy.deepcopy(agent.policy.state_dict()))

        # Apply mutations
        with torch.no_grad():
            for param in new_agent.policy.parameters():
                # Create mask for which weights to mutate
                mask = torch.rand_like(param) < mutation_rate
                # Add Gaussian noise to selected weights
                noise = torch.randn_like(param) * mutation_strength
                param.add_(mask.float() * noise)

        return new_agent

    def _crossover(self, parent1: PPOAgent, parent2: PPOAgent) -> PPOAgent:
        """
        Create offspring by combining weights from two parents.
        Uses uniform crossover (each weight randomly from either parent).
        """
        # Create a new agent with same config
        child = PPOAgent(
            obs_dim=parent1.obs_dim,
            action_dim=parent1.action_dim,
            config=parent1.config,
            device=str(self.device)
        )

        # Combine weights from both parents
        with torch.no_grad():
            child_state = child.policy.state_dict()
            parent1_state = parent1.policy.state_dict()
            parent2_state = parent2.policy.state_dict()

            for key in child_state.keys():
                # Randomly select from either parent
                mask = torch.rand_like(parent1_state[key]) < 0.5
                child_state[key] = torch.where(mask, parent1_state[key], parent2_state[key])

            child.policy.load_state_dict(child_state)

        return child

    def _select_and_reproduce(self, fitnesses: List[float]) -> List[PPOAgent]:
        """
        Select best agents and create next generation.

        Selection strategy:
        1. Keep elite agents unchanged (elitism)
        2. Create remaining through mutation of top performers
        """
        # Sort by fitness (highest first)
        sorted_indices = np.argsort(fitnesses)[::-1]

        new_population = []

        # Elitism: keep top performers unchanged
        for i in range(self.args.elite_count):
            elite_idx = sorted_indices[i]
            # Deep copy the elite agent
            elite = PPOAgent(
                obs_dim=self.population[elite_idx].obs_dim,
                action_dim=self.population[elite_idx].action_dim,
                config=self.population[elite_idx].config,
                device=str(self.device)
            )
            elite.policy.load_state_dict(
                copy.deepcopy(self.population[elite_idx].policy.state_dict())
            )
            new_population.append(elite)

        # Fill remaining slots with mutated copies of top performers
        remaining = self.population_size - self.args.elite_count
        top_half = sorted_indices[:self.population_size // 2]

        for i in range(remaining):
            # Select parent from top performers
            parent_idx = np.random.choice(top_half)
            parent = self.population[parent_idx]

            # Sometimes do crossover
            if np.random.random() < 0.3 and len(top_half) >= 2:
                # Select second parent
                parent2_idx = np.random.choice(top_half)
                while parent2_idx == parent_idx:
                    parent2_idx = np.random.choice(top_half)
                parent2 = self.population[parent2_idx]
                child = self._crossover(parent, parent2)
            else:
                child = self._mutate_agent(
                    parent,
                    self.args.mutation_rate,
                    self.args.mutation_strength
                )

            new_population.append(child)

        return new_population

    def _render_frame(self, training_metrics: dict):
        """Render the current frame with all drones."""
        if self.renderer is None:
            return

        # Get state from first environment
        env = self.envs[0]
        state = env.sim.state
        target = env.target_position

        # Get package info if delivery task
        package = None
        dropzone_radius = 0.3
        if self.task in [TaskType.DELIVERY, TaskType.DELIVERY_ROUTE]:
            package = env.sim.get_package_state()
            dropzone_radius = getattr(env, 'drop_accuracy_radius', 0.3)

        # Get all drone states
        additional_states = [self.envs[i].sim.state for i in range(1, self.population_size)]

        # Render
        self.renderer.render(
            state=state,
            target=target,
            trajectory=env.position_history,
            package=package,
            dropzone_radius=dropzone_radius,
            training_metrics=training_metrics,
            additional_states=additional_states
        )

    def train(self):
        """Main evolutionary training loop."""
        print(f"\n{'='*60}")
        print(f"  EVOLUTIONARY DRONE AI TRAINING")
        print(f"{'='*60}")
        print(f"Device: {get_device_info()}")
        print(f"Task: {self.args.task}, Difficulty: {self.difficulty}")
        print(f"Population size: {self.population_size} drones")
        print(f"Ages: {self.args.num_ages}")
        print(f"Steps per age: {self.args.steps_per_age:,}")
        print(f"Elite count: {self.args.elite_count} (best drones kept unchanged)")
        print(f"Mutation rate: {self.args.mutation_rate}, strength: {self.args.mutation_strength}")
        print(f"{'='*60}\n")

        total_steps = 0

        for age in range(self.args.num_ages):
            self.current_age = age
            print(f"\n--- AGE {age + 1}/{self.args.num_ages} ---")

            # Evaluate all agents
            fitnesses = []
            all_episode_rewards = []

            # Initialize all environments
            observations = []
            for i, env in enumerate(self.envs):
                obs, _ = env.reset(seed=self.args.seed + i + age * 1000)
                observations.append(obs)

            # Track rewards per drone
            drone_rewards = [0.0] * self.population_size
            drone_episodes = [[] for _ in range(self.population_size)]
            episode_rewards = [0.0] * self.population_size

            # Run all drones for steps_per_age steps
            pbar = tqdm(total=self.args.steps_per_age, desc=f"Age {age+1}")

            for step in range(self.args.steps_per_age):
                # Step each drone
                for i in range(self.population_size):
                    agent = self.population[i]
                    env = self.envs[i]
                    obs = observations[i]

                    # Select action
                    action, _ = agent.select_action(obs)

                    # Step environment
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    drone_rewards[i] += reward
                    episode_rewards[i] += reward

                    if done:
                        drone_episodes[i].append(episode_rewards[i])
                        episode_rewards[i] = 0.0
                        observations[i], _ = env.reset()
                    else:
                        observations[i] = next_obs

                total_steps += self.population_size
                pbar.update(1)

                # Render visualization
                if self.renderer is not None and step % self.args.render_freq == 0:
                    training_metrics = {
                        'episodes': sum(len(ep) for ep in drone_episodes),
                        'episode_reward': sum(episode_rewards) / len(episode_rewards),
                        'mean_reward': np.mean([r for eps in drone_episodes for r in eps]) if any(drone_episodes) else 0,
                        'total_steps': total_steps,
                        'age': age + 1,
                    }
                    self._render_frame(training_metrics)

            pbar.close()

            # Calculate fitness for each drone
            fitnesses = drone_rewards
            for i, fitness in enumerate(fitnesses):
                self.fitness_history[i].append(fitness)

            # Log statistics
            best_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[best_idx]
            mean_fitness = np.mean(fitnesses)
            worst_fitness = np.min(fitnesses)

            print(f"  Best drone: #{best_idx + 1} with fitness {best_fitness:.1f}")
            print(f"  Mean fitness: {mean_fitness:.1f}")
            print(f"  Worst fitness: {worst_fitness:.1f}")

            # Track best ever
            if best_fitness > self.best_fitness_ever:
                self.best_fitness_ever = best_fitness
                self.best_agent_ever = copy.deepcopy(self.population[best_idx])
                print(f"  NEW BEST EVER! Fitness: {best_fitness:.1f}")

            # Log to TensorBoard
            self.writer.add_scalar('evolution/best_fitness', best_fitness, age)
            self.writer.add_scalar('evolution/mean_fitness', mean_fitness, age)
            self.writer.add_scalar('evolution/worst_fitness', worst_fitness, age)
            self.writer.add_scalar('evolution/best_ever', self.best_fitness_ever, age)
            self.writer.add_scalar('evolution/fitness_spread', best_fitness - worst_fitness, age)

            # Log fitness for each drone
            for i, fitness in enumerate(fitnesses):
                self.writer.add_scalar(f'drones/drone_{i+1}_fitness', fitness, age)

            # Selection and reproduction (except last age)
            if age < self.args.num_ages - 1:
                print("  Selecting best drones and creating next generation...")
                self.population = self._select_and_reproduce(fitnesses)

            # Save checkpoint
            if (age + 1) % self.args.save_freq == 0 or age == self.args.num_ages - 1:
                self._save_checkpoint(age + 1, fitnesses)

        # Final summary
        print(f"\n{'='*60}")
        print(f"  EVOLUTIONARY TRAINING COMPLETE!")
        print(f"{'='*60}")
        print(f"Best fitness ever: {self.best_fitness_ever:.1f}")
        print(f"Total steps: {total_steps:,}")
        print(f"Saved best agent to: {self.save_dir}/best.pt")
        print(f"{'='*60}\n")

        # Save the best agent
        if self.best_agent_ever is not None:
            self.best_agent_ever.save(str(self.save_dir / 'best.pt'))

    def _save_checkpoint(self, age: int, fitnesses: List[float]):
        """Save checkpoint with best agent and evolution state."""
        # Save best agent from current population
        best_idx = np.argmax(fitnesses)
        best_agent = self.population[best_idx]
        best_agent.save(str(self.save_dir / f'age_{age}_best.pt'))

        # Save evolution state
        state = {
            'age': age,
            'best_fitness_ever': self.best_fitness_ever,
            'fitness_history': self.fitness_history,
            'current_fitnesses': fitnesses,
        }
        state_path = self.save_dir / f'evolution_state_age_{age}.json'
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"  Checkpoint saved: age {age}")


def main():
    """Main entry point."""
    args = parse_args()
    trainer = EvolutionaryTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
