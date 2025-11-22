"""
Hybrid PPO + Evolution Training for Drone AI

This combines the best of both approaches:
- PPO: Fast gradient-based learning (each drone learns efficiently)
- Evolution: Selection pressure (best drones survive and reproduce)

How it works:
1. Each drone has its own neural network (unique brain)
2. During each "age", ALL drones learn via PPO (gradient descent)
3. After the age, evaluate fitness (total reward)
4. Best performers survive, poor ones are replaced with mutated copies of winners
5. Repeat - survival of the fittest with fast learning!

This gives you:
- Speed: PPO gradient-based learning
- Quality: Evolutionary selection pressure
- Visual appeal: Watch drones compete and evolve
"""

import argparse
import copy
import json
from pathlib import Path
from datetime import datetime
from typing import List
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
    parser = argparse.ArgumentParser(description="Hybrid PPO+Evolution Drone AI Training")

    # Environment settings
    parser.add_argument("--task", type=str, default="hover",
                       choices=["hover", "waypoint", "trajectory", "velocity", "delivery", "delivery_route"],
                       help="Training task")
    parser.add_argument("--difficulty", type=float, default=0.3,
                       help="Task difficulty [0, 1]")
    parser.add_argument("--domain-randomization", action="store_true",
                       help="Enable domain randomization for sim-to-real")

    # Hybrid algorithm settings
    parser.add_argument("--population-size", type=int, default=8,
                       help="Number of drones in the population")
    parser.add_argument("--num-ages", type=int, default=50,
                       help="Number of ages (generations) to train")
    parser.add_argument("--steps-per-age", type=int, default=20000,
                       help="Training steps per age before selection")
    parser.add_argument("--elite-count", type=int, default=2,
                       help="Number of top performers to keep unchanged")
    parser.add_argument("--mutation-rate", type=float, default=0.1,
                       help="Probability of mutating each weight")
    parser.add_argument("--mutation-strength", type=float, default=0.05,
                       help="Standard deviation of mutation noise (smaller for fine-tuning)")

    # PPO training settings
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate for PPO")
    parser.add_argument("--n-steps", type=int, default=2048,
                       help="Steps before PPO update")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="PPO minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10,
                       help="PPO epochs per update")

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


class HybridTrainer:
    """
    Hybrid PPO + Evolution training manager.

    Combines fast gradient-based learning with evolutionary selection.
    """

    def __init__(self, args):
        self.args = args
        self.population_size = args.population_size

        # Set up experiment name
        if args.name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.name = f"hybrid_drone_{args.task}_{timestamp}"

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

        # Create population of PPO agents (each learns independently!)
        self.population: List[PPOAgent] = []
        ppo_config = PPOConfig(
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs
        )

        for i in range(self.population_size):
            agent = PPOAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                config=ppo_config,
                device=str(self.device)  # Use resolved device (cuda:0 or cpu)
            )
            self.population.append(agent)

        # Evolution tracking
        self.current_age = 0
        self.best_fitness_ever = float('-inf')
        self.best_agent_ever = None

        # TensorBoard writer
        self.writer = SummaryWriter(str(self.log_dir))

        # Visualization
        self.renderer = None
        if args.render:
            from drone_ai.visualization import DroneRenderer
            self.renderer = DroneRenderer(width=1024, height=768)
            print(f"Live visualization enabled (showing all {self.population_size} drones)")

    def _create_env(self, seed: int = None) -> DroneEnv:
        """Create environment with given seed."""
        env = DroneEnv(
            task=self.task,
            difficulty=self.difficulty,
            domain_randomization=self.args.domain_randomization
        )
        return env

    def _mutate_agent(self, agent: PPOAgent) -> PPOAgent:
        """Create a mutated copy of an agent."""
        new_agent = PPOAgent(
            obs_dim=agent.obs_dim,
            action_dim=agent.action_dim,
            config=agent.config,
            device=str(self.device)
        )

        # Copy weights from parent
        new_agent.policy.load_state_dict(copy.deepcopy(agent.policy.state_dict()))

        # Apply mutations (smaller than pure evolutionary - fine-tuning)
        with torch.no_grad():
            for param in new_agent.policy.parameters():
                mask = torch.rand_like(param) < self.args.mutation_rate
                noise = torch.randn_like(param) * self.args.mutation_strength
                param.add_(mask.float() * noise)

        return new_agent

    def _copy_agent(self, agent: PPOAgent) -> PPOAgent:
        """Create an exact copy of an agent."""
        new_agent = PPOAgent(
            obs_dim=agent.obs_dim,
            action_dim=agent.action_dim,
            config=agent.config,
            device=str(self.device)
        )
        new_agent.policy.load_state_dict(copy.deepcopy(agent.policy.state_dict()))
        return new_agent

    def _select_and_reproduce(self, fitnesses: List[float]) -> List[PPOAgent]:
        """Select best agents and create next generation."""
        sorted_indices = np.argsort(fitnesses)[::-1]  # Best first
        new_population = []

        # Elitism: keep top performers unchanged
        for i in range(self.args.elite_count):
            elite_idx = sorted_indices[i]
            elite = self._copy_agent(self.population[elite_idx])
            new_population.append(elite)

        # Fill remaining with mutated copies of top performers
        remaining = self.population_size - self.args.elite_count
        top_half = sorted_indices[:max(2, self.population_size // 2)]

        for i in range(remaining):
            parent_idx = np.random.choice(top_half)
            child = self._mutate_agent(self.population[parent_idx])
            new_population.append(child)

        return new_population

    def _render_frame(self, training_metrics: dict):
        """Render the current frame with all drones."""
        if self.renderer is None:
            return True

        # Process events first
        if not self.renderer.process_events():
            return False

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
        return True

    def train(self):
        """Main hybrid training loop."""
        print(f"\n{'='*60}")
        print(f"  HYBRID PPO + EVOLUTION TRAINING")
        print(f"{'='*60}")
        print(f"Hardware: {get_device_info()}")
        print(f"Using: {self.device} for neural network training")
        if self.device.type == 'cuda':
            print(f"  GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB allocated")
        print(f"Task: {self.args.task}, Difficulty: {self.difficulty}")
        print(f"Population: {self.population_size} drones (each with unique brain)")
        print(f"Ages: {self.args.num_ages}")
        print(f"Steps per age: {self.args.steps_per_age:,}")
        print()
        print("How it works:")
        print("  1. Each drone learns via PPO (fast gradient descent)")
        print("  2. After each age, best drones survive")
        print("  3. Poor performers replaced with mutated winners")
        print("  4. Best of both worlds: speed + selection pressure!")
        print(f"{'='*60}\n")

        total_steps = 0

        for age in range(self.args.num_ages):
            self.current_age = age
            print(f"\n--- AGE {age + 1}/{self.args.num_ages} ---")

            # Initialize environments - ALL drones spawn at SAME location
            # Use same seed for all so they get same pickup/dropzone positions
            base_seed = self.args.seed + age * 1000
            observations = []

            # Reset first environment to get shared positions
            obs, _ = self.envs[0].reset(seed=base_seed)
            observations.append(obs)

            # Copy positions from first env to all others (same spawn location)
            shared_pickup = self.envs[0].pickup_position.copy()
            shared_dropzone = self.envs[0].dropzone_position.copy()
            shared_target = self.envs[0].target_position.copy()

            for i in range(1, self.population_size):
                env = self.envs[i]
                # Set same positions before reset
                env.pickup_position = shared_pickup.copy()
                env.dropzone_position = shared_dropzone.copy()
                env.target_position = shared_target.copy()
                obs, _ = env.reset(seed=base_seed)  # Same seed = same positions
                observations.append(obs)

            # Track rewards per drone
            drone_total_rewards = [0.0] * self.population_size
            drone_episode_rewards = [0.0] * self.population_size
            drone_episodes_completed = [0] * self.population_size

            # Track which drones are alive (dead drones don't respawn until ALL are dead)
            drone_alive = [True] * self.population_size
            drone_survival_steps = [0] * self.population_size  # How long each drone survived

            # Track steps for PPO updates
            steps_since_update = [0] * self.population_size

            pbar = tqdm(total=self.args.steps_per_age, desc=f"Age {age+1} (PPO learning)")

            for step in range(self.args.steps_per_age):
                # Count alive drones
                alive_count = sum(drone_alive)

                # Step each drone and collect experience
                for i in range(self.population_size):
                    # Skip dead drones
                    if not drone_alive[i]:
                        continue

                    agent = self.population[i]
                    env = self.envs[i]
                    obs = observations[i]

                    # Select action
                    action, action_info = agent.select_action(obs)

                    # Step environment
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    # Store transition for PPO learning
                    agent.store_transition(
                        obs, action, reward,
                        action_info['value'],
                        action_info['log_prob'],
                        done
                    )

                    drone_total_rewards[i] += reward
                    drone_episode_rewards[i] += reward
                    drone_survival_steps[i] += 1
                    steps_since_update[i] += 1

                    if done:
                        # Drone crashed - mark as dead (no respawn until all dead)
                        drone_alive[i] = False
                        drone_episodes_completed[i] += 1
                    else:
                        observations[i] = next_obs

                    # PPO update when enough steps collected
                    if steps_since_update[i] >= self.args.n_steps:
                        agent.update(observations[i])
                        steps_since_update[i] = 0

                # If ALL drones are dead, end this age early
                # (drones stay dead until next age - no mid-age respawns)
                if not any(drone_alive):
                    print(f"\n  All drones crashed at step {step}! Ending age early.")
                    break

                total_steps += alive_count  # Only count steps for alive drones
                pbar.update(1)

                # Render visualization
                if step % self.args.render_freq == 0:
                    training_metrics = {
                        'episodes': sum(drone_episodes_completed),
                        'episode_reward': np.mean([r for r in drone_episode_rewards]),
                        'mean_reward': np.mean(drone_total_rewards),
                        'total_steps': total_steps,
                        'age': age + 1,
                        'alive': sum(drone_alive),
                    }
                    if not self._render_frame(training_metrics):
                        print("\nVisualization closed.")
                        pbar.close()
                        self._save_checkpoint(age + 1, drone_total_rewards)
                        return

            pbar.close()

            # Calculate fitness (combine reward and survival time)
            # Fitness = total_reward + survival_bonus
            fitnesses = []
            for i in range(self.population_size):
                # Reward survival: longer survival = higher fitness bonus
                survival_bonus = drone_survival_steps[i] * 0.01  # Small bonus per step survived
                fitness = drone_total_rewards[i] + survival_bonus
                fitnesses.append(fitness)

            best_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[best_idx]
            mean_fitness = np.mean(fitnesses)
            worst_fitness = np.min(fitnesses)
            best_survival = drone_survival_steps[best_idx]

            print(f"  Best drone: #{best_idx + 1} (fitness: {best_fitness:.1f}, survived: {best_survival} steps)")
            print(f"  Mean fitness: {mean_fitness:.1f}, Worst: {worst_fitness:.1f}")
            print(f"  Episodes (full resets): {sum(drone_episodes_completed)}")

            # Track best ever
            if best_fitness > self.best_fitness_ever:
                self.best_fitness_ever = best_fitness
                self.best_agent_ever = self._copy_agent(self.population[best_idx])
                print(f"  NEW BEST EVER! Fitness: {best_fitness:.1f}")

            # Log to TensorBoard
            self.writer.add_scalar('evolution/best_fitness', best_fitness, age)
            self.writer.add_scalar('evolution/mean_fitness', mean_fitness, age)
            self.writer.add_scalar('evolution/worst_fitness', worst_fitness, age)
            self.writer.add_scalar('evolution/best_ever', self.best_fitness_ever, age)

            # Selection and reproduction (except last age)
            if age < self.args.num_ages - 1:
                print("  Selecting best drones and creating next generation...")
                self.population = self._select_and_reproduce(fitnesses)

            # Save checkpoint
            if (age + 1) % self.args.save_freq == 0 or age == self.args.num_ages - 1:
                self._save_checkpoint(age + 1, fitnesses)

        # Final summary
        print(f"\n{'='*60}")
        print(f"  HYBRID TRAINING COMPLETE!")
        print(f"{'='*60}")
        print(f"Best fitness ever: {self.best_fitness_ever:.1f}")
        print(f"Total steps: {total_steps:,}")
        print(f"Saved best agent to: {self.save_dir}/best.pt")
        print(f"{'='*60}\n")

        # Save the best agent
        if self.best_agent_ever is not None:
            self.best_agent_ever.save(str(self.save_dir / 'best.pt'))

    def _save_checkpoint(self, age: int, fitnesses: List[float]):
        """Save checkpoint with best agent."""
        best_idx = np.argmax(fitnesses)
        best_agent = self.population[best_idx]
        best_agent.save(str(self.save_dir / f'age_{age}_best.pt'))

        state = {
            'age': age,
            'best_fitness_ever': self.best_fitness_ever,
            'current_fitnesses': fitnesses,
        }
        state_path = self.save_dir / f'hybrid_state_age_{age}.json'
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"  Checkpoint saved: age {age}")


def main():
    """Main entry point."""
    args = parse_args()
    trainer = HybridTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
