"""
Learning Sequence - Full Drone AI Training Curriculum

This module runs a complete training curriculum:
1. Hover (basic stabilization)
2. Delivery (pickup and drop)
3. Delivery Route (1km with obstacles)
4. Deployment Ready (real-world simulation)

After all stages, the best 2 drones repeat the process.
Finally, the best drone is saved with a score and grade.
"""

import argparse
import json
import copy
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
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
        return "CPU only"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Drone AI Learning Sequence")

    parser.add_argument("--population-size", type=int, default=6,
                       help="Number of drones")
    parser.add_argument("--ages-per-stage", type=int, default=15,
                       help="Ages per training stage")
    parser.add_argument("--steps-per-age", type=int, default=15000,
                       help="Steps per age")

    parser.add_argument("--save-dir", type=str, default="checkpoints",
                       help="Directory for checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs",
                       help="Directory for logs")

    parser.add_argument("--render", action="store_true",
                       help="Enable visualization")
    parser.add_argument("--render-freq", type=int, default=5,
                       help="Render every N steps")

    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cpu, cuda)")

    return parser.parse_args()


class LearningSequence:
    """Full curriculum training for drone AI."""

    STAGES = [
        {"name": "Hover", "task": "hover", "difficulty": 0.3, "domain_rand": False},
        {"name": "Delivery", "task": "delivery", "difficulty": 0.5, "domain_rand": False},
        {"name": "Delivery Route", "task": "delivery_route", "difficulty": 0.6, "domain_rand": False},
        {"name": "Deployment Ready", "task": "delivery_route", "difficulty": 1.0, "domain_rand": True},
    ]

    def __init__(self, args):
        self.args = args
        self.population_size = args.population_size

        # Set up directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"learning_sequence_{timestamp}"
        self.save_dir = Path(args.save_dir) / self.experiment_name
        self.log_dir = Path(args.log_dir) / self.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set device
        if args.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(args.device)

        # Set seeds
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Create initial population
        self.population: List[PPOAgent] = []
        self._init_population()

        # Visualization
        self.renderer = None
        if args.render:
            from drone_ai.visualization import DroneRenderer
            self.renderer = DroneRenderer(width=1024, height=768)

        # Tracking
        self.stage_scores = []
        self.writer = SummaryWriter(str(self.log_dir))

    def _init_population(self):
        """Initialize population of agents."""
        # Use largest observation space (delivery_route)
        env = DroneEnv(task=TaskType.DELIVERY_ROUTE, difficulty=0.5)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        ppo_config = PPOConfig(
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10
        )

        self.population = []
        for _ in range(self.population_size):
            agent = PPOAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                config=ppo_config,
                device=str(self.device)
            )
            self.population.append(agent)

    def _train_stage(self, stage_info: dict, round_num: int) -> List[float]:
        """Train population on a single stage."""
        stage_name = stage_info["name"]
        task = TaskType(stage_info["task"])
        difficulty = stage_info["difficulty"]
        domain_rand = stage_info["domain_rand"]

        print(f"\n{'='*60}")
        print(f"  STAGE: {stage_name} (Round {round_num})")
        print(f"  Task: {task.value}, Difficulty: {difficulty}")
        print(f"  Domain Randomization: {'ON' if domain_rand else 'OFF'}")
        print(f"{'='*60}")

        # Create environments
        envs = []
        for i in range(self.population_size):
            env = DroneEnv(
                task=task,
                difficulty=difficulty,
                domain_randomization=domain_rand
            )
            envs.append(env)

        # Training loop
        total_rewards = [0.0] * self.population_size

        for age in range(self.args.ages_per_stage):
            print(f"\n  Age {age + 1}/{self.args.ages_per_stage}")

            # Reset all environments at same location
            base_seed = self.args.seed + age * 1000
            observations = []
            obs, _ = envs[0].reset(seed=base_seed)
            observations.append(obs)

            shared_pickup = envs[0].pickup_position.copy()
            shared_dropzone = envs[0].dropzone_position.copy()
            shared_target = envs[0].target_position.copy()

            for i in range(1, self.population_size):
                envs[i].pickup_position = shared_pickup.copy()
                envs[i].dropzone_position = shared_dropzone.copy()
                envs[i].target_position = shared_target.copy()
                obs, _ = envs[i].reset(seed=base_seed)
                observations.append(obs)

            drone_alive = [True] * self.population_size
            drone_rewards = [0.0] * self.population_size
            steps_since_update = [0] * self.population_size

            pbar = tqdm(total=self.args.steps_per_age, desc=f"  Training", leave=False)

            for step in range(self.args.steps_per_age):
                alive_count = sum(drone_alive)

                for i in range(self.population_size):
                    if not drone_alive[i]:
                        continue

                    agent = self.population[i]
                    env = envs[i]
                    obs = observations[i]

                    action, action_info = agent.select_action(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    agent.store_transition(
                        obs, action, reward,
                        action_info['value'],
                        action_info['log_prob'],
                        done
                    )

                    drone_rewards[i] += reward
                    steps_since_update[i] += 1

                    if done:
                        drone_alive[i] = False
                    else:
                        observations[i] = next_obs

                    if steps_since_update[i] >= 2048:
                        agent.update(observations[i])
                        steps_since_update[i] = 0

                # Reset all if all dead
                if not any(drone_alive):
                    reset_seed = base_seed + step
                    obs, _ = envs[0].reset(seed=reset_seed)
                    observations[0] = obs
                    shared_pickup = envs[0].pickup_position.copy()
                    shared_dropzone = envs[0].dropzone_position.copy()
                    shared_target = envs[0].target_position.copy()
                    drone_alive[0] = True

                    for i in range(1, self.population_size):
                        envs[i].pickup_position = shared_pickup.copy()
                        envs[i].dropzone_position = shared_dropzone.copy()
                        envs[i].target_position = shared_target.copy()
                        observations[i], _ = envs[i].reset(seed=reset_seed)
                        drone_alive[i] = True

                pbar.update(1)

                # Render
                if self.renderer and step % self.args.render_freq == 0:
                    if not self.renderer.process_events():
                        pbar.close()
                        return total_rewards

                    state = envs[0].sim.state
                    target = envs[0].target_position
                    additional_states = [envs[i].sim.state for i in range(1, self.population_size)]

                    self.renderer.render(
                        state=state,
                        target=target,
                        trajectory=envs[0].position_history,
                        training_metrics={
                            'stage': stage_name,
                            'age': age + 1,
                            'alive': alive_count
                        },
                        additional_states=additional_states
                    )

            pbar.close()

            # Accumulate rewards
            for i in range(self.population_size):
                total_rewards[i] += drone_rewards[i]

            # Selection at end of age
            if age < self.args.ages_per_stage - 1:
                self._select_and_mutate(drone_rewards)

            best_idx = np.argmax(drone_rewards)
            print(f"    Best: Drone #{best_idx + 1} with {drone_rewards[best_idx]:.1f}")

        return total_rewards

    def _select_and_mutate(self, fitnesses: List[float]):
        """Select best agents and mutate."""
        sorted_indices = np.argsort(fitnesses)[::-1]

        # Keep top 2 unchanged
        new_population = []
        for i in range(2):
            elite_idx = sorted_indices[i]
            elite = self._copy_agent(self.population[elite_idx])
            new_population.append(elite)

        # Mutate rest from top performers
        for i in range(self.population_size - 2):
            parent_idx = sorted_indices[i % 2]
            child = self._mutate_agent(self.population[parent_idx])
            new_population.append(child)

        self.population = new_population

    def _copy_agent(self, agent: PPOAgent) -> PPOAgent:
        """Create exact copy of agent."""
        new_agent = PPOAgent(
            obs_dim=agent.obs_dim,
            action_dim=agent.action_dim,
            config=agent.config,
            device=str(self.device)
        )
        new_agent.policy.load_state_dict(copy.deepcopy(agent.policy.state_dict()))
        return new_agent

    def _mutate_agent(self, agent: PPOAgent) -> PPOAgent:
        """Create mutated copy of agent."""
        new_agent = self._copy_agent(agent)
        with torch.no_grad():
            for param in new_agent.policy.parameters():
                mask = torch.rand_like(param) < 0.1
                noise = torch.randn_like(param) * 0.05
                param.add_(mask.float() * noise)
        return new_agent

    def _evaluate_final(self, agent: PPOAgent) -> Tuple[float, str]:
        """Evaluate final agent and give score/grade."""
        print("\n" + "="*60)
        print("  FINAL EVALUATION")
        print("="*60)

        scores = {}

        for stage in self.STAGES:
            task = TaskType(stage["task"])
            env = DroneEnv(
                task=task,
                difficulty=stage["difficulty"],
                domain_randomization=stage["domain_rand"]
            )

            total_reward = 0.0
            episodes = 5

            for ep in range(episodes):
                obs, _ = env.reset(seed=self.args.seed + ep)
                ep_reward = 0.0

                for step in range(1000):
                    action, _ = agent.select_action(obs)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_reward += reward

                    if terminated or truncated:
                        break

                total_reward += ep_reward

            avg_reward = total_reward / episodes
            scores[stage["name"]] = avg_reward
            print(f"  {stage['name']}: {avg_reward:.1f}")

        # Calculate final score (weighted average)
        weights = {"Hover": 0.15, "Delivery": 0.25, "Delivery Route": 0.30, "Deployment Ready": 0.30}
        final_score = sum(scores[k] * weights[k] for k in scores)

        # Determine grade
        # Ranks from worst to best: W < F < D < C < B < A < S < P
        # +/- variants for all except W and P
        if final_score > 800:
            grade = "P - PERFECT (Flawless drone! Ready for anything!)"
        elif final_score > 700:
            grade = "S+ - SUPREME+ (Near perfect performance!)"
        elif final_score > 600:
            grade = "S - SUPREME (Outstanding results!)"
        elif final_score > 550:
            grade = "S- - SUPREME- (Excellent, almost supreme!)"
        elif final_score > 500:
            grade = "A+ - ALPHA+ (Top tier performer!)"
        elif final_score > 450:
            grade = "A - ALPHA (Dominant performance!)"
        elif final_score > 400:
            grade = "A- - ALPHA- (Strong alpha potential!)"
        elif final_score > 350:
            grade = "B+ - BETTER+ (Impressive improvement!)"
        elif final_score > 300:
            grade = "B - BETTER (Solid performance!)"
        elif final_score > 250:
            grade = "B- - BETTER- (Getting better!)"
        elif final_score > 200:
            grade = "C+ - COOL+ (Pretty cool drone!)"
        elif final_score > 150:
            grade = "C - COOL (Decent, keeps its cool!)"
        elif final_score > 100:
            grade = "C- - COOL- (Barely cool, needs work!)"
        elif final_score > 75:
            grade = "D+ - DELUSIONAL+ (Getting somewhere...)"
        elif final_score > 50:
            grade = "D - DELUSIONAL (Thinks it can fly...)"
        elif final_score > 25:
            grade = "D- - DELUSIONAL- (Very confused drone!)"
        elif final_score > 10:
            grade = "F+ - FAILURE+ (Failed but tried!)"
        elif final_score > 0:
            grade = "F - FAILURE (Complete failure!)"
        elif final_score > -50:
            grade = "F- - FAILURE- (Spectacular failure!)"
        else:
            grade = "W - WORST (Absolutely terrible! Start over!)"

        return final_score, grade, scores

    def run(self):
        """Run the full learning sequence."""
        print("\n" + "="*60)
        print("  DRONE AI LEARNING SEQUENCE")
        print("="*60)
        print(f"Device: {get_device_info()}")
        print(f"Population: {self.population_size} drones")
        print(f"Stages: {len(self.STAGES)}")
        print(f"Ages per stage: {self.args.ages_per_stage}")
        print(f"Steps per age: {self.args.steps_per_age:,}")
        print()
        print("Training curriculum:")
        for i, stage in enumerate(self.STAGES, 1):
            print(f"  {i}. {stage['name']} (difficulty: {stage['difficulty']})")
        print()
        print("After all stages, top 2 drones repeat the process.")
        print("Finally, the best drone is graded!")
        print("="*60)

        # Round 1: All stages with full population
        print("\n" + "#"*60)
        print("  ROUND 1: Full Training")
        print("#"*60)

        round1_scores = []
        for stage in self.STAGES:
            scores = self._train_stage(stage, round_num=1)
            round1_scores.append(scores)

        # Select top 2
        total_scores = [sum(round1_scores[s][i] for s in range(len(self.STAGES)))
                       for i in range(self.population_size)]
        sorted_indices = np.argsort(total_scores)[::-1]

        print("\n" + "-"*60)
        print("  Round 1 Results:")
        for i, idx in enumerate(sorted_indices):
            print(f"    #{i+1}: Drone {idx+1} - Score: {total_scores[idx]:.1f}")

        # Keep only top 2 for round 2
        top2 = [self._copy_agent(self.population[sorted_indices[0]]),
                self._copy_agent(self.population[sorted_indices[1]])]

        # Round 2: Repeat with top 2 (expanded to population size)
        print("\n" + "#"*60)
        print("  ROUND 2: Final Training (Top 2 + Mutations)")
        print("#"*60)

        self.population = []
        self.population.append(top2[0])
        self.population.append(top2[1])
        for i in range(self.population_size - 2):
            self.population.append(self._mutate_agent(top2[i % 2]))

        for stage in self.STAGES:
            self._train_stage(stage, round_num=2)

        # Final evaluation
        best_idx = 0
        best_agent = self.population[best_idx]

        final_score, grade, stage_scores = self._evaluate_final(best_agent)

        # Save the best model
        model_path = self.save_dir / "best_graduated_drone.pt"
        best_agent.save(str(model_path))

        # Save results
        results = {
            "final_score": final_score,
            "grade": grade,
            "stage_scores": stage_scores,
            "timestamp": datetime.now().isoformat()
        }
        results_path = self.save_dir / "graduation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Print final results
        print("\n" + "="*60)
        print("  GRADUATION COMPLETE!")
        print("="*60)
        print(f"\n  Final Score: {final_score:.1f}")
        print(f"  Grade: {grade}")
        print(f"\n  Stage Breakdown:")
        for stage, score in stage_scores.items():
            print(f"    {stage}: {score:.1f}")
        print(f"\n  Model saved to: {model_path}")
        print(f"  Results saved to: {results_path}")
        print("="*60 + "\n")

        return final_score, grade


def main():
    args = parse_args()
    sequence = LearningSequence(args)
    sequence.run()


if __name__ == "__main__":
    main()
