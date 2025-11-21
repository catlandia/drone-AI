"""
Evaluation Script for Trained Drone AI Models

This script provides tools for:
- Evaluating trained models in simulation
- Deploying to real hardware
- Visualizing performance
- Generating metrics and reports
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np

from drone_ai.environment import DroneEnv, TaskType
from drone_ai.agent import PPOAgent
from drone_ai.simulation import DroneSimulation
from drone_ai.hardware import create_interface, DroneInterface


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Drone AI")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")

    # Mode
    parser.add_argument("--mode", type=str, default="simulation",
                       choices=["simulation", "hardware", "visualization"],
                       help="Evaluation mode")

    # Simulation settings
    parser.add_argument("--task", type=str, default="hover",
                       choices=["hover", "waypoint", "velocity", "delivery"],
                       help="Task to evaluate")
    parser.add_argument("--difficulty", type=float, default=0.5,
                       help="Task difficulty")
    parser.add_argument("--n-episodes", type=int, default=100,
                       help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="Maximum steps per episode")

    # Hardware settings
    parser.add_argument("--platform", type=str, default="simulation",
                       choices=["crazyflie", "tello", "simulation"],
                       help="Hardware platform")
    parser.add_argument("--address", type=str, default=None,
                       help="Hardware address (platform specific)")

    # Visualization
    parser.add_argument("--render", action="store_true",
                       help="Enable visualization")
    parser.add_argument("--record", type=str, default=None,
                       help="Record video to file")

    # Output
    parser.add_argument("--output", type=str, default="eval_results.json",
                       help="Output file for results")

    return parser.parse_args()


class SimulationEvaluator:
    """Evaluate trained model in simulation."""

    def __init__(
        self,
        agent: PPOAgent,
        task: TaskType,
        difficulty: float,
        render: bool = False
    ):
        self.agent = agent
        self.task = task
        self.difficulty = difficulty
        self.render = render

        self.env = DroneEnv(
            task=task,
            difficulty=difficulty,
            render_mode="human" if render else None,
            domain_randomization=False
        )

    def evaluate(
        self,
        n_episodes: int = 100,
        max_steps: int = 1000
    ) -> Dict:
        """Run evaluation episodes."""
        results = {
            'rewards': [],
            'lengths': [],
            'successes': [],
            'final_positions': [],
            'final_errors': [],
            'trajectories': []
        }

        for ep in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            trajectory = [info['position'].copy()]
            done = False
            step = 0

            while not done and step < max_steps:
                action, _ = self.agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
                step += 1

                trajectory.append(info['position'].copy())

                if self.render:
                    self.env.render()

            # Record results
            results['rewards'].append(episode_reward)
            results['lengths'].append(step)
            results['successes'].append(info['position_error'] < 0.2)
            results['final_positions'].append(info['position'].tolist())
            results['final_errors'].append(info['position_error'])

            # Store trajectory (subsample for storage)
            traj_subsample = trajectory[::10]
            results['trajectories'].append([p.tolist() for p in traj_subsample])

            if (ep + 1) % 10 == 0:
                print(f"Episode {ep + 1}/{n_episodes}: "
                      f"Reward={episode_reward:.1f}, "
                      f"Error={info['position_error']:.3f}m")

        # Compute summary statistics
        results['summary'] = {
            'mean_reward': float(np.mean(results['rewards'])),
            'std_reward': float(np.std(results['rewards'])),
            'mean_length': float(np.mean(results['lengths'])),
            'success_rate': float(np.mean(results['successes'])),
            'mean_final_error': float(np.mean(results['final_errors'])),
            'std_final_error': float(np.std(results['final_errors'])),
            'min_error': float(np.min(results['final_errors'])),
            'max_error': float(np.max(results['final_errors']))
        }

        return results


class HardwareEvaluator:
    """Deploy and evaluate on real hardware."""

    def __init__(
        self,
        agent: PPOAgent,
        platform: str,
        address: Optional[str] = None
    ):
        self.agent = agent
        self.drone = create_interface(platform)
        self.address = address

        # Safety parameters
        self.max_duration = 60.0  # seconds
        self.safe_bounds = np.array([2.0, 2.0, 2.0])  # meters
        self.min_battery = 20  # percent

    def connect(self) -> bool:
        """Connect to drone."""
        return self.drone.connect(self.address)

    def disconnect(self):
        """Disconnect from drone."""
        self.drone.disconnect()

    def run_episode(
        self,
        target: np.ndarray,
        duration: float = 30.0
    ) -> Dict:
        """Run a single evaluation episode on hardware."""
        results = {
            'positions': [],
            'targets': [],
            'errors': [],
            'timestamps': []
        }

        # Safety checks
        state = self.drone.get_state()
        if state and state.battery_percent < self.min_battery:
            print(f"Battery too low: {state.battery_percent}%")
            return results

        # Takeoff
        print("Taking off...")
        if not self.drone.takeoff():
            print("Takeoff failed!")
            return results

        time.sleep(2.0)  # Stabilize

        print(f"Running episode for {duration}s, target: {target}")
        start_time = time.time()

        try:
            while time.time() - start_time < min(duration, self.max_duration):
                state = self.drone.get_state()
                if state is None:
                    continue

                # Safety check - bounds
                if np.any(np.abs(state.position) > self.safe_bounds):
                    print("Out of bounds - landing!")
                    break

                # Construct observation (same format as simulation)
                obs = self._state_to_observation(state, target)

                # Get action from trained policy
                action, _ = self.agent.select_action(obs, deterministic=True)

                # Send to drone (convert motor commands to attitude if needed)
                self._send_action(action)

                # Record data
                results['positions'].append(state.position.tolist())
                results['targets'].append(target.tolist())
                results['errors'].append(float(np.linalg.norm(state.position - target)))
                results['timestamps'].append(time.time() - start_time)

                time.sleep(0.01)  # 100Hz control loop

        except KeyboardInterrupt:
            print("Interrupted by user")

        finally:
            # Always land
            print("Landing...")
            self.drone.land()
            time.sleep(3.0)

        # Compute summary
        if results['errors']:
            results['summary'] = {
                'mean_error': float(np.mean(results['errors'])),
                'min_error': float(np.min(results['errors'])),
                'max_error': float(np.max(results['errors'])),
                'final_error': float(results['errors'][-1])
            }

        return results

    def _state_to_observation(self, state, target: np.ndarray) -> np.ndarray:
        """Convert drone state to observation vector."""
        # Match simulation observation format
        obs = np.concatenate([
            state.position / 5.0,
            state.velocity / 5.0,
            state.orientation,
            state.angular_velocity / 10.0,
            (target - state.position) / 5.0,
            np.zeros(4)  # Previous action (simplified)
        ]).astype(np.float32)

        return obs

    def _send_action(self, action: np.ndarray):
        """Send action to drone."""
        # For most consumer drones, convert motor commands to attitude
        # This is a simplified conversion - real implementation would
        # use proper motor mixing based on drone configuration

        # Estimate thrust and attitude from motor commands
        thrust = np.mean(action)

        # Simple attitude estimation from differential thrust
        roll = (action[0] + action[2] - action[1] - action[3]) * 0.2
        pitch = (action[0] + action[1] - action[2] - action[3]) * 0.2
        yaw_rate = (action[1] + action[2] - action[0] - action[3]) * 0.5

        self.drone.send_attitude_command(thrust, roll, pitch, yaw_rate)


def main():
    """Main entry point."""
    args = parse_args()

    # Load trained agent
    print(f"Loading model from {args.checkpoint}")
    agent = PPOAgent.from_checkpoint(args.checkpoint)
    print(f"Model loaded. Total training steps: {agent.total_steps:,}")

    if args.mode == "simulation":
        # Simulation evaluation
        evaluator = SimulationEvaluator(
            agent=agent,
            task=TaskType(args.task),
            difficulty=args.difficulty,
            render=args.render
        )

        print(f"\nEvaluating in simulation ({args.n_episodes} episodes)...")
        results = evaluator.evaluate(
            n_episodes=args.n_episodes,
            max_steps=args.max_steps
        )

        # Print summary
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Mean Reward:     {results['summary']['mean_reward']:.2f} +/- {results['summary']['std_reward']:.2f}")
        print(f"Success Rate:    {results['summary']['success_rate']*100:.1f}%")
        print(f"Mean Error:      {results['summary']['mean_final_error']:.3f}m")
        print(f"Min/Max Error:   {results['summary']['min_error']:.3f}m / {results['summary']['max_error']:.3f}m")

    elif args.mode == "hardware":
        # Hardware deployment
        evaluator = HardwareEvaluator(
            agent=agent,
            platform=args.platform,
            address=args.address
        )

        if not evaluator.connect():
            print("Failed to connect to drone!")
            return

        try:
            # Run evaluation
            target = np.array([0.0, 0.0, 1.0])  # Hover at 1m
            results = evaluator.run_episode(target, duration=30.0)

            if 'summary' in results:
                print("\n" + "=" * 50)
                print("HARDWARE EVALUATION RESULTS")
                print("=" * 50)
                print(f"Mean Error:  {results['summary']['mean_error']:.3f}m")
                print(f"Final Error: {results['summary']['final_error']:.3f}m")

        finally:
            evaluator.disconnect()

    elif args.mode == "visualization":
        # Visualization demo
        from drone_ai.visualization import DroneRenderer

        env = DroneEnv(
            task=TaskType(args.task),
            difficulty=args.difficulty,
            render_mode="human"
        )

        obs, info = env.reset()
        done = False

        print("Running visualization... Press ESC to exit")

        while not done:
            action, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            env.render()

    # Save results
    if args.mode in ["simulation", "hardware"]:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            # Convert numpy arrays for JSON serialization
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
