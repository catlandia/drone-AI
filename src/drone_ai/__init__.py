"""
Drone AI - Reinforcement Learning for Autonomous Drone Flight

This package provides a complete framework for training AI agents to fly drones
in simulation, with the goal of transferring learned behaviors to real hardware.

Main Components:
- simulation: Physics-based drone simulation
- environment: Gymnasium-compatible RL environment
- agent: PPO reinforcement learning agent
- hardware: Real drone hardware interfaces
- visualization: 3D flight visualization
"""

__version__ = "0.1.0"
__author__ = "Drone AI Team"

from drone_ai.environment import DroneEnv
from drone_ai.agent import PPOAgent
from drone_ai.simulation import DroneSimulation

__all__ = ["DroneEnv", "PPOAgent", "DroneSimulation", "__version__"]
