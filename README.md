# Drone AI - Learn to Fly with Reinforcement Learning

A complete framework for training AI agents to fly drones using reinforcement learning in simulation, with the goal of deploying learned behaviors to real hardware (sim-to-real transfer).

## Features

- **Realistic Physics Simulation**: Quadcopter dynamics with accurate rigid body physics, motor dynamics, and aerodynamic effects
- **Gymnasium Environment**: Standard RL interface compatible with popular libraries
- **PPO Agent**: Clean implementation of Proximal Policy Optimization for continuous control
- **3D Visualization**: Real-time flight visualization with camera controls and telemetry HUD
- **Domain Randomization**: Built-in support for sim-to-real transfer through parameter randomization
- **Curriculum Learning**: Gradually increase difficulty as the agent improves
- **Hardware Interfaces**: Ready-to-use interfaces for Crazyflie and DJI Tello drones
- **TensorBoard Logging**: Track training progress with detailed metrics

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/drone-AI.git
cd drone-AI

# Install dependencies
pip install -e .

# Or install with development tools
pip install -e ".[dev]"

# For real drone support
pip install -e ".[real]"
```

### Train Your First Agent

```bash
# Basic hover training
python -m drone_ai.train --task hover --total-timesteps 500000

# With visualization (slower but you can watch!)
python -m drone_ai.train --task hover --total-timesteps 100000 --render

# With curriculum learning and domain randomization
python -m drone_ai.train \
    --task hover \
    --total-timesteps 1000000 \
    --curriculum \
    --domain-randomization \
    --name my_experiment
```

### Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir logs/

# Open http://localhost:6006 in your browser
```

### Evaluate Trained Model

```bash
# In simulation
python -m drone_ai.evaluate \
    --checkpoint checkpoints/my_experiment/best.pt \
    --mode simulation \
    --render

# On real hardware (Crazyflie)
python -m drone_ai.evaluate \
    --checkpoint checkpoints/my_experiment/best.pt \
    --mode hardware \
    --platform crazyflie
```

## Project Structure

```
drone-AI/
├── src/drone_ai/
│   ├── __init__.py          # Package initialization
│   ├── simulation.py        # Physics simulation
│   ├── environment.py       # Gymnasium environment
│   ├── agent.py             # PPO implementation
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation and deployment
│   ├── visualization.py     # 3D rendering
│   └── hardware.py          # Real drone interfaces
├── checkpoints/             # Saved models
├── logs/                    # TensorBoard logs
├── pyproject.toml           # Package configuration
└── README.md
```

## Training Tasks

### Hover (Recommended for Starting)
Learn to maintain a stable position in 3D space.
```bash
python -m drone_ai.train --task hover
```

### Waypoint Navigation
Navigate through a series of target positions.
```bash
python -m drone_ai.train --task waypoint --difficulty 0.5
```

### Package Delivery (NEW!)
Pick up a package and deliver it to a drop zone with precision.
```bash
# Train delivery task
python -m drone_ai.train --task delivery --difficulty 0.3 --total-timesteps 2000000

# With curriculum learning for better results
python -m drone_ai.train --task delivery --curriculum --domain-randomization
```

The delivery task has three phases:
1. **Pickup**: Fly to the cyan pickup zone and descend to grab the package
2. **Transport**: Carry the package to the magenta drop zone
3. **Drop**: Release the package over the target (accuracy is rewarded!)

### Velocity Tracking
Maintain a target velocity (useful for trajectory following).
```bash
python -m drone_ai.train --task velocity
```

## Configuration Options

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--task` | hover | Training task type |
| `--difficulty` | 0.3 | Task difficulty (0-1) |
| `--total-timesteps` | 1,000,000 | Total training steps |
| `--n-steps` | 2048 | Steps per update |
| `--batch-size` | 64 | Minibatch size |
| `--lr` | 3e-4 | Learning rate |
| `--curriculum` | False | Enable curriculum learning |
| `--domain-randomization` | False | Enable domain randomization |

### Domain Randomization

Domain randomization helps bridge the sim-to-real gap by training on a distribution of environments:

- Mass: ±20%
- Inertia: ±15%
- Motor thrust coefficient: ±10%
- Drag coefficients: ±30%
- Motor response time: ±25%

## Sim-to-Real Transfer

To deploy trained models to real hardware:

1. **Train with domain randomization**:
   ```bash
   python -m drone_ai.train --domain-randomization --curriculum
   ```

2. **Test extensively in simulation**:
   ```bash
   python -m drone_ai.evaluate --checkpoint best.pt --n-episodes 100
   ```

3. **Deploy to hardware** (start with simulation interface first!):
   ```bash
   # Test deployment code with simulated drone
   python -m drone_ai.evaluate --mode hardware --platform simulation

   # When ready, deploy to real drone
   python -m drone_ai.evaluate --mode hardware --platform crazyflie
   ```

### Supported Hardware

| Platform | Library | Notes |
|----------|---------|-------|
| Crazyflie 2.x | cflib | Best for research, fully open source |
| DJI Tello | djitellopy | Consumer-friendly, limited control |
| MAVLink | (coming soon) | PX4, ArduPilot support |

## Understanding the Code

### Physics Simulation (`simulation.py`)

The drone is modeled as a rigid body with:
- 6 DOF state: position, velocity, orientation (quaternion), angular velocity
- 4 motors in X-configuration with thrust and torque
- First-order motor dynamics
- Quadratic aerodynamic drag
- Ground collision detection

### RL Environment (`environment.py`)

Observation space (19D):
- Position (3): normalized world position
- Velocity (3): normalized world velocity
- Orientation (3): roll, pitch, yaw angles
- Angular velocity (3): body frame rates
- Target relative (3): vector to target
- Previous action (4): last motor commands

Action space (4D):
- Motor commands: normalized [0, 1] for each motor

Reward function:
- Position tracking (main objective)
- Velocity penalty (prefer stability)
- Orientation stability bonus
- Action smoothness (avoid jerky control)
- Alive bonus / crash penalty

### PPO Agent (`agent.py`)

- Actor-Critic network with shared features
- Gaussian policy with learned standard deviation
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Value function clipping

## Tips for Better Results

1. **Start simple**: Begin with hover task at low difficulty
2. **Use curriculum**: Enable `--curriculum` for faster learning
3. **Monitor training**: Watch TensorBoard for reward curves
4. **Domain randomization**: Essential for sim-to-real transfer
5. **Tune carefully**: Position reward weight is most important
6. **Long training**: 1M+ steps often needed for good performance

## Visualization Controls

When running with `--render`:

| Key | Action |
|-----|--------|
| Arrow keys | Rotate camera |
| +/- | Zoom in/out |
| Space | Toggle follow mode |
| R | Reset camera |
| T | Toggle trajectory |
| H | Toggle HUD |
| ESC | Exit |

## Troubleshooting

### Training is slow
- Reduce `--n-steps` or use smaller network
- Disable rendering during training

### Agent crashes immediately
- Start with lower difficulty
- Check reward weights in environment
- Ensure action space bounds are correct

### Sim-to-real gap is large
- Increase domain randomization
- Use curriculum learning
- Tune simulation parameters to match real drone
- Add sensor noise to observations

## Contributing

Contributions are welcome! Areas of interest:
- Additional training tasks (landing, obstacle avoidance)
- More hardware interfaces
- Improved sim-to-real techniques
- Better visualization

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- OpenAI Gym/Gymnasium for the RL environment interface
- PyTorch for neural network implementation
- Bitcraze for Crazyflie hardware and documentation
