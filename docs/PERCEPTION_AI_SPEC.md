# Perception AI Specification

## For Future Development Sessions

This document describes what needs to be built for the Perception AI layer of the drone system.

---

## Current System Architecture

```
Mission Planner    (NOT BUILT)
       |
       v
Path Planner       (NOT BUILT)
       |
       v
Perception AI      <-- BUILD THIS NEXT
       |
       v
Flight Controller  (DONE - "flycontrol")
```

---

## What Flight Controller Expects

The Flight Controller (`src/drone_ai/agent.py`) takes observations and outputs motor commands.

**Current observation format** (from `environment.py`):
- Position (x, y, z)
- Velocity (vx, vy, vz)
- Orientation (quaternion: w, x, y, z)
- Angular velocity (wx, wy, wz)
- Target position (relative)
- Package info (for delivery tasks)

**The Perception AI needs to provide:**
- Obstacle positions in 3D world coordinates
- Obstacle sizes/bounding boxes
- Obstacle types (optional: tree, building, person, etc.)
- Distance to nearest obstacle
- Free/safe directions

---

## Perception AI Requirements

### Input
- Camera frames (RGB images, e.g., 640x480 or 320x240)
- Optional: Stereo camera pair (for depth)
- Optional: Depth camera data
- Drone's current position and orientation (to convert to world coords)

### Output Format
```python
{
    "timestamp": float,  # Time of observation
    "obstacles": [
        {
            "id": int,                    # Tracking ID
            "type": str,                  # "tree", "building", "person", "unknown"
            "position": [x, y, z],        # World coordinates (meters)
            "size": [width, height, depth],  # Bounding box size (meters)
            "velocity": [vx, vy, vz],     # Movement speed (for tracking)
            "confidence": float           # 0.0 to 1.0
        },
        # ... more obstacles
    ],
    "free_directions": [                  # Safe directions to fly
        {"direction": [dx, dy, dz], "distance": float},
        # ...
    ],
    "ground_distance": float,             # Height above ground (meters)
    "nearest_obstacle_distance": float    # Distance to closest obstacle
}
```

### Integration Point
Create a new file: `src/drone_ai/perception.py`

The perception module should:
1. Process camera frames
2. Detect obstacles
3. Estimate depth/distance
4. Track objects across frames
5. Output obstacle data in the format above

---

## Components to Build

### 1. Object Detector
- Detects objects in camera image
- Outputs bounding boxes + class labels
- Can use CNN (YOLO-style architecture)
- Train on aerial/drone images

### 2. Depth Estimator
- Converts 2D detections to 3D positions
- Options:
  - Stereo camera (two cameras, calculate disparity)
  - Monocular depth estimation (single camera + neural network)
  - Depth camera (direct depth data)

### 3. Coordinate Transformer
- Convert camera coordinates to world coordinates
- Needs drone position + orientation
- Formula: world_pos = drone_pos + rotate(camera_to_body, drone_orientation) * depth

### 4. Object Tracker
- Track objects across frames (same object = same ID)
- Predict movement for moving objects
- Use Kalman filter or similar

---

## File Naming Convention

Models should be saved as:
```
{Grade} {DD-MM-YYYY} perception v{N}.pt
```

Examples:
- `C 25-11-2025 perception v1.pt`
- `A 30-11-2025 perception v2.pt`

This matches the Flight Controller format:
- `S+ 22-11-2025 flycontrol v1.pt`

---

## Training Data Needed

For the object detector, you'll need:
- Aerial images from drone perspective
- Labeled bounding boxes for:
  - Trees
  - Buildings
  - People
  - Vehicles
  - Power lines / poles
  - Other drones
  - Birds

Options:
- Use existing datasets (VisDrone, UAV123)
- Generate synthetic data in simulation
- Collect and label custom data

---

## Performance Requirements

- **Speed**: Must run at 15+ FPS on drone hardware (30+ FPS preferred)
- **Latency**: <100ms from frame to output
- **Range**: Detect obstacles from 5m to 100m
- **Accuracy**: Position error <1m at 20m distance

---

## Integration with Existing Code

### Option A: Extend Environment (for simulation)
Add perception to `environment.py`:
```python
def _get_observation(self):
    # ... existing observation code ...

    # Add perception data
    if self.use_perception:
        perception_data = self.perception_ai.process_frame(camera_frame)
        # Add obstacle info to observation
```

### Option B: Separate Module (for real hardware)
```python
from drone_ai.perception import PerceptionAI

perception = PerceptionAI(model_path="A 25-11-2025 perception v1.pt")

while running:
    frame = camera.get_frame()
    obstacles = perception.detect(frame, drone_position, drone_orientation)
    # Pass obstacles to path planner
```

---

## Grading Criteria (for graduation)

The perception AI should be graded on:
- **Detection accuracy**: % of obstacles correctly detected
- **False positive rate**: % of false detections
- **Position accuracy**: Error in estimated 3D position
- **Speed**: Frames per second
- **Tracking consistency**: Same object keeps same ID

Suggested grade thresholds:
- **P (Perfect)**: >95% detection, <2% false positive, <0.5m error
- **S (Supreme)**: >90% detection, <5% false positive, <1m error
- **A (Alpha)**: >80% detection, <10% false positive, <2m error
- **B (Better)**: >70% detection, <15% false positive, <3m error
- (etc.)

---

## DO NOT BREAK

When building Perception AI:
1. Do NOT modify `agent.py` (PPO agent) - it works
2. Do NOT modify `simulation.py` physics - it works
3. Do NOT change the model save format
4. Do NOT change the reward weights in `environment.py`
5. Keep the hybrid training system intact

You CAN:
- Add new files in `src/drone_ai/`
- Add new menu options to `start.py`
- Extend `environment.py` observation space (carefully)
- Create new training scripts for perception

---

## Summary

**Goal**: Build a Perception AI that detects obstacles and outputs their 3D positions.

**Input**: Camera frames + drone pose

**Output**: List of obstacles with positions, sizes, and types

**Format**: Compatible with existing system, saves as `{Grade} {Date} perception v{N}.pt`

**Next steps after Perception**: Path Planner, then Mission Planner

---

*Document created: 22-11-2025*
*For: drone-AI project*
*Current progress: Flight Controller complete (1/4 AI layers)*
