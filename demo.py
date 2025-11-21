#!/usr/bin/env python3
"""
Drone AI Demo Script

Run this script to see the drone simulation in action with a simple
hand-coded controller. This demonstrates the simulation and visualization
without requiring a trained model.

Usage:
    python demo.py [--mode hover|waypoint|circle]
"""

import argparse
import numpy as np
import sys

def run_demo(mode: str = "hover"):
    """Run the drone simulation demo."""

    print("=" * 60)
    print("   DRONE AI - Simulation Demo")
    print("=" * 60)
    print()
    print(f"Mode: {mode}")
    print()
    print("Controls:")
    print("  Arrow keys  - Rotate camera")
    print("  +/-         - Zoom in/out")
    print("  Space       - Toggle follow mode")
    print("  R           - Reset camera")
    print("  T           - Toggle trajectory")
    print("  H           - Toggle HUD")
    print("  ESC         - Exit")
    print()
    print("Starting simulation...")
    print()

    try:
        import pygame
    except ImportError:
        print("ERROR: pygame is required for visualization.")
        print("Install it with: pip install pygame")
        sys.exit(1)

    from drone_ai.simulation import DroneSimulation
    from drone_ai.visualization import DroneRenderer

    # Initialize simulation and renderer
    sim = DroneSimulation()
    renderer = DroneRenderer(width=1024, height=768)

    # Reset simulation
    state = sim.reset(position=np.array([0, 0, 1.0]))
    trajectory = [state.position.copy()]

    # Set target based on mode
    if mode == "hover":
        target = np.array([0.0, 0.0, 1.5])
        print(f"Target: Hover at position {target}")
    elif mode == "waypoint":
        waypoints = [
            np.array([1.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 1.5]),
            np.array([0.0, 1.0, 1.0]),
            np.array([0.0, 0.0, 1.5]),
        ]
        waypoint_idx = 0
        target = waypoints[0]
        print(f"Waypoints: {len(waypoints)} positions to visit")
    elif mode == "circle":
        target = np.array([1.0, 0.0, 1.0])
        print("Target: Circular trajectory")
    else:
        target = np.array([0.0, 0.0, 1.5])

    # Simple PD controller gains
    Kp_pos = 2.0  # Position gain
    Kd_pos = 1.5  # Velocity gain
    Kp_att = 3.0  # Attitude gain
    Kd_att = 0.5  # Angular velocity gain

    # Main loop
    running = True
    step = 0
    hover_action = sim.compute_hover_action()

    while running:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Update target for dynamic modes
        if mode == "waypoint":
            # Check if reached current waypoint
            if np.linalg.norm(state.position - target) < 0.3:
                waypoint_idx = (waypoint_idx + 1) % len(waypoints)
                target = waypoints[waypoint_idx]
                print(f"Reached waypoint! Moving to {target}")

        elif mode == "circle":
            # Circular trajectory
            t = step * 0.01
            radius = 1.5
            target = np.array([
                radius * np.cos(t * 0.5),
                radius * np.sin(t * 0.5),
                1.0 + 0.3 * np.sin(t)
            ])

        # ===== Simple PD Controller =====
        # This is a basic controller to demonstrate the simulation.
        # A trained RL agent would replace this logic.

        # Position error
        pos_error = target - state.position

        # Desired acceleration (PD control)
        desired_acc = Kp_pos * pos_error - Kd_pos * state.velocity

        # Desired thrust (compensate for gravity)
        R = state.get_rotation_matrix()
        gravity_comp = np.array([0, 0, 9.81])
        desired_thrust = gravity_comp + desired_acc

        # Project thrust onto body z-axis
        body_z = R[:, 2]
        thrust = np.dot(desired_thrust, body_z)
        thrust_normalized = np.clip(thrust / 20.0, 0.3, 0.7)  # Normalize

        # Desired attitude (simplified)
        euler = state.get_euler_angles()

        # Roll and pitch to achieve lateral acceleration
        desired_roll = np.clip(-desired_acc[1] / 10.0, -0.3, 0.3)
        desired_pitch = np.clip(desired_acc[0] / 10.0, -0.3, 0.3)

        # Attitude error
        roll_error = desired_roll - euler[0]
        pitch_error = desired_pitch - euler[1]

        # Attitude control (PD)
        roll_cmd = Kp_att * roll_error - Kd_att * state.angular_velocity[0]
        pitch_cmd = Kp_att * pitch_error - Kd_att * state.angular_velocity[1]
        yaw_cmd = -Kd_att * state.angular_velocity[2]  # Just damp yaw

        # Mix to motor commands
        # Motor arrangement: [FL, FR, RL, RR]
        action = np.array([
            thrust_normalized + roll_cmd + pitch_cmd - yaw_cmd,
            thrust_normalized - roll_cmd + pitch_cmd + yaw_cmd,
            thrust_normalized + roll_cmd - pitch_cmd + yaw_cmd,
            thrust_normalized - roll_cmd - pitch_cmd - yaw_cmd,
        ])
        action = np.clip(action, 0, 1)

        # Step simulation
        state = sim.step(action)
        trajectory.append(state.position.copy())

        # Limit trajectory length for rendering performance
        if len(trajectory) > 1000:
            trajectory = trajectory[-500:]

        # Render
        renderer.render(state, target, trajectory)

        # Check for crash
        if sim.is_crashed():
            print("CRASHED! Resetting...")
            state = sim.reset(position=np.array([0, 0, 1.0]))
            trajectory = [state.position.copy()]

        step += 1

        # Print status periodically
        if step % 500 == 0:
            error = np.linalg.norm(state.position - target)
            print(f"Step {step}: Position error = {error:.3f}m, "
                  f"Altitude = {state.position[2]:.2f}m")

    renderer.close()
    print("\nDemo finished!")


def main():
    parser = argparse.ArgumentParser(description="Drone AI Demo")
    parser.add_argument("--mode", type=str, default="hover",
                       choices=["hover", "waypoint", "circle"],
                       help="Demo mode")
    args = parser.parse_args()

    run_demo(args.mode)


if __name__ == "__main__":
    main()
