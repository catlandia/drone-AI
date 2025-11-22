#!/usr/bin/env python3
"""
Drone AI Launcher
=================
Easy-to-use menu to start training, evaluation, or visualization.

Just run: python start.py
"""

import os
import sys
import subprocess


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the program header."""
    print("=" * 60)
    print("            DRONE AI - Delivery Drone Training")
    print("=" * 60)
    print()


def print_menu():
    """Print the main menu."""
    print("What would you like to do?")
    print()
    print("  [1] Train - Hover (basic stabilization)")
    print("  [2] Train - Delivery (pickup and drop)")
    print("  [3] Train - Delivery Route (1km with obstacles)")
    print()
    print("  [4] Watch Demo (visualization only)")
    print("  [5] Evaluate a trained model")
    print()
    print("  [6] Train with LIVE visualization")
    print("  [7] Custom training (advanced)")
    print()
    print("  [0] Exit")
    print()


def run_command(cmd):
    """Run a command and wait for it to complete."""
    print()
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nCommand failed with error: {e}")
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    print()
    input("Press Enter to continue...")


def train_hover():
    """Start hover training."""
    clear_screen()
    print_header()
    print("TRAINING: Hover Task")
    print("The drone will learn to stabilize and hover in place.")
    print()

    steps = input("Training steps (default 100000): ").strip()
    steps = steps if steps else "100000"

    difficulty = input("Difficulty 0.0-1.0 (default 0.3): ").strip()
    difficulty = difficulty if difficulty else "0.3"

    cmd = [
        sys.executable, "-m", "drone_ai.train",
        "--task", "hover",
        "--total-timesteps", steps,
        "--difficulty", difficulty
    ]

    run_command(cmd)


def train_delivery():
    """Start delivery training."""
    clear_screen()
    print_header()
    print("TRAINING: Delivery Task")
    print("The drone will learn to pick up packages and drop them accurately.")
    print()

    steps = input("Training steps (default 500000): ").strip()
    steps = steps if steps else "500000"

    difficulty = input("Difficulty 0.0-1.0 (default 0.5): ").strip()
    difficulty = difficulty if difficulty else "0.5"

    cmd = [
        sys.executable, "-m", "drone_ai.train",
        "--task", "delivery",
        "--total-timesteps", steps,
        "--difficulty", difficulty
    ]

    run_command(cmd)


def train_delivery_route():
    """Start long-range delivery route training."""
    clear_screen()
    print_header()
    print("TRAINING: Delivery Route (Long-Range)")
    print("The drone will learn to:")
    print("  - Fly up to 1km to a dropzone")
    print("  - Navigate through waypoints")
    print("  - Avoid obstacles (trees, buildings)")
    print("  - Drop packages within 5m accuracy")
    print("  - Return to base and reload")
    print()

    steps = input("Training steps (default 1000000): ").strip()
    steps = steps if steps else "1000000"

    difficulty = input("Difficulty 0.0-1.0 (default 0.5): ").strip()
    difficulty = difficulty if difficulty else "0.5"

    randomization = input("Enable domain randomization? (y/n, default n): ").strip().lower()

    cmd = [
        sys.executable, "-m", "drone_ai.train",
        "--task", "delivery_route",
        "--total-timesteps", steps,
        "--difficulty", difficulty
    ]

    if randomization == 'y':
        cmd.append("--domain-randomization")

    run_command(cmd)


def watch_demo():
    """Run visualization demo."""
    clear_screen()
    print_header()
    print("VISUALIZATION DEMO")
    print("Watch a demo of the drone simulation.")
    print()
    print("Controls:")
    print("  Arrow keys - Rotate camera")
    print("  +/- - Zoom in/out")
    print("  1-4 - Switch camera mode")
    print("  T - Toggle trajectory")
    print("  H - Toggle HUD")
    print("  ESC - Exit")
    print()
    input("Press Enter to start...")

    cmd = [sys.executable, "-m", "drone_ai.visualization"]
    run_command(cmd)


def evaluate_model():
    """Evaluate a trained model."""
    clear_screen()
    print_header()
    print("EVALUATE TRAINED MODEL")
    print()

    # List available checkpoints
    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        print("Available checkpoints:")
        for folder in os.listdir(checkpoint_dir):
            folder_path = os.path.join(checkpoint_dir, folder)
            if os.path.isdir(folder_path):
                print(f"  - {folder}")
        print()

    checkpoint = input("Checkpoint path (e.g., checkpoints/drone_hover_xxx/best.pt): ").strip()
    if not checkpoint:
        print("No checkpoint specified.")
        input("Press Enter to continue...")
        return

    task = input("Task (hover/delivery/delivery_route, default hover): ").strip()
    task = task if task else "hover"

    print()
    print("Mode:")
    print("  1. Simulation (run many episodes, get stats)")
    print("  2. Visualization (watch the drone)")
    mode_choice = input("Choose mode (1 or 2, default 2): ").strip()
    mode = "simulation" if mode_choice == "1" else "visualization"

    cmd = [
        sys.executable, "-m", "drone_ai.evaluate",
        "--checkpoint", checkpoint,
        "--task", task,
        "--mode", mode
    ]

    if mode == "visualization":
        cmd.append("--render")

    run_command(cmd)


def train_with_visualization():
    """Start training with live visualization."""
    clear_screen()
    print_header()
    print("TRAINING WITH LIVE VISUALIZATION")
    print("Watch the drone learn in real-time!")
    print()
    print("Note: This is slower than training without visualization.")
    print()

    print("Choose task:")
    print("  1. Hover")
    print("  2. Delivery")
    print("  3. Delivery Route")
    task_choice = input("Task (1-3, default 3): ").strip()

    if task_choice == "1":
        task = "hover"
    elif task_choice == "2":
        task = "delivery"
    else:
        task = "delivery_route"

    steps = input("Training steps (default 100000): ").strip()
    steps = steps if steps else "100000"

    render_freq = input("Render frequency (1=every step, 10=faster, default 5): ").strip()
    render_freq = render_freq if render_freq else "5"

    cmd = [
        sys.executable, "-m", "drone_ai.train",
        "--task", task,
        "--total-timesteps", steps,
        "--render",
        "--render-freq", render_freq,
        "--difficulty", "0.5"
    ]

    run_command(cmd)


def custom_training():
    """Advanced custom training options."""
    clear_screen()
    print_header()
    print("CUSTOM TRAINING (Advanced)")
    print()
    print("Enter the full command arguments:")
    print()
    print("Example arguments:")
    print("  --task delivery_route --total-timesteps 500000 --difficulty 0.7")
    print("  --domain-randomization --curriculum --render")
    print()

    args = input("Arguments: ").strip()
    if not args:
        print("No arguments provided.")
        input("Press Enter to continue...")
        return

    cmd = [sys.executable, "-m", "drone_ai.train"] + args.split()
    run_command(cmd)


def main():
    """Main menu loop."""
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    while True:
        clear_screen()
        print_header()
        print_menu()

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            train_hover()
        elif choice == "2":
            train_delivery()
        elif choice == "3":
            train_delivery_route()
        elif choice == "4":
            watch_demo()
        elif choice == "5":
            evaluate_model()
        elif choice == "6":
            train_with_visualization()
        elif choice == "7":
            custom_training()
        elif choice == "0":
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
