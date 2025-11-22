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


def get_src_path():
    """Get the path to the src directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "src")


def get_env_with_pythonpath():
    """Get environment with src added to PYTHONPATH."""
    env = os.environ.copy()
    src_path = get_src_path()

    # Add src to PYTHONPATH
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = src_path + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = src_path

    return env


def check_dependencies():
    """Check if all required dependencies are installed."""
    missing = []

    required = [
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("gymnasium", "Gymnasium"),
        ("pygame", "Pygame"),
    ]

    for module_name, display_name in required:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(display_name)

    return missing


def get_gpu_status():
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return f"GPU: {gpu_name}"
        else:
            return "CPU only (no GPU - training will be slower)"
    except ImportError:
        return "Unknown (PyTorch not installed)"


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the program header."""
    print("=" * 60)
    print("            DRONE AI - Delivery Drone Training")
    print("=" * 60)
    print(f"  {get_gpu_status()}")
    print()


def print_menu():
    """Print the main menu."""
    print("What would you like to do?")
    print()
    print("  [1] Train - Hover (basic stabilization)")
    print("  [2] Train - Delivery (pickup and drop)")
    print("  [3] Train - Delivery Route (1km with obstacles)")
    print("  [4] Train - Deployment Ready (REAL-WORLD simulation!)")
    print("  [5] Train - FULL LEARNING SEQUENCE (all stages + graduation!)")
    print()
    print("  [6] Watch Demo (no training)")
    print("  [7] Evaluate a trained model")
    print("  [8] Custom training (advanced)")
    print()
    print("  [0] Exit")
    print()
    print("Training uses HYBRID PPO+Evolution - fast learning + best drones survive!")
    print("Watch multiple drones compete and evolve in real-time!")
    print()


def run_command(cmd):
    """Run a command and wait for it to complete."""
    print()
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    try:
        # Use environment with PYTHONPATH set to find drone_ai module
        env = get_env_with_pythonpath()
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"\nCommand failed with error: {e}")
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    print()
    input("Press Enter to continue...")


def get_num_drones_input():
    """Get the number of parallel drones from user."""
    print()
    print("Parallel drones speed up training significantly!")
    print("  1 drone  = normal speed (with visualization)")
    print("  4 drones = ~4x faster")
    print("  8 drones = ~8x faster")
    print("  Note: More drones = faster training but uses more CPU/memory")
    print()
    num_envs = input("Number of parallel drones (default 1, max recommended 16): ").strip()
    if not num_envs:
        return "1"
    try:
        n = int(num_envs)
        if n < 1:
            return "1"
        if n > 32:
            print("Warning: Using more than 32 drones may be unstable")
        return str(n)
    except ValueError:
        return "1"


def get_hybrid_params():
    """Get parameters for hybrid training."""
    print()
    print("HYBRID PPO+EVOLUTION - Best of both worlds!")
    print("  - Each drone learns via PPO (fast gradient descent)")
    print("  - After each age, best drones survive")
    print("  - Poor performers replaced with mutated winners")
    print()

    population = input("Number of drones (default 6): ").strip()
    population = population if population else "6"

    ages = input("Number of ages/generations (default 30): ").strip()
    ages = ages if ages else "30"

    steps_per_age = input("Steps per age (default 20000): ").strip()
    steps_per_age = steps_per_age if steps_per_age else "20000"

    return population, ages, steps_per_age


def train_hover():
    """Start hover training with hybrid PPO+Evolution."""
    clear_screen()
    print_header()
    print("TRAINING: Hover Task (Hybrid PPO+Evolution)")
    print("Drones will learn to stabilize and hover in place.")
    print("Fast PPO learning + evolutionary selection = best results!")
    print()
    print("Camera controls: 1-4 switch views, arrows rotate, +/- zoom")

    difficulty = input("\nDifficulty 0.0-1.0 (default 0.3): ").strip()
    difficulty = difficulty if difficulty else "0.3"

    population, ages, steps_per_age = get_hybrid_params()

    cmd = [
        sys.executable, "-m", "drone_ai.hybrid_train",
        "--task", "hover",
        "--population-size", population,
        "--num-ages", ages,
        "--steps-per-age", steps_per_age,
        "--difficulty", difficulty,
        "--render",
        "--render-freq", "5"
    ]

    run_command(cmd)


def train_delivery():
    """Start delivery training with hybrid PPO+Evolution."""
    clear_screen()
    print_header()
    print("TRAINING: Delivery Task (Hybrid PPO+Evolution)")
    print("Drones will learn to pick up packages and drop them accurately.")
    print("Fast PPO learning + evolutionary selection = best results!")
    print()
    print("Camera controls: 1-4 switch views, arrows rotate, +/- zoom")

    difficulty = input("\nDifficulty 0.0-1.0 (default 0.5): ").strip()
    difficulty = difficulty if difficulty else "0.5"

    population, ages, steps_per_age = get_hybrid_params()

    cmd = [
        sys.executable, "-m", "drone_ai.hybrid_train",
        "--task", "delivery",
        "--population-size", population,
        "--num-ages", ages,
        "--steps-per-age", steps_per_age,
        "--difficulty", difficulty,
        "--render",
        "--render-freq", "5"
    ]

    run_command(cmd)


def train_delivery_route():
    """Start long-range delivery route training with hybrid PPO+Evolution."""
    clear_screen()
    print_header()
    print("TRAINING: Delivery Route - 1km (Hybrid PPO+Evolution)")
    print("Drones will learn to:")
    print("  - Fly up to 1km to a dropzone")
    print("  - Navigate through waypoints")
    print("  - Avoid obstacles (trees, buildings)")
    print("  - Drop packages within 5m accuracy")
    print("  - Return to base and reload")
    print()
    print("Fast PPO learning + evolutionary selection = best results!")
    print("Camera controls: 1-4 switch views, arrows rotate, +/- zoom")

    difficulty = input("\nDifficulty 0.0-1.0 (default 0.5): ").strip()
    difficulty = difficulty if difficulty else "0.5"

    population, ages, steps_per_age = get_hybrid_params()

    cmd = [
        sys.executable, "-m", "drone_ai.hybrid_train",
        "--task", "delivery_route",
        "--population-size", population,
        "--num-ages", ages,
        "--steps-per-age", steps_per_age,
        "--difficulty", difficulty,
        "--render",
        "--render-freq", "5"
    ]

    run_command(cmd)


def train_deployment_ready():
    """Final training mode - maximum real-world simulation."""
    clear_screen()
    print_header()
    print("TRAINING: Deployment Ready (Real-World Simulation)")
    print("=" * 55)
    print()
    print("The ULTIMATE test before real-world deployment!")
    print()
    print("Simulates ALL real-world challenges:")
    print("  - Wind: Variable speed, direction, gusts, turbulence")
    print("  - Sensor noise: GPS drift, IMU noise, motor variance")
    print("  - Battery: Voltage sag under load")
    print("  - Physics: Mass variation, drag changes")
    print("  - Obstacles: Random placement, varying sizes")
    print()
    print("If your drone survives THIS, it's ready for the real world!")
    print()
    print("Camera controls: 1-4 switch views, arrows rotate, +/- zoom")

    print("\nSelect mission type:")
    print("  [1] Hover Challenge - Maintain position despite disturbances")
    print("  [2] Delivery Mission - Pickup and drop with all challenges")
    print("  [3] Full Route - 1km delivery with everything enabled")
    mission = input("Mission (default 3): ").strip()
    if mission == "1":
        task = "hover"
    elif mission == "2":
        task = "delivery"
    else:
        task = "delivery_route"

    population, ages, steps_per_age = get_hybrid_params()

    # Deployment ready uses maximum difficulty and domain randomization
    cmd = [
        sys.executable, "-m", "drone_ai.hybrid_train",
        "--task", task,
        "--population-size", population,
        "--num-ages", ages,
        "--steps-per-age", steps_per_age,
        "--difficulty", "1.0",  # Maximum difficulty
        "--domain-randomization",  # Enable all real-world effects
        "--render",
        "--render-freq", "5"
    ]

    run_command(cmd)


def train_learning_sequence():
    """Full learning sequence - all stages with graduation."""
    clear_screen()
    print_header()
    print("FULL LEARNING SEQUENCE")
    print("=" * 55)
    print()
    print("Complete drone training curriculum with graduation!")
    print()
    print("Training stages:")
    print("  1. Hover - Learn basic stabilization")
    print("  2. Delivery - Learn pickup and drop")
    print("  3. Delivery Route - Learn 1km navigation")
    print("  4. Deployment Ready - Handle real-world challenges")
    print()
    print("Process:")
    print("  - All drones train through each stage")
    print("  - Top 2 drones advance to Round 2")
    print("  - Final evaluation and GRADUATION GRADE!")
    print()
    print("This takes longer but produces the best results!")
    print()

    population = input("Number of drones (default 6): ").strip()
    population = population if population else "6"

    ages = input("Ages per stage (default 15): ").strip()
    ages = ages if ages else "15"

    steps = input("Steps per age (default 15000): ").strip()
    steps = steps if steps else "15000"

    render = input("Show visualization? (y/n, default y): ").strip().lower()
    render_flag = [] if render == 'n' else ["--render", "--render-freq", "5"]

    cmd = [
        sys.executable, "-m", "drone_ai.learning_sequence",
        "--population-size", population,
        "--ages-per-stage", ages,
        "--steps-per-age", steps,
    ] + render_flag

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


def run_installer():
    """Run the dependency installer."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    installer_path = os.path.join(script_dir, "install.py")

    if os.path.exists(installer_path):
        subprocess.run([sys.executable, installer_path])
    else:
        print("Installer not found. Please install dependencies manually:")
        print("  pip install numpy torch gymnasium pygame matplotlib tensorboard pyyaml tqdm")
        input("\nPress Enter to continue...")


def main():
    """Main menu loop."""
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Check for missing dependencies on startup
    missing = check_dependencies()
    if missing:
        clear_screen()
        print_header()
        print("MISSING DEPENDENCIES DETECTED")
        print()
        print(f"The following packages are not installed: {', '.join(missing)}")
        print()
        print("Would you like to run the installer?")
        print("  [Y] Yes, install dependencies")
        print("  [N] No, exit")
        print()
        choice = input("Your choice: ").strip().lower()

        if choice in ['y', 'yes', '']:
            run_installer()
            # Re-check after installation
            missing = check_dependencies()
            if missing:
                print(f"\nStill missing: {', '.join(missing)}")
                print("Please install them manually and try again.")
                input("Press Enter to exit...")
                return
        else:
            print("\nPlease run 'python install.py' to install dependencies.")
            return

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
            train_deployment_ready()
        elif choice == "5":
            train_learning_sequence()
        elif choice == "6":
            watch_demo()
        elif choice == "7":
            evaluate_model()
        elif choice == "8":
            custom_training()
        elif choice == "0":
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
