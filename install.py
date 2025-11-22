#!/usr/bin/env python3
"""
Drone AI Installer
==================
Installs all required dependencies for the Drone AI project.

Just run: python install.py
"""

import subprocess
import sys
import os


def print_header():
    """Print installer header."""
    print("=" * 60)
    print("          DRONE AI - Dependency Installer")
    print("=" * 60)
    print()


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("  ERROR: Python 3.9 or higher is required!")
        print("  Please install a newer version of Python.")
        return False

    print("  OK!")
    return True


def is_package_installed(module_name):
    """Check if a package is already installed."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def install_package(package_name, pip_name=None, module_name=None):
    """Install a single package if not already installed."""
    if pip_name is None:
        pip_name = package_name
    if module_name is None:
        module_name = package_name.lower()

    # Check if already installed
    if is_package_installed(module_name):
        print(f"  {package_name}... already installed")
        return True

    print(f"  {package_name}... installing...", end=" ", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_name, "-q"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("OK")
            return True
        else:
            print("FAILED")
            print(f"    Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"FAILED ({e})")
        return False


def install_dependencies():
    """Install all required dependencies."""
    print("\nChecking and installing dependencies...")
    print()

    # Core dependencies: (display_name, pip_package, import_name)
    packages = [
        ("NumPy", "numpy>=1.24.0", "numpy"),
        ("PyTorch", "torch>=2.0.0", "torch"),
        ("Gymnasium", "gymnasium>=0.29.0", "gymnasium"),
        ("Pygame", "pygame>=2.5.0", "pygame"),
        ("Matplotlib", "matplotlib>=3.7.0", "matplotlib"),
        ("TensorBoard", "tensorboard>=2.14.0", "tensorboard"),
        ("PyYAML", "pyyaml>=6.0", "yaml"),
        ("tqdm", "tqdm>=4.65.0", "tqdm"),
    ]

    failed = []
    for name, pip_name, module_name in packages:
        if not install_package(name, pip_name, module_name):
            failed.append(name)

    return failed


def verify_installation():
    """Verify that all packages are installed correctly."""
    print("\nVerifying installation...")

    packages_to_check = [
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("gymnasium", "Gymnasium"),
        ("pygame", "Pygame"),
        ("matplotlib", "Matplotlib"),
    ]

    all_ok = True
    for module_name, display_name in packages_to_check:
        print(f"  {display_name}...", end=" ", flush=True)
        try:
            __import__(module_name)
            print("OK")
        except ImportError:
            print("MISSING")
            all_ok = False

    return all_ok


def main():
    """Main installer function."""
    print_header()

    # Check Python version
    if not check_python_version():
        input("\nPress Enter to exit...")
        sys.exit(1)

    # Upgrade pip first
    print("\nUpgrading pip...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "-q"],
        capture_output=True
    )
    print("  Done!")

    # Install dependencies
    failed = install_dependencies()

    if failed:
        print(f"\nWARNING: Some packages failed to install: {', '.join(failed)}")
        print("You may need to install them manually.")

    # Verify
    if verify_installation():
        print("\n" + "=" * 60)
        print("  Installation complete!")
        print("=" * 60)
        print()
        print("You can now run the program with:")
        print("  python start.py")
        print()
    else:
        print("\nSome packages are missing. Please check the errors above.")

    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
