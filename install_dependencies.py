#!/usr/bin/env python
"""
Simplified installation script for FEIDEGGER Mamba dataset preparation.

This script installs all necessary dependencies using pip or conda.
"""

import sys
import subprocess
import argparse


def check_python_version():
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required.")
        sys.exit(1)
    print(f"Python version {sys.version_info.major}.{sys.version_info.minor} detected.")


def install_with_pip():
    requirements = [
        "numpy",
        "pandas",
        "tqdm",
        "pillow",
        "requests",
        "torch",
        "torchvision",
        "transformers",
        "huggingface_hub[hf_xet]",
        "h5py",
        "matplotlib"
    ]
    print(f"Installing: {', '.join(requirements)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + requirements)
    print("\nNote: For PyTorch with CUDA, see https://pytorch.org/get-started/locally/")


def install_with_conda(env_name):
    try:
        subprocess.check_call(["conda", "--version"])
    except Exception:
        print("Conda is not installed or not in PATH.")
        sys.exit(1)
    print(f"Creating conda environment '{env_name}' and installing packages...")
    pkgs = [
        "numpy", "pandas", "tqdm", "pillow", "requests", "matplotlib", "h5py"
    ]
    subprocess.check_call(["conda", "create", "-y", "-n", env_name, "python=3.10"])
    subprocess.check_call(["conda", "install", "-y", "-n", env_name] + pkgs)
    subprocess.check_call(["conda", "install", "-y", "-n", env_name, "pytorch", "torchvision", "cpuonly", "-c", "pytorch"])
    subprocess.check_call(["conda", "run", "-n", env_name, "pip", "install", "transformers", "huggingface_hub[hf_xet]"])
    print(f"\nDone! To activate: conda activate {env_name}")


def main():
    parser = argparse.ArgumentParser(description='Install dependencies for FEIDEGGER Mamba dataset preparation')
    parser.add_argument('--conda', action='store_true', help='Use conda to create an environment and install packages')
    parser.add_argument('--env-name', type=str, default='feidegger-mamba', help='Name of the conda environment')
    args = parser.parse_args()
    print("FEIDEGGER Mamba Dataset Preparation - Dependencies Installer")
    print("=" * 60)
    check_python_version()
    if args.conda:
        install_with_conda(args.env_name)
    else:
        install_with_pip()
    print("\nSetup complete! You can now run the dataset preparation script.")


if __name__ == "__main__":
    main()