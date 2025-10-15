#!/usr/bin/env python
"""
Installation script for FEIDEGGER Mamba dataset preparation.

This script helps install all the necessary dependencies for the
FEIDEGGER Mamba dataset preparation and training scripts.
"""

import os
import sys
import subprocess
import argparse


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required.")
        sys.exit(1)
    print(f"Python version {sys.version_info.major}.{sys.version_info.minor} detected.")


def install_requirements():
    """Install the required packages."""
    requirements = [
        "numpy",
        "pandas",
        "tqdm",
        "pillow",
        "requests",
        "torch",
        "torchvision",
        "transformers",
        "huggingface_hub[hf_xet]",  # Added for Xet Storage support
        "h5py",
        "matplotlib",
        "argparse"
    ]
    
    print(f"Installing the following packages: {', '.join(requirements)}")
    
    # Install packages one by one to better handle errors
    for package in requirements:
        try:
            # Special case for PyTorch to guide users
            if package in ["torch", "torchvision"]:
                print(f"\nNote: For {package}, you might want to install it with CUDA support.")
                print(f"Visit https://pytorch.org/get-started/locally/ for installation instructions.")
                print(f"For example, you might use:")
                print(f"pip install {package} --extra-index-url https://download.pytorch.org/whl/cu117")
                print("For this script, we'll install the default version which may not have CUDA support.")
            
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Error: Failed to install {package}. Please install it manually.")
            if package in ["torch", "torchvision"]:
                print("You can visit https://pytorch.org/get-started/locally/ for installation instructions.")


def check_xet_support():
    """Check if Xet Storage support is available."""
    try:
        # Try to import hf_xet
        import importlib.util
        hf_xet_spec = importlib.util.find_spec('hf_xet')
        
        if hf_xet_spec is not None:
            print("✓ Xet Storage support is available (hf_xet is installed)")
            return True
        else:
            # Check if huggingface_hub is installed with hf_xet extra
            import huggingface_hub
            try:
                from huggingface_hub.constants import HF_HUB_XET_ENABLED
                if HF_HUB_XET_ENABLED:
                    print("✓ Xet Storage support is available (huggingface_hub[hf_xet] is installed)")
                    return True
            except ImportError:
                pass
                
            print("✗ Xet Storage support is NOT available")
            print("  For better performance, install the package with:")
            print("  `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`")
            return False
    except ImportError:
        print("✗ Hugging Face Hub is not installed, Xet Storage support unavailable")
        return False


def verify_installation():
    """Verify that all required packages are installed."""
    print("\nVerifying installation...")
    
    packages_to_check = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("tqdm", "tqdm"),
        ("PIL", "pillow"),
        ("requests", "requests"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("huggingface_hub", "huggingface_hub"),
        ("h5py", "h5py"),
        ("matplotlib", "matplotlib")
    ]
    
    all_installed = True
    
    for module_name, package_name in packages_to_check:
        try:
            __import__(module_name)
            print(f"✓ {package_name} is installed")
        except ImportError:
            print(f"✗ {package_name} is NOT installed")
            all_installed = False
    
    # Check Xet Storage support specifically
    print("\nChecking Xet Storage support:")
    check_xet_support()
    
    if all_installed:
        print("\nAll required base packages are successfully installed!")
    else:
        print("\nSome packages are missing. Please install them manually.")


def create_conda_environment(env_name):
    """Create a new conda environment and install requirements."""
    try:
        # Check if conda is available
        subprocess.check_call(["conda", "--version"], stdout=subprocess.PIPE)
        
        print(f"Creating conda environment '{env_name}'...")
        subprocess.check_call(["conda", "create", "-y", "-n", env_name, "python=3.10"])
        
        print(f"Activating environment '{env_name}'...")
        if os.name == 'nt':  # Windows
            activate_cmd = f"conda activate {env_name} && python -m pip install numpy pandas tqdm pillow requests matplotlib h5py"
            pytorch_cmd = f"conda activate {env_name} && conda install -y pytorch torchvision cpuonly -c pytorch"
            transformers_cmd = f"conda activate {env_name} && pip install transformers"
            huggingface_cmd = f"conda activate {env_name} && pip install huggingface_hub[hf_xet]"
            
            print("Installing base packages...")
            subprocess.check_call(activate_cmd, shell=True)
            
            print("Installing PyTorch...")
            subprocess.check_call(pytorch_cmd, shell=True)
            
            print("Installing transformers...")
            subprocess.check_call(transformers_cmd, shell=True)
            
            print("Installing Hugging Face Hub with Xet Storage support...")
            subprocess.check_call(huggingface_cmd, shell=True)
        else:  # Unix/Linux/Mac
            activate_cmd = f"source activate {env_name} && python -m pip install numpy pandas tqdm pillow requests matplotlib h5py"
            pytorch_cmd = f"source activate {env_name} && conda install -y pytorch torchvision cpuonly -c pytorch"
            transformers_cmd = f"source activate {env_name} && pip install transformers"
            huggingface_cmd = f"source activate {env_name} && pip install huggingface_hub[hf_xet]"
            
            print("Installing base packages...")
            subprocess.check_call(activate_cmd, shell=True)
            
            print("Installing PyTorch...")
            subprocess.check_call(pytorch_cmd, shell=True)
            
            print("Installing transformers...")
            subprocess.check_call(transformers_cmd, shell=True)
            
            print("Installing Hugging Face Hub with Xet Storage support...")
            subprocess.check_call(huggingface_cmd, shell=True)
        
        print(f"\nConda environment '{env_name}' has been created and packages installed!")
        print(f"To activate the environment, run: conda activate {env_name}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("Failed to create conda environment. Please create it manually.")
    except FileNotFoundError:
        print("Error: Conda is not installed or not in the PATH.")
        print("Please install conda or use pip installation instead.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Install dependencies for FEIDEGGER Mamba dataset preparation'
    )
    parser.add_argument('--conda', action='store_true',
                        help='Create a conda environment and install packages there')
    parser.add_argument('--env-name', type=str, default='feidegger-mamba',
                        help='Name of the conda environment to create')
    
    args = parser.parse_args()
    
    print("FEIDEGGER Mamba Dataset Preparation - Dependencies Installer")
    print("=" * 60)
    
    print("\nNote: This installer will include Xet Storage support for better performance")
    print("Xet Storage enables faster and more efficient downloads from Hugging Face Hub")
    
    check_python_version()
    
    if args.conda:
        create_conda_environment(args.env_name)
    else:
        install_requirements()
        verify_installation()
    
    print("\nSetup complete! You can now run the dataset preparation script.")


if __name__ == "__main__":
    main()