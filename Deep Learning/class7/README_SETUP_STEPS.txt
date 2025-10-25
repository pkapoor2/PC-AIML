This repository contains helper files for setting up TensorFlow + NumPy.

macOS Apple Silicon (Conda):
  conda env create -f environment_macos_arm64.yml
  conda activate tf-mac-metal

Windows 11 (Conda CPU):
  conda env create -f environment_windows_cpu.yml
  conda activate tf-win

Windows 11 WSL (pip/venv):
  python3 -m venv ~/tf-wsl
  source ~/tf-wsl/bin/activate
  python -m pip install --upgrade pip setuptools wheel
  pip install -r requirements_wsl.txt
