# install_enhanced.sh
#!/bin/bash
echo "Setting up Enhanced CCTV Monitor System..."

# Update system packages
sudo apt-get update

# Install system dependencies
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    ffmpeg

# Check for NVIDIA CUDA toolkit
if nvcc --version &> /dev/null; then
    echo "NVIDIA CUDA toolkit detected. You may be able to leverage GPU acceleration."
else
    echo "NVIDIA CUDA toolkit not detected. Proceeding with CPU-based setup. For GPU acceleration, please install the CUDA toolkit appropriate for your NVIDIA driver."
fi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python packages
pip install -r requirements.txt

echo "Setup complete! To run the system:"
echo "1. source venv/bin/activate"
echo "2. python cctv_monitor.py"