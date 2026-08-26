#!/bin/bash
set -e

echo "=== GT-Quant Milestone 1 Setup ==="
echo "This script prepares your Ubuntu 22.04 server from zero."

# 1. System updates
echo "[*] Updating system..."
sudo apt-get update && sudo apt-get upgrade -y

# 2. Install base dependencies
echo "[*] Installing base packages..."
sudo apt-get install -y     build-essential     software-properties-common     curl     wget     git     tmux     htop     nvtop     jq     unzip     libssl-dev     zlib1g-dev     libbz2-dev     libreadline-dev     libsqlite3-dev     llvm     libncursesw5-dev     xz-utils     tk-dev     libxml2-dev     libxmlsec1-dev     libffi-dev     liblzma-dev

# 3. Install Python 3.11 if not present
if ! command -v python3.11 &> /dev/null; then
    echo "[*] Installing Python 3.11..."
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils
fi

# 4. Install pip for 3.11
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# 5. Create virtual environment
echo "[*] Creating Python venv..."
python3.11 -m venv venv
source venv/bin/activate

# 6. Upgrade pip and install requirements
echo "[*] Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 7. Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "[*] Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "[!] Docker installed. You may need to log out and back in for group changes."
fi

# 8. Install Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "[*] Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# 9. Verify CUDA (should already be present on your RTX machine)
echo "[*] Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "[!] nvidia-smi not found. Install CUDA toolkit if you plan to use GPU training."
fi

# 10. Create data directories
mkdir -p data/raw data/processed

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. source venv/bin/activate"
echo "  2. docker-compose up -d                          (TimescaleDB + Redis + Grafana)"
echo "  3. cd ft_userdata && docker-compose up -d        (Freqtrade + FreqAI)"
echo "  4. See docs/freqai-setup-notes.md for the smoke-test recipe"
