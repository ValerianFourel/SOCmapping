#!/bin/bash
# system_report.sh - Generate system information for README.md

echo "# System Environment Report"
echo ""
echo "Generated on: $(date)"
echo ""

echo "## Operating System"
echo "\`\`\`"
lsb_release -a 2>/dev/null || cat /etc/os-release
echo "\`\`\`"
echo ""

echo "## Hardware Information"
echo "### CPU"
echo "\`\`\`"
lscpu | grep -E "Model name|Architecture|CPU\(s\)|Thread|Core"
echo "\`\`\`"
echo ""

echo "### Memory"
echo "\`\`\`"
free -h
echo "\`\`\`"
echo ""

echo "## GPU Information"
echo "### NVIDIA Driver & CUDA"
echo "\`\`\`"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | head -1
echo "Driver Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)"
echo "CUDA Version: $(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9.]*\).*/\1/')"
echo "\`\`\`"
echo ""

echo "### GPU Configuration"
echo "\`\`\`"
echo "GPU Count: $(nvidia-smi --list-gpus | wc -l)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "\`\`\`"
echo ""

echo "### NVCC Version"
echo "\`\`\`"
nvcc --version 2>/dev/null || echo "NVCC not found in PATH"
echo "\`\`\`"
echo ""

echo "## Python Environment"
echo "### Python Version"
echo "\`\`\`"
python --version
echo "\`\`\`"
echo ""

echo "### Conda Environment"
echo "\`\`\`"
echo "Active environment: $CONDA_DEFAULT_ENV"
conda info --envs | grep '*'
echo "\`\`\`"
echo ""

echo "### Key Python Packages"
echo "\`\`\`"
pip list | grep -E "(torch|tensorflow|cuda|numpy|pandas)" 2>/dev/null || echo "Key packages not found"
echo "\`\`\`"
echo ""

echo "## Node Information"
echo "\`\`\`"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo "Working Directory: $(pwd)"
echo "\`\`\`"
