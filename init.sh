#!/bin/bash
# Setup script for TruFor AI Manipulation Detection

# Create project structure
mkdir -p ai_manip_trufor/configs
mkdir -p ai_manip_trufor/data
mkdir -p ai_manip_trufor/output
mkdir -p ai_manip_trufor/logs
mkdir -p ai_manip_trufor/scripts

# Download TruFor weights (if needed)
if [ ! -f "weights/trufor.pth.tar" ]; then
  echo "Downloading TruFor weights..."
  mkdir -p weights
  wget -q -c https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip -O weights/TruFor_weights.zip
  unzip -q -n weights/TruFor_weights.zip -d weights/ && rm weights/TruFor_weights.zip
fi

# Copy TruFor source files (adjust paths as needed)
cp -r lib ai_manip_trufor/
cp -r models ai_manip_trufor/
cp -r dataset ai_manip_trufor/

# Setup conda environment from YAML
conda env create -f trufor_conda.yaml
echo "Environment 'trufor' created. Activate with: conda activate trufor"

echo "Project setup complete!"