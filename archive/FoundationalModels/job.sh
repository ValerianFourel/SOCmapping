#!/bin/bash


conda init
echo "home"
conda activate spectralgpt


# Execute the Python module with specified parameters
echo "Running LLaVA module..."
export PYTORCH_KERNEL_CACHE_PATH=/home/vfourel/.cache/torch


python embeddings.py
#/home/vfourel/LLaVA/llava/serve/cliAffectnetParallel.py --outputpath /home/vfourel/FaceGPT/Data/LLaVAAnnotations/Conversations/ --model-path liuhaotian/llava-v1.6-34b --folder-section $1 --folder-section-end $1 --load-4bit
# python -m llava.serve.cliAffectnetParallel.py --outputpath ../FaceGPT/Data/LLaVAAnnotations/Conversations/ --model-path liuhaotian/llava-v1.6-34b --folder-section 1197 --folder-section-end 1249 --load-4bit
# Deactivate the environment, if using conda
if type conda >/dev/null 2>&1; then
  conda deactivate
fi

echo "Script execution completed."

