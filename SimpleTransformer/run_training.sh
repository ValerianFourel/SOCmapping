#!/bin/bash

# Define arrays for the parameters to loop through
target_transforms=("none" "log" "normalize")
loss_types=("mse" "composite_l2")

# Loop through each combination of target_transform and loss_type
for transform in "${target_transforms[@]}"; do
    for loss in "${loss_types[@]}"; do
        echo "Running train.py with target_transform=$transform and loss_type=$loss"
        accelerate launch --multi_gpu train.py \
            --loss_type "$loss" \
            --loss_alpha 0.5 \
            --target_transform "$transform"
        
        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "Completed: target_transform=$transform, loss_type=$loss"
        else
            echo "Error: Failed for target_transform=$transform, loss_type=$loss"
        fi
    done
done

echo "All training runs completed."
