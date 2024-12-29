import os
import shutil
from pathlib import Path

def reconstruct_split_folders(*part_paths, output_dir="reconstructed"):
    """
    Reconstruct split folders from multiple parts.

    Args:
        *part_paths: Variable number of paths to the unzipped part folders
        output_dir: Directory where the reconstructed structure will be created
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for part_path in part_paths:
        # Convert to Path object
        part_dir = Path(part_path)

        # Get the first directory inside the unzipped folder
        try:
            first_subdir = next(p for p in part_dir.iterdir() if p.is_dir())
        except StopIteration:
            print(f"No subdirectories found in {part_path}")
            continue

        # Walk through all files in the first subdirectory
        for root, _, files in os.walk(first_subdir):
            for file in files:
                # Get the full source path
                src_path = os.path.join(root, file)

                # Calculate relative path from the first subdirectory
                rel_path = os.path.relpath(src_path, first_subdir)

                # Create destination path
                dst_path = os.path.join(output_dir, rel_path)

                # Create destination directory if it doesn't exist
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                # Copy the file to its new location
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {rel_path}")

# Example usage:
# reconstruct_split_folders(
#     "/path/to/part1",
#     "/path/to/part2",
#     "/path/to/part3",
#     output_dir="reconstructed_folder"
# )

# Example with three parts
reconstruct_split_folders(
    "/home/vfourel/SOCmapping/Data/Coordinates-20241211T155342Z-001",
    "/home/vfourel/SOCmapping/Data/Coordinates-20241211T155342Z-002",
    output_dir="/home/vfourel/SOCmapping/Data/Coordinates"
)

