import numpy as np
import os
import argparse
import glob

def merge_results():
    parser = argparse.ArgumentParser(description='Merge land cover results from multiple runs')
    parser.add_argument('--input-dir', type=str, default='./landcover_results',
                        help='Input directory containing partial results')
    parser.add_argument('--output-file', type=str, default='landcover_values_merged.npy',
                        help='Output filename for merged results')

    args = parser.parse_args()

    # Find all result files
    pattern = os.path.join(args.input_dir, 'landcover_values_*.npy')
    files = sorted(glob.glob(pattern))

    print(f"Found {len(files)} result files to merge")

    # Extract ranges and sort
    file_info = []
    for file in files:
        basename = os.path.basename(file)
        parts = basename.replace('landcover_values_', '').replace('.npy', '').split('_')
        start, end = int(parts[0]), int(parts[1])
        file_info.append((start, end, file))

    file_info.sort(key=lambda x: x[0])

    # Check for gaps or overlaps
    print("\nChecking for gaps or overlaps...")
    expected_start = 0
    for i, (start, end, file) in enumerate(file_info):
        if start != expected_start:
            print(f"WARNING: Gap or overlap detected! Expected start: {expected_start}, actual start: {start}")
        expected_start = end

    # Load and merge
    merged_data = []
    for start, end, file in file_info:
        print(f"Loading {file} (indices {start}-{end})")
        # Use allow_pickle=True to load object arrays containing None values
        data = np.load(file, allow_pickle=True)
        merged_data.extend(data)
        print(f"  Loaded {len(data)} values")

    # Convert to numpy array
    merged_array = np.array(merged_data, dtype=object)

    # Save merged result
    output_path = os.path.join(args.input_dir, args.output_file)
    np.save(output_path, merged_array)

    print(f"\nMerged results saved to: {output_path}")
    print(f"Total points: {len(merged_array)}")

    # Convert None to a numeric value for counting
    non_none_mask = np.array([x is not None for x in merged_array])
    print(f"Points with data: {np.sum(non_none_mask)}")
    print(f"Points without data: {np.sum(~non_none_mask)}")

    # Generate final summary
    valid_values = merged_array[non_none_mask]
    if len(valid_values) > 0:
        # Convert to numeric array for unique counting
        valid_numeric = np.array([float(x) for x in valid_values])
        unique, counts = np.unique(valid_numeric, return_counts=True)

        # Dictionary of class values and their descriptions
        class_dict = {
            10: 'Tree cover',
            20: 'Shrubland',
            30: 'Grassland',
            40: 'Cropland',
            50: 'Built-up',
            60: 'Bare / sparse vegetation',
            70: 'Snow and ice',
            80: 'Permanent water bodies',
            90: 'Herbaceous wetland',
            95: 'Mangroves',
            100: 'Moss and lichen'
        }

        summary_path = os.path.join(args.input_dir, 'summary_merged.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Merged Land Cover Class Distribution:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total points: {len(merged_array):,}\n")
            f.write(f"Points with data: {np.sum(non_none_mask):,}\n")
            f.write(f"Points without data: {np.sum(~non_none_mask):,}\n")
            f.write("-" * 50 + "\n")

            for val, count in zip(unique, counts):
                class_name = class_dict.get(int(val), f'Unknown class {int(val)}')
                percentage = (count / len(merged_array)) * 100
                f.write(f"{class_name}: {count:,} points ({percentage:.2f}%)\n")

        print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    merge_results()
