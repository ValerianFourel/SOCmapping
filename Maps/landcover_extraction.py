import ee
import numpy as np
import os
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from tqdm import tqdm

# Initialize the Earth Engine API
ee.Initialize()

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

def process_batch(batch_data):
    """Process a single batch of coordinates."""
    batch_idx, coords_batch, batch_size = batch_data

    try:
        # Re-initialize EE for each process
        ee.Initialize()

        # Load WorldCover image
        worldcover = ee.ImageCollection("ESA/WorldCover/v100") \
            .filterDate('2020-01-01', '2021-01-01') \
            .first() \
            .select('Map')

        # Create feature collection from coordinates
        points = []
        for i, (lat, lon) in enumerate(coords_batch):
            point = ee.Feature(ee.Geometry.Point([lon, lat]), {'index': i})
            points.append(point)

        fc = ee.FeatureCollection(points)

        # Sample the image at all points
        sampled = worldcover.sampleRegions(
            collection=fc,
            scale=10,
            geometries=False
        )

        # Get results
        results = sampled.getInfo()

        # Extract values in order
        values = [None] * len(coords_batch)
        for feature in results['features']:
            idx = feature['properties']['index']
            values[idx] = feature['properties'].get('Map', None)

        return batch_idx, values, None

    except Exception as e:
        error_msg = f"Error in batch {batch_idx}: {str(e)}"
        return batch_idx, [None] * len(coords_batch), error_msg

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract land cover values for coordinates')
    parser.add_argument('--start', type=int, default=0, 
                        help='Start index (default: 0)')
    parser.add_argument('--end', type=int, default=None, 
                        help='End index (default: all coordinates)')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Batch size for Earth Engine requests (default: 500)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: number of CPU cores)')
    parser.add_argument('--output-dir', type=str, default='./landcover_results',
                        help='Output directory (default: ./landcover_results)')

    args = parser.parse_args()

    # Load coordinates
    coordinates_path = "/home/vfourel/SOCProject/SOCmapping/BaselinesXGBoostAndRF/RandomForestFinalResults2023/coordinates_1mil_rf.npy"
    all_coordinates = np.load(coordinates_path)
    all_coordinates = [(y, x) for (x, y) in all_coordinates]
    # Apply start and end indices
    end_idx = args.end if args.end is not None else len(all_coordinates)
    coordinates = all_coordinates[args.start:end_idx]

    print(f"Processing coordinates from index {args.start} to {end_idx}")
    print(f"Total coordinates to process: {len(coordinates)}")
    print(f"Batch size: {args.batch_size}")

    # Determine number of workers
    num_workers = args.workers if args.workers else cpu_count()
    print(f"Using {num_workers} worker processes")

    # Prepare batches
    batches = []
    for i in range(0, len(coordinates), args.batch_size):
        batch = coordinates[i:i + args.batch_size]
        batches.append((i // args.batch_size, batch, args.batch_size))

    print(f"Total batches: {len(batches)}")

    # Process batches in parallel
    results_dict = {}
    errors = []

    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better progress tracking
        with tqdm(total=len(batches), desc="Processing batches") as pbar:
            for batch_idx, values, error in pool.imap_unordered(process_batch, batches):
                results_dict[batch_idx] = values
                if error:
                    errors.append(error)
                pbar.update(1)

    # Combine results in correct order
    landcover_values = []
    for i in range(len(batches)):
        landcover_values.extend(results_dict[i])

    # Convert to numpy array
    landcover_array = np.array(landcover_values)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save results with range in filename
    output_filename = f"landcover_values_{args.start}_{end_idx}.npy"
    output_path = os.path.join(args.output_dir, output_filename)
    np.save(output_path, landcover_array)

    print(f"\nResults saved to: {output_path}")
    print(f"Total points processed: {len(landcover_array)}")
    print(f"Points with data: {np.sum(landcover_array != None)}")

    # Save errors if any
    if errors:
        error_path = os.path.join(args.output_dir, f"errors_{args.start}_{end_idx}.txt")
        with open(error_path, 'w') as f:
            for error in errors:
                f.write(error + '\n')
        print(f"Errors saved to: {error_path}")

    # Save summary
    unique, counts = np.unique(landcover_array[landcover_array != None], return_counts=True)
    summary = {}
    for val, count in zip(unique, counts):
        if val in class_dict:
            summary[class_dict[val]] = count

    summary_path = os.path.join(args.output_dir, f"summary_{args.start}_{end_idx}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Land Cover Class Distribution (indices {args.start}-{end_idx}):\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total points: {len(landcover_array)}\n")
        f.write(f"Points with data: {np.sum(landcover_array != None)}\n")
        f.write(f"Points without data: {np.sum(landcover_array == None)}\n")
        f.write("-" * 50 + "\n")
        for class_name, count in sorted(summary.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(landcover_array)) * 100
            f.write(f"{class_name}: {count:,} points ({percentage:.2f}%)\n")

    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
