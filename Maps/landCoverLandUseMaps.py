import ee
import numpy as np
import os
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geopandas as gpd
from datetime import datetime
import matplotlib.colors as mcolors

# Initialize the Earth Engine API
ee.Initialize()

# Dictionary of class values and their descriptions with BOLD, CONTRASTING colors
class_dict = {
    10: {'name': 'Tree cover', 'color': '#003d00'},  # Very dark green
    20: {'name': 'Shrubland', 'color': '#228B22'},  # Forest green
    30: {'name': 'Grassland', 'color': '#32CD32'},  # Lime green
    40: {'name': 'Cropland', 'color': '#FFD700'},   # Gold
    50: {'name': 'Built-up', 'color': '#FF1493'},   # Deep pink
    60: {'name': 'Bare / sparse vegetation', 'color': '#DEB887'},  # Burlywood
    70: {'name': 'Snow and ice', 'color': '#E0FFFF'},  # Light cyan
    80: {'name': 'Permanent water bodies', 'color': '#0000CD'},  # Medium blue
    90: {'name': 'Herbaceous wetland', 'color': '#00FFFF'},  # Cyan
    95: {'name': 'Mangroves', 'color': '#008080'},  # Teal
    100: {'name': 'Moss and lichen', 'color': '#ADFF2F'}  # Green yellow
}

def load_bavaria_boundary():
    """Load Bavaria boundary from GeoJSON."""
    bavaria = gpd.read_file('https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/4_niedrig.geo.json')
    bavaria = bavaria[bavaria['name'] == 'Bayern']
    return bavaria

def filter_coordinates_to_bavaria(coordinates, bavaria_boundary):
    """Filter coordinates to only include those within Bavaria."""
    from shapely.geometry import Point

    # Create points from coordinates
    points = [Point(lon, lat) for lat, lon in coordinates]

    # Check which points are within Bavaria
    bavaria_geom = bavaria_boundary.geometry.iloc[0]
    within_bavaria = [bavaria_geom.contains(point) for point in points]

    # Filter coordinates
    filtered_coords = coordinates[within_bavaria]

    print(f"Original coordinates: {len(coordinates)}")
    print(f"Coordinates within Bavaria: {len(filtered_coords)}")

    return filtered_coords

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

def create_landcover_visualization(coordinates, landcover_values, bavaria_boundary, save_path):
    """
    Create and save land cover map visualization of Bavaria with HIGHLY VISIBLE POINTS.

    Parameters:
    coordinates (numpy.array): Array of coordinates (latitude, longitude)
    landcover_values (numpy.array): Array of land cover class values
    bavaria_boundary (GeoDataFrame): Bavaria boundary geometry
    save_path (str): Directory path where the images should be saved
    """

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directories if they don't exist
    os.makedirs(save_path, exist_ok=True)

    # Filter out None values
    valid_mask = np.array(landcover_values) != None
    valid_coords = coordinates[valid_mask]
    valid_values = np.array(landcover_values)[valid_mask]

    print(f"Valid land cover samples: {len(valid_values)}")

    # Get Bavaria bounds for proper aspect ratio
    bounds = bavaria_boundary.total_bounds
    min_lon, min_lat, max_lon, max_lat = bounds

    # Calculate aspect ratio - Bavaria is wider than it is tall
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    aspect_ratio = lon_range / lat_range

    print(f"Bavaria bounds: Lon {min_lon:.2f}-{max_lon:.2f}, Lat {min_lat:.2f}-{max_lat:.2f}")
    print(f"Aspect ratio (lon/lat): {aspect_ratio:.2f}")

    # Create the plot with PROPER RECTANGULAR SHAPE for Bavaria
    fig, ax = plt.subplots(figsize=(18, 12), dpi=300)  # Made wider for Bavaria's shape
    fig.patch.set_facecolor('white')

    # Plot Bavaria boundary with thick, dark border
    bavaria_boundary.boundary.plot(ax=ax, linewidth=4, edgecolor='black', alpha=1.0)

    # Create scatter plot for each land cover class with MUCH LARGER, VISIBLE POINTS
    unique_classes = np.unique(valid_values)
    print(f"Found land cover classes: {unique_classes}")

    # Sort classes by count to plot larger classes first (better visibility)
    class_counts = {}
    for class_val in unique_classes:
        mask = valid_values == class_val
        class_counts[class_val] = np.sum(mask)

    # Sort by count (largest first) so smaller classes appear on top
    sorted_classes = sorted(unique_classes, key=lambda x: class_counts[x], reverse=True)

    for class_val in sorted_classes:
        if class_val in class_dict:
            mask = valid_values == class_val
            class_coords = valid_coords[mask]

            if len(class_coords) > 0:
                # MUCH LARGER, MORE VISIBLE POINTS
                ax.scatter(class_coords[:, 1], class_coords[:, 0], 
                          c=class_dict[class_val]['color'], 
                          s=3.0,  # INCREASED from 0.5 to 3.0
                          alpha=0.9,  # INCREASED opacity
                          edgecolors='white',  # WHITE BORDER for contrast
                          linewidths=0.1,  # Thin white border
                          label=f"{class_dict[class_val]['name']} ({int(class_val)}) - {len(class_coords):,} pts")
        else:
            # Handle unknown classes
            mask = valid_values == class_val
            class_coords = valid_coords[mask]
            if len(class_coords) > 0:
                ax.scatter(class_coords[:, 1], class_coords[:, 0], 
                          c='darkgray', 
                          s=3.0,  # INCREASED size
                          alpha=0.9,  # INCREASED opacity
                          edgecolors='black',
                          linewidths=0.1,
                          label=f"Unknown ({int(class_val)}) - {len(class_coords):,} pts")

    # Set PROPER limits to show Bavaria nicely
    ax.set_xlim(min_lon - 0.02, max_lon + 0.02)
    ax.set_ylim(min_lat - 0.02, max_lat + 0.02)

    # Set plot properties with LARGER FONTS
    ax.set_xlabel('Longitude', fontsize=16, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=16, fontweight='bold')
    ax.set_title('Land Cover Classification - Bavaria (ESA WorldCover 2020)\n', 
                fontsize=20, fontweight='bold', pad=20)

    # Add BETTER legend with smaller font to fit more classes
    legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, 
                      markerscale=2.0,  # Make legend markers bigger
                      frameon=True, fancybox=True, shadow=True)
    legend.set_title('Land Cover Classes (with point counts)', prop={'size': 12, 'weight': 'bold'})
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)

    # DO NOT force equal aspect - let Bavaria show its natural elongated shape
    # ax.set_aspect('equal')  # REMOVED THIS LINE

    # Add subtle grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # Make tick labels larger
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Save the plot with HIGH QUALITY
    output_file = os.path.join(save_path, f'bavaria_landcover_VISIBLE_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved HIGHLY VISIBLE land cover map: {output_file}")

    # Also save as PDF for highest quality
    pdf_file = os.path.join(save_path, f'bavaria_landcover_VISIBLE_{timestamp}.pdf')
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved VISIBLE land cover map (PDF): {pdf_file}")

    plt.show()

    # Create ENHANCED class statistics
    create_enhanced_class_statistics(valid_values, save_path, timestamp, bavaria_boundary)

def create_enhanced_class_statistics(landcover_values, save_path, timestamp, bavaria_boundary):
    """Create and save ENHANCED statistics about land cover classes."""

    # Count occurrences of each class
    unique, counts = np.unique(landcover_values, return_counts=True)

    # Calculate percentages
    total_points = len(landcover_values)
    percentages = (counts / total_points) * 100

    # Estimate Bavaria's area for density calculations
    bavaria_area_km2 = bavaria_boundary.to_crs('EPSG:3857').area.iloc[0] / 1e6  # Convert to km²

    # Create ENHANCED statistics DataFrame
    stats_data = []
    for class_val, count, percentage in zip(unique, counts, percentages):
        if class_val in class_dict:
            class_name = class_dict[class_val]['name']
            color = class_dict[class_val]['color']
        else:
            class_name = f"Unknown Class {int(class_val)}"
            color = 'gray'

        stats_data.append({
            'Class_Value': int(class_val),
            'Class_Name': class_name,
            'Color': color,
            'Point_Count': count,
            'Percentage': round(percentage, 2),
            'Density_per_km2': round(count / bavaria_area_km2, 1),
            'Est_Coverage_km2': round((percentage/100) * bavaria_area_km2, 1)
        })

    stats_df = pd.DataFrame(stats_data)
    stats_df = stats_df.sort_values('Point_Count', ascending=False)

    # Save ENHANCED statistics
    stats_file = os.path.join(save_path, f'bavaria_landcover_ENHANCED_stats_{timestamp}.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"Saved ENHANCED statistics: {stats_file}")

    # Print statistics
    print("\n" + "="*80)
    print("ENHANCED LAND COVER STATISTICS FOR BAVARIA")
    print("="*80)
    print(f"Total sample points: {total_points:,}")
    print(f"Bavaria area: ~{bavaria_area_km2:,.0f} km²")
    print("-"*80)
    for _, row in stats_df.head(10).iterrows():  # Show top 10
        print(f"{row['Class_Name']:25} | {row['Point_Count']:7,} pts ({row['Percentage']:5.1f}%) | ~{row['Est_Coverage_km2']:6,.0f} km²")

    # Create BETTER pie chart with enhanced visuals
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=300)
    fig.patch.set_facecolor('white')

    # Enhanced pie chart
    colors = [row['Color'] for _, row in stats_df.iterrows()]
    labels = [f"{row['Class_Name']}\n{row['Point_Count']:,} pts" for _, row in stats_df.iterrows()]
    sizes = [row['Point_Count'] for _, row in stats_df.iterrows()]

    wedges, texts, autotexts = ax1.pie(sizes, labels=None, colors=colors, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})

    ax1.set_title(f'Land Cover Distribution - Bavaria\n{total_points:,} Sample Points', 
                 fontsize=16, fontweight='bold', pad=20)

    # Enhanced bar chart
    y_pos = np.arange(len(stats_df))
    bars = ax2.barh(y_pos, stats_df['Percentage'], color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=0.5)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{row['Class_Name']}\n({row['Point_Count']:,} pts)" 
                        for _, row in stats_df.iterrows()], fontsize=10)
    ax2.set_xlabel('Percentage of Total Points (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Detailed Breakdown by Point Count', fontsize=16, fontweight='bold')

    # Add percentage labels on bars
    for i, (bar, pct, count) in enumerate(zip(bars, stats_df['Percentage'], stats_df['Point_Count'])):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%\n({count:,})', ha='left', va='center', 
                fontweight='bold', fontsize=9)

    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_xlim(0, max(stats_df['Percentage']) * 1.2)

    plt.tight_layout()

    # Save ENHANCED charts
    chart_file = os.path.join(save_path, f'bavaria_landcover_ENHANCED_charts_{timestamp}.png')
    plt.savefig(chart_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved ENHANCED charts: {chart_file}")

    plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract and visualize land cover for Bavaria with HIGHLY VISIBLE POINTS')
    parser.add_argument('--start', type=int, default=0, 
                        help='Start index (default: 0)')
    parser.add_argument('--end', type=int, default=None, 
                        help='End index (default: all coordinates)')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Batch size for Earth Engine requests (default: 500)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: number of CPU cores)')
    parser.add_argument('--output-dir', type=str, default='./bavaria_landcover_VISIBLE',
                        help='Output directory (default: ./bavaria_landcover_VISIBLE)')
    parser.add_argument('--coordinates-path', type=str, 
                        default="/home/vfourel/SOCProject/SOCmapping/BaselinesXGBoostAndRF/RandomForestFinalResults2023/coordinates_1mil_rf.npy",
                        help='Path to coordinates file')

    args = parser.parse_args()

    # Load Bavaria boundary
    print("🗺️  Loading Bavaria boundary...")
    bavaria_boundary = load_bavaria_boundary()

    # Load coordinates
    print("📍 Loading coordinates...")
    all_coordinates = np.load(args.coordinates_path)
    all_coordinates = [(y, x) for (x, y) in all_coordinates]  # Convert to (lat, lon)
    all_coordinates = np.array(all_coordinates)

    # Filter coordinates to Bavaria
    print("🎯 Filtering coordinates to Bavaria...")
    bavaria_coordinates = filter_coordinates_to_bavaria(all_coordinates, bavaria_boundary)

    if len(bavaria_coordinates) == 0:
        print("❌ No coordinates found within Bavaria boundary!")
        return

    # Apply start and end indices
    end_idx = args.end if args.end is not None else len(bavaria_coordinates)
    coordinates = bavaria_coordinates[args.start:end_idx]

    print(f"🔄 Processing coordinates from index {args.start} to {end_idx}")
    print(f"📊 Total coordinates to process: {len(coordinates):,}")
    print(f"📦 Batch size: {args.batch_size}")

    # Determine number of workers
    num_workers = args.workers if args.workers else cpu_count()
    print(f"⚡ Using {num_workers} worker processes")

    # Prepare batches
    batches = []
    for i in range(0, len(coordinates), args.batch_size):
        batch = coordinates[i:i + args.batch_size]
        batches.append((i // args.batch_size, batch, args.batch_size))

    print(f"📋 Total batches: {len(batches)}")

    # Process batches in parallel
    results_dict = {}
    errors = []

    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(batches), desc="🛰️  Extracting land cover data") as pbar:
            for batch_idx, values, error in pool.imap_unordered(process_batch, batches):
                results_dict[batch_idx] = values
                if error:
                    errors.append(error)
                pbar.update(1)

    # Combine results in correct order
    all_landcover_values = []
    for i in range(len(batches)):
        if i in results_dict:
            all_landcover_values.extend(results_dict[i])
        else:
            # Handle missing batches
            batch_size = len(batches[i][1])
            all_landcover_values.extend([None] * batch_size)

    # Print any errors
    if errors:
        print(f"\n⚠️  Encountered {len(errors)} errors:")
        for error in errors[:3]:  # Show first 3 errors
            print(f"  {error}")
        if len(errors) > 3:
            print(f"  ... and {len(errors) - 3} more errors")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save coordinates and landcover values
    coords_file = os.path.join(args.output_dir, f'bavaria_coordinates_{timestamp}.npy')
    landcover_file = os.path.join(args.output_dir, f'bavaria_landcover_values_{timestamp}.npy')

    np.save(coords_file, coordinates)
    np.save(landcover_file, np.array(all_landcover_values))

    print(f"💾 Saved coordinates: {coords_file}")
    print(f"💾 Saved landcover values: {landcover_file}")

    # Create HIGHLY VISIBLE visualizations
    print("🎨 Creating HIGHLY VISIBLE visualizations...")
    create_landcover_visualization(coordinates, all_landcover_values, bavaria_boundary, args.output_dir)

    print(f"\n✅ Processing complete!")
    print(f"📁 Results saved in: {args.output_dir}")
    print("🔍 Points are now MUCH MORE VISIBLE!")

if __name__ == "__main__":
    main()
