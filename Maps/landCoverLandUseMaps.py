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
ee.Initialize(project="sgtmodel")

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
        # Re-initialize EE for each process with project parameter
        ee.Initialize(project="sgtmodel")

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

def create_landcover_visualization(coordinates, landcover_values, bavaria_boundary, save_path, point_area_km2=1.0):
    """
    Create and save land cover map visualization of Bavaria with RESEARCH PAPER QUALITY.

    Parameters:
    coordinates (numpy.array): Array of coordinates (latitude, longitude)
    landcover_values (numpy.array): Array of land cover class values
    bavaria_boundary (GeoDataFrame): Bavaria boundary geometry
    save_path (str): Directory path where the images should be saved
    point_area_km2 (float): Area represented by each point in km² (default: 1.0 km²)
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

    # Check if we have any valid data
    if len(valid_values) == 0:
        print("❌ No valid land cover data retrieved! Check Earth Engine authentication and permissions.")
        return

    # Get Bavaria bounds for proper aspect ratio
    bounds = bavaria_boundary.total_bounds
    min_lon, min_lat, max_lon, max_lat = bounds

    # Calculate aspect ratio - Bavaria is wider than it is tall
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    aspect_ratio = lon_range / lat_range

    print(f"Bavaria bounds: Lon {min_lon:.2f}-{max_lon:.2f}, Lat {min_lat:.2f}-{max_lat:.2f}")
    print(f"Aspect ratio (lon/lat): {aspect_ratio:.2f}")

    # Create the plot with RESEARCH PAPER QUALITY - wider to accommodate legend
    fig = plt.figure(figsize=(24, 14), dpi=300)  # Much wider for external legend
    fig.patch.set_facecolor('white')
    
    # Create axes with space for external legend on the right
    ax = plt.axes([0.08, 0.10, 0.65, 0.80])  # [left, bottom, width, height] - reduced width for legend space

    # Plot Bavaria boundary with thick, dark border
    bavaria_boundary.boundary.plot(ax=ax, linewidth=4, edgecolor='black', alpha=1.0)

    # Create scatter plot for each land cover class with VISIBLE POINTS
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
                # Calculate area in km²
                area_km2 = len(class_coords) * point_area_km2
                
                # VISIBLE POINTS with clear borders
                ax.scatter(class_coords[:, 1], class_coords[:, 0], 
                          c=class_dict[class_val]['color'], 
                          s=3.0,
                          alpha=0.9,
                          edgecolors='white',
                          linewidths=0.1,
                          label=f"{class_dict[class_val]['name']} - ~{area_km2:,.0f} km²")
        else:
            # Handle unknown classes
            mask = valid_values == class_val
            class_coords = valid_coords[mask]
            if len(class_coords) > 0:
                area_km2 = len(class_coords) * point_area_km2
                
                ax.scatter(class_coords[:, 1], class_coords[:, 0], 
                          c='darkgray', 
                          s=3.0,
                          alpha=0.9,
                          edgecolors='black',
                          linewidths=0.1,
                          label=f"Unknown ({int(class_val)}) - ~{area_km2:,.0f} km²")

    # Set PROPER limits to show Bavaria nicely
    ax.set_xlim(min_lon - 0.02, max_lon + 0.02)
    ax.set_ylim(min_lat - 0.02, max_lat + 0.02)

    # RESEARCH PAPER QUALITY text - MUCH LARGER FONTS
    ax.set_xlabel('Longitude (°)', fontsize=24, fontweight='bold', labelpad=12)
    ax.set_ylabel('Latitude (°)', fontsize=24, fontweight='bold', labelpad=12)
    ax.set_title('Land Cover Classification - Bavaria (ESA WorldCover 2020)', 
                fontsize=28, fontweight='bold', pad=25)

    # LARGER tick labels for research paper quality
    ax.tick_params(axis='both', which='major', labelsize=20, width=1.5, length=8)

    # Add EXTERNAL LEGEND on the right side - RESEARCH PAPER QUALITY
    legend = ax.legend(
        bbox_to_anchor=(1.05, 1.0),  # Position outside plot area on the right
        loc='upper left',
        fontsize=16,  # Large, readable font
        markerscale=3.0,  # Make legend markers much bigger
        frameon=True,
        fancybox=True,
        shadow=True,
        borderpad=1.2,  # More padding inside legend box
        labelspacing=1.0,  # More space between legend entries
        handletextpad=1.0,  # More space between marker and text
        title_fontsize=18
    )
    legend.set_title('Land Cover Classes\n(Estimated Area)', 
                    prop={'size': 18, 'weight': 'bold'})
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.98)
    legend.get_frame().set_linewidth(2.0)  # Thicker frame for legend

    # Add subtle grid with RESEARCH PAPER appropriate style
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=1.0, color='gray')

    # Save the plot with HIGH QUALITY for research papers
    output_file = os.path.join(save_path, f'bavaria_landcover_PAPER_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    print(f"Saved RESEARCH PAPER quality land cover map: {output_file}")

    # Also save as PDF for publication quality
    pdf_file = os.path.join(save_path, f'bavaria_landcover_PAPER_{timestamp}.pdf')
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white', pad_inches=0.2)
    print(f"Saved PUBLICATION quality land cover map (PDF): {pdf_file}")

    # Also save as EPS for LaTeX documents
    eps_file = os.path.join(save_path, f'bavaria_landcover_PAPER_{timestamp}.eps')
    plt.savefig(eps_file, format='eps', bbox_inches='tight', facecolor='white', pad_inches=0.2)
    print(f"Saved EPS for LaTeX: {eps_file}")

    plt.close()

    # Create ENHANCED class statistics
    create_enhanced_class_statistics(valid_values, point_area_km2, save_path, timestamp, bavaria_boundary)

def create_enhanced_class_statistics(landcover_values, point_area_km2, save_path, timestamp, bavaria_boundary):
    """Create and save ENHANCED statistics about land cover classes with RESEARCH PAPER QUALITY."""

    # Count occurrences of each class
    unique, counts = np.unique(landcover_values, return_counts=True)

    # Calculate percentages and areas
    total_points = len(landcover_values)
    percentages = (counts / total_points) * 100
    
    # Calculate areas based on point_area_km2
    areas_km2 = counts * point_area_km2

    # Estimate Bavaria's area for reference
    bavaria_area_km2 = bavaria_boundary.to_crs('EPSG:3857').area.iloc[0] / 1e6  # Convert to km²

    # Create ENHANCED statistics DataFrame
    stats_data = []
    for class_val, count, percentage, area_km2 in zip(unique, counts, percentages, areas_km2):
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
            'Area_km2': round(area_km2, 1),
            'Percentage': round(percentage, 2)
        })

    stats_df = pd.DataFrame(stats_data)
    
    # Only sort if we have data
    if len(stats_df) > 0:
        stats_df = stats_df.sort_values('Area_km2', ascending=False)

    # Save ENHANCED statistics
    stats_file = os.path.join(save_path, f'bavaria_landcover_PAPER_stats_{timestamp}.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"Saved RESEARCH PAPER statistics: {stats_file}")

    # Print statistics
    print("\n" + "="*80)
    print("RESEARCH PAPER QUALITY LAND COVER STATISTICS FOR BAVARIA")
    print("="*80)
    print(f"Total sample points: {total_points:,}")
    print(f"Point area: {point_area_km2} km² per point")
    print(f"Total mapped area: ~{total_points * point_area_km2:,.0f} km²")
    print(f"Bavaria total area: ~{bavaria_area_km2:,.0f} km²")
    print("-"*80)
    for _, row in stats_df.head(10).iterrows():  # Show top 10
        print(f"{row['Class_Name']:25} | {row['Point_Count']:7,} pts | ~{row['Area_km2']:10,.0f} km² ({row['Percentage']:5.1f}%)")

    # Only create charts if we have data
    if len(stats_df) == 0:
        print("⚠️  No data to create charts")
        return

    # Create RESEARCH PAPER QUALITY charts
    fig = plt.figure(figsize=(24, 12), dpi=300)
    fig.patch.set_facecolor('white')
    
    # Create two subplots with proper spacing
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    # Enhanced pie chart with RESEARCH PAPER fonts
    colors = [row['Color'] for _, row in stats_df.iterrows()]
    labels = [f"{row['Class_Name']}" for _, row in stats_df.iterrows()]
    sizes = [row['Area_km2'] for _, row in stats_df.iterrows()]

    wedges, texts, autotexts = ax1.pie(sizes, labels=None, colors=colors, autopct='%1.1f%%',
                                      startangle=90, 
                                      textprops={'fontsize': 16, 'fontweight': 'bold'},
                                      pctdistance=0.85)

    # Make percentage text more visible
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(16)

    ax1.set_title(f'Land Cover Distribution - Bavaria\n~{total_points * point_area_km2:,.0f} km² Mapped', 
                 fontsize=24, fontweight='bold', pad=25)

    # Add legend for pie chart - outside on the right
    ax1.legend(wedges, labels, loc='center left', bbox_to_anchor=(1.0, 0.5),
              fontsize=16, frameon=True, shadow=True, 
              title='Land Cover Classes', title_fontsize=18)

    # Enhanced bar chart with RESEARCH PAPER fonts
    y_pos = np.arange(len(stats_df))
    bars = ax2.barh(y_pos, stats_df['Area_km2'], color=colors, alpha=0.85, 
                   edgecolor='black', linewidth=1.5)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{row['Class_Name']}" for _, row in stats_df.iterrows()], 
                        fontsize=16, fontweight='medium')
    ax2.set_xlabel('Area (km²)', fontsize=22, fontweight='bold', labelpad=12)
    ax2.set_title('Detailed Breakdown by Area', fontsize=24, fontweight='bold', pad=20)

    # Add area and percentage labels on bars - LARGER FONT
    for i, (bar, area, pct) in enumerate(zip(bars, stats_df['Area_km2'], stats_df['Percentage'])):
        ax2.text(bar.get_width() + max(stats_df['Area_km2']) * 0.01, 
                bar.get_y() + bar.get_height()/2, 
                f'{area:,.0f} km²\n({pct:.1f}%)', 
                ha='left', va='center', 
                fontweight='bold', fontsize=14)

    ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1.0)
    ax2.set_xlim(0, max(stats_df['Area_km2']) * 1.25)  # More space for labels
    ax2.tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout()

    # Save RESEARCH PAPER QUALITY charts
    chart_file = os.path.join(save_path, f'bavaria_landcover_PAPER_charts_{timestamp}.png')
    plt.savefig(chart_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved RESEARCH PAPER quality charts: {chart_file}")

    # Save as PDF for publication
    pdf_chart_file = os.path.join(save_path, f'bavaria_landcover_PAPER_charts_{timestamp}.pdf')
    plt.savefig(pdf_chart_file, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved PUBLICATION quality charts (PDF): {pdf_chart_file}")

    plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract and visualize land cover for Bavaria - RESEARCH PAPER QUALITY')
    parser.add_argument('--start', type=int, default=0, 
                        help='Start index (default: 0)')
    parser.add_argument('--end', type=int, default=None, 
                        help='End index (default: all coordinates)')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Batch size for Earth Engine requests (default: 500)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: number of CPU cores)')
    parser.add_argument('--output-dir', type=str, default='./bavaria_landcover_PAPER',
                        help='Output directory (default: ./bavaria_landcover_PAPER)')
    parser.add_argument('--point-area', type=float, default=1.0,
                        help='Area represented by each point in km² (default: 1.0)')
    parser.add_argument('--coordinates-path', type=str, 
                        default="/home/valerian/SGTPublication/Weights-ResidualsModels-MappingInference-SOCmapping/BaselinesXGBoostAndRF/RandomForestFinalResults2023/coordinates_1mil_rf.npy",
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
    print(f"📏 Point area: {args.point_area} km² per point")

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
        
        # Check if ALL batches failed
        valid_count = sum(1 for val in all_landcover_values if val is not None)
        if valid_count == 0:
            print("\n❌ ALL BATCHES FAILED! This is likely an Earth Engine authentication issue.")
            print("💡 Please check your Earth Engine project permissions.")
            print(f"   Project ID: sgtmodel")
            return

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

    # Create RESEARCH PAPER QUALITY visualizations
    print("🎨 Creating RESEARCH PAPER QUALITY visualizations...")
    create_landcover_visualization(coordinates, all_landcover_values, bavaria_boundary, 
                                   args.output_dir, args.point_area)

    print(f"\n✅ Processing complete!")
    print(f"📁 Results saved in: {args.output_dir}")
    print("📄 RESEARCH PAPER QUALITY figures generated (PNG, PDF, and EPS formats)!")

if __name__ == "__main__":
    main()