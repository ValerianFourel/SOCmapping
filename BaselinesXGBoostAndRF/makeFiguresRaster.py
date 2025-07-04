import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Data extracted from your results
window_sizes = [5, 11, 21, 31]  # pixel dimensions
spatial_sizes = [1.25, 2.75, 5.25, 7.75]  # km dimensions
window_labels = [f"{w}×{w}\n({s:.2f}km)" for w, s in zip(window_sizes, spatial_sizes)]

# XGBoost metrics
xgb_train_r2 = [0.8922, 0.8914, 0.8910, 0.8914]
xgb_train_r2_std = [0.0012, 0.0011, 0.0014, 0.0013]
xgb_val_r2 = [0.2774, 0.2584, 0.2811, 0.2640]
xgb_val_r2_std = [0.0412, 0.0267, 0.0251, 0.0481]
xgb_val_mae = [7.9392, 8.0976, 7.5397, 7.8728]
xgb_val_mae_std = [0.2897, 0.5327, 0.3092, 0.3600]

# Random Forest metrics
rf_train_r2 = [0.7878, 0.7948, 0.8086, 0.8083]
rf_train_r2_std = [0.0021, 0.0033, 0.0027, 0.0022]
rf_val_r2 = [0.2715, 0.2579, 0.3015, 0.2647]
rf_val_r2_std = [0.0514, 0.0297, 0.0272, 0.0453]
rf_val_mae = [7.1997, 7.2134, 6.6739, 7.0780]
rf_val_mae_std = [0.2135, 0.2395, 0.1845, 0.2922]

# Create the main figure
fig = plt.figure(figsize=(20, 14))

# ============ MAIN PERFORMANCE PLOTS ============
# Plot 1: R² Score Comparison
ax1 = plt.subplot(2, 3, 1)
x_pos = np.arange(len(window_sizes))

ax1.errorbar(x_pos, xgb_val_r2, yerr=xgb_val_r2_std, 
             fmt='o-', linewidth=3, markersize=8, capsize=5,
             label='XGBoost', color='#2E86C1', alpha=0.8)
ax1.errorbar(x_pos, rf_val_r2, yerr=rf_val_r2_std, 
             fmt='s-', linewidth=3, markersize=8, capsize=5,
             label='Random Forest', color='#E74C3C', alpha=0.8)

ax1.set_xticks(x_pos)
ax1.set_xticklabels(window_labels, fontsize=10)
ax1.set_ylabel('Validation R² Score', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance vs Window Size', fontsize=14, fontweight='bold')
ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
ax1.grid(True, alpha=0.3)

# Highlight best performance
best_xgb_idx = np.argmax(xgb_val_r2)
best_rf_idx = np.argmax(rf_val_r2)
ax1.scatter(best_xgb_idx, xgb_val_r2[best_xgb_idx], s=150, color='gold', 
           edgecolor='#2E86C1', linewidth=2, zorder=5)
ax1.scatter(best_rf_idx, rf_val_r2[best_rf_idx], s=150, color='gold', 
           edgecolor='#E74C3C', linewidth=2, zorder=5)

# Plot 2: MAE Comparison
ax2 = plt.subplot(2, 3, 2)
ax2.errorbar(x_pos, xgb_val_mae, yerr=xgb_val_mae_std, 
             fmt='o-', linewidth=3, markersize=8, capsize=5,
             label='XGBoost', color='#2E86C1', alpha=0.8)
ax2.errorbar(x_pos, rf_val_mae, yerr=rf_val_mae_std, 
             fmt='s-', linewidth=3, markersize=8, capsize=5,
             label='Random Forest', color='#E74C3C', alpha=0.8)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(window_labels, fontsize=10)
ax2.set_ylabel('Validation MAE', fontsize=12, fontweight='bold')
ax2.set_title('Mean Absolute Error vs Window Size', fontsize=14, fontweight='bold')
ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
ax2.grid(True, alpha=0.3)

# Plot 3: Model Stability
ax3 = plt.subplot(2, 3, 3)
xgb_cv = np.array(xgb_val_r2_std) / np.array(xgb_val_r2) * 100
rf_cv = np.array(rf_val_r2_std) / np.array(rf_val_r2) * 100

bars1 = ax3.bar(x_pos - 0.2, xgb_cv, 0.35, label='XGBoost', 
                color='#2E86C1', alpha=0.7, edgecolor='black')
bars2 = ax3.bar(x_pos + 0.2, rf_cv, 0.35, label='Random Forest', 
                color='#E74C3C', alpha=0.7, edgecolor='black')

ax3.set_xticks(x_pos)
ax3.set_xticklabels(window_labels, fontsize=10)
ax3.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
ax3.set_title('Model Stability Across Runs', fontsize=14, fontweight='bold')
ax3.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
ax3.grid(True, alpha=0.3)

# Plot 4: Spatial Coverage
ax4 = plt.subplot(2, 3, 4)
spatial_coverage = [s**2 for s in spatial_sizes]  # km²
colors = plt.cm.viridis(np.linspace(0, 1, len(window_sizes)))

bars = ax4.bar(x_pos, spatial_coverage, color=colors, alpha=0.8, 
               edgecolor='black', linewidth=2)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(window_labels, fontsize=10)
ax4.set_ylabel('Spatial Coverage (km²)', fontsize=12, fontweight='bold')
ax4.set_title('Area Covered by Each Window', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Add value labels on bars
for bar, coverage in zip(bars, spatial_coverage):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{coverage:.1f}', ha='center', va='bottom', 
             fontsize=11, fontweight='bold')

# Plot 5: Performance Summary
ax5 = plt.subplot(2, 3, 5)
models = ['XGBoost', 'Random Forest']
best_r2_values = [max(xgb_val_r2), max(rf_val_r2)]
best_windows_text = [f"{window_sizes[best_xgb_idx]}×{window_sizes[best_xgb_idx]}", 
                     f"{window_sizes[best_rf_idx]}×{window_sizes[best_rf_idx]}"]

bars = ax5.bar(models, best_r2_values, 
               color=['#2E86C1', '#E74C3C'], alpha=0.7, 
               edgecolor='black', linewidth=2)
ax5.set_ylabel('Best Validation R²', fontsize=12, fontweight='bold')
ax5.set_title('Best Performance Comparison', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Add annotations
for i, (bar, window, r2) in enumerate(zip(bars, best_windows_text, best_r2_values)):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{window}\n{r2:.3f}', ha='center', va='bottom', 
             fontsize=11, fontweight='bold')

# ============ SPECTACULAR 3D WINDOW HIERARCHY VISUALIZATION ============
ax6 = plt.subplot(2, 3, 6, projection='3d')

# Define dramatic color scheme
colors_3d = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # Red to green gradient
edge_colors = ['#C0392B', '#16A085', '#2980B9', '#27AE60']

# Define positions for a diamond/square arrangement
positions = [
    (0, 1.5, 0),     # 5x5 - top (smallest, highest detail)
    (-1, 0, 0),      # 11x11 - middle left
    (1, 0, 0),       # 21x21 - middle right  
    (0, -1.5, 0)     # 31x31 - bottom (largest, broadest coverage)
]

# Heights representing detail level (inverse relationship with size)
heights = [1.2, 0.8, 0.5, 0.3]  # Tallest for smallest window
widths = [0.4, 0.7, 1.0, 1.3]   # Width represents spatial coverage

# Create the 3D windows with dramatic visual hierarchy
for i, (size, spatial, color, edge_color, pos, height, width) in enumerate(
    zip(window_sizes, spatial_sizes, colors_3d, edge_colors, positions, heights, widths)):

    x, y, z = pos

    # Create 3D rectangular prism (window representation)
    ax6.bar3d(x - width/2, y - width/2, z, width, width, height,
              color=color, alpha=0.85, edgecolor=edge_color, linewidth=2)

    # Add detailed labels with performance info
    label_text = f'{size}×{size} pixels\n{spatial:.1f}km coverage\nR²={rf_val_r2[i]:.3f}'
    ax6.text(x, y, height + 0.2, label_text, 
             ha='center', va='bottom', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor=edge_color))

    # Add connecting lines to show hierarchy
    if i < len(positions) - 1:
        next_pos = positions[i + 1]
        ax6.plot([x, next_pos[0]], [y, next_pos[1]], [height/2, heights[i+1]/2], 
                 'k--', alpha=0.5, linewidth=2)

# Add performance indicators as floating elements
performance_positions = [(1.5, 1.5, 0.6), (1.5, 0, 0.4), (1.5, -1.5, 0.2)]
performance_values = [max(rf_val_r2), np.mean(rf_val_r2), min(rf_val_r2)]
performance_labels = ['BEST', 'AVG', 'MIN']

for pos, val, label in zip(performance_positions, performance_values, performance_labels):
    x, y, z = pos
    # Create small performance indicator cubes
    ax6.bar3d(x, y, z, 0.2, 0.2, val, color='gold', alpha=0.8, edgecolor='orange')
    ax6.text(x, y, z + val + 0.1, f'{label}\n{val:.3f}', 
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Customize the 3D plot extensively
ax6.set_xlabel('Spatial Distribution', fontsize=12, fontweight='bold')
ax6.set_ylabel('Processing Hierarchy', fontsize=12, fontweight='bold')
ax6.set_zlabel('Detail Resolution\n(Processing Depth)', fontsize=12, fontweight='bold')
ax6.set_title('🛰️ 3D Window Size Hierarchy\nFrom Finest Detail (Top) to Broadest Coverage (Bottom)', 
              fontsize=16, fontweight='bold', pad=20)

# Set dramatic axis limits for better visual impact
ax6.set_xlim(-2, 2.5)
ax6.set_ylim(-2.5, 2.5)
ax6.set_zlim(0, 1.8)

# Custom tick labels for better understanding
ax6.set_xticks([-1, 0, 1])
ax6.set_xticklabels(['Left\nWindow', 'Center\nLine', 'Right\nWindow'])
ax6.set_yticks([-1.5, 0, 1.5])
ax6.set_yticklabels(['Largest\n(31×31)', 'Medium\n(11×21)', 'Smallest\n(5×5)'])
ax6.set_zticks([0, 0.5, 1.0, 1.5])
ax6.set_zticklabels(['Broad\nCoverage', 'Medium\nDetail', 'High\nDetail', 'Finest\nDetail'])

# Set optimal viewing angle for dramatic effect
ax6.view_init(elev=20, azim=45)

# Add a custom legend with detailed explanations
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor=colors_3d[0], edgecolor=edge_colors[0], 
                 label='5×5: Highest Detail, Smallest Coverage'),
    plt.Rectangle((0, 0), 1, 1, facecolor=colors_3d[1], edgecolor=edge_colors[1], 
                 label='11×11: High Detail, Small Coverage'),
    plt.Rectangle((0, 0), 1, 1, facecolor=colors_3d[2], edgecolor=edge_colors[2], 
                 label='21×21: Medium Detail, Medium Coverage'),
    plt.Rectangle((0, 0), 1, 1, facecolor=colors_3d[3], edgecolor=edge_colors[3], 
                 label='31×31: Lower Detail, Broadest Coverage'),
]

ax6.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.1, 1), 
           fontsize=10, frameon=True, fancybox=True, shadow=True)

# Add grid for better depth perception
ax6.grid(True, alpha=0.2)

# Add main title
fig.suptitle('🛰️ Satellite Data Window Size Analysis\nComplete Performance & Spatial Hierarchy Visualization', 
             fontsize=20, fontweight='bold', y=0.96)

# Adjust layout to give more space to the 3D plot
plt.tight_layout()

# Enhanced summary text with 3D insights
summary_text = (
    f"KEY FINDINGS & 3D INSIGHTS:\n"
    f"📊 Best XGBoost: R² = {max(xgb_val_r2):.3f} at {window_sizes[best_xgb_idx]}×{window_sizes[best_xgb_idx]} pixels\n"
    f"📊 Best Random Forest: R² = {max(rf_val_r2):.3f} at {window_sizes[best_rf_idx]}×{window_sizes[best_rf_idx]} pixels\n"
    f"📈 Optimal window: {window_sizes[best_rf_idx]}×{window_sizes[best_rf_idx]} pixels ({spatial_sizes[best_rf_idx]:.2f}km)\n"
    f"🎯 Sweet spot: Medium-scale windows balance detail and context\n"
    f"🔍 3D Hierarchy: Height ∝ Detail Level, Width ∝ Coverage Area\n"
    f"🏗️ Architecture: Pyramid structure shows detail-coverage tradeoff"
)

fig.text(0.02, 0.02, summary_text, fontsize=12, 
         bbox=dict(boxstyle="round,pad=0.6", facecolor="lightcyan", alpha=0.95, edgecolor="navy"),
         verticalalignment='bottom')

# Save the figure with high quality
plt.savefig('satellite_window_analysis_spectacular_3d.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# Show the plot
plt.show()

# Print enhanced summary table
print("\n" + "="*90)
print("🛰️  SATELLITE WINDOW SIZE PERFORMANCE & SPATIAL HIERARCHY SUMMARY")
print("="*90)

# Create enhanced summary DataFrame
summary_data = []
for i, (ws, ss) in enumerate(zip(window_sizes, spatial_sizes)):
    summary_data.append({
        'Window_Size': f'{ws}×{ws}',
        'Spatial_Coverage_km': f'{ss:.2f}',
        'Area_km2': f'{ss**2:.1f}',
        'Detail_Level': ['Finest', 'High', 'Medium', 'Broad'][i],
        'XGB_Val_R2': f'{xgb_val_r2[i]:.4f}±{xgb_val_r2_std[i]:.4f}',
        'RF_Val_R2': f'{rf_val_r2[i]:.4f}±{rf_val_r2_std[i]:.4f}',
        '3D_Height': f'{heights[i]:.1f}',
        '3D_Width': f'{widths[i]:.1f}'
    })

df = pd.DataFrame(summary_data)
print(df.to_string(index=False))

print(f"\n🏆 OPTIMAL CONFIGURATIONS:")
print(f"   XGBoost: {window_sizes[best_xgb_idx]}×{window_sizes[best_xgb_idx]} pixels "
      f"({spatial_sizes[best_xgb_idx]:.2f}km) → R² = {max(xgb_val_r2):.4f}")
print(f"   Random Forest: {window_sizes[best_rf_idx]}×{window_sizes[best_rf_idx]} pixels "
      f"({spatial_sizes[best_rf_idx]:.2f}km) → R² = {max(rf_val_r2):.4f}")

print(f"\n🏗️ 3D VISUALIZATION ARCHITECTURE:")
print(f"   • Pyramid Structure: Smallest windows at top (highest detail)")
print(f"   • Height Dimension: Represents processing detail intensity")
print(f"   • Width/Depth: Represents spatial coverage area")
print(f"   • Color Gradient: From fine detail (red) to broad coverage (green)")
print(f"   • Performance Indicators: Gold cubes show R² performance levels")
print(f"   • Hierarchical Flow: Connected windows show progression")

print(f"\n📈 SPATIAL-PERFORMANCE INSIGHTS:")
print(f"   • Detail vs Coverage Tradeoff clearly visualized")
print(f"   • Optimal performance occurs at medium scales (21×21)")
print(f"   • Diminishing returns for very large windows")
print(f"   • Small windows provide detail but limited context")
