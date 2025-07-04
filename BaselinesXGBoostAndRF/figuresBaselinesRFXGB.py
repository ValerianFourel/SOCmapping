import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Data extracted from your results
n_estimators = [5, 10, 25, 50, 100, 500, 1000, 2000]

# XGBoost metrics
xgb_train_r2 = [0.8351, 0.8574, 0.8797, 0.8865, 0.8865, 0.8869, 0.8869, 0.8869]
xgb_val_r2 = [0.1600, 0.1955, 0.2192, 0.2285, 0.2591, 0.2589, 0.2589, 0.2589]
xgb_train_mae = [8.7026, 6.3575, 3.4068, 2.4768, 2.2710, 2.1388, 2.1371, 2.1371]
xgb_val_mae = [7.9862, 7.5836, 7.6038, 7.8809, 8.3674, 8.4390, 8.4404, 8.4404]

# Random Forest metrics
rf_train_r2 = [0.7724, 0.7935, 0.8027, 0.8048, 0.7971, 0.8013, 0.8011, 0.8011]
rf_val_r2 = [0.1537, 0.1895, 0.2001, 0.1999, 0.2627, 0.2787, 0.2805, 0.2804]
rf_train_mae = [5.4969, 5.3497, 5.2472, 5.2530, 5.3179, 5.2696, 5.2778, 5.2779]
rf_val_mae = [8.1563, 7.9308, 7.5560, 7.5012, 7.2462, 7.1976, 7.1726, 7.1685]

# Create the figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Performance vs Number of Estimators', 
             fontsize=16, fontweight='bold', y=0.98)

# Plot 1: R² Score
ax1.plot(n_estimators, xgb_train_r2, 'o-', linewidth=2, markersize=6, 
         label='XGBoost Train', color='#2E86C1', alpha=0.8)
ax1.plot(n_estimators, xgb_val_r2, 's--', linewidth=2, markersize=6, 
         label='XGBoost Validation', color='#2E86C1', alpha=0.6)
ax1.plot(n_estimators, rf_train_r2, 'o-', linewidth=2, markersize=6, 
         label='Random Forest Train', color='#E74C3C', alpha=0.8)
ax1.plot(n_estimators, rf_val_r2, 's--', linewidth=2, markersize=6, 
         label='Random Forest Validation', color='#E74C3C', alpha=0.6)

ax1.set_xscale('log')
ax1.set_xlabel('Number of Estimators (log scale)', fontsize=11)
ax1.set_ylabel('R² Score', fontsize=11)
ax1.set_title('R² Score: Training vs Validation', fontsize=12, fontweight='bold')
ax1.legend(frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3)

# Plot 2: MAE
ax2.plot(n_estimators, xgb_train_mae, 'o-', linewidth=2, markersize=6, 
         label='XGBoost Train', color='#2E86C1', alpha=0.8)
ax2.plot(n_estimators, xgb_val_mae, 's--', linewidth=2, markersize=6, 
         label='XGBoost Validation', color='#2E86C1', alpha=0.6)
ax2.plot(n_estimators, rf_train_mae, 'o-', linewidth=2, markersize=6, 
         label='Random Forest Train', color='#E74C3C', alpha=0.8)
ax2.plot(n_estimators, rf_val_mae, 's--', linewidth=2, markersize=6, 
         label='Random Forest Validation', color='#E74C3C', alpha=0.6)

ax2.set_xscale('log')
ax2.set_xlabel('Number of Estimators (log scale)', fontsize=11)
ax2.set_ylabel('Mean Absolute Error', fontsize=11)
ax2.set_title('MAE: Training vs Validation', fontsize=12, fontweight='bold')
ax2.legend(frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3)

# Plot 3: Overfitting Analysis (Gap between train and validation R²)
xgb_gap = np.array(xgb_train_r2) - np.array(xgb_val_r2)
rf_gap = np.array(rf_train_r2) - np.array(rf_val_r2)

ax3.plot(n_estimators, xgb_gap, 'o-', linewidth=2, markersize=6, 
         label='XGBoost', color='#2E86C1', alpha=0.8)
ax3.plot(n_estimators, rf_gap, 'o-', linewidth=2, markersize=6, 
         label='Random Forest', color='#E74C3C', alpha=0.8)

ax3.set_xscale('log')
ax3.set_xlabel('Number of Estimators (log scale)', fontsize=11)
ax3.set_ylabel('Training - Validation R²', fontsize=11)
ax3.set_title('Overfitting Gap (Train R² - Validation R²)', fontsize=12, fontweight='bold')
ax3.legend(frameon=True, fancybox=True, shadow=True)
ax3.grid(True, alpha=0.3)

# Plot 4: Performance Saturation Analysis
# Calculate relative improvement for training R²
xgb_improvement = [(val - xgb_train_r2[0])/xgb_train_r2[0] * 100 for val in xgb_train_r2]
rf_improvement = [(val - rf_train_r2[0])/rf_train_r2[0] * 100 for val in rf_train_r2]

ax4.plot(n_estimators, xgb_improvement, 'o-', linewidth=2, markersize=6, 
         label='XGBoost', color='#2E86C1', alpha=0.8)
ax4.plot(n_estimators, rf_improvement, 'o-', linewidth=2, markersize=6, 
         label='Random Forest', color='#E74C3C', alpha=0.8)

# Add saturation threshold line
ax4.axhline(y=max(xgb_improvement) * 0.95, color='gray', linestyle=':', alpha=0.7, 
           label='95% of Max Improvement')

ax4.set_xscale('log')
ax4.set_xlabel('Number of Estimators (log scale)', fontsize=11)
ax4.set_ylabel('Relative Improvement (%)', fontsize=11)
ax4.set_title('Training R² Relative Improvement (Saturation Effect)', fontsize=12, fontweight='bold')
ax4.legend(frameon=True, fancybox=True, shadow=True)
ax4.grid(True, alpha=0.3)

# Adjust layout and add annotations
plt.tight_layout()

# Add text annotations highlighting key insights
fig.text(0.02, 0.02, 
         '• Training performance improves and saturates around 100-500 estimators\n'
         '• Validation performance shows overfitting with increased complexity\n'
         '• XGBoost shows stronger overfitting tendency than Random Forest',
         fontsize=10, style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

# Save the figure
plt.savefig('model_performance_saturation.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# Show the plot
plt.show()

# Create a summary table
print("\n" + "="*80)
print("PERFORMANCE SATURATION ANALYSIS")
print("="*80)

# Find saturation points (where improvement becomes minimal)
for i, estimators in enumerate(n_estimators[1:], 1):
    xgb_improvement_rate = abs(xgb_train_r2[i] - xgb_train_r2[i-1])
    rf_improvement_rate = abs(rf_train_r2[i] - rf_train_r2[i-1])

    if xgb_improvement_rate < 0.001:
        print(f"XGBoost saturates around {n_estimators[i-1]} estimators")
        break

print(f"\nBest validation R² scores:")
print(f"XGBoost: {max(xgb_val_r2):.4f} at {n_estimators[xgb_val_r2.index(max(xgb_val_r2))]} estimators")
print(f"Random Forest: {max(rf_val_r2):.4f} at {n_estimators[rf_val_r2.index(max(rf_val_r2))]} estimators")

