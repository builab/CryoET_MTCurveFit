#!/usr/bin/env python3
"""
Visualize Euler angles (Rot, Tilt, Psi) for helical tubes from RELION star file.

This script creates a multi-panel visualization showing how the three Euler angles
vary along each helical tube. Each tube is colored consistently across all subplots.

The --fit_line option now:
1. Fits a polynomial and plots the **fitted line** (not residuals).
2. Implements **robust outlier detection** (MAD Z-score > 3.5), highlighting outliers in red.
3. Calculates the **individual RMSE** for each tube.

The output logic is:
- Plot: If --output is provided, saves to file. Otherwise, attempts to display interactively (plt.show()).
- RMSE: If --out_rmse is provided, saves to CSV. Otherwise, prints the table to the console.

Usage:
    python visualize_angles.py input.star --fit_line --output plot.png --out_rmse rmse.csv

@Builab 2025
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import starfile
from pathlib import Path
import pandas as pd


def read_star(file_path: str):
    """
    Read STAR file into DataFrame.
    """
    df = starfile.read(file_path)
    if 'rlnCoordinateX' not in df and 'particles' in df:
        df = df['particles']
    return df


def fit_polynomial(x, y, order=2):
    """
    Fit polynomial to data and calculate residuals.
    """
    coeffs = np.polyfit(x, y, order)
    poly = np.poly1d(coeffs)
    fitted_y = poly(x)
    residuals = y - fitted_y
    rmse = np.sqrt(np.mean(residuals**2))
    return fitted_y, residuals, rmse


def robust_mad_outlier_detection(residuals, threshold=3.5):
    """
    Detects outliers using the Modified Z-score based on the Median Absolute Deviation (MAD).
    """
    if len(residuals) < 5: 
        return np.zeros_like(residuals, dtype=bool)

    median_residual = np.median(residuals)
    mad = np.median(np.abs(residuals - median_residual))
    
    if mad == 0:
        return np.abs(residuals - median_residual) > 1e-6 

    # 0.6745 = 1/1.4826 for consistency with Gaussian Z-score
    modified_z_score = 0.6745 * (residuals - median_residual) / mad
    
    is_outlier = np.abs(modified_z_score) > threshold
    return is_outlier


def plot_tube_angles(star_path, output_path=None, fit_line=False, rmse_output_path=None):
    """
    Plot Euler angles for all helical tubes in a star file.
    """
    # Read star file
    print(f"Reading star file: {star_path}")
    df = read_star(star_path)
    
    required_cols = ['rlnHelicalTubeID', 'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Total particles: {len(df)}")
    print(f"Total tubes: {df['rlnHelicalTubeID'].nunique()}")
    
    grouped = df.groupby('rlnHelicalTubeID')
    n_tubes = len(grouped)
    
    # Create color map
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_tubes, 20)))
    if n_tubes > 20:
        colors = plt.cm.hsv(np.linspace(0, 1, n_tubes))
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    if fit_line:
        fig.suptitle(f'Euler Angle Polynomial Fit & Outlier Detection (Order 2, Z-score > 3.5)\n{Path(star_path).name}', 
                     fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f'Euler Angles for Helical Tubes\n{Path(star_path).name}', 
                     fontsize=14, fontweight='bold')
    
    angle_names = ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
    angle_labels = ['Rot (°)', 'Tilt (°)', 'Psi (°)']
    
    tube_metrics = []
    outlier_legend_added = False
    
    # Plot each tube
    for idx, (tube_id, tube_data) in enumerate(grouped):
        color = colors[idx % len(colors)]
        particle_indices = np.arange(len(tube_data))
        tube_metric = {'rlnHelicalTubeID': tube_id}
        
        for ax_idx, (angle_col, angle_label) in enumerate(zip(angle_names, angle_labels)):
            angles = tube_data[angle_col].values
            
            if fit_line:
                # Fit polynomial (Order 2)
                fitted_angles, residuals, rmse = fit_polynomial(particle_indices, angles, order=2)
                tube_metric[f'RMSE_{angle_col}'] = rmse
                
                # Robust Outlier Detection
                is_outlier = robust_mad_outlier_detection(residuals)
                
                inlier_indices = particle_indices[~is_outlier]
                inlier_angles = angles[~is_outlier]
                outlier_indices = particle_indices[is_outlier]
                outlier_angles = angles[is_outlier]
                
                # Plot fitted line
                axes[ax_idx].plot(particle_indices, fitted_angles, 
                                color=color, alpha=0.9, linewidth=2.0, linestyle='-',
                                label=f'Tube {tube_id}' if ax_idx == 0 and n_tubes <= 10 else None)
                
                # Plot inliers (original angles)
                axes[ax_idx].scatter(inlier_indices, inlier_angles, 
                                   color=color, alpha=0.5, s=10)
                                   
                # Plot outliers (original angles) in RED
                outlier_label = 'Outlier (Red)' if ax_idx == 0 and not outlier_legend_added else None
                axes[ax_idx].scatter(outlier_indices, outlier_angles, 
                                   color='red', marker='o', alpha=1.0, s=20, zorder=5,
                                   label=outlier_label)
                if outlier_label:
                    outlier_legend_added = True
            
            else:
                # Plot raw angles
                axes[ax_idx].plot(particle_indices, angles, 
                                color=color, alpha=0.7, linewidth=1.5,
                                label=f'Tube {tube_id}' if ax_idx == 0 else None)
                axes[ax_idx].scatter(particle_indices, angles, 
                                   color=color, alpha=0.5, s=10)
        
        if fit_line:
            tube_metrics.append(tube_metric)

    
    # Format subplots
    for ax_idx, (angle_col, angle_label) in enumerate(zip(angle_names, angle_labels)):
        axes[ax_idx].set_xlabel('Particle Index (within tube)', fontsize=11)
        axes[ax_idx].set_ylabel(angle_label, fontsize=11)
        
        axes[ax_idx].grid(True, alpha=0.3)
        
        # Set y-axis limits 
        if angle_col == 'rlnAngleTilt':
            axes[ax_idx].set_ylim(-10, 190)
        else:
            axes[ax_idx].set_ylim(-190, 190)
        
        # Add a horizontal line at y=0
        y_min, y_max = axes[ax_idx].get_ylim()
        if y_min < 0 < y_max:
            axes[ax_idx].axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)

    
    if n_tubes <= 10 or fit_line:
        axes[0].legend(loc='upper right', fontsize=8, ncol=2)
    
    plt.tight_layout()
    
    # --- PLOT SAVING LOGIC: Save if output_path is provided, otherwise show (interactive) ---
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        # Reverting to interactive mode for local execution as requested
        plt.show() 
    
    plt.close()
    
    # --- RMSE REPORTING LOGIC: Save to CSV if path provided, otherwise print to console ---
    if fit_line and tube_metrics:
        metrics_df = pd.DataFrame(tube_metrics)
        metrics_df.columns = ['TubeID', 'RMSE_Rot_deg', 'RMSE_Tilt_deg', 'RMSE_Psi_deg']

        if rmse_output_path:
            # Save to CSV
            metrics_df.to_csv(rmse_output_path, index=False, float_format='%.2f')
            print(f"\nIndividual Tube RMSE saved to CSV: {rmse_output_path}")
        else:
            # Print to console
            print("\n--- Individual Tube RMSE (Tube Quality Check) ---")
            metrics_df.columns = ['TubeID', 'RMSE_Rot (°)', 'RMSE_Tilt (°)', 'RMSE_Psi (°)']
            
            # Format to 2 decimal places for printing
            for col in [c for c in metrics_df.columns if c.startswith('RMSE_')]:
                metrics_df[col] = metrics_df[col].map('{:.2f}'.format)
            print(metrics_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Euler angles for helical tubes from RELION star file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display fit/outliers plot interactively AND print RMSE to console
  python visualize_star_angles.py particles.star --fit_line
  
  # Save fit/outliers plot to file AND save RMSE to CSV
  python visualize_star_angles.py particles.star --fit_line --output fit_plot.png --out_rmse tube_quality.csv
        """
    )
    
    parser.add_argument('input', type=str,
                       help='Input star file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output plot file (PNG, PDF, SVG). If not specified, displays interactively (plt.show()).')
    parser.add_argument('--fit_line', action='store_true',
                       help='Fit polynomial (order 2), plot the fitted line, highlight outliers (Robust Z-score > 3.5) in red, and calculate individual tube RMSE.')
    parser.add_argument('--out_rmse', type=str, default=None,
                       help='Output file to save individual tube RMSE values (CSV format). If not specified, prints to console.')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        parser.error(f"Input file not found: {args.input}")
    
    try:
        plot_tube_angles(args.input, args.output, args.fit_line, args.out_rmse)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())