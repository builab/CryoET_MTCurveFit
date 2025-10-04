#!/usr/bin/env python3
"""
CLI wrapper for the full filament processing pipeline:
1. Initial Curve Fitting (Clustering)
2. Cleaning (Overlap Removal)
3. Connection (Trajectory Extrapolation)
@Builab 2025
"""

import sys
import os
import argparse
import pandas as pd

# Add parent directory to path to import utils modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.fit import fit_curves
from utils.connect import (
    find_line_connections,
    merge_connected_lines,
    refit_and_resample_tubes
)
from utils.clean import (
    calculate_all_overlaps,
    identify_tubes_to_delete,
    remove_overlapping_tubes
)
from utils.io import read_star, write_star, load_coordinates


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Full pipeline: Fit -> Clean -> Connect helical tubes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  1. Fit    : Clusters particles into tubes and resamples
  2. Clean  : Removes overlapping shorter tubes
  3. Connect: Merges broken tube segments via extrapolation

Example:
  %(prog)s input.star --angpix 14 --sample_step 82 --min_seed 6 \\
           --clean_dist_thres 50 --dist_extrapolate 1500 --overlap_thres 80
        """
    )
    
    parser.add_argument('input_star', help='Input STAR file (coordinates in pixels)')
    
    # Core parameters
    core_group = parser.add_argument_group('Core Parameters')
    core_group.add_argument('--angpix', type=float, default=14.00,
                        help='Pixel size (Angstroms/pixel) (default: 14.00)')
    core_group.add_argument('--poly_order', type=int, default=3,
                        help='Polynomial order for fitting (default: 3)')
    core_group.add_argument('--min_seed', type=int, default=6,
                        help='Minimum seed points (default: 6)')
    core_group.add_argument('--sample_step', type=float, default=82.0,
                        help='Resampling step (Angstroms) (default: 82.0)')
    
    # Cleaning parameters
    clean_group = parser.add_argument_group('Cleaning Parameters')
    clean_group.add_argument('--clean_dist_thres', type=float, default=50,
                        help='Overlap removal threshold (Angstroms) (default: 50)')
    clean_group.add_argument('--clean_margin', type=float, default=500,
                        help='Bounding box margin (Angstroms) (default: 500)')
    
    # Connection parameters
    connect_group = parser.add_argument_group('Connection Parameters')
    connect_group.add_argument('--dist_extrapolate', type=float, required=True,
                        help='Initial extrapolation distance (Angstroms) (REQUIRED)')
    connect_group.add_argument('--overlap_thres', type=float, required=True,
                        help='Connection overlap threshold (Angstroms) (REQUIRED)')
    connect_group.add_argument('--conn_iter', type=int, default=2,
                        help='Connection iterations (default: 2)')
    connect_group.add_argument('--dist_iter_scale', type=float, default=1.5,
                        help='Distance scale factor per iteration (default: 1.5)')
    
    # Advanced fit parameters (with defaults)
    parser.add_argument('--max_dis_to_line_ang', type=float, default=50, help=argparse.SUPPRESS)
    parser.add_argument('--min_dis_neighbor_seed_ang', type=float, default=60, help=argparse.SUPPRESS)
    parser.add_argument('--max_dis_neighbor_seed_ang', type=float, default=320, help=argparse.SUPPRESS)
    parser.add_argument('--poly_order_seed', type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument('--max_seed_fitting_error', type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument('--max_angle_change_per_4nm', type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument('--max_dis_to_curve_ang', type=float, default=80, help=argparse.SUPPRESS)
    parser.add_argument('--min_dis_neighbor_curve_ang', type=float, default=60, help=argparse.SUPPRESS)
    parser.add_argument('--max_dis_neighbor_curve_ang', type=float, default=320, help=argparse.SUPPRESS)
    parser.add_argument('--min_number_growth', type=int, default=0, help=argparse.SUPPRESS)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()


def print_step_header(step_num: int, step_name: str) -> None:
    """Print formatted step header."""
    print("\n" + "="*80)
    print(f"STEP {step_num}: {step_name}")
    print("="*80)


def print_summary(title: str, items: list) -> None:
    """Print formatted summary section."""
    print(f"\n{title}")
    print("-" * len(title))
    for item in items:
        print(f"  {item}")


def run_fitting(file_path: str, args: argparse.Namespace) -> pd.DataFrame:
    """STEP 1: Run initial curve fitting and clustering."""
    print_step_header(1, "CURVE FITTING & CLUSTERING")
    
    pixel_size = args.angpix
    
    # Load coordinates
    coords, tomo_name, detector_pixel_size = load_coordinates(file_path, pixel_size)
    if coords is None:
        raise ValueError("Failed to load coordinates from input file")
    
    print(f"Loaded {len(coords)} particles from {tomo_name}")
    
    # Run fitting
    df_resam, _, cluster_count = fit_curves(
        coords=coords,
        tomo_name=tomo_name,
        angpix=pixel_size,
        poly_order=args.poly_order,
        sample_step=args.sample_step / pixel_size,
        integration_step=1.0 / pixel_size,
        min_seed=args.min_seed,
        max_distance_to_line=args.max_dis_to_line_ang / pixel_size,
        min_distance_in_extension_seed=args.min_dis_neighbor_seed_ang / pixel_size,
        max_distance_in_extension_seed=args.max_dis_neighbor_seed_ang / pixel_size,
        poly_order_seed=args.poly_order_seed,
        seed_evaluation_constant=args.max_seed_fitting_error,
        max_angle_change_per_4nm=args.max_angle_change_per_4nm,
        max_distance_to_curve=args.max_dis_to_curve_ang / pixel_size,
        min_distance_in_extension=args.min_dis_neighbor_curve_ang / pixel_size,
        max_distance_in_extension=args.max_dis_neighbor_curve_ang / pixel_size,
        min_number_growth=args.min_number_growth,
        detector_pixel_size=detector_pixel_size
    )
    
    print_summary("Fitting Results", [
        f"Tubes found: {cluster_count}",
        f"Particles resampled: {len(df_resam)}"
    ])
    
    return df_resam


def run_cleaning(df_input: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """STEP 2: Remove overlapping tubes."""
    print_step_header(2, "CLEANING (OVERLAP REMOVAL)")
    
    n_tubes_before = df_input['rlnHelicalTubeID'].nunique()
    n_particles_before = len(df_input)
    
    print(f"Distance threshold: {args.clean_dist_thres} Å")
    print(f"Bounding box margin: {args.clean_margin} Å")
    
    # Calculate overlaps
    overlap_results = calculate_all_overlaps(
        df=df_input,
        margin=args.clean_margin,
        angpix=args.angpix
    )
    
    if len(overlap_results) == 0:
        print("\nNo overlapping tubes found - skipping cleaning")
        return df_input
    
    # Remove overlapping tubes
    tubes_to_delete = identify_tubes_to_delete(
        overlap_results=overlap_results,
        distance_threshold=args.clean_dist_thres
    )
    
    if len(tubes_to_delete) == 0:
        print(f"\nNo tubes meet removal criteria (distance < {args.clean_dist_thres} Å)")
        return df_input
    
    df_filtered = remove_overlapping_tubes(df=df_input, tubes_to_delete=tubes_to_delete)
    
    print_summary("Cleaning Results", [
        f"Tubes removed: {len(tubes_to_delete)}",
        f"Particles removed: {n_particles_before - len(df_filtered)}",
        f"Remaining tubes: {df_filtered['rlnHelicalTubeID'].nunique()}",
        f"Remaining particles: {len(df_filtered)}"
    ])
    
    return df_filtered


def run_connection(df_input: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """STEP 3: Connect broken tube segments."""
    print_step_header(3, "CONNECTION (TRAJECTORY EXTRAPOLATION)")
    
    n_tubes_before = df_input['rlnHelicalTubeID'].nunique()
    df_current = df_input.copy()
    current_dist_extrapolate = args.dist_extrapolate
    total_merges = 0
    
    print(f"Initial extrapolation: {args.dist_extrapolate} Å")
    print(f"Overlap threshold: {args.overlap_thres} Å")
    print(f"Max iterations: {args.conn_iter}")
    
    # Iterative connection
    for i in range(1, args.conn_iter + 1):
        print(f"\n  Iteration {i} (distance: {current_dist_extrapolate:.1f} Å)")
        
        n_tubes_iter_start = df_current['rlnHelicalTubeID'].nunique()
        
        connections = find_line_connections(
            df=df_current,
            angpix=args.angpix,
            overlap_thres=args.overlap_thres,
            min_seed=args.min_seed,
            dist_extrapolate=current_dist_extrapolate,
            poly_order_seed=args.poly_order_seed
        )
        
        if not connections:
            print(f"    No connections found")
            break
        
        df_merged, _ = merge_connected_lines(df_current, connections)
        n_tubes_iter_end = df_merged['rlnHelicalTubeID'].nunique()
        merges_this_iter = n_tubes_iter_start - n_tubes_iter_end
        total_merges += merges_this_iter
        
        print(f"    Merged: {merges_this_iter} tube groups")
        print(f"    Remaining: {n_tubes_iter_end} tubes")
        
        if n_tubes_iter_end == n_tubes_iter_start:
            print(f"    Converged - stopping")
            df_current = df_merged
            break
        
        df_current = df_merged
        
        if i < args.conn_iter:
            current_dist_extrapolate *= args.dist_iter_scale
    
    # Final refit and resample
    if total_merges > 0:
        print(f"\n  Refitting and resampling merged tubes...")
        df_final = refit_and_resample_tubes(
            df_input=df_current,
            poly_order=args.poly_order,
            sample_step=args.sample_step,
            angpix=args.angpix
        )
    else:
        print(f"\n  No merges performed - skipping refit")
        df_final = df_current
    
    print_summary("Connection Results", [
        f"Tube groups merged: {total_merges}",
        f"Final tubes: {df_final['rlnHelicalTubeID'].nunique()}",
        f"Final particles: {len(df_final)}"
    ])
    
    return df_final


def main() -> None:
    """Run the full pipeline."""
    args = get_args()
    
    base = os.path.splitext(args.input_star)[0]
    output_star = f"{base}_processed.star"
    
    # Print header
    print("="*80)
    print("FILAMENT PROCESSING PIPELINE")
    print("="*80)
    print(f"Input: {args.input_star}")
    print(f"Pixel size: {args.angpix} Å/px")
    print(f"Sample step: {args.sample_step} Å")
    print(f"Polynomial order: {args.poly_order}")
    
    try:
        # Run pipeline
        df_fitted = run_fitting(args.input_star, args)
        df_cleaned = run_cleaning(df_fitted, args)
        df_final = run_connection(df_cleaned, args)
        
        # Save output
        if not df_final.empty:
            write_star(df_final, output_star, overwrite=True)
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETE")
            print("="*80)
            print_summary("Final Output", [
                f"File: {output_star}",
                f"Tubes: {df_final['rlnHelicalTubeID'].nunique()}",
                f"Particles: {len(df_final)}"
            ])
        else:
            print("\nWarning: Pipeline produced no output particles")
            
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
