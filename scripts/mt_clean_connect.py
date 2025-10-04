#!/usr/bin/env python3
"""
CLI wrapper for cleaning (removing overlaps) and connecting (extrapolating 
trajectories) broken helical tubes sequentially.
@Builab 2025
"""

import sys
import os
import argparse
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

# Add parent directory to path to import utils modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import necessary functions from utils
try:
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
    from utils.io import read_star, write_star
except ImportError as e:
    print(f"Error importing utility modules: {e}")
    print("Ensure that 'utils/connect.py', 'utils/clean.py', and 'utils/io.py' are accessible.")
    sys.exit(1)


def get_args() -> argparse.Namespace:
    """Parse command-line arguments, combining clean and connect parameters."""
    parser = argparse.ArgumentParser(
        description='Sequentially clean (remove overlaps) and connect broken helical tubes (STAR file utility)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Process Order:
1. **Clean:** Tubes with significant overlap (avg_distance < --dist_thres) are removed.
2. **Connect:** The remaining tubes are iteratively connected over increasing distances.

Cleaning Examples:
  --clean_dist_thres 20 --clean_margin 50
  
Connection Examples:
  --dist_extrapolate 100.0 --overlap_thres 10.0 --min_seed 5 --conn_iter 3 --dist_iter_scale 2.0 --poly_order_seed 2 --poly_order 3 --sample_step 20.0
        """
    )
    
    parser.add_argument('input_star', help='Input STAR file (coordinates in pixels)')
    parser.add_argument('--angpix', type=float, required=True,
                        help='Pixel size (Angstroms/pixel)')
    
    # --- CLEANING PARAMETERS ---
    cleaning_group = parser.add_argument_group('Cleaning Parameters (Overlap Removal)')
    cleaning_group.add_argument('--clean_dist_thres', type=float, default=50,
                                help='Distance threshold (Angstroms) for overlap removal (default: 50)')
    cleaning_group.add_argument('--clean_margin', type=float, default=500,
                                help='Margin (Angstroms) for bounding box overlap check (default: 500)')
    cleaning_group.add_argument('--clean_csv', action='store_true',
                                help='Save overlap results to CSV file')
    
    # --- CONNECTION PARAMETERS ---
    connection_group = parser.add_argument_group('Connection Parameters (Trajectory Extrapolation)')
    connection_group.add_argument('--poly_order_seed', type=int, default=2, choices=[1, 2],
                                  help='Polynomial degree for fitting segment ends (default: 2)')
    connection_group.add_argument('--min_seed', type=int, default=5,
                                  help='Minimum points from line end for polynomial fit (default: 5)')
    connection_group.add_argument('--dist_extrapolate', type=float, required=True,
                                  help='Initial distance (Angstroms) to project trajectory forward')
    connection_group.add_argument('--overlap_thres', type=float, required=True,
                                  help='Maximum distance (Angstroms) for overlap detection')
    connection_group.add_argument('--conn_iter', type=int, default=2,
                                  help='Maximum number of iterative connection runs (default: 2)')
    connection_group.add_argument('--dist_iter_scale', type=float, default=1.5,
                                  help='Scale factor for dist_extrapolate in subsequent iterations (default: 1.5)')
    connection_group.add_argument('--poly_order', type=int, default=3,
                                  help='Polynomial degree for final curve fitting (default: 3)')
    connection_group.add_argument('--sample_step', type=float, default=82.0,
                                  help='Distance (Angstroms) between resampled particles (default: 82.0)')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()


def run_cleaning(df_input: pd.DataFrame, args: argparse.Namespace, base: str) -> pd.DataFrame:
    """Run the overlap calculation and filtering step (mt_clean logic)."""
    
    print("\n" + "="*80)
    print("STEP 1: REMOVING OVERLAPPING LINES (CLEANING)")
    print("="*80)
    print(f"Distance threshold: {args.clean_dist_thres} Angstroms")
    
    original_n_tubes = df_input['rlnHelicalTubeID'].nunique()
    
    # Calculate overlaps
    overlap_results = calculate_all_overlaps(
        df=df_input,
        margin=args.clean_margin,
        angpix=args.angpix
    )
    
    if len(overlap_results) == 0:
        print("\nNo overlapping tube pairs found based on the bounding box margin.")
        return df_input
    
    # Identify and remove tubes to delete
    tubes_to_delete = identify_tubes_to_delete(
        overlap_results=overlap_results,
        distance_threshold=args.clean_dist_thres
    )
    
    if len(tubes_to_delete) == 0:
        print(f"\nNo tubes found to delete (avg distance < {args.clean_dist_thres} Angstroms).")
        return df_input
        
    df_filtered = remove_overlapping_tubes(
        df=df_input,
        tubes_to_delete=tubes_to_delete
    )
    
    print(f"\n{'='*30} CLEANING SUMMARY {'='*30}")
    print(f"Original: {len(df_input)} particles from {original_n_tubes} tubes")
    print(f"Filtered: {len(df_filtered)} particles from {df_filtered['rlnHelicalTubeID'].nunique()} tubes")
    print(f"Removed: {len(df_input) - len(df_filtered)} particles from {len(tubes_to_delete)} tubes")
    print('='*80)
    
    if args.clean_csv:
        csv_filename = f"{base}_overlap_results.csv"
        overlap_results.to_csv(csv_filename, index=False)
        print(f"Saved overlap results to '{csv_filename}'")
        
    return df_filtered


def run_connection(df_input: pd.DataFrame, args: argparse.Namespace, base: str) -> pd.DataFrame:
    """Run the iterative connection and final refitting step (mt_connect logic)."""
    
    print("\n" + "="*80)
    print("STEP 2: CONNECTING BROKEN LINES (EXTRAPOLATION)")
    print("="*80)
    
    df_current = df_input.copy()
    original_n_tubes = df_current['rlnHelicalTubeID'].nunique()
    current_dist_extrapolate = args.dist_extrapolate
    all_connections = []
    
    print(f"Tubes starting connection: {original_n_tubes}")
    print(f"Max iterations: {args.conn_iter}")
    
    # Run iterative connection process
    for i in range(1, args.conn_iter + 1):
        print(f"\n--- Running Iteration {i} ---")
        print(f"Extrapolation Distance: {current_dist_extrapolate:.2f} Angstroms")
        
        n_tubes_before = df_current['rlnHelicalTubeID'].nunique()
        
        connections = find_line_connections(
            df=df_current,
            angpix=args.angpix,
            overlap_thres=args.overlap_thres,
            min_seed=args.min_seed,  # Use conn_min_seed from combined args
            dist_extrapolate=current_dist_extrapolate,
            poly_order_seed=args.poly_order_seed
        )
        
        if not connections:
            print(f"No connections found in Iteration {i}. Stopping iterations.")
            break
        
        df_merged, _ = merge_connected_lines(df_current, connections)
        n_tubes_after = df_merged['rlnHelicalTubeID'].nunique()
        
        print(f"  Merged {n_tubes_before - n_tubes_after} pairs/groups.")
        print(f"  Tubes remaining: {n_tubes_after}")
        
        # Add iteration info to connection records
        for conn in connections:
            conn['iteration'] = i
            conn['dist_extrapolate_used'] = current_dist_extrapolate
        all_connections.extend(connections)
        
        if n_tubes_after == n_tubes_before:
            print(f"Convergence reached (no new connections in Iteration {i}). Stopping.")
            df_current = df_merged
            break
        
        df_current = df_merged
        
        # Scale distance for next iteration
        if i < args.conn_iter:
            current_dist_extrapolate *= args.dist_iter_scale

    final_n_tubes_merged = df_current['rlnHelicalTubeID'].nunique()
    
    # Final output
    print(f"\n{'='*30} CONNECTION SUMMARY {'='*30}")
    if final_n_tubes_merged < original_n_tubes:
        # Post-merging refit and resample
        df_final = refit_and_resample_tubes(
            df_input=df_current,
            poly_order=args.poly_order,
            sample_step=args.sample_step,
            angpix=args.angpix
        )
        
        csv_output_file = f"{base}_connections.csv"
        
        print(f"Tubes before connection: {original_n_tubes}")
        print(f"Tubes after merge: {final_n_tubes_merged}")
        print(f"Total tubes merged: {original_n_tubes - final_n_tubes_merged}")
        print(f"Final particle count after resampling: {len(df_final)}")
        
        # Save connection details to CSV
        if all_connections:
            connections_df = pd.DataFrame(all_connections)
            csv_data = connections_df[['iteration', 'tube_id1', 'tube_id2', 
                                      'simple_end_distance', 'min_overlap_dist', 
                                      'connection_type', 'dist_extrapolate_used']]
            csv_data = csv_data.sort_values(by=['iteration', 'min_overlap_dist'])
            csv_data.to_csv(csv_output_file, index=False, float_format='%.3f')
            print(f"Saved connection details to '{csv_output_file}'")
            
        return df_final
        
    else:
        print("NO CONNECTIONS FOUND AFTER ALL ITERATIONS")
        print('='*80)
        return df_current


def main() -> None:
    """Main function to run script from command line."""
    args = get_args()
    
    # Setup
    base = os.path.splitext(args.input_star)[0]
    output_star = f"{base}_cleaned_connected.star"
    
    print("="*80)
    print(f"Processing STAR file: {args.input_star}")
    print(f"Pixel Size (angpix): {args.angpix} Angstroms/pixel")
    print("="*80)
    
    try:
        df_initial = read_star(args.input_star)
        print(f"Loaded {len(df_initial)} particles from {df_initial['rlnHelicalTubeID'].nunique()} tubes")
    except Exception as e:
        print(f"Error reading star file: {e}")
        sys.exit(1)
    
    # --- 1. CLEANING STEP ---
    df_cleaned = run_cleaning(df_initial, args, base)
    
    # --- 2. CONNECTION STEP ---
    if len(df_cleaned) > 0:
        df_final = run_connection(df_cleaned, args, base)
    else:
        print("\nSkipping connection step: No particles remaining after cleaning.")
        df_final = df_cleaned
    
    # --- 3. FINAL OUTPUT ---
    if len(df_final) > 0:
        write_star(df_final, output_star, overwrite=True)
        print(f"\nFINAL OUTPUT saved to '{output_star}'")
    else:
        print("\nFINAL OUTPUT: Empty DataFrame. No STAR file saved.")


if __name__ == "__main__":
    main()