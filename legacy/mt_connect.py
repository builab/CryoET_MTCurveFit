#!/usr/bin/env python3
"""
CLI wrapper for connecting broken helical tubes using trajectory extrapolation.
@Builab 2025
"""

import sys
import os
import argparse
import pandas as pd

from typing import Dict, Any, Tuple, Optional, List

# Add parent directory to path to import utils module
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.connect import (
    find_line_connections,
    merge_connected_lines,
    refit_and_resample_tubes
)
from utils.io import read_star, write_star


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Connect broken helical tubes using trajectory extrapolation (STAR file utility)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool runs iteratively to connect segments over increasing distances. The 
--dist_extrapolate is scaled by --dist_iter_scale for each subsequent iteration.
Finally, a high-order curve fitting and resampling step is performed.

Examples:
  # Run 3 iterations (seed poly order 2), increasing search distance by 2.0x.
  # Final fit uses poly order 3 and resamples every 20.0 Angstroms.
  %(prog)s input.star --angpix 14.0 --dist_extrapolate 100.0 --overlap_thres 10.0 --min_seed 5 --iter 3 --dist_iter_scale 2.0 --poly_order_seed 2 --poly_order 3 --sample_step 20.0
        """
    )
    
    parser.add_argument('input_star', help='Input STAR file (coordinates in pixels)')
    parser.add_argument('--angpix', type=float, required=True,
                        help='Pixel size (Angstroms/pixel)')
    
    # Extrapolation parameters
    parser.add_argument('--poly_order_seed', type=int, default=2, choices=[1, 2],
                        help='Polynomial degree for fitting segment ends (default: 2)')
    parser.add_argument('--min_seed', type=int, default=5,
                        help='Minimum points from line end for polynomial fit (default: 5)')
    parser.add_argument('--dist_extrapolate', type=float, required=True,
                        help='Initial distance (Angstroms) to project trajectory forward')
    parser.add_argument('--overlap_thres', type=float, required=True,
                        help='Maximum distance (Angstroms) for overlap detection')
    
    # Iteration parameters
    parser.add_argument('--iter', type=int, default=2,
                        help='Maximum number of iterative connection runs (default: 2)')
    parser.add_argument('--dist_iter_scale', type=float, default=1.5,
                        help='Scale factor for dist_extrapolate in subsequent iterations (default: 1.5)')
    
    # Final refitting/resampling parameters
    parser.add_argument('--poly_order', type=int, default=3,
                        help='Polynomial degree for final curve fitting (default: 3)')
    parser.add_argument('--sample_step', type=float, default=20.0,
                        help='Distance (Angstroms) between resampled particles (default: 20.0)')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()


def run_connection_step(
    df_input: pd.DataFrame,
    current_dist_extrapolate: float,
    args: argparse.Namespace,
    iteration: int
) -> Tuple[pd.DataFrame, List[Dict], int]:
    """
    Perform one iteration of connection finding and merging.
    
    Args:
        df_input: Input DataFrame.
        current_dist_extrapolate: Current extrapolation distance.
        args: Command-line arguments.
        iteration: Current iteration number.
    
    Returns:
        Tuple of (merged DataFrame, connections found, tubes before merging).
    """
    print(f"\n--- Running Iteration {iteration} ---")
    print(f"Extrapolation Distance: {current_dist_extrapolate:.2f} Angstroms")
    
    original_n_tubes = df_input['rlnHelicalTubeID'].nunique()
    
    connections = find_line_connections(
        df=df_input,
        angpix=args.angpix,
        overlap_thres=args.overlap_thres,
        min_seed=args.min_seed,
        dist_extrapolate=current_dist_extrapolate,
        poly_order_seed=args.poly_order_seed
    )
    
    if not connections:
        print(f"No connections found in Iteration {iteration}.")
        return df_input, [], original_n_tubes
    
    df_merged, _ = merge_connected_lines(df_input, connections)
    final_n_tubes = df_merged['rlnHelicalTubeID'].nunique()
    
    print(f"  Merged {original_n_tubes - final_n_tubes} pairs/groups.")
    print(f"  Tubes remaining: {final_n_tubes}")
    
    # Add iteration info to connection records
    for conn in connections:
        conn['iteration'] = iteration
        conn['dist_extrapolate_used'] = current_dist_extrapolate

    return df_merged, connections, original_n_tubes


def main() -> None:
    """Main function to run script from command line."""
    args = get_args()
    
    # Initial setup
    base = os.path.splitext(args.input_star)[0]
    
    try:
        df_current = read_star(args.input_star)
    except Exception as e:
        print(f"Error reading star file: {e}")
        sys.exit(1)
    
    original_n_tubes = df_current['rlnHelicalTubeID'].nunique()
    current_dist_extrapolate = args.dist_extrapolate
    all_connections = []
    
    print("Iterative Filament Connect")
    print(f"Input file: {args.input_star}")
    print(f"Original tubes: {original_n_tubes}")
    print(f"Max iterations: {args.iter}")
    print(f"Seed Poly Order: {args.poly_order_seed}")
    print(f"Final Poly Order: {args.poly_order}")
    print(f"Resampling Step: {args.sample_step:.2f} Angstroms")
    
    # Run iterative connection process
    for i in range(1, args.iter + 1):
        df_merged, connections_found, n_tubes_before = run_connection_step(
            df_current, current_dist_extrapolate, args, i
        )

        all_connections.extend(connections_found)
        
        # Check for convergence
        n_tubes_after = df_merged['rlnHelicalTubeID'].nunique()
        if n_tubes_after == n_tubes_before:
            print(f"\nConvergence reached (no new connections in Iteration {i}). Stopping.")
            df_current = df_merged
            break
        
        df_current = df_merged
        
        # Scale distance for next iteration
        if i < args.iter:
            current_dist_extrapolate *= args.dist_iter_scale

    final_n_tubes_merged = df_current['rlnHelicalTubeID'].nunique()
    
    # Final output
    if final_n_tubes_merged < original_n_tubes:
        # Post-merging refit and resample
        df_current = refit_and_resample_tubes(
            df_input=df_current,
            poly_order=args.poly_order,
            sample_step=args.sample_step,
            angpix=args.angpix
        )
        
        output_star = f"{base}_connected.star"
        csv_output_file = f"{base}_connections.csv"
        
        print(f"FINAL PROCESSING SUMMARY")
        print(f"Original tubes: {original_n_tubes}")
        print(f"Tubes after merge: {df_current['rlnHelicalTubeID'].nunique()}")
        print(f"Total tubes merged: {original_n_tubes - final_n_tubes_merged}")
        print(f"Final particle count: {len(df_current)}")
        
        # Save final STAR file
        write_star(df_current, output_star, overwrite=True)
        print(f"\nSaved connected tubes to '{output_star}'")
        
        # Save connection details to CSV
        if all_connections:
            connections_df = pd.DataFrame(all_connections)
            csv_data = connections_df[['iteration', 'tube_id1', 'tube_id2', 
                                      'simple_end_distance', 'min_overlap_dist', 
                                      'connection_type', 'dist_extrapolate_used']]
            csv_data = csv_data.sort_values(by=['iteration', 'min_overlap_dist'])
            csv_data.to_csv(csv_output_file, index=False, float_format='%.3f')
            print(f"Saved connection details to '{csv_output_file}'")
        
    else:
        print("NO CONNECTIONS FOUND AFTER ALL ITERATIONS")


if __name__ == "__main__":
    main()
