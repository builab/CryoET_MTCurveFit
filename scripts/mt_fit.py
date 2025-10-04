#!/usr/bin/env python3
"""
CLI wrapper for filament processing pipeline with subcommands:
- fit     : Initial curve fitting and clustering
- clean   : Remove overlapping tubes
- connect : Connect broken tube segments
- pipeline: Run full pipeline (fit -> clean -> connect)
Not yet tested clearly

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


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments common to multiple subcommands."""
    parser.add_argument('input', help='Input STAR file')
    parser.add_argument('-o', '--output', help='Output STAR file (default: auto-generated)')
    parser.add_argument('--angpix', type=float, default=14.00,
                       help='Pixel size (Angstroms/pixel) (default: 14.00)')


def add_fit_arguments(parser: argparse.ArgumentParser) -> None:
    """Add fitting-specific arguments."""
    parser.add_argument('--poly_order', type=int, default=3,
                       help='Polynomial order for fitting (default: 3)')
    parser.add_argument('--min_seed', type=int, default=6,
                       help='Minimum seed points (default: 6)')
    parser.add_argument('--sample_step', type=float, default=82.0,
                       help='Resampling step (Angstroms) (default: 82.0)')
    
    # Advanced fit parameters (hidden)
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


def add_clean_arguments(parser: argparse.ArgumentParser) -> None:
    """Add cleaning-specific arguments."""
    parser.add_argument('--dist_thres', type=float, default=50,
                       help='Overlap removal threshold (Angstroms) (default: 50)')
    parser.add_argument('--margin', type=float, default=500,
                       help='Bounding box margin (Angstroms) (default: 500)')


def add_connect_arguments(parser: argparse.ArgumentParser) -> None:
    """Add connection-specific arguments."""
    parser.add_argument('--dist_extrapolate', type=float, required=True,
                       help='Initial extrapolation distance (Angstroms) (REQUIRED)')
    parser.add_argument('--overlap_thres', type=float, required=True,
                       help='Connection overlap threshold (Angstroms) (REQUIRED)')
    parser.add_argument('--iterations', type=int, default=2,
                       help='Connection iterations (default: 2)')
    parser.add_argument('--dist_scale', type=float, default=1.5,
                       help='Distance scale factor per iteration (default: 1.5)')
    parser.add_argument('--min_seed', type=int, default=6,
                       help='Minimum seed points (default: 6)')
    parser.add_argument('--poly_order', type=int, default=3,
                       help='Polynomial order for refitting (default: 3)')
    parser.add_argument('--sample_step', type=float, default=82.0,
                       help='Resampling step for refitting (Angstroms) (default: 82.0)')
    parser.add_argument('--poly_order_seed', type=int, default=3, help=argparse.SUPPRESS)

# To be added
#def add_predict_arguments(parser: argparse.ArgumentParser) -> None:

def run_fitting(file_path: str, args: argparse.Namespace, step_num: int = None) -> pd.DataFrame:
    """Run initial curve fitting and clustering."""
    if step_num is not None:
        print_step_header(step_num, "CURVE FITTING & CLUSTERING")
    else:
        print("="*80)
        print("CURVE FITTING & CLUSTERING")
        print("="*80)
    
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


def run_cleaning(df_input: pd.DataFrame, args: argparse.Namespace, step_num: int = None) -> pd.DataFrame:
    """Remove overlapping tubes."""
    if step_num is not None:
        print_step_header(step_num, "CLEANING (OVERLAP REMOVAL)")
    else:
        print("="*80)
        print("CLEANING (OVERLAP REMOVAL)")
        print("="*80)
    
    n_tubes_before = df_input['rlnHelicalTubeID'].nunique()
    n_particles_before = len(df_input)
    
    print(f"Distance threshold: {args.dist_thres} Å")
    print(f"Bounding box margin: {args.margin} Å")
    
    # Calculate overlaps
    overlap_results = calculate_all_overlaps(
        df=df_input,
        margin=args.margin,
        angpix=args.angpix
    )
    
    if len(overlap_results) == 0:
        print("\nNo overlapping tubes found - skipping cleaning")
        return df_input
    
    # Remove overlapping tubes
    tubes_to_delete = identify_tubes_to_delete(
        overlap_results=overlap_results,
        distance_threshold=args.dist_thres
    )
    
    if len(tubes_to_delete) == 0:
        print(f"\nNo tubes meet removal criteria (distance < {args.dist_thres} Å)")
        return df_input
    
    df_filtered = remove_overlapping_tubes(df=df_input, tubes_to_delete=tubes_to_delete)
    
    print_summary("Cleaning Results", [
        f"Tubes removed: {len(tubes_to_delete)}",
        f"Particles removed: {n_particles_before - len(df_filtered)}",
        f"Remaining tubes: {df_filtered['rlnHelicalTubeID'].nunique()}",
        f"Remaining particles: {len(df_filtered)}"
    ])
    
    return df_filtered


def run_connection(df_input: pd.DataFrame, args: argparse.Namespace, step_num: int = None) -> pd.DataFrame:
    """Connect broken tube segments."""
    if step_num is not None:
        print_step_header(step_num, "CONNECTION (TRAJECTORY EXTRAPOLATION)")
    else:
        print("="*80)
        print("CONNECTION (TRAJECTORY EXTRAPOLATION)")
        print("="*80)
    
    n_tubes_before = df_input['rlnHelicalTubeID'].nunique()
    df_current = df_input.copy()
    current_dist_extrapolate = args.dist_extrapolate
    total_merges = 0
    
    print(f"Initial extrapolation: {args.dist_extrapolate} Å")
    print(f"Overlap threshold: {args.overlap_thres} Å")
    print(f"Max iterations: {args.iterations}")
    
    # Iterative connection
    for i in range(1, args.iterations + 1):
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
        
        if i < args.iterations:
            current_dist_extrapolate *= args.dist_scale
    
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

# To be added
#def run_predict

def cmd_fit(args: argparse.Namespace) -> None:
    """Execute fit subcommand."""
    output_file = args.output or f"{os.path.splitext(args.input)[0]}_fitted.star"
    
    try:
        df_fitted = run_fitting(args.input, args)
        
        if not df_fitted.empty:
            write_star(df_fitted, output_file, overwrite=True)
            print(f"\n✓ Output saved to: {output_file}")
        else:
            print("\nWarning: No output particles generated")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def cmd_clean(args: argparse.Namespace) -> None:
    """Execute clean subcommand."""
    output_file = args.output or f"{os.path.splitext(args.input)[0]}_cleaned.star"
    
    try:
        # Read input
        df_input = read_star(args.input)
        if df_input is None or df_input.empty:
            raise ValueError("Failed to load input file or file is empty")
        
        df_cleaned = run_cleaning(df_input, args)
        
        if not df_cleaned.empty:
            write_star(df_cleaned, output_file, overwrite=True)
            print(f"\n✓ Output saved to: {output_file}")
        else:
            print("\nWarning: No output particles remaining after cleaning")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def cmd_connect(args: argparse.Namespace) -> None:
    """Execute connect subcommand."""
    output_file = args.output or f"{os.path.splitext(args.input)[0]}_connected.star"
    
    try:
        # Read input
        df_input = read_star(args.input)
        if df_input is None or df_input.empty:
            raise ValueError("Failed to load input file or file is empty")
        
        df_connected = run_connection(df_input, args)
        
        if not df_connected.empty:
            write_star(df_connected, output_file, overwrite=True)
            print(f"\n✓ Output saved to: {output_file}")
        else:
            print("\nWarning: No output particles generated")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

# To be added
#def cmd_predict

def cmd_pipeline(args: argparse.Namespace) -> None:
    """Execute full pipeline."""
    output_file = args.output or f"{os.path.splitext(args.input)[0]}_processed.star"
    
    # Print header
    print("="*80)
    print("FILAMENT PROCESSING PIPELINE")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Pixel size: {args.angpix} Å/px")
    print(f"Sample step: {args.sample_step} Å")
    print(f"Polynomial order: {args.poly_order}")
    
    try:
        # Run pipeline
        df_fitted = run_fitting(args.input, args, step_num=1)
        df_cleaned = run_cleaning(df_fitted, args, step_num=2)
        df_final = run_connection(df_cleaned, args, step_num=3)
        
        # Save output
        if not df_final.empty:
            write_star(df_final, output_file, overwrite=True)
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETE")
            print("="*80)
            print_summary("Final Output", [
                f"File: {output_file}",
                f"Tubes: {df_final['rlnHelicalTubeID'].nunique()}",
                f"Particles: {len(df_final)}"
            ])
        else:
            print("\nWarning: Pipeline produced no output particles")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def main() -> None:
    """Main entry point with subcommand parsing."""
    parser = argparse.ArgumentParser(
        description='Filament processing toolkit for cryo-ET data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  fit       Initial curve fitting and clustering
  clean     Remove overlapping tubes
  connect   Connect broken tube segments
  pipeline  Run full pipeline (fit -> clean -> connect)

Examples:
  # Run individual steps
  %(prog)s fit input.star --angpix 14 --sample_step 82
  %(prog)s clean fitted.star --dist_thres 50
  %(prog)s connect cleaned.star --dist_extrapolate 1500 --overlap_thres 80
  
  # Run full pipeline
  %(prog)s pipeline input.star --angpix 14 --sample_step 82 \\
           --dist_thres 50 --dist_extrapolate 1500 --overlap_thres 80
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # FIT subcommand
    fit_parser = subparsers.add_parser('fit', help='Initial curve fitting and clustering')
    add_common_arguments(fit_parser)
    add_fit_arguments(fit_parser)
    fit_parser.set_defaults(func=cmd_fit)
    
    # CLEAN subcommand
    clean_parser = subparsers.add_parser('clean', help='Remove overlapping tubes')
    add_common_arguments(clean_parser)
    add_clean_arguments(clean_parser)
    clean_parser.set_defaults(func=cmd_clean)
    
    # CONNECT subcommand
    connect_parser = subparsers.add_parser('connect', help='Connect broken tube segments')
    add_common_arguments(connect_parser)
    add_connect_arguments(connect_parser)
    connect_parser.set_defaults(func=cmd_connect)
    
    # PIPELINE subcommand
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
    add_common_arguments(pipeline_parser)
    add_fit_arguments(pipeline_parser)
    add_clean_arguments(pipeline_parser)
    
    # Connection args for pipeline (with different names to avoid conflicts)
    pipeline_parser.add_argument('--dist_extrapolate', type=float, required=True,
                                help='Initial extrapolation distance (Angstroms) (REQUIRED)')
    pipeline_parser.add_argument('--overlap_thres', type=float, required=True,
                                help='Connection overlap threshold (Angstroms) (REQUIRED)')
    pipeline_parser.add_argument('--iterations', type=int, default=2,
                                help='Connection iterations (default: 2)')
    pipeline_parser.add_argument('--dist_scale', type=float, default=1.5,
                                help='Distance scale factor per iteration (default: 1.5)')
    pipeline_parser.set_defaults(func=cmd_pipeline)
    
    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Execute subcommand
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
