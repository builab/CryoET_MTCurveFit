#!/usr/bin/env python
# coding: utf-8

"""
CLI wrapper for curve fitting of 3D coordinates from STAR template matching files.
@Builab 2025
"""

import sys
import os
import argparse

# Add parent directory to path to import star_pipeline module
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.fit import fit_curves
from utils.io import load_coordinates, write_star


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=f"About this program:\n"
                    "This script performs multi-curve fitting of 3D coordinates from STAR files.\n"
                    "It identifies filamentous structures, clusters the points, and generates resampled coordinates.\n"
                    "Adapted from https://github.com/PengxinChai/multi-curve-fitting",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('files', nargs='*', help="Input coordinate files (.star only).")

    # General options
    gen_group = parser.add_argument_group('General options')
    gen_group.add_argument("--angpix", type=float, default=14.00,
                          help="Pixel size of the micrograph/coordinate, in Angstroms.")
    gen_group.add_argument("--sample_step", type=float, default=82,
                          help="Final sampling step for resampling, in Angstroms.")
    gen_group.add_argument("--poly_order", type=int, default=3,
                          help="Polynomial factor for curve growth and final resampling.")

    # Seed searching options
    seed_group = parser.add_argument_group('Options for seed searching and evaluation')
    seed_group.add_argument("--min_seed", type=int, default=6,
                           help="Minimum number of points to form a valid seed.")
    seed_group.add_argument("--max_dis_to_line_ang", type=float, default=50,
                           help="Max distance from initial seed line, in Angstroms.")
    seed_group.add_argument("--min_dis_neighbor_seed_ang", type=float, default=60,
                           help="Min distance between neighboring seed points, in Angstroms.")
    seed_group.add_argument("--max_dis_neighbor_seed_ang", type=float, default=320,
                           help="Max distance between neighboring seed points, in Angstroms.")
    seed_group.add_argument("--poly_order_seed", type=int, default=3,
                           help="Polynomial factor for seed quality evaluation.")
    seed_group.add_argument("--max_seed_fitting_error", type=float, default=1.0,
                           help="Maximum fitting error for valid seed.")
    seed_group.add_argument("--max_angle_change_per_4nm", type=float, default=0.5,
                           help="Curvature restriction: max angle change per 4nm, in degrees.")

    # Growth options
    growth_group = parser.add_argument_group('Options for seed growth')
    growth_group.add_argument("--max_dis_to_curve_ang", type=float, default=80,
                             help="Max distance from growing curve, in Angstroms.")
    growth_group.add_argument("--min_dis_neighbor_curve_ang", type=float, default=60,
                             help="Min distance between neighbors during growth, in Angstroms.")
    growth_group.add_argument("--max_dis_neighbor_curve_ang", type=float, default=320,
                             help="Max distance between neighbors during growth, in Angstroms.")
    growth_group.add_argument("--min_number_growth", type=int, default=0,
                             help="Min points added during growth phase.")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()


def process_file(file_path: str, args: argparse.Namespace) -> None:
    """Orchestrate the entire process for a single file: load, fit, write."""
    print(f"Processing file: {file_path}")
    base_name = os.path.splitext(file_path)[0]
    
    # Convert Angstrom to pixel values
    pixel_size = args.angpix
    
    try:
        coords, tomo_name, detector_pixel_size = load_coordinates(file_path, pixel_size)
        if coords is None:
            print(f"  - Skipping {file_path} due to missing essential columns.")
            return
        print(f"  - Loaded {len(coords)} particles from {tomo_name} "
              f"(pixel size: {detector_pixel_size}).")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    # Call core engine with explicit parameters
    df_resam, assigned_clusters, cluster_count = fit_curves(
        coords=coords,
        tomo_name=tomo_name,
        angpix=pixel_size,
        poly_order=args.poly_order,
        sample_step=args.sample_step / pixel_size,
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
    
    # Write results
    suffix = "_init_fit"
    if cluster_count == 0 or df_resam.empty:
        print("  - No clusters found, no output file written.")
        return
    output_file = f"{base_name}{suffix}.star"
    write_star(df_resam, output_file, overwrite=True)
    print(f"  - Written: {output_file} [{len(df_resam)} points, {cluster_count} clusters]")


def main() -> None:
    """Main function to run script from command line."""
    args = get_args()

    print(f"Starting curve fitting with the following parameters:")
    for key, val in vars(args).items():
        if key != 'files':
            print(f"  --{key}: {val}")

    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        process_file(file_path, args)

if __name__ == "__main__":
    main()