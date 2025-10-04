#!/usr/bin/env python3
"""
CLI wrapper for helical tube overlap calculator and filter.
@Builab 2025
"""

import sys
import os
import argparse

# Add parent directory to path to import utils module
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.clean import (
    calculate_all_overlaps,
    identify_tubes_to_delete,
    remove_overlapping_tubes
)
from utils.io import read_star, write_star


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate helical tube overlaps and remove shorter overlapping tubes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dist_thres 20 --margin 50 --angpix 14.00 input.star
  %(prog)s --dist_thres 15 --margin 30 --angpix 10.00 input.star --csv
        """
    )
    
    parser.add_argument('input_star', help='Input STAR file')
    parser.add_argument('--dist_thres', type=float, default=50,
                        help='Distance threshold (Angstroms) for overlap detection')
    parser.add_argument('--margin', type=float, default=500,
                        help='Margin (Angstroms) for bounding box overlap check')
    parser.add_argument('--angpix', type=float, required=True,
                        help='Pixel size (Angstroms/pixel)')
    parser.add_argument('--csv', action='store_true',
                        help='Save overlap results to CSV file')
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    return parser.parse_args()


def main() -> None:
    """Main function to run script from command line."""
    args = get_args()
    
    # Set default output name: XXX.star -> XXX_filtered{threshold}A.star
    base = os.path.splitext(args.input_star)[0]
    output_star = f"{base}_filtered{args.dist_thres}A.star"
    
    # Print parameters
    print("Removing overlapping lines")
    print(f"Input file: {args.input_star}")
    print(f"Distance threshold: {args.dist_thres} Angstroms")
    print(f"Bounding box margin: {args.margin} Angstroms")
    print(f"Pixel size (angpix): {args.angpix} Angstroms/pixel")
    
    # Read the star file
    try:
        df = read_star(args.input_star)
        print(f"\nLoaded {len(df)} particles from {df['rlnHelicalTubeID'].nunique()} tubes")
    except Exception as e:
        print(f"Error reading star file: {e}")
        sys.exit(1)
    
    # Calculate overlaps
    overlap_results = calculate_all_overlaps(
        df=df,
        margin=args.margin,
        angpix=args.angpix
    )
    
    # Display results
    if len(overlap_results) > 0:
        print("\n" + "="*80)
        print("OVERLAP RESULTS (sorted by average distance)")
        print("="*80)
        print(overlap_results.to_string(index=False))
        
        # Identify tubes to delete
        tubes_to_delete = identify_tubes_to_delete(
            overlap_results=overlap_results,
            distance_threshold=args.dist_thres
        )
        
        print(f"\n\n{'='*80}")
        print(f"TUBES TO DELETE (avg distance < {args.dist_thres} Angstroms)")
        print(f"{'='*80}")
        print(f"Number of tubes to delete: {len(tubes_to_delete)}")
        if len(tubes_to_delete) > 0:
            print(f"Tube IDs to delete: {sorted(tubes_to_delete)}")
            
            # Show details of overlapping tubes
            overlapping_details = overlap_results[
                overlap_results['avg_distance'] < args.dist_thres
            ]
            print(f"\nDetails of overlapping tubes:")
            print(overlapping_details.to_string(index=False))
        
        # Remove overlapping tubes
        df_filtered = remove_overlapping_tubes(
            df=df,
            tubes_to_delete=tubes_to_delete
        )
        
        print(f"\n{'='*80}")
        print(f"FILTERING SUMMARY")
        print(f"{'='*80}")
        print(f"Original: {len(df)} particles from {df['rlnHelicalTubeID'].nunique()} tubes")
        print(f"Filtered: {len(df_filtered)} particles from {df_filtered['rlnHelicalTubeID'].nunique()} tubes")
        print(f"Removed: {len(df) - len(df_filtered)} particles from {len(tubes_to_delete)} tubes")
        
        # Save results
        if args.csv:
            csv_filename = f"{base}_overlap_results.csv"
            overlap_results.to_csv(csv_filename, index=False)
            print(f"\nSaved overlap results to '{csv_filename}'")
        
        write_star(df_filtered, output_star, overwrite=True)
        print(f"Saved filtered star file to '{output_star}'")
    else:
        print("\nNo overlapping tube pairs found.")
        print("All tubes are sufficiently separated based on the bounding box margin.")
        print("No filtering needed - original file is already optimal.")


if __name__ == "__main__":
    main()
