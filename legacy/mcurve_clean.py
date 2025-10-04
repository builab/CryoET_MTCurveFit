#!/usr/bin/env python3
"""
Helical Tube Overlap Calculator and Filter

Calculates overlaps between helical tubes and removes shorter overlapping tubes.
Distance is calculated from shorter lines to longer lines only.
"""

import argparse
import starfile
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import sys

def read_star_file(filename):
    """Read star file and return dataframe"""
    try:
        df = starfile.read(filename)
        return df
    except Exception as e:
        print(f"Error reading star file: {e}")
        sys.exit(1)

def get_line_bounding_boxes(df, tube_ids, angpix):
    """
    Step 1: Calculate bounding boxes for each line to quickly identify
    potentially overlapping lines
    """
    bounding_boxes = {}
    
    for tube_id in tube_ids:
        tube_data = df[df['rlnHelicalTubeID'] == tube_id]
        # Convert to real coordinates by multiplying with angpix
        coords = tube_data[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values * angpix
        
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        
        bounding_boxes[tube_id] = {
            'min': min_coords,
            'max': max_coords,
            'center': (min_coords + max_coords) / 2,
            'n_points': len(coords)
        }
    
    return bounding_boxes

def boxes_overlap_with_margin(box1, box2, margin):
    """
    Check if two bounding boxes overlap with a margin
    Returns True if boxes are close enough to warrant detailed comparison
    """
    # Check if boxes are separated in any dimension
    for i in range(3):  # x, y, z
        if box1['max'][i] + margin < box2['min'][i] or box2['max'][i] + margin < box1['min'][i]:
            return False
    return True

def identify_line_pairs_to_compare(bounding_boxes, margin):
    """
    Step 1: Identify pairs of lines that are close enough to compare
    Returns list of tuples: (shorter_tube_id, longer_tube_id)
    """
    tube_ids = list(bounding_boxes.keys())
    pairs_to_compare = []
    
    for i, tube_id1 in enumerate(tube_ids):
        for tube_id2 in tube_ids[i+1:]:
            if boxes_overlap_with_margin(bounding_boxes[tube_id1], 
                                        bounding_boxes[tube_id2], 
                                        margin):
                # Order by length: shorter first, longer second
                n_points1 = bounding_boxes[tube_id1]['n_points']
                n_points2 = bounding_boxes[tube_id2]['n_points']
                
                if n_points1 <= n_points2:
                    pairs_to_compare.append((tube_id1, tube_id2))
                else:
                    pairs_to_compare.append((tube_id2, tube_id1))
    
    return pairs_to_compare

def calculate_distance_shorter_to_longer(coords_shorter, coords_longer):
    """
    Step 2: Calculate the average minimum distance from each point in shorter line
    to the nearest point in longer line
    """
    # Build KDTree for efficient nearest neighbor search
    tree_longer = cKDTree(coords_longer)
    
    # For each point in shorter line, find distance to nearest point in longer line
    distances, _ = tree_longer.query(coords_shorter)
    
    # Return average distance
    return distances.mean()

def calculate_all_overlaps(df, margin, angpix):
    """
    Main function to calculate overlaps between all helical tubes
    Distance is calculated from shorter line to longer line only
    """
    # Get unique tube IDs
    tube_ids = df['rlnHelicalTubeID'].unique()
    print(f"Found {len(tube_ids)} helical tubes")
    
    # Step 1: Get bounding boxes and identify pairs to compare
    print("\nStep 1: Identifying line pairs to compare...")
    bounding_boxes = get_line_bounding_boxes(df, tube_ids, angpix)
    pairs_to_compare = identify_line_pairs_to_compare(bounding_boxes, margin)
    print(f"Found {len(pairs_to_compare)} pairs close enough to compare")
    print(f"(Skipped {len(tube_ids)*(len(tube_ids)-1)//2 - len(pairs_to_compare)} distant pairs)")
    
    # Step 2: Calculate detailed distances for identified pairs
    print("\nStep 2: Calculating distances from shorter to longer lines...")
    results = []
    
    for shorter_id, longer_id in pairs_to_compare:
        # Get coordinates for both lines (convert to real coordinates)
        coords_shorter = df[df['rlnHelicalTubeID'] == shorter_id][
            ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']
        ].values * angpix
        
        coords_longer = df[df['rlnHelicalTubeID'] == longer_id][
            ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']
        ].values * angpix
        
        # Calculate distance from shorter to longer
        avg_distance = calculate_distance_shorter_to_longer(coords_shorter, coords_longer)
        
        results.append({
            'shorter_tube_id': shorter_id,
            'longer_tube_id': longer_id,
            'n_points_shorter': len(coords_shorter),
            'n_points_longer': len(coords_longer),
            'avg_distance': avg_distance
        })
    
    # Convert to DataFrame for easy viewing
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values('avg_distance')
    
    return results_df

def identify_tubes_to_delete(overlap_results, distance_threshold):
    """
    Identify which shorter tubes should be deleted based on overlap
    A shorter tube is marked for deletion if its average distance to a longer tube
    is below the threshold
    """
    if len(overlap_results) == 0:
        return set()
    
    overlapping = overlap_results[overlap_results['avg_distance'] < distance_threshold]
    tubes_to_delete = set(overlapping['shorter_tube_id'].unique())
    
    return tubes_to_delete

def remove_overlapping_tubes(df, tubes_to_delete):
    """
    Remove particles belonging to overlapping shorter tubes
    Returns filtered dataframe with original coordinates (not multiplied by angpix)
    """
    df_filtered = df[~df['rlnHelicalTubeID'].isin(tubes_to_delete)].copy()
    return df_filtered

def main():
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
    
    args = parser.parse_args()
    
    # Set default output name: XXX.star -> XXX_filter.star
    import os
    base = os.path.splitext(args.input_star)[0]
    output_star = f"{base}_filtered{args.dist_thres}A.star"
    
    # Print parameters
    print("="*80)
    print("HELICAL TUBE OVERLAP CALCULATOR")
    print("="*80)
    print(f"Input file: {args.input_star}")
    print(f"Distance threshold: {args.dist_thres} Angstroms")
    print(f"Bounding box margin: {args.margin} Angstroms")
    print(f"Pixel size (angpix): {args.angpix} Angstroms/pixel")
    print("="*80)
    
    # Read the star file
    df = read_star_file(args.input_star)
    print(f"\nLoaded {len(df)} particles from {df['rlnHelicalTubeID'].nunique()} tubes")
    
    # Calculate overlaps
    overlap_results = calculate_all_overlaps(df, args.margin, args.angpix)
    
    # Display results
    if len(overlap_results) > 0:
        print("\n" + "="*80)
        print("OVERLAP RESULTS (sorted by average distance)")
        print("="*80)
        print(overlap_results.to_string(index=False))
        
        # Identify tubes to delete
        tubes_to_delete = identify_tubes_to_delete(overlap_results, args.dist_thres)
        
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
        df_filtered = remove_overlapping_tubes(df, tubes_to_delete)
        
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
        
        starfile.write(df_filtered, output_star, overwrite=True)
        print(f"Saved filtered star file to '{output_star}'")
    else:
        print("\nNo overlapping tube pairs found.")
        print("All tubes are sufficiently separated based on the bounding box margin.")
        print("No filtering needed - original file is already optimal.")

if __name__ == "__main__":
    main()