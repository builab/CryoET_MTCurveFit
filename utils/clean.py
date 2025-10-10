#!/usr/bin/env python3

"""
Cleaning overlapping & short helical tubes.

This module provides functionality to:
- Calculate overlapped tubes within a certain distance and filter out
- Implement a quick bounding box for screening close tubes
- Filter tubes with smaller number of particles

@Builab 2025
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from typing import List, Tuple, Set, Dict, Any


def get_tube_bounding_boxes(
    df: pd.DataFrame,
    tube_ids: np.ndarray,
    angpix: float
) -> Dict[int, Dict[str, Any]]:
    """
    Calculate bounding boxes for each helical tube.
    
    Args:
        df: DataFrame with particle coordinates.
        tube_ids: Array of unique tube IDs.
        angpix: Pixel size in Angstroms.
    
    Returns:
        Dictionary mapping tube_id to bounding box info.
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


def boxes_overlap_with_margin(
    box1: Dict[str, Any],
    box2: Dict[str, Any],
    margin: float
) -> bool:
    """
    Check if two bounding boxes overlap with a margin.
    
    Args:
        box1: First bounding box dictionary.
        box2: Second bounding box dictionary.
        margin: Margin in Angstroms to extend bounding boxes.
    
    Returns:
        True if boxes are close enough to warrant detailed comparison.
    """
    # Check if boxes are separated in any dimension
    for i in range(3):  # x, y, z
        if box1['max'][i] + margin < box2['min'][i] or box2['max'][i] + margin < box1['min'][i]:
            return False
    return True


def identify_tube_pairs_to_compare(
    bounding_boxes: Dict[int, Dict[str, Any]],
    margin: float
) -> List[Tuple[int, int]]:
    """
    Identify pairs of helical tubes that are close enough to compare.
    
    Args:
        bounding_boxes: Dictionary of bounding boxes for all tubes.
        margin: Margin in Angstroms for overlap detection.
    
    Returns:
        List of tuples (shorter_tube_id, longer_tube_id).
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


def calculate_distance_shorter_to_longer(
    coords_shorter: np.ndarray,
    coords_longer: np.ndarray
) -> float:
    """
    Calculate average minimum distance from shorter tube to longer tube.
    
    Args:
        coords_shorter: Coordinates of shorter tube (N x 3).
        coords_longer: Coordinates of longer tube (M x 3).
    
    Returns:
        Average distance from points in shorter tube to nearest points in longer tube.
    """
    # Build KDTree for efficient nearest neighbor search
    tree_longer = cKDTree(coords_longer)
    
    # For each point in shorter line, find distance to nearest point in longer line
    distances, _ = tree_longer.query(coords_shorter)
    
    # Return average distance
    return distances.mean()


def calculate_all_overlaps(
    df: pd.DataFrame,
    margin: float,
    angpix: float
) -> pd.DataFrame:
    """
    Calculate overlaps between all helical tubes.
    
    Args:
        df: DataFrame with particle data.
        margin: Margin in Angstroms for bounding box overlap check.
        angpix: Pixel size in Angstroms.
    
    Returns:
        DataFrame with overlap results containing columns:
        - shorter_tube_id
        - longer_tube_id
        - n_points_shorter
        - n_points_longer
        - avg_distance
    """
    # Get unique tube IDs
    tube_ids = df['rlnHelicalTubeID'].unique()
    print(f"Found {len(tube_ids)} helical tubes")
    
    # Step 1: Get bounding boxes and identify pairs to compare
    print("\nStep 1: Identifying tube pairs to compare...")
    bounding_boxes = get_tube_bounding_boxes(df, tube_ids, angpix)
    pairs_to_compare = identify_tube_pairs_to_compare(bounding_boxes, margin)
    print(f"Found {len(pairs_to_compare)} pairs close enough to compare")
    print(f"(Skipped {len(tube_ids)*(len(tube_ids)-1)//2 - len(pairs_to_compare)} distant pairs)")
    
    # Step 2: Calculate detailed distances for identified pairs
    print("\nStep 2: Calculating distances from shorter to longer tubes...")
    results = []
    
    for shorter_id, longer_id in pairs_to_compare:
        # Get coordinates for both tubes (convert to real coordinates)
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


def identify_tubes_to_delete(
    overlap_results: pd.DataFrame,
    distance_threshold: float
) -> Set[int]:
    """
    Identify which shorter tubes should be deleted based on overlap.
    
    Args:
        overlap_results: DataFrame with overlap calculations.
        distance_threshold: Distance threshold in Angstroms.
    
    Returns:
        Set of tube IDs to delete.
    """
    if len(overlap_results) == 0:
        return set()
    
    overlapping = overlap_results[overlap_results['avg_distance'] < distance_threshold]
    tubes_to_delete = set(overlapping['shorter_tube_id'].unique())
    
    return tubes_to_delete


def remove_overlapping_tubes(
    df: pd.DataFrame,
    tubes_to_delete: Set[int]
) -> pd.DataFrame:
    """
    Remove particles belonging to overlapping shorter tubes.
    
    Args:
        df: Original DataFrame.
        tubes_to_delete: Set of tube IDs to remove.
    
    Returns:
        Filtered DataFrame with overlapping tubes removed.
    """
    df_filtered = df[~df['rlnHelicalTubeID'].isin(tubes_to_delete)].copy()
    return df_filtered


def filter_short_tubes(df: pd.DataFrame, min_part_per_tube: int) -> pd.DataFrame:
    """
    Filter out tubes (rlnHelicalTubeID groups) with fewer than min_part_per_tube particles.
    If min_part_per_tube == 0, returns df unchanged.
    """
    if df.empty:
        print("⚠️ Input DataFrame is empty — skipping short tube filtering.")
        return df

    if 'rlnHelicalTubeID' not in df.columns:
        raise ValueError("DataFrame must contain a 'rlnHelicalTubeID' column.")

    if not isinstance(min_part_per_tube, int) or min_part_per_tube < 0:
        raise ValueError("min_part_per_tube must be a non-negative integer.")

    if min_part_per_tube == 0:
        #print("Minimum particles per tube = 0 → skipping short tube filtering.")
        return df.copy()

    df_filtered = (
        df.groupby('rlnHelicalTubeID')
        .filter(lambda x: len(x) >= min_part_per_tube)
        .reset_index(drop=True)
    )
    return df_filtered