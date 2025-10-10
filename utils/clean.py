#!/usr/bin/env python3
"""
Clean overlapping and short helical tubes.

This module provides functionality to:
- Detect and filter overlapping tube segments
- Use bounding box optimization for efficient spatial screening
- Remove tubes with insufficient particle counts

@Builab 2025
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from typing import List, Tuple, Set, Dict, Any

from .io import validate_dataframe

class BoundingBox:
    """3D bounding box for spatial screening."""
    
    def __init__(self, coords: np.ndarray):
        """
        Initialize bounding box from coordinates.
        
        Args:
            coords: Array of 3D coordinates (N x 3).
        """
        self.min = coords.min(axis=0)
        self.max = coords.max(axis=0)
        self.center = (self.min + self.max) / 2
        self.n_points = len(coords)
    
    def overlaps_with_margin(self, other: 'BoundingBox', margin: float) -> bool:
        """
        Check if this box overlaps with another box within a margin.
        
        Args:
            other: Another bounding box.
            margin: Margin in Angstroms to extend boxes.
        
        Returns:
            True if boxes are within margin distance.
        """
        for dim in range(3):  # x, y, z
            if (self.max[dim] + margin < other.min[dim] or 
                other.max[dim] + margin < self.min[dim]):
                return False
        return True


def extract_tube_coordinates(
    df: pd.DataFrame,
    tube_id: int,
    angpix: float
) -> np.ndarray:
    """
    Extract and convert coordinates for a specific tube.
    
    Args:
        df: DataFrame containing particle data.
        tube_id: Tube ID to extract.
        angpix: Pixel size in Angstroms for conversion.
    
    Returns:
        Coordinates array (N x 3) in Angstroms.
    """
    tube_data = df[df['rlnHelicalTubeID'] == tube_id]
    coords = tube_data[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
    return coords * angpix


def compute_bounding_boxes(
    df: pd.DataFrame,
    tube_ids: np.ndarray,
    angpix: float
) -> Dict[int, BoundingBox]:
    """
    Calculate bounding boxes for all helical tubes.
    
    Args:
        df: DataFrame with particle coordinates.
        tube_ids: Array of unique tube IDs.
        angpix: Pixel size in Angstroms.
    
    Returns:
        Dictionary mapping tube_id to BoundingBox object.
    """
    bounding_boxes = {}
    
    for tube_id in tube_ids:
        coords = extract_tube_coordinates(df, tube_id, angpix)
        bounding_boxes[tube_id] = BoundingBox(coords)
    
    return bounding_boxes


def find_proximate_tube_pairs(
    bounding_boxes: Dict[int, BoundingBox],
    margin: float
) -> List[Tuple[int, int]]:
    """
    Identify pairs of tubes that are spatially close enough to compare.
    
    Uses bounding box pre-screening to avoid expensive distance calculations
    for tubes that are far apart.
    
    Args:
        bounding_boxes: Dictionary of bounding boxes for all tubes.
        margin: Margin in Angstroms for overlap detection.
    
    Returns:
        List of tuples (shorter_tube_id, longer_tube_id) ordered by point count.
    """
    tube_ids = list(bounding_boxes.keys())
    proximate_pairs = []
    
    for i, tube_id1 in enumerate(tube_ids):
        box1 = bounding_boxes[tube_id1]
        
        for tube_id2 in tube_ids[i + 1:]:
            box2 = bounding_boxes[tube_id2]
            
            if box1.overlaps_with_margin(box2, margin):
                # Order by length: shorter tube first
                if box1.n_points <= box2.n_points:
                    proximate_pairs.append((tube_id1, tube_id2))
                else:
                    proximate_pairs.append((tube_id2, tube_id1))
    
    return proximate_pairs


def compute_proximity_score(
    coords_query: np.ndarray,
    coords_reference: np.ndarray
) -> float:
    """
    Calculate average minimum distance from query points to reference points.
    
    For each point in the query set, finds the nearest point in the reference
    set and returns the mean of these minimum distances.
    
    Args:
        coords_query: Query coordinates (N x 3) in Angstroms.
        coords_reference: Reference coordinates (M x 3) in Angstroms.
    
    Returns:
        Average nearest-neighbor distance in Angstroms.
    """
    tree = cKDTree(coords_reference)
    distances, _ = tree.query(coords_query)
    return np.mean(distances)


def analyze_tube_overlaps(
    df: pd.DataFrame,
    margin: float,
    angpix: float
) -> pd.DataFrame:
    """
    Analyze overlaps between all helical tubes using spatial screening.
    
    Uses a two-stage approach:
    1. Bounding box screening to identify candidate pairs
    2. Detailed distance calculation only for proximate pairs
    
    Args:
        df: DataFrame with particle data.
        margin: Margin in Angstroms for bounding box screening.
        angpix: Pixel size in Angstroms.
    
    Returns:
        DataFrame with overlap analysis containing columns:
        - shorter_tube_id: ID of the shorter tube
        - longer_tube_id: ID of the longer tube
        - n_points_shorter: Number of points in shorter tube
        - n_points_longer: Number of points in longer tube
        - avg_distance: Average distance from shorter to longer tube
    """
    tube_ids = df['rlnHelicalTubeID'].unique()
    total_tubes = len(tube_ids)
    
    print(f"\nAnalyzing {total_tubes} helical tubes for overlaps...")
    print(f"{'='*60}")
    
    # Stage 1: Bounding box screening
    print("\nStage 1: Spatial screening with bounding boxes")
    bounding_boxes = compute_bounding_boxes(df, tube_ids, angpix)
    proximate_pairs = find_proximate_tube_pairs(bounding_boxes, margin)
    
    total_possible = total_tubes * (total_tubes - 1) // 2
    skipped = total_possible - len(proximate_pairs)
    
    print(f"  Candidate pairs: {len(proximate_pairs)}")
    print(f"  Skipped distant pairs: {skipped}")
    print(f"  Efficiency: {100 * len(proximate_pairs) / total_possible:.1f}% "
          f"of pairs require detailed analysis")
    
    if not proximate_pairs:
        print("\n✓ No overlapping tubes detected")
        return pd.DataFrame()
    
    # Stage 2: Detailed distance calculation
    print(f"\nStage 2: Computing distances for {len(proximate_pairs)} candidate pairs")
    results = []
    
    for shorter_id, longer_id in proximate_pairs:
        coords_shorter = extract_tube_coordinates(df, shorter_id, angpix)
        coords_longer = extract_tube_coordinates(df, longer_id, angpix)
        
        avg_distance = compute_proximity_score(coords_shorter, coords_longer)
        
        results.append({
            'shorter_tube_id': shorter_id,
            'longer_tube_id': longer_id,
            'n_points_shorter': len(coords_shorter),
            'n_points_longer': len(coords_longer),
            'avg_distance': avg_distance
        })
    
    # Create results DataFrame sorted by distance
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('avg_distance').reset_index(drop=True)
    
    print(f"✓ Analysis complete")
    
    return results_df


def identify_overlapping_tubes(
    overlap_analysis: pd.DataFrame,
    distance_threshold: float
) -> Set[int]:
    """
    Identify shorter tubes that overlap with longer tubes.
    
    Args:
        overlap_analysis: DataFrame from analyze_tube_overlaps.
        distance_threshold: Maximum average distance in Angstroms to 
                          consider tubes as overlapping.
    
    Returns:
        Set of shorter tube IDs that should be removed.
    """
    if overlap_analysis.empty:
        return set()
    
    overlapping = overlap_analysis[overlap_analysis['avg_distance'] < distance_threshold]
    
    if not overlapping.empty:
        tubes_to_remove = set(overlapping['shorter_tube_id'].unique())
        print(f"\nIdentified {len(tubes_to_remove)} overlapping tubes to remove")
        return tubes_to_remove
    
    return set()


def remove_tubes_by_id(
    df: pd.DataFrame,
    tube_ids_to_remove: Set[int]
) -> pd.DataFrame:
    """
    Remove all particles belonging to specified tubes.
    
    Args:
        df: Original DataFrame.
        tube_ids_to_remove: Set of tube IDs to remove.
    
    Returns:
        Filtered DataFrame with specified tubes removed.
    """
    if not tube_ids_to_remove:
        return df.copy()
    
    df_filtered = df[~df['rlnHelicalTubeID'].isin(tube_ids_to_remove)].copy()
    
    particles_removed = len(df) - len(df_filtered)
    tubes_removed = len(tube_ids_to_remove)
    
    print(f"  Removed {tubes_removed} tubes ({particles_removed} particles)")
    
    return df_filtered


def filter_short_tubes(
    df: pd.DataFrame,
    min_particles: int
) -> pd.DataFrame:
    """
    Remove tubes with fewer than the minimum number of particles.
    
    Args:
        df: DataFrame with particle data.
        min_particles: Minimum number of particles required per tube.
                      If 0, no filtering is performed.
    
    Returns:
        Filtered DataFrame with short tubes removed.
    
    Raises:
        ValueError: If min_particles is negative or DataFrame is missing
                   required columns.
    """
    if df.empty:
        print("⚠️  Input DataFrame is empty - skipping short tube filtering")
        return df
    
    if 'rlnHelicalTubeID' not in df.columns:
        raise ValueError("DataFrame must contain 'rlnHelicalTubeID' column")
    
    if not isinstance(min_particles, int) or min_particles < 0:
        raise ValueError("min_particles must be a non-negative integer")
    
    if min_particles == 0:
        return df.copy()
    
    # Count particles per tube
    tube_counts = df.groupby('rlnHelicalTubeID').size()
    tubes_before = len(tube_counts)
    
    # Filter tubes with sufficient particles
    df_filtered = (
        df.groupby('rlnHelicalTubeID')
        .filter(lambda group: len(group) >= min_particles)
        .reset_index(drop=True)
    )
    
    tubes_after = df_filtered['rlnHelicalTubeID'].nunique()
    tubes_removed = tubes_before - tubes_after
    particles_removed = len(df) - len(df_filtered)
    
    if tubes_removed > 0:
        print(f"\nShort tube filtering (minimum {min_particles} particles):")
        print(f"  Removed {tubes_removed} tubes ({particles_removed} particles)")
    
    return df_filtered


boxes_overlap_with_margin = lambda box1, box2, margin: (
    BoundingBox(np.array([box1['min']])).overlaps_with_margin(
        BoundingBox(np.array([box2['min']])), margin
    ) if isinstance(box1, dict) else box1.overlaps_with_margin(box2, margin)
)

def clean_tubes(
    df: pd.DataFrame,
    angpix: float,
    distance_threshold: float,
    margin: float = 50.0
) -> pd.DataFrame:
    """
    Comprehensive tube cleaning pipeline.
    
    Performs two cleaning operations:
    1. Removes overlapping shorter tubes
    2. Removes tubes with insufficient particles
    
    Args:
        df: DataFrame with particle data.
        distance_threshold: Maximum average distance in Angstroms for overlap.
        min_particles: Minimum particles required per tube.
        angpix: Pixel size in Angstroms.
        margin: Margin for bounding box screening (default: 50 Angstroms).
    
    Returns:
        Cleaned DataFrame.
    """
    print("\n" + "="*60)
    print("TUBE CLEANING PIPELINE")
    print("="*60)
    
    validate_dataframe(df, ['rlnHelicalTubeID'])
    
    tubes_initial = df['rlnHelicalTubeID'].nunique()
    particles_initial = len(df)
    
    print(f"\nInitial data: {tubes_initial} tubes, {particles_initial} particles")
    
    # Step 1: Remove overlapping tubes
    print(f"\n[1/2] Overlap detection (threshold: {distance_threshold:.1f} Å)")
    print("-" * 60)
    
    overlap_analysis = analyze_tube_overlaps(df, margin, angpix)
    overlapping_tubes = identify_overlapping_tubes(overlap_analysis, distance_threshold)
    df = remove_tubes_by_id(df, overlapping_tubes)
    
    
    # Summary
    tubes_final = df['rlnHelicalTubeID'].nunique()
    particles_final = len(df)
    
    print("\n" + "="*60)
    print("CLEANING SUMMARY")
    print("="*60)
    print(f"  Tubes:     {tubes_initial} → {tubes_final} "
          f"({tubes_initial - tubes_final} removed)")
    print(f"  Particles: {particles_initial} → {particles_final} "
          f"({particles_initial - particles_final} removed)")
    print("="*60 + "\n")
    
    return df