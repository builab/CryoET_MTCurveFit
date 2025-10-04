#!/usr/bin/env python3
"""
Core functions for connecting broken helical tubes using trajectory extrapolation.
@Builab 2025
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two 3D points."""
    return np.linalg.norm(p1 - p2)


def get_line_info(
    df: pd.DataFrame,
    tube_id: int,
    angpix: float
) -> Optional[Dict[str, Any]]:
    """
    Get coordinates and metadata for a helical tube.
    
    Args:
        df: DataFrame with particle data.
        tube_id: Tube ID to extract.
        angpix: Pixel size in Angstroms.
    
    Returns:
        Dictionary with tube info or None if insufficient points.
    """
    tube_data = df[df['rlnHelicalTubeID'] == tube_id].copy()
    tube_data = tube_data.sort_index()
    # Scale coordinates by angpix to work in Angstroms
    coords = tube_data[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values * angpix
    
    if len(coords) < 2:
        return None
    
    tomo_name = tube_data['rlnTomoName'].iloc[0] if 'rlnTomoName' in tube_data.columns and not tube_data['rlnTomoName'].empty else "Unknown"
    detector_pixel_size = tube_data['rlnDetectorPixelSize'].iloc[0] if 'rlnDetectorPixelSize' in tube_data.columns and not tube_data['rlnDetectorPixelSize'].empty else None

    return {
        'tube_id': tube_id,
        'coords': coords,  # Coords in Angstroms
        'n_points': len(coords),
        'tomo_name': tomo_name,
        'detector_pixel_size': detector_pixel_size
    }


def fit_and_extrapolate(
    coords: np.ndarray,
    min_seed: int,
    dist_extrapolate: float,
    poly_order_seed: int
) -> Optional[np.ndarray]:
    """
    Fit polynomial to last min_seed coordinates and extrapolate forward.
    
    Args:
        coords: Coordinate array (N x 3) in Angstroms.
        min_seed: Number of points to use for fitting.
        dist_extrapolate: Total distance to extrapolate in Angstroms.
        poly_order_seed: Polynomial order for fitting.
    
    Returns:
        Extrapolated coordinates or None if fitting fails.
    """
    N = len(coords)
    if N < poly_order_seed + 1 or min_seed < poly_order_seed + 1:
        return None
    
    # Calculate average step size from seed points
    seed_coords = coords[N - min_seed:N, :]
    step_distances = np.linalg.norm(seed_coords[1:] - seed_coords[:-1], axis=1)
    
    if len(step_distances) == 0:
        avg_step_dist = np.linalg.norm(coords[N-1] - coords[N-2]) if N >= 2 else 0
    else:
        avg_step_dist = np.mean(step_distances)

    if avg_step_dist <= 0:
        n_steps = 10 
    else:
        n_steps = max(1, int(np.ceil(dist_extrapolate / avg_step_dist)))

    n_extrapolate = min(n_steps, 50)

    # Fit and extrapolate
    t_fit = np.arange(N - min_seed, N)
    t_extrapolate = np.arange(N, N + n_extrapolate)
    extrapolated_coords = np.zeros((n_extrapolate, 3))
    
    for i in range(3):
        y_fit = coords[N - min_seed:N, i]
        try:
            p = np.polyfit(t_fit, y_fit, poly_order_seed)
        except Exception:
            return None
            
        extrapolated_coords[:, i] = np.polyval(p, t_extrapolate)
        
    return extrapolated_coords


def check_extrapolation_overlap(
    extrapolated_coords: Optional[np.ndarray],
    target_coords: np.ndarray,
    overlap_threshold: float
) -> Tuple[bool, float]:
    """
    Check if extrapolated path overlaps with target segment.
    
    Args:
        extrapolated_coords: Extrapolated coordinates.
        target_coords: Target segment coordinates.
        overlap_threshold: Maximum distance for overlap in Angstroms.
    
    Returns:
        Tuple of (overlap_found, min_distance).
    """
    if extrapolated_coords is None or len(target_coords) == 0:
        return False, float('inf')

    dist_matrix = np.linalg.norm(
        extrapolated_coords[:, None, :] - target_coords[None, :, :], axis=2
    )
    
    min_distance = np.min(dist_matrix)
    overlap_found = min_distance <= overlap_threshold
    
    return overlap_found, min_distance


def check_connection_compatibility_extrapolate(
    line1_info: Dict[str, Any],
    line2_info: Dict[str, Any],
    end1: str,
    end2: str,
    overlap_threshold: float,
    min_seed: int,
    dist_extrapolate: float,
    poly_order_seed: int
) -> Tuple[bool, float, bool, bool, float]:
    """
    Check if line1 can connect to line2 via trajectory extrapolation.
    
    Args:
        line1_info: Info dict for first tube.
        line2_info: Info dict for second tube.
        end1: 'start' or 'end' of line1.
        end2: 'start' or 'end' of line2.
        overlap_threshold: Distance threshold in Angstroms.
        min_seed: Number of points for fitting.
        dist_extrapolate: Extrapolation distance in Angstroms.
        poly_order_seed: Polynomial order.
    
    Returns:
        Tuple of (can_connect, min_distance, reverse1, reverse2, simple_end_distance).
    """
    coords1 = line1_info['coords']
    coords2 = line2_info['coords']
    
    # Calculate simple end-to-end distance
    P1 = coords1[-1] if end1 == 'end' else coords1[0]
    P2 = coords2[0] if end2 == 'start' else coords2[-1]
    simple_end_distance = np.linalg.norm(P1 - P2)

    # Determine which points to fit on Line 1
    if end1 == 'end':
        fit_coords1 = coords1
        reverse1 = False 
    else:
        fit_coords1 = np.flipud(coords1)
        reverse1 = True
    
    # Extrapolate Line 1's trajectory
    extrapolated_coords = fit_and_extrapolate(
        fit_coords1, min_seed, dist_extrapolate, poly_order_seed
    )
    
    if extrapolated_coords is None:
        return False, float('inf'), False, False, simple_end_distance

    # Determine target points on Line 2
    target_buffer = min_seed * 2 
    
    if end2 == 'start':
        target_coords2 = coords2[:target_buffer] 
        reverse2 = False
    else:
        target_coords2 = np.flipud(coords2[-target_buffer:])
        reverse2 = True
    
    # Check for overlap
    overlap_found, min_distance = check_extrapolation_overlap(
        extrapolated_coords, target_coords2, overlap_threshold
    )
    
    if not overlap_found:
        return False, min_distance, False, False, simple_end_distance

    return True, min_distance, reverse1, reverse2, simple_end_distance


def find_line_connections(
    df: pd.DataFrame,
    angpix: float,
    overlap_thres: float,
    min_seed: int,
    dist_extrapolate: float,
    poly_order_seed: int
) -> List[Dict[str, Any]]:
    """
    Find all possible connections between tubes using extrapolation.
    
    Args:
        df: DataFrame with particle data.
        angpix: Pixel size in Angstroms.
        overlap_thres: Overlap threshold in Angstroms.
        min_seed: Number of points for fitting.
        dist_extrapolate: Extrapolation distance in Angstroms.
        poly_order_seed: Polynomial order for fitting.
    
    Returns:
        List of connection dictionaries.
    """
    tube_ids = df['rlnHelicalTubeID'].unique()
    
    line_info = {}
    for tube_id in tube_ids:
        info = get_line_info(df, tube_id, angpix)
        if info is not None:
            line_info[tube_id] = info
    
    connections = []
    tube_ids_list = list(tube_ids)
    
    for i, tube_id1 in enumerate(tube_ids_list):
        if tube_id1 not in line_info:
            continue
        
        for tube_id2 in tube_ids_list[i+1:]:
            if tube_id2 not in line_info:
                continue
            
            connection_types = [
                ('end', 'start', 'Line1_end -> Line2_start'),
                ('end', 'end', 'Line1_end -> Line2_end (reverse Line2)'),
                ('start', 'start', 'Line1_start -> Line2_start (reverse Line1)'),
                ('start', 'end', 'Line1_start -> Line2_end (reverse both)'),
            ]
            
            best_connection = None
            best_score = float('inf')
            
            for end1, end2, desc in connection_types:
                can_connect, min_distance, reverse1, reverse2, simple_end_distance = check_connection_compatibility_extrapolate(
                    line_info[tube_id1], line_info[tube_id2],
                    end1, end2, overlap_thres, min_seed, dist_extrapolate, poly_order_seed
                )
                
                if can_connect:
                    score = min_distance
                    if score < best_score:
                        best_score = score
                        best_connection = {
                            'tube_id1': tube_id1,
                            'tube_id2': tube_id2,
                            'min_overlap_dist': min_distance,
                            'simple_end_distance': simple_end_distance,
                            'reverse1': reverse1,
                            'reverse2': reverse2,
                            'connection_type': desc,
                            'n_points1': line_info[tube_id1]['n_points'],
                            'n_points2': line_info[tube_id2]['n_points']
                        }
            
            if best_connection is not None:
                connections.append(best_connection)
    
    return connections


def merge_connected_lines(
    df: pd.DataFrame,
    connections: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """
    Merge tubes that should be connected using union-find.
    
    Args:
        df: DataFrame with particle data.
        connections: List of connection dictionaries.
    
    Returns:
        Tuple of (merged DataFrame, tube_id mapping).
    """
    parent = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x
    
    for conn in connections:
        union(conn['tube_id1'], conn['tube_id2'])
    
    groups = {}
    for tube_id in df['rlnHelicalTubeID'].unique():
        root = find(tube_id)
        if root not in groups:
            groups[root] = []
        groups[root].append(tube_id)
    
    df_merged = df.copy()
    tube_id_mapping = {}
    new_tube_id = 1
    
    for _, tube_ids in groups.items():
        for tube_id in tube_ids:
            tube_id_mapping[tube_id] = new_tube_id
        new_tube_id += 1
    
    df_merged['rlnHelicalTubeID'] = df_merged['rlnHelicalTubeID'].map(tube_id_mapping)
    
    return df_merged, tube_id_mapping


def fit_and_resample_single_tube(
    tube_data: pd.DataFrame,
    poly_order: int,
    sample_step: float,
    integration_step: float,
    angpix: float,
    current_tube_id: int
) -> List[Dict[str, Any]]:
    """
    Fit polynomial to merged tube and resample.
    
    Args:
        tube_data: DataFrame with tube particle data (coordinates in pixels).
        poly_order: Polynomial order for fitting.
        sample_step: Resampling step in Angstroms.
        integration_step: Integration step in Angstroms.
        angpix: Pixel size in Angstroms.
        current_tube_id: Tube ID for output.
    
    Returns:
        List of resampled points (coordinates in pixels).
    """
    # Import resample from fit to avoid circular import
    from .fit import resample
    
    # Convert to Angstroms for fitting/resampling
    coords_angstrom = tube_data[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values * angpix
    
    N = len(coords_angstrom)
    if N < poly_order + 1:
        print(f"  Warning: Tube {current_tube_id} has {N} points, too few for poly order {poly_order}. Skipping refit.")
        return []

    # Get metadata
    tomo_name = tube_data['rlnTomoName'].iloc[0] if 'rlnTomoName' in tube_data.columns and not tube_data['rlnTomoName'].empty else "Unknown"
    detector_pixel_size = tube_data['rlnDetectorPixelSize'].iloc[0] if 'rlnDetectorPixelSize' in tube_data.columns and not tube_data['rlnDetectorPixelSize'].empty else None

    # Determine independent variable by largest range
    ranges = np.ptp(coords_angstrom, axis=0)
    
    if ranges[0] >= ranges[1]:  # X is major axis
        independent_var = coords_angstrom[:, 0]
        poly_xy = np.poly1d(np.polyfit(independent_var, coords_angstrom[:, 1], poly_order))
        poly_k = np.poly1d(np.polyfit(independent_var, coords_angstrom[:, 2], poly_order))
        mode = 1
    else:  # Y is major axis
        independent_var = coords_angstrom[:, 1]
        poly_xy = np.poly1d(np.polyfit(independent_var, coords_angstrom[:, 0], poly_order))
        poly_k = np.poly1d(np.polyfit(independent_var, coords_angstrom[:, 2], poly_order))
        mode = 2

    start = independent_var.min()
    end = independent_var.max()
    
    # Resample (output in Angstroms)
    resampled_points_angstrom = resample(
        poly_xy, poly_k, start, end, mode,
        current_tube_id - 1,  # cluster_id is 0-indexed
        tomo_name, sample_step, integration_step,
        detector_pixel_size
    )
    
    # Convert back to pixels
    for point in resampled_points_angstrom:
        point['rlnCoordinateX'] /= angpix
        point['rlnCoordinateY'] /= angpix
        point['rlnCoordinateZ'] /= angpix
        
    return resampled_points_angstrom


def refit_and_resample_tubes(
    df_input: pd.DataFrame,
    poly_order: int,
    sample_step: float,
    angpix: float
) -> pd.DataFrame:
    """
    Renumber tube IDs consecutively and refit/resample all tubes.
    
    Args:
        df_input: DataFrame with merged tubes.
        poly_order: Polynomial order for final fitting.
        sample_step: Resampling step in Angstroms.
        angpix: Pixel size in Angstroms.
    
    Returns:
        DataFrame with resampled particles (coordinates in pixels).
    """
    print(f"\n--- Post-Merging Refit/Resample Step ---")
    print(f"  Target Polynomial Order for Final Fit: {poly_order}")
    print(f"  Resampling Step Distance: {sample_step:.2f} Angstroms")

    # Renumber tube IDs consecutively
    unique_ids = df_input['rlnHelicalTubeID'].unique()
    id_map = {old_id: new_id + 1 for new_id, old_id in enumerate(unique_ids)}
    df_renumbered = df_input.copy()
    df_renumbered['rlnHelicalTubeID'] = df_renumbered['rlnHelicalTubeID'].map(id_map)
    
    all_resampled_points = []
    integration_step = sample_step / 10.0  # Small step for integration
    
    for old_id, new_id in id_map.items():
        tube_data = df_renumbered[df_renumbered['rlnHelicalTubeID'] == new_id]
        
        resampled_data = fit_and_resample_single_tube(
            tube_data, poly_order, sample_step, integration_step, angpix, new_id
        )
        all_resampled_points.extend(resampled_data)

    if all_resampled_points:
        df_resampled = pd.DataFrame(all_resampled_points)
        print(f"  Successfully resampled {df_resampled['rlnHelicalTubeID'].nunique()} tubes into {len(df_resampled)} particles.")
        
        # Standardize column order
        required_cols = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ', 
                         'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi', 
                         'rlnHelicalTubeID', 'rlnTomoName']
        if 'rlnDetectorPixelSize' in df_resampled.columns:
            required_cols.append('rlnDetectorPixelSize')
             
        for col in required_cols:
            if col not in df_resampled.columns:
                df_resampled[col] = 0.0

        return df_resampled[required_cols]
    else:
        print("  Resampling failed. Returning original merged data (renumbered).")
        return df_renumbered
