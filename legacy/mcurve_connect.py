#!/usr/bin/env python3
"""
Connect Broken Helical Tubes using Trajectory Extrapolation (Iterative)

Identifies and connects helical tubes that are broken into segments.
Uses polynomial extrapolation of one segment's end to check for overlap
with another segment's end/start. This is robust for curved lines.
The process is run iteratively with increasing search distance to connect large filaments.
Includes a final step for high-order curve fitting and resampling on merged tubes,
outputting coordinates in pixels.
"""

import argparse
import starfile
import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, Any, Tuple, Optional, List
import math # Used by the resample function

# --- Helper Functions (Resampling Dependencies) ---

def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculates the Euclidean distance between two 3D points."""
    return np.linalg.norm(p1 - p2)

def resample(
    poly_xy: np.poly1d,
    poly_k: np.poly1d,
    start: float,
    end: float,
    mode: int,
    cluster_id: int,
    tomo_name: str,
    detector_pixel_size: Optional[float],
    params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Resample points along fitted 3D curve at specified step size."""
    # NOTE: This function's logic is derived from user provided structure.
    # Coordinates in this function are assumed to be in Angstroms.
    resampled_points = []
    accumulation = 0.0
    current_val = start
    step = params['intergration_step']
    sample_step = params['sample_step']

    def get_coords(val: float) -> np.ndarray:
        if mode == 1:  # y=f(x)
            x, y = val, poly_xy(val)
        else:  # x=f(y)
            y, x = val, poly_xy(val)
        k = poly_k(val)
        return np.array([x, y, k])

    # Initialize current_pos before the loop
    if current_val < end:
        current_pos = get_coords(current_val)
    else:
        return resampled_points

    while current_val < end:
        next_val = current_val + step
        
        # Clamp next_val to end if overshot due to step size
        if next_val > end:
            next_val = end
            
        next_pos = get_coords(next_val)
        
        # Ensure distance calculation uses the helper function
        dist = distance(current_pos, next_pos)
        accumulation += dist
        
        if accumulation >= sample_step:
            accumulation = 0.0
            
            # Calculate angles
            dx = next_pos[0] - current_pos[0]
            dy = next_pos[1] - current_pos[1]
            d_xy = np.sqrt(dx**2 + dy**2)
            
            # Angle in XY plane (rlnAngleYX)
            angle_yx = math.degrees(math.atan2(dy, dx))
            
            # Angle with respect to XY plane (rlnAngleZXY)
            dz = next_pos[2] - current_pos[2]
            angle_zxy = math.degrees(math.atan2(dz, d_xy)) if d_xy != 0 else 0

            # Coordinates are in Angstroms here
            point_data = {
                'rlnCoordinateX': current_pos[0],
                'rlnCoordinateY': current_pos[1],
                'rlnCoordinateZ': current_pos[2],
                'rlnAngleRot': 0.0,
                'rlnAngleTilt': angle_zxy + 90.0,
                'rlnAnglePsi': angle_yx,
                'rlnHelicalTubeID': cluster_id + 1,
                'rlnTomoName': tomo_name
            }
            
            if detector_pixel_size is not None:
                point_data['rlnDetectorPixelSize'] = detector_pixel_size
                
            resampled_points.append(point_data)

        current_pos = next_pos
        current_val = next_val

        # If we reached the end, break the loop
        if current_val >= end:
             break
             
    return resampled_points


# --- Core Logic Functions (Renamed/Modified) ---

def read_star_file(filename):
    """Read star file and return dataframe"""
    try:
        df = starfile.read(filename)
        if isinstance(df, dict):
            data_key = next(iter(df.keys()))
            df = df[data_key]
        
        required_cols = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ', 'rlnHelicalTubeID']
        for col in required_cols:
            if col not in df.columns:
                print(f"Error: Required column '{col}' not found in STAR file.")
                sys.exit(1)
                
        return df
    except Exception as e:
        print(f"Error reading star file: {e}")
        sys.exit(1)

def get_line_info(df, tube_id, angpix):
    """
    Get all coordinates and number of points for a tube, scaled by angpix 
    (i.e., coordinates converted to Angstroms).
    """
    tube_data = df[df['rlnHelicalTubeID'] == tube_id].copy()
    tube_data = tube_data.sort_index()
    # Scale coordinates by angpix to work in Angstroms for distance checks
    coords = tube_data[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values * angpix
    
    if len(coords) < 2:
        return None
    
    tomo_name = tube_data['rlnTomoName'].iloc[0] if 'rlnTomoName' in tube_data.columns and not tube_data['rlnTomoName'].empty else "Unknown"
    detector_pixel_size = tube_data['rlnDetectorPixelSize'].iloc[0] if 'rlnDetectorPixelSize' in tube_data.columns and not tube_data['rlnDetectorPixelSize'].empty else None

    return {
        'tube_id': tube_id,
        'coords': coords, # Coords in Angstroms
        'n_points': len(coords),
        'tomo_name': tomo_name,
        'detector_pixel_size': detector_pixel_size
    }

def fit_and_extrapolate(coords, min_seed, dist_extrapolate, poly_order_seed):
    """
    Fits a polynomial (poly_order_seed) to the last min_seed coordinates and 
    extrapolates forward by a total distance of dist_extrapolate.
    """
    N = len(coords)
    if N < poly_order_seed + 1 or min_seed < poly_order_seed + 1:
        return None
    
    # --- 1. Calculate average step size from seed points ---
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

    # --- 2. Fit and Extrapolate ---
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

# (check_extrapolation_overlap, check_connection_compatibility_extrapolate, find_line_connections, 
# and merge_connected_lines remain logically identical to the last version but use poly_order_seed)

def check_extrapolation_overlap(extrapolated_coords, target_coords, overlap_threshold):
    """
    Checks if the extrapolated path overlaps with the target segment's points.
    Returns: (overlap_found, min_distance)
    """
    if extrapolated_coords is None or len(target_coords) == 0:
        return False, float('inf')

    dist_matrix = np.linalg.norm(
        extrapolated_coords[:, None, :] - target_coords[None, :, :], axis=2
    )
    
    min_distance = np.min(dist_matrix)
    overlap_found = min_distance <= overlap_threshold
    
    return overlap_found, min_distance

def check_connection_compatibility_extrapolate(line1_info, line2_info, end1, end2, 
                                              overlap_threshold, min_seed, dist_extrapolate, poly_order_seed):
    """
    Check if line1 can connect to line2 via trajectory extrapolation.
    Uses poly_order_seed.
    Returns: (can_connect, min_distance, reverse1, reverse2, simple_end_distance)
    """
    coords1 = line1_info['coords']
    coords2 = line2_info['coords']
    
    # --- 0. Calculate Simple End-to-End Distance (for reporting) ---
    P1 = coords1[-1] if end1 == 'end' else coords1[0]
    P2 = coords2[0] if end2 == 'start' else coords2[-1]
    simple_end_distance = np.linalg.norm(P1 - P2)
    # -------------------------------------------------------------

    # 1. Determine which points to fit on Line 1 (source)
    if end1 == 'end':
        fit_coords1 = coords1
        reverse1 = False 
    else:
        fit_coords1 = np.flipud(coords1)
        reverse1 = True
    
    # 2. Extrapolate Line 1's trajectory
    extrapolated_coords = fit_and_extrapolate(fit_coords1, min_seed, dist_extrapolate, poly_order_seed)
    
    if extrapolated_coords is None:
        return False, float('inf'), False, False, simple_end_distance

    # 3. Determine target points on Line 2 (buffer is 2x seed points)
    target_buffer = min_seed * 2 
    
    if end2 == 'start':
        target_coords2 = coords2[:target_buffer] 
        reverse2 = False
    else:
        target_coords2 = np.flipud(coords2[-target_buffer:])
        reverse2 = True
    
    # 4. Check for overlap
    overlap_found, min_distance = check_extrapolation_overlap(
        extrapolated_coords, target_coords2, overlap_threshold
    )
    
    if not overlap_found:
        return False, min_distance, False, False, simple_end_distance

    return True, min_distance, reverse1, reverse2, simple_end_distance


def find_line_connections(df, angpix, overlap_thres, min_seed, dist_extrapolate, poly_order_seed):
    """
    Find all possible connections between lines using extrapolation check.
    Uses poly_order_seed.
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
        if tube_id1 not in line_info: continue
        
        for tube_id2 in tube_ids_list[i+1:]:
            if tube_id2 not in line_info: continue
            
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

def merge_connected_lines(df, connections):
    """
    Merge lines that should be connected into unified tubes using union-find.
    Returns: (df_merged, original_tube_id_mapping)
    """
    parent = {}
    
    def find(x):
        if x not in parent: parent[x] = x
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y: parent[root_y] = root_x
    
    for conn in connections:
        union(conn['tube_id1'], conn['tube_id2'])
    
    groups = {}
    for tube_id in df['rlnHelicalTubeID'].unique():
        root = find(tube_id)
        if root not in groups: groups[root] = []
        groups[root].append(tube_id)
    
    df_merged = df.copy()
    tube_id_mapping = {}
    new_tube_id = 1
    
    for _, tube_ids in groups.items():
        for tube_id in tube_ids:
            tube_id_mapping[tube_id] = new_tube_id
        new_tube_id += 1
    
    # Apply the mapping to group the tubes for the next iteration step.
    df_merged['rlnHelicalTubeID'] = df_merged['rlnHelicalTubeID'].map(tube_id_mapping)
    
    return df_merged, tube_id_mapping

def run_connection_step(df_input, current_dist_extrapolate, args, iteration):
    """Performs one iteration of connection finding and merging."""
    print(f"\n--- Running Iteration {iteration} ---")
    print(f"Extrapolation Distance: {current_dist_extrapolate:.2f} Angstroms")
    
    original_n_tubes = df_input['rlnHelicalTubeID'].nunique()
    
    connections = find_line_connections(
        df_input, args.angpix, args.overlap_thres, args.min_seed, 
        current_dist_extrapolate, args.poly_order_seed
    )
    
    if not connections:
        print(f"No connections found in Iteration {iteration}.")
        return df_input, [], original_n_tubes
    
    df_merged, _ = merge_connected_lines(df_input, connections)
    final_n_tubes = df_merged['rlnHelicalTubeID'].nunique()
    
    print(f"  Merged {original_n_tubes - final_n_tubes} pairs/groups.")
    print(f"  Tubes remaining: {final_n_tubes}")
    
    # Add iteration number to connection records
    for conn in connections:
        conn['iteration'] = iteration
        conn['dist_extrapolate_used'] = current_dist_extrapolate

    return df_merged, connections, original_n_tubes

def fit_and_resample_single_tube(
    tube_data: pd.DataFrame,
    poly_order: int,
    sample_step: float,
    angpix: float,
    current_tube_id: int
) -> List[Dict[str, Any]]:
    """
    Fits a polynomial to the coordinates of a merged tube and resamples them 
    using the provided 'resample' logic.
    """
    # NOTE: The input tube_data coordinates are still in pixels.
    
    # 1. Prepare data (convert to Angstroms for fitting/resampling)
    coords_angstrom = tube_data[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values * angpix
    
    N = len(coords_angstrom)
    if N < poly_order + 1:
        print(f"  Warning: Tube {current_tube_id} has {N} points, too few for poly order {poly_order}. Skipping refit.")
        return []

    # Get metadata for resampling function
    tomo_name = tube_data['rlnTomoName'].iloc[0] if 'rlnTomoName' in tube_data.columns and not tube_data['rlnTomoName'].empty else "Unknown"
    detector_pixel_size = tube_data['rlnDetectorPixelSize'].iloc[0] if 'rlnDetectorPixelSize' in tube_data.columns and not tube_data['rlnDetectorPixelSize'].empty else None

    # Determine independent variable (X or Y) by finding the axis with the largest range
    ranges = np.ptp(coords_angstrom, axis=0)
    
    if ranges[0] >= ranges[1]: # X is the major axis or equal -> fit Y=f(X), Z=f(X) (mode 1)
        independent_var = coords_angstrom[:, 0] # X
        poly_xy = np.poly1d(np.polyfit(independent_var, coords_angstrom[:, 1], poly_order)) # Y(X)
        poly_k = np.poly1d(np.polyfit(independent_var, coords_angstrom[:, 2], poly_order))  # Z(X)
        mode = 1
    else: # Y is the major axis -> fit X=f(Y), Z=f(Y) (mode 2)
        independent_var = coords_angstrom[:, 1] # Y
        poly_xy = np.poly1d(np.polyfit(independent_var, coords_angstrom[:, 0], poly_order)) # X(Y)
        poly_k = np.poly1d(np.polyfit(independent_var, coords_angstrom[:, 2], poly_order))  # Z(Y)
        mode = 2

    # Resampling limits
    start = independent_var.min()
    end = independent_var.max()
    
    # Define placeholder params needed by 'resample'
    params = {
        'sample_step': sample_step,      # User-defined resampling distance (Angstroms)
        'intergration_step': sample_step / 10.0 # Small step for arc length integration (Angstroms)
    }

    # Resample points (output coordinates are in Angstroms)
    resampled_points_angstrom = resample(
        poly_xy,
        poly_k,
        start,
        end,
        mode,
        current_tube_id - 1, # cluster_id is 0-indexed in the original resample function input
        tomo_name,
        detector_pixel_size,
        params
    )
    
    # 2. Convert resampled coordinates from Angstroms back to Pixels (MANDATORY REQUEST)
    for point in resampled_points_angstrom:
        point['rlnCoordinateX'] /= angpix
        point['rlnCoordinateY'] /= angpix
        point['rlnCoordinateZ'] /= angpix
        
    return resampled_points_angstrom


def refit_and_resample_tubes(df_input: pd.DataFrame, poly_order: int, sample_step: float, angpix: float) -> pd.DataFrame:
    """
    Renumber rlnHelicalTubeID consecutively, then refit and resample all tubes.
    """
    print(f"\n--- Post-Merging Refit/Resample Step ---")
    print(f"  Target Polynomial Order for Final Fit: {poly_order}")
    print(f"  Resampling Step Distance: {sample_step:.2f} Angstroms")

    # 1. Renumber the rlnHelicalTubeID consecutively starting from 1
    unique_ids = df_input['rlnHelicalTubeID'].unique()
    id_map = {old_id: new_id + 1 for new_id, old_id in enumerate(unique_ids)}
    df_renumbered = df_input.copy()
    df_renumbered['rlnHelicalTubeID'] = df_renumbered['rlnHelicalTubeID'].map(id_map)
    
    all_resampled_points = []
    
    # 2. Refitting and Resampling
    
    for old_id, new_id in id_map.items():
        # Select the points for the current renumbered tube
        tube_data = df_renumbered[df_renumbered['rlnHelicalTubeID'] == new_id]
        
        resampled_data = fit_and_resample_single_tube(
            tube_data,
            poly_order,
            sample_step,
            angpix,
            new_id
        )
        all_resampled_points.extend(resampled_data)

    if all_resampled_points:
        df_resampled = pd.DataFrame(all_resampled_points)
        print(f"  Successfully resampled {df_resampled['rlnHelicalTubeID'].nunique()} tubes into {len(df_resampled)} particles.")
        
        # Standardize STAR column order/types
        required_cols = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ', 
                         'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi', 
                         'rlnHelicalTubeID', 'rlnTomoName']
        if 'rlnDetectorPixelSize' in df_resampled.columns:
             required_cols.append('rlnDetectorPixelSize')
             
        # Filter and reorder columns
        for col in required_cols:
            if col not in df_resampled.columns:
                df_resampled[col] = 0.0 # Add missing columns

        return df_resampled[required_cols]
    else:
        print("  Resampling failed or resulted in no points. Returning original merged data (renumbered).")
        # Ensure that if refitting fails, we still return the renumbered, merged coordinates in pixels.
        df_renumbered['rlnCoordinateX'] = df_renumbered['rlnCoordinateX'] / angpix
        df_renumbered['rlnCoordinateY'] = df_renumbered['rlnCoordinateY'] / angpix
        df_renumbered['rlnCoordinateZ'] = df_renumbered['rlnCoordinateZ'] / angpix
        return df_renumbered

# --- Main Logic ---

def main():
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
    
    parser.add_argument('input_star', help='Input STAR file containing particle data with rlnHelicalTubeID (coordinates in pixels)')
    parser.add_argument('--angpix', type=float, required=True,
                        help='Pixel size (Angstroms/pixel)')
    
    # Extrapolation parameters
    parser.add_argument('--poly_order_seed', type=int, default=2, choices=[1, 2],
                        help='Polynomial degree for fitting the segment ends during connection search (default: 2)')
    parser.add_argument('--min_seed', type=int, default=5,
                        help='Minimum number of points required from the end of the line to use for the polynomial fit (default: 5)')
    parser.add_argument('--dist_extrapolate', type=float, required=True,
                        help='Initial total distance (Angstroms) to project the trajectory forward (defines max search distance for Iteration 1).')
    parser.add_argument('--overlap_thres', type=float, required=True,
                        help='Maximum allowed distance (Angstroms) for an overlap between an extrapolated point and a target point.')
    
    # Iteration parameters
    parser.add_argument('--iter', type=int, default=2,
                        help='Maximum number of iterative connection runs (default: 2)')
    parser.add_argument('--dist_iter_scale', type=float, default=1.5,
                        help='Scale factor for dist_extrapolate in subsequent iterations (Iter_N = dist_iter_scale * Iter_{N-1}) (default: 1.5)')
    
    # Final Refitting/Resampling parameters (NEW/MODIFIED)
    parser.add_argument('--poly_order', type=int, default=3,
                        help='Polynomial degree for final post-merging curve fitting (default: 3)')
    parser.add_argument('--sample_step', type=float, default=20.0,
                        help='Distance in Angstroms between resampled particles after final curve fitting (default: 20.0)')


    args = parser.parse_args()
    
    # Initial setup
    base = os.path.splitext(args.input_star)[0]
    df_current = read_star_file(args.input_star)
    original_n_tubes = df_current['rlnHelicalTubeID'].nunique()
    current_dist_extrapolate = args.dist_extrapolate
    all_connections = []
    
    print("="*80)
    print("HELICAL TUBE CONNECTION (ITERATIVE EXTRAPOLATION)")
    print("="*80)
    print(f"Input file: {args.input_star}")
    print(f"Original tubes: {original_n_tubes}")
    print(f"Max iterations: {args.iter}")
    print(f"Seed Poly Order: {args.poly_order_seed}")
    print(f"Final Poly Order: {args.poly_order}")
    print(f"Resampling Step: {args.sample_step:.2f} Angstroms")
    print("="*80)
    
    
    for i in range(1, args.iter + 1):
        df_merged, connections_found, n_tubes_before = run_connection_step(
            df_current, current_dist_extrapolate, args, i
        )

        all_connections.extend(connections_found)
        
        # Check for convergence or no change
        n_tubes_after = df_merged['rlnHelicalTubeID'].nunique()
        if n_tubes_after == n_tubes_before:
            print(f"\nConvergence reached (no new connections found in Iteration {i}). Stopping.")
            df_current = df_merged
            break
        
        # Prepare for next iteration
        df_current = df_merged
        
        # Scale the distance for the next iteration (only if more runs are scheduled)
        if i < args.iter:
            current_dist_extrapolate *= args.dist_iter_scale

    
    final_n_tubes_merged = df_current['rlnHelicalTubeID'].nunique()
    
    # --- Final Output ---
    
    if final_n_tubes_merged < original_n_tubes:
        # --- NEW STEP: Post-Merging Refit and Resample ---
        # Perform final processing on the fully merged tubes, converts coordinates to pixels
        df_current = refit_and_resample_tubes(df_current, args.poly_order, args.sample_step, args.angpix)
        
        output_star = f"{base}_final_merged_resampled.star"
        csv_output_file = f"{base}_all_connections.csv"
        
        print(f"\n{'='*80}")
        print(f"FINAL PROCESSING SUMMARY")
        print(f"{'='*80}")
        print(f"Original tubes: {original_n_tubes}")
        print(f"Tubes after merge and renumber: {df_current['rlnHelicalTubeID'].nunique()}")
        print(f"Total tubes merged: {original_n_tubes - final_n_tubes_merged} over {i} runs.")
        print(f"Final particle count after resampling: {len(df_current)}")
        
        # Save final STAR file
        starfile.write(df_current, output_star, overwrite=True)
        print(f"\nSaved final connected (and renumbered/resampled) tubes to '{output_star}'")
        
        # Save aggregated CSV
        connections_df = pd.DataFrame(all_connections)
        csv_data = connections_df[['iteration', 'tube_id1', 'tube_id2', 'simple_end_distance', 
                                   'min_overlap_dist', 'connection_type', 'dist_extrapolate_used']]
        csv_data = csv_data.sort_values(by=['iteration', 'min_overlap_dist'])
        
        # Handle case where connections_df might be empty if no connections were found
        if not connections_df.empty:
            csv_content = csv_data.to_csv(index=False, float_format='%.3f')
            with open(csv_output_file, 'w') as f:
                 f.write(csv_content)
            print(f"Saved detailed connection summary to '{csv_output_file}'")
        
    else:
        print("\n" + "="*80)
        print("NO CONNECTIONS FOUND AFTER ALL ITERATIONS")
        print("="*80)

if __name__ == "__main__":
    main()
