#!/usr/bin/env python
# coding: utf-8

"""
mcurve_fitting_3D.py
Multi-curve fitting of 3D coordinates from STAR files for filamentous structures.
"""

import sys
import os
import math
import argparse
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import starfile


class TerminalColors:
    """ANSI color codes for terminal output."""
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=f"{TerminalColors.BOLD}About this program:{TerminalColors.END}\n"
                    "This script performs multi-curve fitting of 3D coordinates from STAR files.\n"
                    "It identifies filamentous structures, clusters the points, and generates resampled coordinates.\n"
                    "Outputs resampled STAR files only.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('files', nargs='*', help="Input coordinate files (.star only).")

    # General options
    gen_group = parser.add_argument_group('General options')
    gen_group.add_argument("--pixel_size_ang", type=float, default=14.00,
                          help="Pixel size of the micrograph/coordinate, in Angstroms.")
    gen_group.add_argument("--sample_step_ang", type=float, default=82,
                          help="Final sampling step for resampling, in Angstroms.")
    gen_group.add_argument("--intergration_step_ang", type=float, default=1,
                          help="Integration step for curve length calculation, in Angstroms.")
    gen_group.add_argument("--poly_expon", type=int, default=3,
                          help="Polynomial factor for curve growth and final resampling.")

    # Seed searching options
    seed_group = parser.add_argument_group('Options for seed searching and evaluation')
    seed_group.add_argument("--min_number_seed", type=int, default=5,
                           help="Minimum number of points to form a valid seed.")
    seed_group.add_argument("--max_dis_to_line_ang", type=float, default=50,
                           help="Max distance from initial seed line, in Angstroms.")
    seed_group.add_argument("--min_dis_neighbor_seed_ang", type=float, default=60,
                           help="Min distance between neighboring seed points, in Angstroms.")
    seed_group.add_argument("--max_dis_neighbor_seed_ang", type=float, default=320,
                           help="Max distance between neighboring seed points, in Angstroms.")
    seed_group.add_argument("--poly_expon_seed", type=int, default=2,
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


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points (2D or 3D)."""
    return np.linalg.norm(p1 - p2)


def find_seed(
    i: int,
    j: int,
    coords: np.ndarray,
    assigned_clusters: np.ndarray,
    params: Dict[str, Any]
) -> List[int]:
    """
    Find an initial seed of collinear points starting from points i and j.
    
    Returns:
        List of indices forming a valid seed, or empty list if no valid seed found.
    """
    if assigned_clusters[i] != -1 or assigned_clusters[j] != -1:
        return []

    seed_indices = [i, j]
    p1, p2 = coords[i, :2], coords[j, :2]
    k1, k2 = coords[i, 2], coords[j, 2]

    # Line equation: ax + by + c = 0
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = -p1[0] * p2[1] + p2[0] * p1[1]
    norm_factor = np.sqrt(a**2 + b**2)
    
    if norm_factor == 0:
        return []

    # Check Z-slice distance for initial pair
    if abs(k1 - k2) > params['max_distance_to_line']:
        return []
    
    k_avg = (k1 + k2) / 2
    
    unassigned_mask = (assigned_clusters == -1)
    unassigned_indices = np.where(unassigned_mask)[0]
    
    potential_points_mask = np.ones(len(coords), dtype=bool)
    potential_points_mask[seed_indices] = False

    while True:
        found_new_point = False
        
        # Vectorized distance calculation
        candidate_mask = unassigned_mask & potential_points_mask
        candidate_coords = coords[candidate_mask]
        
        dist_to_line = np.abs(a * candidate_coords[:, 0] + 
                             b * candidate_coords[:, 1] + c) / norm_factor
        delta_z = np.abs(candidate_coords[:, 2] - k_avg)
        
        line_candidates_mask = ((dist_to_line < params['max_distance_to_line']) & 
                               (delta_z < params['max_distance_to_line']))
        
        candidate_indices = unassigned_indices[potential_points_mask[unassigned_mask]][line_candidates_mask]

        for k in candidate_indices:
            # Vectorized distance calculation to all seed points
            seed_coords = coords[seed_indices, :2]
            distances = np.linalg.norm(seed_coords - coords[k, :2], axis=1)
            min_dist = np.min(distances)

            if params['min_distance_in_extension_seed'] < min_dist < params['max_distance_in_extension_seed']:
                seed_indices.append(k)
                potential_points_mask[k] = False
                
                if len(seed_indices) >= params['min_number_seed']:
                    return seed_indices
                
                found_new_point = True
                break

        if not found_new_point:
            break
            
    return seed_indices if len(seed_indices) >= params['min_number_seed'] else []


def angle_evaluate(
    poly: np.poly1d,
    point: float,
    mode: int,
    params: Dict[str, Any]
) -> int:
    """
    Evaluate curvature of polynomial fit.
    
    Returns:
        1 if curvature is acceptable, 0 otherwise.
    """
    evaluation_step = 40 / params['pixel_size_ang']
    step = params['intergration_step']
    
    def get_next_pos(val: float) -> Tuple[float, float]:
        next_val = val + step
        return (next_val, poly(next_val)) if mode == 1 else (poly(next_val), next_val)

    def get_slope(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        return np.inf if abs(dx) < 1e-4 else dy / dx
    
    # Calculate initial position and angle
    current_pos = (point, poly(point)) if mode == 1 else (poly(point), point)
    next_pos = get_next_pos(point)
    angle_one = math.degrees(math.atan(get_slope(current_pos, next_pos)))

    # Move along curve for evaluation_step distance
    accumulation = 0.0
    while accumulation < evaluation_step:
        next_pos = get_next_pos(current_pos[0] if mode == 1 else current_pos[1])
        accumulation += distance(np.array(current_pos), np.array(next_pos))
        current_pos = next_pos
    
    # Calculate final angle
    final_next_pos = get_next_pos(current_pos[0] if mode == 1 else current_pos[1])
    angle_two = math.degrees(math.atan(get_slope(current_pos, final_next_pos)))
    
    return 1 if abs(angle_two - angle_one) < params['max_angle_change_per_4nm'] else 0


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
    resampled_points = []
    accumulation = 0.0
    current_val = start
    step = params['intergration_step']

    def get_coords(val: float) -> np.ndarray:
        if mode == 1:  # y=f(x)
            x, y = val, poly_xy(val)
        else:  # x=f(y)
            y, x = val, poly_xy(val)
        k = poly_k(val)
        return np.array([x, y, k])

    current_pos = get_coords(current_val)

    while current_val < end:
        next_val = current_val + step
        next_pos = get_coords(next_val)
        
        dist = distance(current_pos, next_pos)
        accumulation += dist
        
        if accumulation >= params['sample_step']:
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

            point_data = {
                'rlnCoordinateX': current_pos[0],
                'rlnCoordinateY': current_pos[1],
                'rlnCoordinateZ': current_pos[2],
                'rlnAngleRot': 0,
                'rlnAngleTilt': angle_zxy + 90,
                'rlnAnglePsi': angle_yx,
                'rlnHelicalTubeID': cluster_id,
                'rlnTomoName': tomo_name
            }
            
            if detector_pixel_size is not None:
                point_data['rlnDetectorPixelSize'] = detector_pixel_size
                
            resampled_points.append(point_data)

        current_pos = next_pos
        current_val = next_val
        
    return resampled_points


def seed_extension(
    seed_indices: List[int],
    coords: np.ndarray,
    assigned_clusters: np.ndarray,
    cluster_id: int,
    tomo_name: str,
    detector_pixel_size: Optional[float],
    params: Dict[str, Any]
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    Extend seed by iteratively fitting polynomial and adding nearby points.
    
    Returns:
        Tuple of (final cluster indices, resampled points) or ([], []) on failure.
    """
    cluster_indices = list(seed_indices)
    
    # Determine fitting mode
    cluster_coords = coords[cluster_indices]
    delta_x = np.ptp(cluster_coords[:, 0])
    delta_y = np.ptp(cluster_coords[:, 1])
    mode = 1 if delta_x >= delta_y else 0  # 1: y=f(x), 0: x=f(y)

    # Set up variables for fitting
    if mode == 1:
        ind_vars = cluster_coords[:, 0]
        dep_vars_xy = cluster_coords[:, 1]
    else:
        ind_vars = cluster_coords[:, 1]
        dep_vars_xy = cluster_coords[:, 0]
    dep_vars_k = cluster_coords[:, 2]

    # --- Seed Evaluation ---
    # 1. Fit with lower order polynomial
    poly_seed_xy = np.poly1d(np.polyfit(ind_vars, dep_vars_xy, params['poly_expon_seed']))
    
    # 2. Check fitting error
    errors = np.abs(poly_seed_xy(ind_vars) - dep_vars_xy)
    if np.any(errors >= params['seed_evaluation_constant']):
        return [], []

    # 3. Check curvature
    poly_xy_final = np.poly1d(np.polyfit(ind_vars, dep_vars_xy, params['poly_expon']))
    mid_point = (np.min(ind_vars) + np.max(ind_vars)) / 2
    if not angle_evaluate(poly_xy_final, mid_point, mode, params):
        return [], []

    # --- Curve Growth ---
    while True:
        grew = False
        unassigned_indices = np.where(assigned_clusters == -1)[0]
        
        if len(unassigned_indices) == 0:
            break
        
        # Re-fit polynomial with all current cluster points
        all_cluster_coords = coords[cluster_indices]
        if mode == 1:
            ind_vars_all = all_cluster_coords[:, 0]
            dep_vars_xy_all = all_cluster_coords[:, 1]
        else:
            ind_vars_all = all_cluster_coords[:, 1]
            dep_vars_xy_all = all_cluster_coords[:, 0]
        dep_vars_k_all = all_cluster_coords[:, 2]

        poly_xy_growth = np.poly1d(np.polyfit(ind_vars_all, dep_vars_xy_all, params['poly_expon']))
        poly_k_growth = np.poly1d(np.polyfit(ind_vars_all, dep_vars_k_all, params['poly_expon']))

        for k in unassigned_indices:
            p_k = coords[k]
            ind_var_k = p_k[0] if mode == 1 else p_k[1]
            
            dist_to_curve_xy = abs(poly_xy_growth(ind_var_k) - (p_k[1] if mode == 1 else p_k[0]))
            dist_to_curve_k = abs(poly_k_growth(ind_var_k) - p_k[2])
            
            if (dist_to_curve_xy < params['max_distance_to_curve'] and 
                dist_to_curve_k < params['max_distance_to_curve']):
                
                # Vectorized distance calculation
                cluster_coords_2d = coords[cluster_indices, :2]
                distances = np.linalg.norm(cluster_coords_2d - p_k[:2], axis=1)
                min_dist_to_cluster = np.min(distances)
                
                if (params['min_distance_in_extension'] < min_dist_to_cluster < 
                    params['max_distance_in_extension']):
                    cluster_indices.append(k)
                    assigned_clusters[k] = -2  # Provisional assignment
                    grew = True

        if not grew:
            break

    # --- Final Evaluation and Resampling ---
    if len(cluster_indices) - len(seed_indices) >= params['min_number_growth']:
        print(f"  - Seed extension successful. Cluster {cluster_id} found with "
              f"{len(cluster_indices)} points.")
        
        final_coords = coords[cluster_indices]
        
        if mode == 1:
            ind = final_coords[:, 0]
            dep_xy = final_coords[:, 1]
            dep_k = final_coords[:, 2]
        else:
            ind = final_coords[:, 1]
            dep_xy = final_coords[:, 0]
            dep_k = final_coords[:, 2]
        
        poly_final_xy = np.poly1d(np.polyfit(ind, dep_xy, params['poly_expon']))
        poly_final_k = np.poly1d(np.polyfit(ind, dep_k, params['poly_expon']))
        
        resampled_data = resample(
            poly_final_xy, poly_final_k, np.min(ind), np.max(ind),
            mode, cluster_id, tomo_name, detector_pixel_size, params
        )
        return cluster_indices, resampled_data
    else:
        # Revert provisional assignments
        for idx in cluster_indices:
            if assigned_clusters[idx] == -2:
                assigned_clusters[idx] = -1
        return [], []


def fit_curves(
    coords: np.ndarray,
    tomo_name: str,
    detector_pixel_size: Optional[float],
    params: Dict[str, Any],
    cluster_id_offset: int = 0
) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """
    Core computational engine for curve fitting (I/O-free).

    Args:
        coords: Array of shape (N, 3) with X, Y, Z coordinates.
        tomo_name: Tomogram name for output.
        detector_pixel_size: Detector pixel size for output.
        params: Processing parameters dictionary.
        cluster_id_offset: Offset for cluster IDs.

    Returns:
        Tuple of (resampled DataFrame, cluster assignments, cluster count).
    """
    total_number = len(coords)
    assigned_clusters = np.full(total_number, -1, dtype=int)
    cluster_id_counter = 0
    all_resampled_points = []

    # Main clustering loop
    for i in range(total_number):
        if assigned_clusters[i] != -1:
            continue
            
        for j in range(i + 1, total_number):
            if assigned_clusters[j] != -1:
                continue

            dist_ij = distance(coords[i, :2], coords[j, :2])
            
            if (params['min_distance_in_extension_seed'] < dist_ij < 
                params['max_distance_in_extension_seed']):
                
                seed_indices = find_seed(i, j, coords, assigned_clusters, params)
                
                if seed_indices:
                    current_cluster_id = cluster_id_counter + cluster_id_offset
                    final_indices, resampled_data = seed_extension(
                        seed_indices, coords, assigned_clusters,
                        current_cluster_id, tomo_name, detector_pixel_size, params
                    )
                    
                    if final_indices:
                        for idx in final_indices:
                            assigned_clusters[idx] = current_cluster_id
                        all_resampled_points.extend(resampled_data)
                        cluster_id_counter += 1
    
    df_resam = pd.DataFrame(all_resampled_points)
    return df_resam, assigned_clusters, cluster_id_counter


def validate_star_file(df: pd.DataFrame, file_path: str) -> bool:
    """
    Validate STAR file contains required columns.
    
    Returns:
        True if valid, False otherwise.
    """
    essential_columns = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']
    optional_columns = [
        'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi', 'rlnLCCmax',
        'rlnDetectorPixelSize', 'rlnMicrographName', 'rlnTomoName'
    ]
    
    # Check essential columns
    missing_essential = [col for col in essential_columns if col not in df.columns]
    
    if missing_essential:
        print(f"{TerminalColors.RED}Error: {file_path} is missing essential columns: "
              f"{', '.join(missing_essential)}. Skipping file.{TerminalColors.END}")
        return False
    
    # Warn about optional columns
    missing_optional = [col for col in optional_columns if col not in df.columns]
    
    if missing_optional:
        print(f"{TerminalColors.CYAN}Warning: {file_path} is missing optional columns: "
              f"{', '.join(missing_optional)}{TerminalColors.END}")
    
    return True


def load_coordinates(
    file_path: str,
    pixel_size_ang: float
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[float]]:
    """
    Load coordinates from STAR file into NumPy array.
    
    Returns:
        Tuple of (coordinates array, tomogram name, detector pixel size).
    """
    if not file_path.endswith(".star"):
        raise ValueError(f"Unsupported file format: {file_path}. Only .star files supported.")
    
    df = starfile.read(file_path)
    if 'rlnCoordinateX' not in df and 'particles' in df:
        df = df['particles']
    
    if not validate_star_file(df, file_path):
        return None, None, None
    
    coords = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy(dtype=float)
    
    # Handle detector pixel size
    if 'rlnDetectorPixelSize' in df.columns:
        detector_pixel_size = df['rlnDetectorPixelSize'].iloc[0]
    else:
        detector_pixel_size = pixel_size_ang
        print(f"  - rlnDetectorPixelSize not found, using --pixel_size_ang: {pixel_size_ang}")
    
    # Handle tomogram name (priority: rlnMicrographName > rlnTomoName > filename)
    tomo_name = None
    
    if 'rlnMicrographName' in df.columns:
        tomo_name = df['rlnMicrographName'].iloc[0]
    elif 'rlnTomoName' in df.columns:
        tomo_name = df['rlnTomoName'].iloc[0]
        if tomo_name.endswith('.tomostar'):
            tomo_name = tomo_name[:-9]
            print(f"  - Removed .tomostar extension from rlnTomoName: {tomo_name}")
    
    if tomo_name is None:
        tomo_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"  - No rlnMicrographName or rlnTomoName found, using filename: {tomo_name}")
    
    return coords, tomo_name, detector_pixel_size


def write_outputs(base_name: str, df_resam: pd.DataFrame, cluster_count: int) -> None:
    """Write resampled STAR file with _init_fit suffix."""
    if cluster_count == 0 or df_resam.empty:
        print("  - No clusters found, no output file written.")
        return
        
    print(f"  - Found {cluster_count} clusters. Writing output file...")
    
    output_file = f"{base_name}_init_fit.star"
    starfile.write(df_resam, output_file, overwrite=True)
    print(f"  - Written: {output_file} [{len(df_resam)} points, {cluster_count} clusters]")


def process_file(file_path: str, params: Dict[str, Any]) -> None:
    """Orchestrate the entire process for a single file: load, fit, write."""
    print(f"{TerminalColors.GREEN}Processing file: {file_path}{TerminalColors.END}")
    base_name = os.path.splitext(file_path)[0]
    
    try:
        coords, tomo_name, detector_pixel_size = load_coordinates(file_path, params['pixel_size_ang'])
        if coords is None:
            print(f"  - Skipping {file_path} due to missing essential columns.")
            return
        print(f"  - Loaded {len(coords)} particles from {tomo_name} "
              f"(pixel size: {detector_pixel_size}).")
    except Exception as e:
        print(f"{TerminalColors.RED}Error loading {file_path}: {e}{TerminalColors.END}")
        return

    # Call core engine
    df_resam, assigned_clusters, cluster_count = fit_curves(
        coords, tomo_name, detector_pixel_size, params
    )
    
    # Write results
    write_outputs(base_name, df_resam, cluster_count)


def main() -> None:
    """Main function to run script from command line."""
    args = get_args()
    
    # Convert Angstrom values to pixel values
    pixel_size = args.pixel_size_ang
    params = {
        "pixel_size_ang": pixel_size,
        "poly_expon": args.poly_expon,
        "sample_step": args.sample_step_ang / pixel_size,
        "intergration_step": args.intergration_step_ang / pixel_size,
        "min_number_seed": args.min_number_seed,
        "max_distance_to_line": args.max_dis_to_line_ang / pixel_size,
        "min_distance_in_extension_seed": args.min_dis_neighbor_seed_ang / pixel_size,
        "max_distance_in_extension_seed": args.max_dis_neighbor_seed_ang / pixel_size,
        "poly_expon_seed": args.poly_expon_seed,
        "seed_evaluation_constant": args.max_seed_fitting_error,
        "max_angle_change_per_4nm": args.max_angle_change_per_4nm,
        "max_distance_to_curve": args.max_dis_to_curve_ang / pixel_size,
        "min_distance_in_extension": args.min_dis_neighbor_curve_ang / pixel_size,
        "max_distance_in_extension": args.max_dis_neighbor_curve_ang / pixel_size,
        "min_number_growth": args.min_number_growth,
    }

    print(f"{TerminalColors.BOLD}Starting curve fitting with the following parameters:"
          f"{TerminalColors.END}")
    for key, val in vars(args).items():
        if key != 'files':
            print(f"  --{key}: {val}")

    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"{TerminalColors.RED}File not found: {file_path}{TerminalColors.END}")
            continue
        process_file(file_path, params)

    print(f"\n{TerminalColors.BOLD}Finished!{TerminalColors.END}")

if __name__ == "__main__":
    main()
