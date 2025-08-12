#!/usr/bin/env python
# coding: utf-8

# mcurve_fitting_3D.py
# Check prefer input from pytom_tm
# If there is no _rlnAutopickFigureOfMerit, then not process
import sys
import os
import math
import argparse
import numpy as np
import pandas as pd
import starfile

# --- Utility and ANSI Color Class ---
class color:
    """A simple class for adding color to terminal output using ANSI codes."""
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def get_args():
    """Parse command-line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description=f"{color.BOLD}About this program:{color.END}\n"
                    "This script performs multi-curve fitting of 3D coordinates from STAR files.\n"
                    "It identifies filamentous structures, clusters the points, and generates resampled coordinates.\n"
                    "Outputs resampled STAR files only.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('files', nargs='*', help="Input coordinate files (.star only).")

    # General options
    gen_group = parser.add_argument_group('General options')
    gen_group.add_argument("--pixel_size_ang", type=float, default=14.00, help="Pixel size of the micrograph/coordinate, in Angstroms.")
    gen_group.add_argument("--sample_step_ang", type=float, default=82, help="Final sampling step for resampling, in Angstroms (e.g., length of a repeating unit).")
    gen_group.add_argument("--intergration_step_ang", type=float, default=1, help="Integration step for curve length calculation, in Angstroms. Smaller is more accurate but slower.")
    gen_group.add_argument("--poly_expon", type=int, default=3, help="Polynomial factor for curve growth and final resampling.")

    # Seed searching options
    seed_group = parser.add_argument_group('Options for seed searching and evaluation')
    seed_group.add_argument("--min_number_seed", type=int, default=5, help="Minimum number of points to form a valid seed.")
    seed_group.add_argument("--max_dis_to_line_ang", type=float, default=50, help="Max distance a point can be from the initial seed line, in Angstroms.")
    seed_group.add_argument("--min_dis_neighbor_seed_ang", type=float, default=60, help="Minimum distance between neighboring points in a seed, in Angstroms.")
    seed_group.add_argument("--max_dis_neighbor_seed_ang", type=float, default=320, help="Maximum distance between neighboring points in a seed, in Angstroms.")
    seed_group.add_argument("--poly_expon_seed", type=int, default=2, help="Polynomial factor for evaluating the seed's quality.")
    seed_group.add_argument("--max_seed_fitting_error", type=float, default=1.0, help="Maximum fitting error allowed for an initial seed to be considered valid.")
    seed_group.add_argument("--max_angle_change_per_4nm", type=float, default=0.5, help="Curvature restriction: max angle change per 4nm, in degrees. Use a large value to disable.")

    # Growth options
    growth_group = parser.add_argument_group('Options for seed growth')
    growth_group.add_argument("--max_dis_to_curve_ang", type=float, default=80, help="Max distance a point can be from the growing curve to be added, in Angstroms.")
    growth_group.add_argument("--min_dis_neighbor_curve_ang", type=float, default=60, help="Minimum distance between neighboring points during curve growth, in Angstroms.")
    growth_group.add_argument("--max_dis_neighbor_curve_ang", type=float, default=320, help="Maximum distance between neighboring points during curve growth, in Angstroms.")
    growth_group.add_argument("--min_number_growth", type=int, default=0, help="Minimum number of points that must be added during the growth phase.")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()

# (Helper functions like distance, etc. remain the same)
def distance(p1, p2):
    """
    Estimate distance in both 2D & 3D as long as input is numpy array
    """
    return np.linalg.norm(p1 - p2)

def find_seed(i, j, coords, assigned_clusters, params):
    """
    Tries to find an initial seed of points starting from two points (i and j).
    A seed is a line-like collection of unassigned points.
    """
    if assigned_clusters[i] != -1 or assigned_clusters[j] != -1:
        return []

    # Initial points for the seed
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

    # Check Z-slice distance for the initial pair
    if abs(k1 - k2) > params['max_distance_to_line']:
        return []
    
    k_avg = (k1 + k2) / 2
    
    unassigned_mask = (assigned_clusters == -1)
    unassigned_indices = np.where(unassigned_mask)[0]
    
    potential_points_mask = np.ones(len(coords), dtype=bool)
    potential_points_mask[seed_indices] = False

    while True:
        found_new_point = False
        
        # Vectorized distance calculation to the line
        candidate_coords = coords[unassigned_mask & potential_points_mask]
        dist_to_line = np.abs(a * candidate_coords[:, 0] + b * candidate_coords[:, 1] + c) / norm_factor
        delta_z = np.abs(candidate_coords[:, 2] - k_avg)
        
        # Filter points close to the line in XY and Z
        line_candidates_mask = (dist_to_line < params['max_distance_to_line']) & \
                               (delta_z < params['max_distance_to_line'])
        
        candidate_indices = unassigned_indices[potential_points_mask[unassigned_mask]][line_candidates_mask]

        for k in candidate_indices:
            # Check distance to existing points in the seed
            distances = [distance(coords[k, :2], coords[idx, :2]) for idx in seed_indices]
            min_dist = np.min(distances)

            if params['min_distance_in_extension_seed'] < min_dist < params['max_distance_in_extension_seed']:
                seed_indices.append(k)
                potential_points_mask[k] = False
                if len(seed_indices) >= params['min_number_seed']:
                    return seed_indices # Found a seed of required size
                
                # Restart search with the new point included
                found_new_point = True
                break # break from inner loop to re-evaluate candidates

        if not found_new_point:
            break # No more points can be added
            
    return seed_indices if len(seed_indices) >= params['min_number_seed'] else []


def angle_evaluate(poly, point, mode, params):
    """
    Evaluates the curvature of the polynomial fit.
    Returns 1 if curvature is acceptable, 0 otherwise.
    """
    evaluation_step = 40 / params['pixel_size_ang']
    step = params['intergration_step']
    
    # Mode 1: y = f(x), Mode 0: x = f(y)
    # The logic is the same, just swapping x and y
    def get_next_pos(val):
        return (val + step, poly(val + step)) if mode == 1 else (poly(val + step), val + step)

    def get_slope(p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        if abs(dx) < 1e-4: return np.inf
        return dy / dx
        
    # Find first point
    accumulation = 0
    current_pos = (point, poly(point)) if mode == 1 else (poly(point), point)
    
    # Calculate initial angle
    next_pos = get_next_pos(point)
    angle_one = math.degrees(math.atan(get_slope(current_pos, next_pos)))

    # Move along the curve for `evaluation_step`
    while accumulation < evaluation_step:
        next_pos = get_next_pos(current_pos[0] if mode == 1 else current_pos[1])
        accumulation += distance(np.array(current_pos), np.array(next_pos))
        current_pos = next_pos
    
    # Calculate final angle
    final_next_pos = get_next_pos(current_pos[0] if mode == 1 else current_pos[1])
    angle_two = math.degrees(math.atan(get_slope(current_pos, final_next_pos)))
    
    return 1 if abs(angle_two - angle_one) < params['max_angle_change_per_4nm'] else 0


def resample(poly_xy, poly_k, start, end, mode, cluster_id, tomo_name, detector_pixel_size, params):
    """
    Resamples points along the fitted 3D curve at a specified step size.
    """
    resampled_points = []
    accumulation = 0
    
    current_val = start
    step = params['intergration_step']

    # Mode 1: y=f(x), k=f(x); Mode 0: x=f(y), k=f(y)
    def get_coords(val):
        if mode == 1: # y=f(x)
            x = val
            y = poly_xy(x)
        else: # x=f(y)
            y = val
            x = poly_xy(y)
        k = poly_k(val)
        return np.array([x, y, k])

    current_pos = get_coords(current_val)

    while current_val < end:
        next_val = current_val + step
        next_pos = get_coords(next_val)
        
        dist = distance(current_pos, next_pos)
        accumulation += dist
        
        if accumulation >= params['sample_step']:
            accumulation = 0
            
            # Calculate angles
            d_xy = distance(current_pos[:2], next_pos[:2])
            
            # Angle in XY plane (rlnAngleYX)
            slope_xy = (next_pos[1] - current_pos[1]) / (next_pos[0] - current_pos[0]) if (next_pos[0] - current_pos[0]) != 0 else np.inf
            angle_yx = math.degrees(math.atan(slope_xy))
            
            # Angle with respect to XY plane (rlnAngleZXY)
            slope_k = (next_pos[2] - current_pos[2]) / d_xy if d_xy != 0 else np.inf
            angle_zxy = math.degrees(math.atan(slope_k))

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
            
            # Add detector pixel size if available
            if detector_pixel_size is not None:
                point_data['rlnDetectorPixelSize'] = detector_pixel_size
                
            resampled_points.append(point_data)

        current_pos = next_pos
        current_val = next_val
        
    return resampled_points


def seed_extension(seed_indices, coords, assigned_clusters, cluster_id, tomo_name, detector_pixel_size, params):
    """
    Extends a seed by iteratively fitting a polynomial and adding nearby points.
    Returns a tuple: (list of final cluster indices, list of resampled points)
    or ([], []) if the extension fails.
    """
    cluster_indices = list(seed_indices)
    
    # Determine fitting mode (x-dependent vs y-dependent)
    cluster_coords = coords[cluster_indices]
    delta_x = np.ptp(cluster_coords[:, 0])
    delta_y = np.ptp(cluster_coords[:, 1])
    mode = 1 if delta_x >= delta_y else 0 # 1: y=f(x), 0: x=f(y)

    # Independent and dependent variables for fitting
    # ind_vars: x or y coords, dep_vars_xy: y or x coords, dep_vars_k: z coords
    if mode == 1:
        ind_vars = cluster_coords[:, 0]
        dep_vars_xy = cluster_coords[:, 1]
    else:
        ind_vars = cluster_coords[:, 1]
        dep_vars_xy = cluster_coords[:, 0]
    dep_vars_k = cluster_coords[:, 2]

    # --- Seed Evaluation ---
    # 1. Fit seed points with a lower order polynomial
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
        
        # Re-fit polynomial with all current points in cluster
        all_cluster_coords = coords[cluster_indices]
        if mode == 1:
            ind_vars_all, dep_vars_xy_all = all_cluster_coords[:, 0], all_cluster_coords[:, 1]
        else:
            ind_vars_all, dep_vars_xy_all = all_cluster_coords[:, 1], all_cluster_coords[:, 0]
        dep_vars_k_all = all_cluster_coords[:, 2]

        poly_xy_growth = np.poly1d(np.polyfit(ind_vars_all, dep_vars_xy_all, params['poly_expon']))
        poly_k_growth = np.poly1d(np.polyfit(ind_vars_all, dep_vars_k_all, params['poly_expon']))

        for k in unassigned_indices:
            p_k = coords[k]
            
            # Select correct independent variable for evaluation
            ind_var_k = p_k[0] if mode == 1 else p_k[1]
            
            dist_to_curve_xy = abs(poly_xy_growth(ind_var_k) - (p_k[1] if mode == 1 else p_k[0]))
            dist_to_curve_k = abs(poly_k_growth(ind_var_k) - p_k[2])
            
            if dist_to_curve_xy < params['max_distance_to_curve'] and dist_to_curve_k < params['max_distance_to_curve']:
                min_dist_to_cluster = np.min([distance(p_k[:2], coords[idx, :2]) for idx in cluster_indices])
                
                if params['min_distance_in_extension'] < min_dist_to_cluster < params['max_distance_in_extension']:
                    cluster_indices.append(k)
                    # Mark as provisionally assigned to prevent being added to another curve in the same growth loop
                    assigned_clusters[k] = -2 
                    grew = True

        if not grew:
            break

    # --- Final Evaluation and Resampling ---
    if len(cluster_indices) - len(seed_indices) >= params['min_number_growth']:
        print(f"  - Seed extension successful. Cluster {cluster_id} found with {len(cluster_indices)} points.")
        final_coords = coords[cluster_indices]
        
        if mode == 1:
            ind, dep_xy, dep_k = final_coords[:, 0], final_coords[:, 1], final_coords[:, 2]
        else:
            ind, dep_xy, dep_k = final_coords[:, 1], final_coords[:, 0], final_coords[:, 2]
        
        poly_final_xy = np.poly1d(np.polyfit(ind, dep_xy, params['poly_expon']))
        poly_final_k = np.poly1d(np.polyfit(ind, dep_k, params['poly_expon']))
        
        resampled_data = resample(poly_final_xy, poly_final_k, np.min(ind), np.max(ind), mode, cluster_id, tomo_name, detector_pixel_size, params)
        return cluster_indices, resampled_data
    else:
        # Growth failed, revert provisional assignments
        for idx in cluster_indices:
            if assigned_clusters[idx] == -2:
                assigned_clusters[idx] = -1
        return [], []

# --- Core Computational Engine ---

def fit_curves(coords, tomo_name, detector_pixel_size, params, cluster_id_offset=0):
    """
    Core computational engine for curve fitting.
    This function is I/O-free.

    Args:
        coords (np.ndarray): A NumPy array of shape (N, 3) with X, Y, Z coordinates.
        tomo_name (str): The tomogram name to include in output.
        detector_pixel_size (float or None): The detector pixel size to include in output.
        params (dict): A dictionary of processing parameters.
        cluster_id_offset (int): An integer to offset the cluster IDs, useful for combining results.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Resampled points (_resam_Zscore).
            - np.ndarray: An array of cluster assignments for each input coordinate.
            - int: The total number of new clusters found.
    """
    total_number = len(coords)
    assigned_clusters = np.full(total_number, -1, dtype=int)
    cluster_id_counter = 0
    
    all_resampled_points = []

    # Main clustering loop
    for i in range(total_number):
        for j in range(i + 1, total_number):
            if assigned_clusters[i] != -1 or assigned_clusters[j] != -1:
                continue

            dist_ij = distance(coords[i, :2], coords[j, :2])
            
            if params['min_distance_in_extension_seed'] < dist_ij < params['max_distance_in_extension_seed']:
                seed_indices = find_seed(i, j, coords, assigned_clusters, params)
                
                if seed_indices:
                    current_cluster_id = cluster_id_counter + cluster_id_offset
                    final_indices, resampled_data = seed_extension(
                        seed_indices, coords, assigned_clusters, current_cluster_id, tomo_name, detector_pixel_size, params
                    )
                    
                    if final_indices:
                        for idx in final_indices:
                            assigned_clusters[idx] = current_cluster_id
                        all_resampled_points.extend(resampled_data)
                        cluster_id_counter += 1
    
    df_resam = pd.DataFrame(all_resampled_points)
    
    return df_resam, assigned_clusters, cluster_id_counter


# --- I/O and Orchestration Functions ---

def validate_star_file(df, file_path):
    """
    Validates that the STAR file contains required columns.
    Returns True if valid, False otherwise.
    """
    # Essential columns - must be present
    essential_columns = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']
    
    # Optional columns - show warning if missing
    optional_columns = [
        'rlnAngleRot',
        'rlnAngleTilt', 
        'rlnAnglePsi',
        'rlnLCCmax',
        'rlnDetectorPixelSize',
        'rlnMicrographName',
        'rlnTomoName'
    ]
    
    # Check essential columns
    missing_essential = []
    for col in essential_columns:
        if col not in df.columns:
            missing_essential.append(col)
    
    if missing_essential:
        print(f"{color.RED}Error: {file_path} is missing essential columns: {', '.join(missing_essential)}. Skipping file.{color.END}")
        return False
    
    # Check optional columns and warn
    missing_optional = []
    for col in optional_columns:
        if col not in df.columns:
            missing_optional.append(col)
    
    if missing_optional:
        print(f"{color.CYAN}Warning: {file_path} is missing optional columns: {', '.join(missing_optional)}{color.END}")
    
    return True

def load_coordinates(file_path, pixel_size_ang):
    """Loads coordinates from a .star file into a NumPy array after validation."""
    if not file_path.endswith(".star"):
        raise ValueError(f"Unsupported file format: {file_path}. Only .star files are supported.")
    
    df = starfile.read(file_path)
    if 'rlnCoordinateX' not in df and 'particles' in df:
        df = df['particles']
    
    # Validate required columns
    if not validate_star_file(df, file_path):
        return None, None, None
    
    coords = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy(dtype=float)
    
    # Handle detector pixel size
    detector_pixel_size = None
    if 'rlnDetectorPixelSize' in df.columns:
        detector_pixel_size = df['rlnDetectorPixelSize'].iloc[0]
    else:
        detector_pixel_size = pixel_size_ang
        print(f"  - rlnDetectorPixelSize not found, using --pixel_size_ang: {pixel_size_ang}")
    
    # Handle tomogram name
    tomo_name = None
    
    # Priority 1: Check for rlnMicrographName
    if 'rlnMicrographName' in df.columns:
        tomo_name = df['rlnMicrographName'].iloc[0]
    # Priority 2: Check for rlnTomoName
    elif 'rlnTomoName' in df.columns:
        tomo_name = df['rlnTomoName'].iloc[0]
        # Remove .tomostar extension if present
        if tomo_name.endswith('.tomostar'):
            tomo_name = tomo_name[:-9]  # Remove '.tomostar'
            print(f"  - Removed .tomostar extension from rlnTomoName: {tomo_name}")
    
    if tomo_name is None:
        # Fallback to filename without extension
        tomo_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"  - No rlnMicrographName or rlnTomoName found, using filename: {tomo_name}")
    
    return coords, tomo_name, detector_pixel_size

def write_outputs(base_name, df_resam, cluster_count):
    """
    Writes only the resampled STAR file with _init_fit suffix.
    """
    print(f"  - Found {cluster_count} clusters. Writing output file...")
    
    # Write only Resampled ZScore STAR file with new naming
    if not df_resam.empty:
        output_file = f"{base_name}_init_fit.star"
        starfile.write(df_resam, output_file, overwrite=True)
        print(f"  - Written: {output_file} [{len(df_resam)} points, {cluster_count} clusters]")
    else:
        print(f"  - No clusters found, no output file written.")

def process_file(file_path, params):
    """
    Orchestrates the entire process for a single file: load, fit, write.
    """
    print(f"{color.GREEN}Processing file: {file_path}{color.END}")
    base_name = os.path.splitext(file_path)[0]
    
    try:
        coords, tomo_name, detector_pixel_size = load_coordinates(file_path, params['pixel_size_ang'])
        if coords is None:
            print(f"  - Skipping {file_path} due to missing essential columns.")
            return
        print(f"  - Loaded {len(coords)} particles from {tomo_name} (pixel size: {detector_pixel_size}).")
    except Exception as e:
        print(f"{color.RED}Error loading {file_path}: {e}{color.END}")
        return

    # Call the core engine
    df_resam, assigned_clusters, cluster_count = fit_curves(coords, tomo_name, detector_pixel_size, params)
    
    # Write the results
    write_outputs(base_name, df_resam, cluster_count)

# --- Main Execution Block ---

def main():
    """Main function to run the script from the command line."""
    args = get_args() # Assume get_args() is defined as before
    
    # Convert Angstrom values to pixel values based on pixel size
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

    print(f"{color.BOLD}Starting curve fitting with the following parameters:{color.END}")
    for key, val in vars(args).items():
        if key != 'files':
            print(f"  --{key}: {val}")

    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"{color.RED}File not found: {file_path}{color.END}")
            continue
        process_file(file_path, params)

    print(f"\n{color.BOLD}Finished!{color.END}")
    print(f"{color.CYAN}<<<<< If you find this script useful, please acknowledge... >>>>>{color.END}")

if __name__ == "__main__":
    # This block only runs when the script is executed directly
    # It does not run when the script is imported
    main()