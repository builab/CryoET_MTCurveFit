#!/usr/bin/env python3
"""
Connect Broken Helical Tubes using Trajectory Extrapolation (Iterative)

Identifies and connects helical tubes that are broken into segments.
Uses polynomial extrapolation of one segment's end to check for overlap
with another segment's end/start. This is robust for curved lines.
The process is run iteratively with increasing search distance to connect large filaments.
"""

import argparse
import starfile
import pandas as pd
import numpy as np
import sys
import os

# --- Helper Functions ---
# (read_star_file, get_line_info, fit_and_extrapolate, check_extrapolation_overlap, 
#  check_connection_compatibility_extrapolate, merge_connected_lines are omitted for brevity 
#  but remain identical to the previous version.)

def read_star_file(filename):
    """Read star file and return dataframe"""
    try:
        # Check if the file contains a loop table (Micrograph/Particle/Tube data)
        df = starfile.read(filename)
        if isinstance(df, dict):
            # Try to find the data block, typically the largest one
            data_key = next(iter(df.keys()))
            df = df[data_key]
        
        # Ensure required columns are present
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
    Get all coordinates and number of points for a tube, scaled by angpix.
    """
    tube_data = df[df['rlnHelicalTubeID'] == tube_id].copy()
    tube_data = tube_data.sort_index()
    coords = tube_data[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values * angpix
    
    if len(coords) < 2:
        return None
    
    return {
        'tube_id': tube_id,
        'coords': coords,
        'n_points': len(coords)
    }

def fit_and_extrapolate(coords, min_seed, dist_extrapolate, poly_order):
    """
    Fits a polynomial to the last min_seed coordinates and extrapolates forward 
    by a total distance of dist_extrapolate.
    """
    N = len(coords)
    if N < poly_order + 1 or min_seed < poly_order + 1:
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
            p = np.polyfit(t_fit, y_fit, poly_order)
        except Exception:
            return None
            
        extrapolated_coords[:, i] = np.polyval(p, t_extrapolate)
        
    return extrapolated_coords

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
                                              overlap_threshold, min_seed, dist_extrapolate, poly_order):
    """
    Check if line1 can connect to line2 via trajectory extrapolation.
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
    extrapolated_coords = fit_and_extrapolate(fit_coords1, min_seed, dist_extrapolate, poly_order)
    
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


def find_line_connections(df, angpix, overlap_thres, min_seed, dist_extrapolate, poly_order):
    """
    Find all possible connections between lines using extrapolation check.
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
                            'min_overlap_dist': min_distance, # Renamed for clarity in connections list
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

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(
        description='Connect broken helical tubes using trajectory extrapolation (STAR file utility)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool runs iteratively to connect segments over increasing distances. The 
--dist_extrapolate is scaled by --dist_iter_scale for each subsequent iteration.

Examples:
  # Run 3 iterations, increasing search distance by 2.0x each time.
  %(prog)s input.star --angpix 14.0 --dist_extrapolate 100.0 --overlap_thres 10.0 --min_seed 5 --iter 3 --dist_iter_scale 2.0
        """
    )
    
    parser.add_argument('input_star', help='Input STAR file containing particle data with rlnHelicalTubeID')
    parser.add_argument('--angpix', type=float, required=True,
                        help='Pixel size (Angstroms/pixel)')
    
    # Extrapolation parameters
    parser.add_argument('--poly_order_seed', type=int, default=1, choices=[1, 2],
                        help='Polynomial degree for fitting (1=linear, 2=quadratic) (default: 1)')
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
    print(f"Distance scale factor: {args.dist_iter_scale}")
    print(f"Initial Extrapolation Distance: {current_dist_extrapolate:.2f} Angstroms")
    print(f"Overlap threshold: {args.overlap_thres} Angstroms")
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
            # Reset rlnHelicalTubeID to ensure a clean sequential ID set for the next run's pairing.
            # This is implicitly handled by the merge_connected_lines output, but good practice.
            # df_current is already the result of merge_connected_lines, so IDs are sequential.

    
    final_n_tubes = df_current['rlnHelicalTubeID'].nunique()
    
    # --- Final Output ---
    
    if final_n_tubes < original_n_tubes:
        output_star = f"{base}_final_merged.star"
        csv_output_file = f"{base}_all_connections.csv"
        
        print(f"\n{'='*80}")
        print(f"FINAL CONNECTION SUMMARY")
        print(f"{'='*80}")
        print(f"Original tubes: {original_n_tubes}")
        print(f"Final tubes: {final_n_tubes}")
        print(f"Total tubes merged: {original_n_tubes - final_n_tubes} over {i} runs.")
        
        # Save final STAR file
        starfile.write(df_current, output_star, overwrite=True)
        print(f"\nSaved final connected tubes to '{output_star}'")
        
        # Save aggregated CSV
        connections_df = pd.DataFrame(all_connections)
        csv_data = connections_df[['iteration', 'tube_id1', 'tube_id2', 'simple_end_distance', 
                                   'min_overlap_dist', 'connection_type', 'dist_extrapolate_used']]
        csv_data = csv_data.sort_values(by=['iteration', 'min_overlap_dist'])
        csv_content = csv_data.to_csv(index=False, float_format='%.3f')

        print(f"Saved detailed connection summary to '{csv_output_file}'")
        
    else:
        print("\n" + "="*80)
        print("NO CONNECTIONS FOUND AFTER ALL ITERATIONS")
        print("="*80)

if __name__ == "__main__":
    main()
