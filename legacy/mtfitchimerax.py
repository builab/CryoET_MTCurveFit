#!/usr/bin/env python
# coding: utf-8

"""
ChimeraX command wrapper for the full helical tube processing pipeline:
1. Initial Curve Fitting (Clustering)
2. Cleaning (Overlap Removal)
3. Connection (Trajectory Extrapolation)

Somehow it is not the same when run command line and inside ChimeraX

Usage in ChimeraX:
runscript mt_fit_all_chimerax.py #1.2.1 voxelSize 14 sampleStep 82 [minSeed 6] [...]
where #1.2.1 is the particle coordinate model ID.

@Builab 2025
"""

import sys, os
import argparse
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

# Add parent directory to path to import utils modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add this directory to the Python path
sys.path.append(script_dir)

# Configuration: Change these settings as needed
TEMPDIR = "/tmp"
CLEANUP_TEMP_FILES = True  # Set to False to keep temporary files for debugging

# Import utils modules
try:
    from utils.fit import fit_curves
    from utils.connect import (
        find_line_connections,
        merge_connected_lines,
        refit_and_resample_tubes
    )
    from utils.clean import (
        calculate_all_overlaps,
        identify_tubes_to_delete,
        remove_overlapping_tubes
    )
    from utils.io import read_star, write_star, load_coordinates
except ImportError as e:
    print(f"Failed to import utils: {e}")
    sys.exit(1)

# Check if running in ChimeraX context
try:
    from chimerax.core.commands import run
    CHIMERAX_AVAILABLE = True
    print("ChimeraX environment detected")
except ImportError:
    print("Warning: Not running in ChimeraX environment")
    CHIMERAX_AVAILABLE = False
    
def idstr2tuple(model_id_string):
    model_id = tuple(int(i) for i in model_id_string.lstrip("#").split('.'))
    return model_id

def get_model_name(model_id):
# Script to get model name from model ID string like #1.2.1
    #print(model_id)
    for model in session.models:
        print(model.id)
        if model.id == model_id:
            print(f'Found model name {model.name}')
            return model.name
    print("ERROR: No model id found")
    return None

def parse_arguments():
    """Parse command line arguments for voxelSize, minseed, sampleStep, poly."""
    
    if len(sys.argv) < 2:
        print("Usage: python mtfitchimerax.py <input_model_id> [voxelSize <value> minseed <value> sampleStep <value> poly <value> cleanDistThres <value> distExtrapolate <value> overlapThres <value>]")
        sys.exit(1)
    
    input_model_id = sys.argv[1]
    args = sys.argv[2:]  # Remaining optional arguments

    # Default values
    voxel_size = 14.0
    min_seed = 5
    sample_step = 82
    poly = 3
    clean_dist_thres = 50
    dist_extrapolate = 2000
    overlap_thres = 80
    

    # Process arguments in pairs
    i = 0
    while i < len(args):
        key = args[i]
        if i + 1 < len(args):
            value = args[i + 1]
            if key == "voxelSize":
                try:
                    voxel_size = float(value)
                except ValueError:
                    print(f"Error: Invalid voxelSize '{value}'. Using default {voxel_size}")
            elif key == "minseed":
                try:
                    min_seed = int(value)
                except ValueError:
                    print(f"Error: Invalid minseed '{value}'. Using default {min_seed}")
            elif key == "sampleStep":
                try:
                    sample_step = float(value)
                except ValueError:
                    print(f"Error: Invalid sampleStep '{value}'. Using default {sample_step}")
            elif key == "poly":
                try:
                    poly = int(value)
                except ValueError:
                    print(f"Error: Invalid poly '{value}'. Using default {poly}")
            elif key == "cleanDistThres":
                try:
                    clean_dist_thres = int(value)
                except ValueError:
                    print(f"Error: Invalid cleanDistThres '{value}'. Using default {clean_dist_thres}")
            elif key == "distExtrapolate":
                try:
                    dist_extrapolate = int(value)
                except ValueError:
                    print(f"Error: Invalid distExtrapolate '{value}'. Using default {dist_extrapolate}")
            elif key == "overlapThres":
                try:
                    overlap_thres = int(value)
                except ValueError:
                    print(f"Error: Invalid overlapThres '{value}'. Using default {overlap_thres}")
            i += 2
        else:
            print(f"Warning: Argument '{args[i]}' has no value, skipping.")
            i += 1

    return input_model_id, voxel_size, min_seed, sample_step, poly, clean_dist_thres, dist_extrapolate, overlap_thres

def cleanup_temp_files(file_list):
    """Delete temporary files."""
    if not CLEANUP_TEMP_FILES:
        print("Cleanup disabled - temporary files retained")
        return
    
    for file_path in file_list:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️  Deleted temporary file: {file_path}")
        except Exception as e:
            print(f"Warning: Could not delete {file_path}: {e}")

def run_fitting(file_path: str, args: argparse.Namespace) -> pd.DataFrame:
    """STEP 1: Load raw coordinates, run curve fitting, and return resampled data."""
    base_name = os.path.splitext(file_path)[0]
    pixel_size = args.angpix
    
    try:
        # load_coordinates converts raw X,Y,Z from Angstroms to Pixels if needed.
        coords, tomo_name, detector_pixel_size = load_coordinates(file_path, pixel_size)
        if coords is None:
            raise ValueError("No coordinates loaded or file is invalid.")
        print(f"Loaded {len(coords)} raw particles from {tomo_name}.")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        sys.exit(1)

    # Convert Angstroms to Pixels for fit_curves call (as required by fit.py)
    # integration_step is set to 1.0 (pixel) based on the first user request.
    df_resam, assigned_clusters, cluster_count = fit_curves(
        coords=coords,
        tomo_name=tomo_name,
        angpix=pixel_size,
        poly_order=args.poly_order,
        sample_step=args.sample_step / pixel_size, # Angstroms to Pixels
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
        detector_pixel_size=detector_pixel_size,
        integration_step=1.0, # Defaulted to 1.0 pixel
    )
    
    # Save intermediate file
    if cluster_count == 0 or df_resam.empty:
        print("  - No clusters found after fitting. Pipeline stops.")
        return pd.DataFrame()

    print(f"Tubes fitted: {cluster_count}")
    print(f"Resampled particles: {len(df_resam)}")
    
    return df_resam
    
def run_cleaning(df_input: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """STEP 2: Run the overlap calculation and filtering step (mt_clean logic)."""
    
    print("\n" + "="*80)
    print("STEP 2: REMOVING OVERLAPPING LINES (CLEANING)")
    print("="*80)
    print(f"Distance threshold: {args.clean_dist_thres} Angstroms")
    
    original_n_tubes = df_input['rlnHelicalTubeID'].nunique()
    
    # Calculate overlaps
    overlap_results = calculate_all_overlaps(
        df=df_input,
        margin=args.clean_margin,
        angpix=args.angpix
    )
    
    if len(overlap_results) == 0:
        print("\nNo significant overlapping tube pairs found.")
        return df_input
    
    # Identify and remove tubes to delete
    tubes_to_delete = identify_tubes_to_delete(
        overlap_results=overlap_results,
        distance_threshold=args.clean_dist_thres
    )
    
    if len(tubes_to_delete) == 0:
        print(f"\nNo tubes found to delete (avg distance < {args.clean_dist_thres} Angstroms).")
        return df_input
        
    df_filtered = remove_overlapping_tubes(
        df=df_input,
        tubes_to_delete=tubes_to_delete
    )
    
    print(f"\n{'='*30} CLEANING SUMMARY {'='*30}")
    print(f"Original: {len(df_input)} particles from {original_n_tubes} tubes")
    print(f"Filtered: {len(df_filtered)} particles from {df_filtered['rlnHelicalTubeID'].nunique()} tubes")
    print(f"Removed: {len(df_input) - len(df_filtered)} particles from {len(tubes_to_delete)} tubes")
    print('='*80)
        
    return df_filtered


def run_connection(df_input: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """STEP 3: Run the iterative connection and final refitting step (mt_connect logic)."""
    
    print("\n" + "="*80)
    print("STEP 3: CONNECTING BROKEN LINES (EXTRAPOLATION)")
    print("="*80)
    
    df_current = df_input.copy()
    original_n_tubes = df_current['rlnHelicalTubeID'].nunique()
    current_dist_extrapolate = args.dist_extrapolate
    all_connections = []
    
    print(f"Tubes starting connection: {original_n_tubes}")
    print(f"Max iterations: {args.conn_iter}")
    
    # Run iterative connection process
    for i in range(1, args.conn_iter + 1):
        print(f"\n--- Running Iteration {i} ---")
        print(f"Extrapolation Distance: {current_dist_extrapolate:.2f} Angstroms")
        
        n_tubes_before = df_current['rlnHelicalTubeID'].nunique()
        
        connections = find_line_connections(
            df=df_current,
            angpix=args.angpix,
            overlap_thres=args.overlap_thres,
            min_seed=args.min_seed,
            dist_extrapolate=current_dist_extrapolate,
            poly_order_seed=args.poly_order_seed
        )
        
        if not connections:
            print(f"No connections found in Iteration {i}. Stopping iterations.")
            break
        
        df_merged, _ = merge_connected_lines(df_current, connections)
        n_tubes_after = df_merged['rlnHelicalTubeID'].nunique()
        
        print(f"  Merged {n_tubes_before - n_tubes_after} pairs/groups.")
        print(f"  Tubes remaining: {n_tubes_after}")
        
        # Add iteration info to connection records
        for conn in connections:
            conn['iteration'] = i
            conn['dist_extrapolate_used'] = current_dist_extrapolate
        all_connections.extend(connections)
        
        if n_tubes_after == n_tubes_before:
            print(f"Convergence reached (no new connections in Iteration {i}). Stopping.")
            df_current = df_merged
            break
        
        df_current = df_merged
        
        # Scale distance for next iteration
        if i < args.conn_iter:
            current_dist_extrapolate *= args.dist_iter_scale

    final_n_tubes_merged = df_current['rlnHelicalTubeID'].nunique()
    
    # Final output
    print(f"\n{'='*30} CONNECTION SUMMARY {'='*30}")
    if final_n_tubes_merged < original_n_tubes:
        # Post-merging refit and resample
        df_final = refit_and_resample_tubes(
            df_input=df_current,
            poly_order=args.poly_order,
            sample_step=args.sample_step,
            angpix=args.angpix
        )
        
        #csv_output_file = f"{base}_connections.csv"
        
        print(f"Tubes before connection: {original_n_tubes}")
        print(f"Tubes after merge: {final_n_tubes_merged}")
        print(f"Total tubes merged: {original_n_tubes - final_n_tubes_merged}")
        print(f"Final particle count after resampling: {len(df_final)}")
        
        # Save connection details to CSV
        #if all_connections:
        #    connections_df = pd.DataFrame(all_connections)
        #    csv_data = connections_df[['iteration', 'tube_id1', 'tube_id2', 
        #                              'simple_end_distance', 'min_overlap_dist', 
        #                              'connection_type', 'dist_extrapolate_used']]
        #    csv_data = csv_data.sort_values(by=['iteration', 'min_overlap_dist'])
        #    csv_data.to_csv(csv_output_file, index=False, float_format='%.3f')
        #    print(f"Saved connection details to '{csv_output_file}'")
            
        return df_final
        
    else:
        return df_current


def main():
    # Parse command line arguments
    input_model_id, voxel_size, min_seed, sample_step, poly, clean_dist_thres, dist_extrapolate, overlap_thres= parse_arguments()
    
    args = argparse.Namespace(
        # Initial Seed Search & Evaluation
        max_dis_to_line_ang=50,
        min_dis_neighbor_seed_ang=60,
        max_dis_neighbor_seed_ang=320,
        poly_order_seed=3, # Used for both Fit and Connect
        max_seed_fitting_error=1.0,
        max_angle_change_per_4nm=0.5,
        # Curve Growth
        max_dis_to_curve_ang=80,
        min_dis_neighbor_curve_ang=60,
        max_dis_neighbor_curve_ang=320,
        min_number_growth=0,
        # MT_CLEAN DEFAULTS
        clean_margin=500,
        # MT_CONNECT DEFAULTS
        conn_iter=2,
        dist_iter_scale=1.5,
        #Extra
        angpix=voxel_size,
        min_seed=min_seed,
        sample_step=sample_step,
        poly_order=poly,
        clean_dist_thres=clean_dist_thres,
        dist_extrapolate=dist_extrapolate,
        overlap_thres=overlap_thres
    )
        
    print(f'Input model ID: {input_model_id}')
    print(f'Voxel size (Angstrom): {voxel_size}')
    print(f'Min seed: {min_seed}')
    print(f'Sample Step (Angstrom): {sample_step}')
    print(f'Polynominal Fitting: {poly}')
    print(f'Clean Distance Threshold (Angstrom): {clean_dist_thres}')
    print(f'Distance Extrapolate (Angstrom): {dist_extrapolate}')
    print(f'Overlap Threshold for Connecting (Angstrom): {overlap_thres}')
    
    input_star_file = get_model_name(idstr2tuple(input_model_id))
    
    

    # Ensure TEMPDIR exists
    os.makedirs(TEMPDIR, exist_ok=True)
    print(f"Using temporary directory: {TEMPDIR}")
    
    # List to track temporary files for cleanup
    temp_files = []
    
    # Get the absolute path for the input star file in TEMPDIR
    if CHIMERAX_AVAILABLE:
        try:
            input_star_file = os.path.join(TEMPDIR, f'{input_star_file}')
            temp_files.append(input_star_file)  # Track for cleanup
            print(f'Saving particle list to {input_star_file}')
            run(session, f'save "{input_star_file}" partlist {input_model_id}')
            print(f"Successfully saved {input_star_file}")
        except Exception as e:
            print(f"Error running ChimeraX command: {e}")
            return
    else:
        print("Skipping ChimeraX save command - not in ChimeraX environment")
        # For testing outside ChimeraX, you could specify a test file here
        # input_star_file = "/path/to/your/test.star"



    # Process the star file if we have one
    if input_star_file is None:
        print("No input star file available for processing")
        return
        
    try:
        # --- 1. FITTING STEP ---
        df_fitted = run_fitting(input_star_file, args)

        # --- 2. CLEANING STEP ---
        df_cleaned = run_cleaning(df_fitted, args)
    
        # --- 3. CONNECTION STEP ---
        if not df_cleaned.empty:
            df_final = run_connection(df_cleaned, args)
        else:
            print("\nSkipping connection step: No particles remaining after cleaning.")
            df_final = df_cleaned
    
        # --- 4. FINAL OUTPUT ---
        base_name = os.path.splitext(os.path.basename(input_star_file))[0]
                
        # Generate output filename with _init_fit suffix
        output_base_name = f"{base_name}_fitted.star"
                
        # Create full output path in TEMPDIR
        output_star_file = os.path.join(TEMPDIR, output_base_name)
        write_star(df_final, output_star_file, overwrite=True)
        print(f"\nFINAL OUTPUT saved to '{output_star_file}'")
        
        try:
            if os.path.exists(output_star_file):
                model = run(session, f'open "{output_star_file}"')[0]
                model_id = f"#{model.id[0]}.{model.id[1]}.{model.id[2]}"
                print(f'Fitted star file loaded as {model_id}')
                    
                # Clean up temporary files after successful loading
                cleanup_temp_files(temp_files)
            else:
                print(f"Warning: Output file {output_star_file} not found")
        except Exception as e:
            print(f"Error loading results into ChimeraX: {e}")
            cleanup_temp_files(temp_files)

        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        cleanup_temp_files(temp_files)
        return

# Execute main logic - this runs whether imported or executed directly
main()