#!/usr/bin/env python
# coding: utf-8

"""
ChimeraX command wrapper for the full helical tube processing pipeline:
1. Initial Curve Fitting (Clustering)
2. Cleaning (Overlap Removal)
3. Connection (Trajectory Extrapolation)
4. Predict (Use Angle from Template)


Instead of doing internally, call mt_fit_simple.py to process. Make thing a lot simpler 
and avoid using too many duplicate scripts.

Usage in ChimeraX:
runscript mtfitchimerax.py #1.2.1 voxelSize 14 sampleStep 82 [minSeed 6] [...]
where #1.2.1 is the particle coordinate model ID.

@Builab 2025
"""

import sys, os
import argparse
import pandas as pd
import subprocess
import shlex

from typing import Dict, Any, Tuple, Optional, List

# Add parent directory to path to import utils modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

 
#PYTHON_EXECUTABLE="/opt/anaconda3/bin/python"
PYTHON_EXECUTABLE="/Applications/ChimeraX-1.10.1.app/Contents/bin/python3.11"

print(f"Using {PYTHON_EXECUTABLE}")

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add this directory to the Python path
sys.path.append(script_dir)

# Configuration: Change these settings as needed
TEMPDIR = "/tmp"
CLEANUP_TEMP_FILES = True  # Set to False to keep temporary files for debugging


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
        print("Usage: runscript mtfitchimerax.py <input_model_id> [voxelSize <value> minseed <value> sampleStep <value> poly <value> cleanDistThres <value> distExtrapolate <value> overlapThres <value> minPart <value>]")
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
    min_part_per_line = 5

    

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
            elif key == "minPart":
                try:
                    min_part_per_line = int(value)
                except ValueError:
                    print(f"Error: Invalid overlapThres '{value}'. Using default {overlap_thres}")
            elif key == "neighborRad":
                try:
                    neighbor_rad = int(value)
                except ValueError:
                    print(f"Error: Invalid overlapThres '{value}'. Using default {overlap_thres}")
            i += 2
        else:
            print(f"Warning: Argument '{args[i]}' has no value, skipping.")
            i += 1

    return input_model_id, voxel_size, min_seed, sample_step, poly, clean_dist_thres, dist_extrapolate, overlap_thres, min_part_per_line, neighbor_rad

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

def run_mtfit_simple(input_star_file, angpix, sample_step, min_seed, poly_order, clean_dist_thres, dist_extrapolate, overlap_thres, min_part_per_line, neighbor_rad):
    """
    Constructs and runs the mt_fit_simple.py command with specified parameters.
    """
    
    # 1. Define the base command and arguments as a list
    # It's best practice to pass the command and arguments as a list.
    command_list = [
        PYTHON_EXECUTABLE,
        f"{script_dir}/mt_fit.py",
        "pipeline",
        input_star_file,  # Positional argument
        f"--angpix", str(angpix),
        f"--sample_step", str(sample_step),
        f"--min_seed", str(min_seed),
        f"--poly_order", str(poly_order),
        f"--dist_thres", str(clean_dist_thres),
        f"--dist_extrapolate", str(dist_extrapolate),
        f"--overlap_thres", str(overlap_thres),
        f"--min_part_per_line", str(min_part_per_line),
        f"--neighbor_rad", str(neighbor_rad),
        f"--template", input_star_file,
    ]
    
    # 2. Execute the command
    print(f"Executing command: {' '.join(command_list)}")
    
    try:
        # We use check=True to raise a CalledProcessError if the command fails (returns non-zero exit code)
        result = subprocess.run(
            command_list, 
            check=True, 
            capture_output=True, 
            text=True  # Decodes stdout and stderr as text
        )
        
        print("\n--- Command Successful ---")
        #print("STDOUT:\n", result.stdout) # Uncomment to see output
        #print("STDERR:\n", result.stderr) # Uncomment to see errors
        print(f"Return Code: {result.returncode}")
        
        return result
        
    except subprocess.CalledProcessError as e:
        print("\n--- Command Failed ---")
        print(f"Error: {e}")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        return None
    except FileNotFoundError:
        print(f"\n--- Execution Error ---")
        print(f"Error: The program 'mt_fit_simple.py' was not found. Check your PATH or environment.")
        return None
        
def main():
    # Parse command line arguments
    input_model_id, voxel_size, min_seed, sample_step, poly, clean_dist_thres, dist_extrapolate, overlap_thres, min_part_per_line, neighbor_rad = parse_arguments()
    

    print(f'Input model ID: {input_model_id}')
    print(f'Voxel size (Angstrom): {voxel_size}')
    print(f'Minimum number of seed: {min_seed}')
    print(f'Sample Step (Angstrom): {sample_step}')
    print(f'Polynominal Fitting: {poly}')
    print(f'Clean Distance Threshold (Angstrom): {clean_dist_thres}')
    print(f'Distance Extrapolate (Angstrom): {dist_extrapolate}')
    print(f'Overlap Threshold for Connecting (Angstrom): {overlap_thres}')
    print(f'Minimum particles per line: {min_part_per_line}')
    print(f'Neighbor radius (Angstrom): {neighbor_rad}')


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

    # Process the star file if we have one
    if input_star_file is None:
        print("No input star file available for processing")
        return
        
    try:
        # --- 1. FITTING ---
        run_result = run_mtfit_simple(
            input_star_file, 
            voxel_size, 
            sample_step, 
            min_seed, 
            poly, 
            clean_dist_thres, 
            dist_extrapolate, 
            overlap_thres,
            min_part_per_line,
            neighbor_rad
        )
        # --- 4. FINAL OUTPUT ---
        base_name = os.path.splitext(os.path.basename(input_star_file))[0]
                                
        # Create full output path in TEMPDIR
        output_star_file = os.path.join(TEMPDIR, f"{base_name}_processed.star")
        
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