# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.commands import CmdDesc      # Command description
from chimerax.atomic import AtomsArg            # Collection of atoms argument
from chimerax.core.commands import BoolArg      # Boolean argument
from chimerax.core.commands import ColorArg     # Color argument
from chimerax.core.commands import IntArg       # Integer argument
from chimerax.core.commands import EmptyArg     # (see below)
from chimerax.core.commands import Or, Bounded  # Argument modifiers

import sys, os
import argparse
import pandas as pd
import subprocess
import shlex

from typing import Dict, Any, Tuple, Optional, List

# Add parent directory to path to import utils modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ==========================================================================
# Functions and descriptions for registering using ChimeraX bundle API
# ==========================================================================

def mtfit(session, input_model_id, voxel_size,
         min_seed,
         sample_step,
         poly=3,
         clean_dist_thres=50,
         dist_extrapolate=2000,
         overlap_thres=80):
    """Fit MT line"""
    input_star_file = get_model_name(idstr2tuple(input_model_id))
        
    # Ensure TEMPDIR exists
    os.makedirs(TEMPDIR, exist_ok=True)
    print(f"Using temporary directory: {TEMPDIR}")
    
    # List to track temporary files for cleanup
    temp_files = []
    
    # Get the absolute path for the input star file in TEMPDIR
    input_star_file = os.path.join(TEMPDIR, f'{input_star_file}')
    temp_files.append(input_star_file)  # Track for cleanup
    print(f'Saving particle list to {input_star_file}')
    run(session, f'save "{input_star_file}" partlist {input_model_id}')
    print(f"Successfully saved {input_star_file}")

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
            overlap_thres
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

#mtfit_desc = CmdDesc(required=[("atoms", Or(AtomsArg, EmptyArg),
#					("voxelSize", VoxelSizeArg),
#					("sampleStep", SampleStepArg)])

# ==========================================================================
# Functions intended only for internal use by bundle
# ==========================================================================

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

def run_mtfit_simple(input_star_file, angpix, sample_step, min_seed, poly_order, clean_dist_thres, dist_extrapolate, overlap_thres):
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
        f"--overlap_thres", str(overlap_thres)
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

    return