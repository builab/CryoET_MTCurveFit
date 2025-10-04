#!/usr/bin/env python
# coding: utf-8
# ChimeraX version

"""
Run mcurve_fitting_3D.py within ChimeraX
@Builab 2025
"""
import sys, os

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add this directory to the Python path
sys.path.append(script_dir)

# Configuration: Change these settings as needed
TEMPDIR = "/tmp"
CLEANUP_TEMP_FILES = True  # Set to False to keep temporary files for debugging

# Add error handling and debugging
try:
    import mcurve_fitting_3D
    print("Successfully imported mcurve_fitting_3D")
except ImportError as e:
    print(f"Failed to import mcurve_fitting_3D: {e}")
    print(f"Python path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
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
        print("Usage: python mcurvefit.py <input_model_id> [voxelSize <value> minseed <value> sampleStep <value> poly <value>]")
        sys.exit(1)
    
    input_model_id = sys.argv[1]
    args = sys.argv[2:]  # Remaining optional arguments

    # Default values
    voxel_size = 14.0
    min_seed = 5
    sample_step = 82
    poly = 3

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
            i += 2
        else:
            print(f"Warning: Argument '{args[i]}' has no value, skipping.")
            i += 1

    return input_model_id, voxel_size, min_seed, sample_step, poly

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

def main():
    # Parse command line arguments
    input_model_id, voxel_size, min_seed, sample_step, poly= parse_arguments()
    
    print(f'Input model ID: {input_model_id}')
    print(f'Voxel size (Angstrom): {voxel_size}')
    print(f'Min seed: {min_seed}')
    print(f'Sample Step (Angstrom): {sample_step}')
    print(f'Polynominal Fitting: {poly}')
    
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

    def process_star_file(input_file, **kwargs):
        """
        Process a STAR file using the mcurve fitting algorithm.
        
        Args:
            input_file (str): Absolute path to input .star file
            **kwargs: Parameters for the algorithm
        """
        # Set default parameters
        default_params = {
            'pixel_size_ang': 14.0,
            'sample_step_ang': 82,
            'intergration_step_ang': 1,
            'poly_expon': 3,
            'min_number_seed': 5,
            'max_dis_to_line_ang': 50,
            'min_dis_neighbor_seed_ang': 60,
            'max_dis_neighbor_seed_ang': 320,
            'poly_expon_seed': 2,
            'max_seed_fitting_error': 1.0,
            'max_angle_change_per_4nm': 0.5,
            'max_dis_to_curve_ang': 80,
            'min_dis_neighbor_curve_ang': 60,
            'max_dis_neighbor_curve_ang': 320,
            'min_number_growth': 0
        }
        
        # Update with user parameters
        default_params.update(kwargs)
        
        # Convert to processing parameters
        pixel_size = default_params['pixel_size_ang']
        params = {
            "pixel_size_ang": pixel_size,
            "poly_expon": default_params['poly_expon'],
            "sample_step": default_params['sample_step_ang'] / pixel_size,
            "intergration_step": default_params['intergration_step_ang'] / pixel_size,
            "min_number_seed": default_params['min_number_seed'],
            "max_distance_to_line": default_params['max_dis_to_line_ang'] / pixel_size,
            "min_distance_in_extension_seed": default_params['min_dis_neighbor_seed_ang'] / pixel_size,
            "max_distance_in_extension_seed": default_params['max_dis_neighbor_seed_ang'] / pixel_size,
            "poly_expon_seed": default_params['poly_expon_seed'],
            "seed_evaluation_constant": default_params['max_seed_fitting_error'],
            "max_angle_change_per_4nm": default_params['max_angle_change_per_4nm'],
            "max_distance_to_curve": default_params['max_dis_to_curve_ang'] / pixel_size,
            "min_distance_in_extension": default_params['min_dis_neighbor_curve_ang'] / pixel_size,
            "max_distance_in_extension": default_params['max_dis_neighbor_curve_ang'] / pixel_size,
            "min_number_growth": default_params['min_number_growth'],
        }
        
        print(f"Processing: {input_file}")
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file {input_file} does not exist")
            return None
        
        try:
            # Load coordinates with error handling
            try:
                coords, tomo_name, detector_pixel_size = mcurve_fitting_3D.load_coordinates(
                    input_file, params['pixel_size_ang']
                )
            except Exception as e:    
                print(f"Error loading coordinates: {e}")
                return None
            
            if coords is None:
                print(f"Skipping {input_file} due to missing essential columns.")
                return None
                
            print(f"Loaded {len(coords)} particles from {tomo_name} (pixel size: {detector_pixel_size})")
            
            # Perform curve fitting with error handling
            try:
                df_resam, assigned_clusters, cluster_count = mcurve_fitting_3D.fit_curves(
                    coords, tomo_name, detector_pixel_size, params
                )
            except Exception as e:
                print(f"Error during curve fitting: {e}")
                return None
            
            # Write output with error handling
            try:
                # Use TEMPDIR for all outputs
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                
                # Generate output filename with _init_fit suffix
                output_base_name = f"{base_name}_init_fit.star"
                
                # Create full output path in TEMPDIR
                output_star_file = os.path.join(TEMPDIR, output_base_name)
                temp_files.append(output_star_file)  # Track for cleanup
                
                print(f"Writing output to: {output_star_file}")
                mcurve_fitting_3D.write_outputs(os.path.splitext(input_star_file)[0], df_resam, cluster_count)
                print(f"Successfully wrote outputs for {output_base_name} in {TEMPDIR}")
                
                return df_resam, assigned_clusters, cluster_count, output_star_file
                
            except Exception as e:
                print(f"Error writing outputs: {e}")
                return None
            
        except Exception as e:
            print(f"Unexpected error during processing: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Process the star file if we have one
    if input_star_file is None:
        print("No input star file available for processing")
        return
        
    try:
        result = process_star_file(
            input_star_file,
            pixel_size_ang=voxel_size,
            min_number_seed=min_seed
        )
        
        if result is None:
            print("Processing failed")
            cleanup_temp_files(temp_files)
            return
            
        df_resam, assigned_clusters, cluster_count, output_star_file = result
        print("Processing completed successfully")
        
        # Load results back into ChimeraX if available
        if CHIMERAX_AVAILABLE:
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
        else:
            # If not in ChimeraX, still offer to clean up
            cleanup_temp_files(temp_files)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        cleanup_temp_files(temp_files)
        return

# Execute main logic - this runs whether imported or executed directly
main()