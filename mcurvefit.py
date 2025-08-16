#!/usr/bin/env python
# coding: utf-8
# First working version
# ChimeraX version

# testing mcurve_fitting_3D.py for ChimeraX
import sys, os

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add this directory to the Python path
sys.path.append(script_dir)

# Add error handling and debugging
try:
    import mcurve_fitting_modified
    print("Successfully imported mcurve_fitting_modified")
except ImportError as e:
    print(f"Failed to import mcurve_fitting_modified: {e}")
    print(f"Python path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
    sys.exit(1)

# Configuration: Change this path as needed
TEMPDIR = "/tmp"

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

def main():
    if len(sys.argv) < 2:
        print("Usage: python mcurvefit_simple.py <input_model_id>")
        sys.exit(1)
    
    input_model_id = sys.argv[1]
    print(f'Input model ID: {input_model_id}')

    input_star_file = get_model_name(idstr2tuple(input_model_id))

    # Ensure TEMPDIR exists
    os.makedirs(TEMPDIR, exist_ok=True)
    print(f"Using temporary directory: {TEMPDIR}")
    
    # Get the absolute path for the input star file in TEMPDIR
    if CHIMERAX_AVAILABLE:
        try:
            input_star_file = os.path.join(TEMPDIR, f'{input_star_file}')
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
                coords, tomo_name, detector_pixel_size = mcurve_fitting_modified.load_coordinates(
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
                df_resam, assigned_clusters, cluster_count = mcurve_fitting_modified.fit_curves(
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
                
                print(f"Writing output to: {output_star_file}")
                mcurve_fitting_modified.write_outputs(os.path.splitext(input_star_file)[0], df_resam, cluster_count)
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
            pixel_size_ang=1.0,
            min_number_seed=5
        )
        
        if result is None:
            print("Processing failed")
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
                else:
                    print(f"Warning: Output file {output_star_file} not found")
            except Exception as e:
                print(f"Error loading results into ChimeraX: {e}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return

# Execute main logic - this runs whether imported or executed directly
main()
