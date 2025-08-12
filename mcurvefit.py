#!/usr/bin/env python
# coding: utf-8

# testing mcurve_fitting_3D.py for ChimeraX
import sys, os

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add this directory to the Python path
sys.path.append(script_dir)

import mcurve_fitting_modified

input_model_id = sys.argv[1]

print(f'{input_model_id}')

# This load to ChimeraX (still required interface)
#print('save tmp.star partlist %s' % input_model_id)
from chimerax.core.commands import run
run(session, 'save tmp.star partlist %s' % input_model_id)

original_argv = sys.argv.copy()
sys.argv = [
    'mcurve_fitting_modified.py',
    'tmp.star',
    '--pixel_size_ang', '1.0',
    '--sample_step_ang', '82',
    '--min_number_seed', '5'
]

try:
    mcurve_fitting_modified.main()
finally:
    sys.argv = original_argv  # Restore original argv
    
    
model = run(session, 'open tmp_init_fit.star')[0]
model_id = model.id

formatted_string = f"#{model_id[0]}.{model_id[1]}.{model_id[2]}"

print(f'Fitted star file loaded as {formatted_string}')
