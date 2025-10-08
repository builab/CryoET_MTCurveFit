"""
Star pipeline processing modules for filamentous structure analysis.
@Builab 2025
"""

# Core computational functions
from .fit import (
    distance,
    find_seed,
    angle_evaluate,
    resample,
    seed_extension,
    fit_curves
)

# I/O utilities
from .io import (
    validate_star_file,
    load_coordinates,
    read_star,
    write_star
)

from .clean import (
    get_line_bounding_boxes,
    boxes_overlap_with_margin,
    identify_line_pairs_to_compare,
    calculate_distance_shorter_to_longer,
    calculate_all_overlaps,
    identify_tubes_to_delete,
    remove_overlapping_tubes
)

from .connect import (
    find_line_connections,
    merge_connected_lines,
    refit_and_resample_tubes
)

from .view import (
    view_star_df
)

__all__ = [
    # Fit functions
    'distance',
    'find_seed',
    'angle_evaluate',
    'resample',
    'seed_extension',
    'fit_curves',
    # Clean functions
    'get_line_bounding_boxes',
    'boxes_overlap_with_margin',
    'identify_line_pairs_to_compare',
    'calculate_distance_shorter_to_longer',
    'calculate_all_overlaps',
    'identify_tubes_to_delete',
    'remove_overlapping_tubes',
    #Connect functions
    'find_line_connections',
    'merge_connected_lines',
    'refit_and_resample_tubes',
    # I/O functions,
    'validate_star_file',
    'load_coordinates',
    'read_star',
    'write_star',
    # View functions
    'view_star_df' 
]

__version__ = '1.0.0'
