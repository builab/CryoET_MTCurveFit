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
    validate_dataframe,
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
    remove_overlapping_tubes,
    filter_short_lines
)

from .connect import (
    find_line_connections,
    merge_connected_lines,
    refit_and_resample_tubes
)

from .predict import (
    snap_by_filament_median,
    map_local_avg_angles,
    lcc_filter
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
    'filter_short_lines',
    #Connect functions
    'find_line_connections',
    'merge_connected_lines',
    'refit_and_resample_tubes',
    # Predict functions
    'snap_by_filament_median',
    'map_local_avg_angles',
    'lcc_filter',
    # I/O functions,
    'validate_dataframe',
    'load_coordinates',
    'read_star',
    'write_star',
]

__version__ = '1.0.0'
