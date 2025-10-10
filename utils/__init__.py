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
    get_tube_bounding_boxes,
    boxes_overlap_with_margin,
    identify_tube_pairs_to_compare,
    calculate_distance_shorter_to_longer,
    calculate_all_overlaps,
    identify_tubes_to_delete,
    remove_overlapping_tubes,
    filter_short_tubes
)

from .connect import (
    find_tube_connections,
    merge_connected_tubes,
    refit_and_resample_all_tubes
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
    'find_tube_connections',
    'merge_connected_tubes',
    'refit_and_resample_all_tubes',
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
