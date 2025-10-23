"""
Star pipeline processing modules for filamentous structure analysis.
@Builab 2025
"""

# Core computational functions
from .fit import fit_curves

# I/O utilities
from .io import (
    validate_dataframe,
    load_coordinates,
    read_star,
    write_star,
    combine_star_files
)

from .clean import (
    clean_tubes,
    filter_short_tubes
)

from .connect import (
    connect_tubes
)

from .predict import (
    predict_angles
)

__all__ = [
    # Fit functions
    'fit_curves',
    # Clean functions
    'clean_tubes',
    'filter_short_tubes',
    #Connect functions
    'connect_tubes',
    # Predict functions
    'predict_angles',
    # I/O functions,
    'validate_dataframe',
    'load_coordinates',
    'read_star',
    'write_star',
]

__version__ = '1.0.0'
