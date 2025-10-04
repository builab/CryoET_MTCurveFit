#!/usr/bin/env python
# coding: utf-8

"""
I/O utilities for STAR file processing.
@Builab 2025
"""

import os
import re
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import starfile


def validate_star_file(df: pd.DataFrame, file_path: str) -> bool:
    """
    Validate STAR file contains required columns.
    
    Args:
        df: DataFrame to validate.
        file_path: Path to file (for error messages).
    
    Returns:
        True if valid, False otherwise.
    """
    essential_columns = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']
    optional_columns = [
        'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi', 'rlnLCCmax',
        'rlnDetectorPixelSize', 'rlnMicrographName', 'rlnTomoName'
    ]
    
    # Check essential columns
    missing_essential = [col for col in essential_columns if col not in df.columns]
    
    if missing_essential:
        print(f"Error: {file_path} is missing essential columns: "
              f"{', '.join(missing_essential)}. Skipping file.")
        return False
    
    # Warn about optional columns
    missing_optional = [col for col in optional_columns if col not in df.columns]
    
    if missing_optional:
        print(f"Warning: {file_path} is missing optional columns: "
              f"{', '.join(missing_optional)}")
    
    return True


def load_coordinates(
    file_path: str,
    angpix: float
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[float]]:
    """
    Load coordinates from STAR file into NumPy array.
    
    Args:
        file_path: Path to STAR file.
        angpix: Default pixel size if not found in file.
    
    Returns:
        Tuple of (coordinates array, tomogram name, detector pixel size).
        Returns (None, None, None) if file is invalid.
    """
    if not file_path.endswith(".star"):
        raise ValueError(f"Unsupported file format: {file_path}. Only .star files supported.")
    
    df = starfile.read(file_path)
    if 'rlnCoordinateX' not in df and 'particles' in df:
        df = df['particles']
    
    if not validate_star_file(df, file_path):
        return None, None, None
    
    coords = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy(dtype=float)
    
    # Handle detector pixel size
    if 'rlnDetectorPixelSize' in df.columns:
        detector_pixel_size = df['rlnDetectorPixelSize'].iloc[0]
    else:
        detector_pixel_size = angpix
        print(f"  - rlnDetectorPixelSize not found, using --angpix: {angpix}")
    
    # Handle tomogram name (priority: rlnMicrographName > rlnTomoName > filename)
    tomo_name = None
    
    if 'rlnMicrographName' in df.columns:
        tomo_name = df['rlnMicrographName'].iloc[0]
    elif 'rlnTomoName' in df.columns:
        tomo_name = df['rlnTomoName'].iloc[0]
        if tomo_name.endswith('.tomostar'):
            tomo_name = tomo_name[:-9]
            print(f"  - Removed .tomostar extension from rlnTomoName: {tomo_name}")
    
    if tomo_name is None:
        match = re.match(r"^(.+?_\d{2,3})", os.path.basename(file_path))
        if match:
            tomo_name = match.group(1)
        else:
            tomo_name = os.path.splitext(os.path.basename(file_path))[0]

        print(f"  - No rlnMicrographName or rlnTomoName found, using modified filename: {tomo_name}")
    
    return coords, tomo_name, detector_pixel_size


def read_star(file_path: str) -> pd.DataFrame:
    """
    Read STAR file into DataFrame.
    
    Args:
        file_path: Path to STAR file.
    
    Returns:
        DataFrame containing STAR file data.
    """
    df = starfile.read(file_path)
    if 'rlnCoordinateX' not in df and 'particles' in df:
        df = df['particles']
    return df


def write_star(df: pd.DataFrame, file_path: str, overwrite: bool = True) -> None:
    """
    Write DataFrame to STAR file.
    
    Args:
        df: DataFrame to write.
        file_path: Output file path.
        overwrite: Whether to overwrite existing file.
    """
    starfile.write(df, file_path, overwrite=overwrite)

