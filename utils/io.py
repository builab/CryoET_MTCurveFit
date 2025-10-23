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
from pathlib import Path


def validate_dataframe(df: pd.DataFrame, required_columns: list = None) -> None:
    """
    Validate DataFrame is not None or empty, and has required columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    name : str
        Name of the DataFrame for error messages
    required_columns : list, optional
        List of required column names
    Returns:
        True if valid, False otherwise.        
    
    Raises
    ------
    ValueError
        If DataFrame is None, empty, or missing required columns
    """
    if df is None:
        raise ValueError(f"DataFrame is None")
        return False
        
    if df.empty:
        raise ValueError(f"DataFrame is empty")
        return False
        
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
            return False
            
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
    
    if not validate_dataframe(df, ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']):
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
    

def convert_to_relionwarp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert STAR file DataFrame to relionwarp format.
    
    Args:
        df: Input DataFrame with particle data.
    
    Returns:
        DataFrame in relionwarp format.
    """
    # Create output DataFrame with required columns
    output_df = pd.DataFrame()
    
    # Handle rlnTomoName - add .tomostar extension if not present
    if 'rlnTomoName' in df.columns:
        output_df['rlnTomoName'] = df['rlnTomoName'].apply(
            lambda x: x if x.endswith('.tomostar') else f"{x}.tomostar"
        )
    else:
        raise ValueError("Input STAR file missing rlnTomoName column")
    
    # Copy coordinate columns
    for col in ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']:
        if col in df.columns:
            output_df[col] = df[col]
        else:
            raise ValueError(f"Input STAR file missing {col} column")
    
    # Copy angle columns
    for col in ['rlnAngleTilt', 'rlnAnglePsi', 'rlnAngleRot']:
        if col in df.columns:
            output_df[col] = df[col]
        else:
            raise ValueError(f"Input STAR file missing {col} column")
    
    # Set origin columns to 0
    output_df['rlnOriginXAngst'] = 0.0
    output_df['rlnOriginYAngst'] = 0.0
    output_df['rlnOriginZAngst'] = 0.0
    
    # Copy rlnHelicalTubeID if present
    if 'rlnHelicalTubeID' in df.columns:
        output_df['rlnHelicalTubeID'] = df['rlnHelicalTubeID']
    
    # Handle pixel size: prefer rlnImagePixelSize, fallback to rlnDetectorPixelSize
    if 'rlnImagePixelSize' in df.columns:
        output_df['rlnImagePixelSize'] = df['rlnImagePixelSize']
    elif 'rlnDetectorPixelSize' in df.columns:
        output_df['rlnImagePixelSize'] = df['rlnDetectorPixelSize']
    else:
        raise ValueError("Input STAR file missing both rlnImagePixelSize and rlnDetectorPixelSize columns")
    
    return output_df


def combine_star_files(input_patterns: list, output_file: str) -> None:
    """
    Combine multiple STAR files into a single relionwarp file.
    
    Args:
        input_patterns: List of file paths or glob patterns to match input STAR files.
        output_file: Output file path for combined relionwarp STAR file.
    """
    # Collect all input files (expanding patterns if needed)
    input_files = []
    for pattern in input_patterns:
        # Check if it's an exact file path
        if Path(pattern).exists():
            input_files.append(pattern)
        else:
            # Try as a glob pattern
            matched_files = glob.glob(pattern)
            if matched_files:
                input_files.extend(matched_files)
            else:
                print(f"Warning: No files found for pattern: {pattern}")
    
    # Remove duplicates and sort
    input_files = sorted(set(input_files))
    
    if not input_files:
        print(f"Error: No input files found")
        sys.exit(1)
    
    print(f"Found {len(input_files)} files to process")
    
    # List to store all DataFrames
    all_dfs = []
    
    # Read and convert each file
    for file_path in input_files:
        print(f"Processing: {file_path}")
        try:
            df = read_star(file_path)
            
            # Validate required columns
            required_cols = [
                'rlnTomoName',
                'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ',
                'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi'
            ]
            validate_dataframe(df, required_cols)
            
            # Check for at least one pixel size column
            if 'rlnImagePixelSize' not in df.columns and 'rlnDetectorPixelSize' not in df.columns:
                raise ValueError("Missing both rlnImagePixelSize and rlnDetectorPixelSize columns")
            
            # Convert to relionwarp format
            converted_df = convert_to_relionwarp(df)
            all_dfs.append(converted_df)
            print(f"  - Added {len(converted_df)} particles")
            
        except Exception as e:
            print(f"  - Error processing {file_path}: {e}")
            print(f"  - Skipping this file...")
            continue
    
    if not all_dfs:
        print("Error: No valid STAR files were processed")
        sys.exit(1)
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal particles combined: {len(combined_df)}")
    
    # Write output file
    print(f"Writing output to: {output_file}")
    write_star(combined_df, output_file, overwrite=True)
    print("Done!")

