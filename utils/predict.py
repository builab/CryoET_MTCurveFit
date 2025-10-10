#!/usr/bin/env python3
# predict_pipeline.py
# One-shot pipeline:
#   1) LCC filter (keep top 80% if available)
#   2) Map local-averaged angles from filtered source to template geometry
#   3) Per-filament median snap (Rot/Tilt/Psi) with max_delta=20°, min_pts=5
#
# Usage:
#   python predict_pipeline.py \
#     --particles CCDC147C_001_particles.star \
#     --template  CCDC147C_001_particles_fitted_cleaned_connected.star \
#     --output    CCDC147C_001_particles_predicted.star
#
# Optional:
#   --write-intermediates  (saves *_filtered80.star and *_mapped.star)

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree


COORDS = ["rlnCoordinateX","rlnCoordinateY","rlnCoordinateZ"]
ANGLES = ["rlnAngleRot","rlnAngleTilt","rlnAnglePsi"]


# ---------------- Step 1: LCC filter (keep top 80%) ----------------
    
def lcc_filter(df_input: pd.DataFrame,
               df_template: pd.DataFrame,
               angpix: float,
               neighbor_rad: float,
               keep_top: float = 80.0) -> pd.DataFrame:
    """
    Filter df_template particles based on LCC scores within neighborhood of df_input particles.
    
    For each rlnHelicalTubeID in df_input:
    1. Find all template particles within neighbor_rad distance of ANY input particle with that tube ID
    2. Sort those template particles by rlnLCCmax (descending)
    3. Keep only the top `keep_top` percent
    
    Parameters
    ----------
    df_input : pd.DataFrame
        Input dataframe with particles (must have COORDS and rlnHelicalTubeID)
    df_template : pd.DataFrame
        Template dataframe with particles (must have COORDS, rlnHelicalTubeID, rlnLCCmax)
    angpix : float
        Angstroms per pixel for converting radius
    neighbor_rad : float
        Neighborhood radius in Angstroms
    keep_top : float
        Percentage of top LCC particles to keep (0-100)
    
    Returns
    -------
    pd.DataFrame
        Filtered template dataframe containing only high-LCC neighbors
    """
    # Validate inputs
    for c in COORDS:
        if c not in df_input.columns:
            raise ValueError(f"Missing coordinate column {c} in df_input")
        if c not in df_template.columns:
            raise ValueError(f"Missing coordinate column {c} in df_template")
    
    if 'rlnHelicalTubeID' not in df_input.columns:
        raise ValueError("df_input must have 'rlnHelicalTubeID' column")
    if 'rlnLCCmax' not in df_template.columns:
        raise ValueError("df_template must have 'rlnLCCmax' column")
    
    if not 0 < keep_top <= 100:
        raise ValueError(f"keep_top must be between 0 and 100, got {keep_top}")
    
    # Convert radius to pixels
    rad_px = neighbor_rad / angpix
    
    # Get unique tube IDs from input
    tube_ids = df_input['rlnHelicalTubeID'].unique()
    
    # Build KDTree for all template particles (do this once)
    template_xyz = df_template[COORDS].to_numpy(float)
    tree = cKDTree(template_xyz)
    
    print(f"[LCC Filter] Processing {len(tube_ids)} tube IDs...")
    print(f"[LCC Filter] Radius: {neighbor_rad} Å ({rad_px:.2f} px), keeping top {keep_top}%")
    
    # Store indices of template particles to keep
    keep_indices = set()
    
    for tube_id in tube_ids:
        # Get input particles for this tube
        input_tube = df_input[df_input['rlnHelicalTubeID'] == tube_id]
        
        # Extract coordinates
        input_xyz = input_tube[COORDS].to_numpy(float)
        
        # Find all template particles within neighbor_rad of ANY input particle in this tube
        # query_ball_point returns list of neighbor indices for each input point
        neighbors_list = tree.query_ball_point(input_xyz, r=rad_px)
        
        # Flatten and get unique neighbor indices
        neighbor_indices = set()
        for neighbors in neighbors_list:
            neighbor_indices.update(neighbors)
        
        if len(neighbor_indices) == 0:
            continue
        
        # Get the template particles that are neighbors
        neighbor_mask = np.zeros(len(df_template), dtype=bool)
        neighbor_mask[list(neighbor_indices)] = True
        template_neighbors = df_template[neighbor_mask]
        
        # Sort by LCC (descending) and keep top percentage
        template_sorted = template_neighbors.sort_values('rlnLCCmax', ascending=False)
        n_keep = max(1, int(np.ceil(len(template_sorted) * keep_top / 100.0)))
        template_keep = template_sorted.iloc[:n_keep]
        
        # Add indices to keep set
        keep_indices.update(template_keep.index)
    
    # Filter template dataframe
    df_filtered = df_template.loc[list(keep_indices)].copy()
    
    
    return df_filtered
    

# ---------------- Utils: circular mean ----------------
def circ_mean_deg(vals, w=None):
    vals = np.asarray(vals, float)
    r = np.deg2rad(vals)
    if w is None:
        s = np.sin(r).mean(); c = np.cos(r).mean()
    else:
        w = np.asarray(w, float); w = w / (w.sum() + 1e-12)
        s = (np.sin(r)*w).sum(); c = (np.cos(r)*w).sum()
    return (np.rad2deg(np.arctan2(s, c)) + 360.0) % 360.0

def circ_diff(a, b):
    d = a - b
    return (d + 180) % 360 - 180

def circ_mean(vals):
    vals = np.radians(vals)
    s, c = np.sin(vals).mean(), np.cos(vals).mean()
    return np.degrees(np.arctan2(s, c)) % 360

# ---------------- Step 2: Map local-avg angles from template to input ----------------
def map_local_avg_angles(df_input: pd.DataFrame,
                         df_tpl: pd.DataFrame,
                         angpix: float = 14.0,
                         radiusA: float = 100.0,
                         k: int = 8,
                         weight_by_distance: bool = True) -> pd.DataFrame:
    """
    Map angles from df_tpl to df_input based on spatial proximity.
    
    For each particle in df_input, find neighbors in df_tpl within radiusA distance,
    and assign angles using circular mean (weighted by distance if specified).
    """
    # Check required columns
    for c in COORDS:
        if c not in df_input.columns or c not in df_tpl.columns:
            raise ValueError(f"Missing coordinate column {c} in input or template dataframe.")
    
    for a in ANGLES:
        if a not in df_tpl.columns:
            raise ValueError(f"Missing angle column {a} in template dataframe.")
    
    # Initialize angles in input if missing
    for a in ANGLES:
        if a not in df_input.columns:
            df_input[a] = 0.0
    
    # Extract coordinates
    input_xyz = df_input[COORDS].to_numpy(float)
    tpl_xyz = df_tpl[COORDS].to_numpy(float)
    rad_px = radiusA / angpix
    
    # Prepare output
    out = df_input.copy()
    for a in ANGLES:
        out[a] = 0.0
    
    fallbacks = {a: 0 for a in ANGLES}
    
    # Process each input particle
    for i, p in enumerate(input_xyz):
        diffs = tpl_xyz - p
        dist = np.linalg.norm(diffs, axis=1)
        order = np.argsort(dist)
        nn = order[:max(1, k)]
        d = dist[nn]
        
        # Check which neighbors are within radius
        within = d <= rad_px
        snapped = False
        
        if not np.any(within):
            # No neighbors in radius - use closest one
            within = np.zeros_like(d, bool)
            within[0] = True
            snapped = True
        
        nn = nn[within]
        d = d[within]
        
        # Compute weights
        w = 1.0 / (d + 1e-9) if (weight_by_distance and len(d) > 1) else None
        
        # Assign angles using circular mean
        for a in ANGLES:
            vals = df_tpl[a].astype(float).to_numpy()[nn]  # FIXED: use df_tpl instead of df_input
            out.iloc[i, out.columns.get_loc(a)] = circ_mean_deg(vals, w=w)  # More efficient indexing
            if snapped:
                fallbacks[a] += 1
    
    if any(v > 0 for v in fallbacks.values()):
        print(f"[Map] Fallbacks (no neighbors in radius): {fallbacks}")
    
    return out

# ---------------- Step 3: Filament median snap ----------------
def snap_by_filament_median(df: pd.DataFrame,
                            max_delta_deg: float = 20.0,
                            min_points_per_filament: int = 5) -> pd.DataFrame:
    if 'rlnHelicalTubeID' not in df.columns:
        print(f"[Snap] No rlnHelicalTubeID column; treating all as one filament.")
        df = df.copy()
        df['rlnHelicalTubeID'] = 1

    out = df.copy()
    counts = {a: 0 for a in ANGLES}
    for fid, grp in out.groupby('rlnHelicalTubeID'):
        if len(grp) < min_points_per_filament:
            continue
        for a in ANGLES:
            vals = grp[a].astype(float).to_numpy()
            med  = circ_mean(vals)
            delta = np.abs(circ_diff(vals, med))
            mask = delta > max_delta_deg
            if np.any(mask):
                out.loc[grp.index[mask], a] = med
                counts[a] += int(mask.sum())
    print(f"[Snap] max_delta={max_delta_deg}°, min_pts={min_points_per_filament} → {counts}")
    return out
