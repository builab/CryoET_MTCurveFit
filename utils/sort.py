#!/usr/bin/env python3
"""
SORT (cross-section → ellipse → reorder) **using existing angles, no polarity**

What this does (matches your clarified workflow):
1) **Find a cross section**: build a plane perpendicular to the filament bundle using
   the local direction of the *shortest* filament (geom logic).
   For each filament, pick the point closest to that plane.
2) **Rotate only that cross section** into the xyz frame using the per-row
   Rot/Tilt/Psi from the *processed* STAR (we don't derive angles; we trust them).
   Rotation order matches your geom convention: Rz(-Psi) → Rx(-Tilt) → Rz(-Rot).
   This yields a dummy cross section where Z is ~constant, so we can use (X,Y).
3) **Ellipse fit** on (X,Y) of the rotated cross section (geom.fit_ellipse),
   compute a parametric angle for each point (geom.angle_along_ellipse), then
   sort by that angle to obtain a consistent **rlnCrossSectionOrder** 1..N.
4) **Map order back** to the *original, unrotated* particles by filament-id.
   Nothing in the coordinates/angles is changed in the output — only the new
   `rlnCrossSectionOrder` column is added.

Usage:
  python sort.py \
    --input CCDC147C_001_particles_processed.star \
    --output CCDC147C_001_particles_sorted.star \
    [--filament-col rlnHelicalTubeID] \
    [--plot cs_ellipse.png]

No polarity handling, no shortest-distance reorder; ellipse-only as requested.
"""

import argparse
import os
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import starfile

# ------------------------
# Minimal geom functions
# (copied/adapted from geom.py to keep behavior consistent)
# ------------------------

def calculate_perpendicular_distance(point, plane_normal, reference_point):
    return np.abs(np.dot(plane_normal, point - reference_point)) / np.linalg.norm(plane_normal)


def calculate_normal_vector(filament_points: np.ndarray) -> np.ndarray:
    vectors = np.diff(filament_points, axis=0)
    normal_vector = np.sum(vectors, axis=0)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    if normal_vector[2] < 0:
        normal_vector = -normal_vector
    return normal_vector


def find_shortest_filament(data: pd.DataFrame) -> Tuple[int, np.ndarray]:
    shortest_length, shortest_midpoint, shortest_filament_id = float('inf'), None, None
    for filament_id, group in data.groupby('rlnHelicalTubeID'):
        filament_points = group[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
        min_point, max_point = filament_points.min(axis=0), filament_points.max(axis=0)
        length = np.linalg.norm(max_point - min_point)
        if length < shortest_length:
            shortest_length = length
            shortest_midpoint = (min_point + max_point) / 2
            shortest_filament_id = filament_id
    return shortest_filament_id, shortest_midpoint


def find_cross_section_points(data: pd.DataFrame, plane_normal: np.ndarray, reference_point: np.ndarray) -> pd.DataFrame:
    cross_section = []
    grouped = data.groupby('rlnHelicalTubeID')
    for filament_id, group in grouped:
        pts = group[['rlnCoordinateX','rlnCoordinateY','rlnCoordinateZ']].values
        dists = np.array([calculate_perpendicular_distance(p, plane_normal, reference_point) for p in pts])
        closest = group.iloc[np.argmin(dists)].copy()
        cross_section.append(closest)
    return pd.DataFrame(cross_section)

# ---- rotations (Z–X–Z) ----

def Rz(theta):
    c = np.cos(theta); s = np.sin(theta)
    return np.array([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]])

def Rx(theta):
    c = np.cos(theta); s = np.sin(theta)
    return np.array([[1.0,0.0,0.0],[0.0,c,-s],[0.0,s,c]])


def rotate_row_to_xyz(row) -> np.ndarray:
    v = np.array([row['rlnCoordinateX'], row['rlnCoordinateY'], row['rlnCoordinateZ']], float)
    psi = np.radians(row['rlnAnglePsi'])
    tilt = np.radians(row['rlnAngleTilt'])
    rot = np.radians(row['rlnAngleRot'])
    v1 = Rz(-psi) @ v
    v2 = Rx(-tilt) @ v1
    v3 = Rz(-rot) @ v2
    return v3

# ---- ellipse fit (from geom.fit_ellipse / angle_along_ellipse) ----

def fit_ellipse(x, y):
    # Converted from geom.fit_ellipse (Fitzgibbon-style)
    x = np.asarray(x).flatten(); y = np.asarray(y).flatten()
    x_mean, y_mean = x.mean(), y.mean()
    x0, y0 = x - x_mean, y - y_mean
    D = np.vstack([x0*x0, x0*y0, y0*y0, x0, y0, np.ones_like(x0)]).T
    S = D.T @ D
    C = np.zeros((6,6)); C[0,2] = C[2,0] = 2; C[1,1] = -1
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(S) @ C)
    a = eigvecs[:, np.argmax(np.real(eigvals))].real
    A,B,Cc,Dc,Ec,Fc = a
    phi = 0.5 * np.arctan2(B, A - Cc)
    cos_t, sin_t = np.cos(phi), np.sin(phi)
    Ap = A*cos_t**2 + B*cos_t*sin_t + Cc*sin_t**2
    Cp = A*sin_t**2 - B*cos_t*sin_t + Cc*cos_t**2
    Dp = Dc*cos_t + Ec*sin_t
    Ep = -Dc*sin_t + Ec*cos_t
    Xc = -Dp/(2*Ap); Yc = -Ep/(2*Cp)
    Fp = Fc + Ap*Xc**2 + Cp*Yc**2 + Dp*Xc + Ep*Yc
    a_len = np.sqrt(-Fp/Ap); b_len = np.sqrt(-Fp/Cp)
    return { 'X0': Xc + x_mean, 'Y0': Yc + y_mean, 'a': a_len, 'b': b_len, 'phi': phi }


def angle_along_ellipse(center, axes, angle, points):
    cos_angle = np.cos(-angle); sin_angle = np.sin(-angle)
    a,b = axes
    ts = []
    for (x,y) in points:
        x_trans = x - center[0]; y_trans = y - center[1]
        xr = x_trans * cos_angle - y_trans * sin_angle
        yr = x_trans * sin_angle + y_trans * cos_angle
        t = np.arctan2(yr / b, xr / a)
        ts.append(t + angle)
    return np.array(ts)

# ------------------------
# STAR helpers
# ------------------------

FILAMENT_CANDIDATES = ['rlnHelicalTubeID','rlnFilamentID','rlnMicrotubuleID','filament_id','mt_id']

def read_particles_table(path: str) -> Tuple[Dict, str, pd.DataFrame]:
    data = starfile.read(path)
    if isinstance(data, pd.DataFrame):
        return data, 'data_particles', data
    assert isinstance(data, dict)
    key = None
    for k,v in data.items():
        if isinstance(v, pd.DataFrame) and 'rlnCoordinateX' in v.columns:
            key = k; break
    if key is None:
        for k,v in data.items():
            if isinstance(v, pd.DataFrame):
                key = k; break
    if key is None:
        raise ValueError('No table found in STAR')
    return data, key, data[key]


def write_particles_table(all_tables: Dict, key: str, df_out: pd.DataFrame, out_path: str) -> None:
    if isinstance(all_tables, dict):
        all_tables[key] = df_out
        starfile.write(all_tables, out_path, overwrite=True)
    else:
        starfile.write(df_out, out_path, overwrite=True)

# ------------------------
# Optional plot
# ------------------------

def maybe_plot(xs, ys, labels, out_png: Optional[str]):
    if not out_png: return
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(xs, ys)
        for i,(x,y) in enumerate(zip(xs,ys)):
            ax.text(x, y, str(labels[i]), fontsize=9)
        ax.set_xlabel('X (Å)'); ax.set_ylabel('Y (Å)'); ax.set_aspect('equal','box')
        fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    except Exception as e:
        print(f"[warn] plot failed: {e}")

# ------------------------
# Core pipeline
# ------------------------

def main():
    ap = argparse.ArgumentParser(description='Cross-section ellipse sort using provided angles')
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--filament-col', default=None)
    ap.add_argument('--plot', default=None, help='Optional path to save a cross-section XY plot')
    args = ap.parse_args()

    tables, key, df = read_particles_table(args.input)

    # sanity columns
    need = ['rlnCoordinateX','rlnCoordinateY','rlnCoordinateZ','rlnAngleRot','rlnAngleTilt','rlnAnglePsi']
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f'Missing required columns: {miss}')

    # choose filament column
    fcol = args.filament_col
    if fcol is None:
        for c in FILAMENT_CANDIDATES:
            if c in df.columns:
                fcol = c; break
    if fcol is None:
        raise ValueError('Could not detect filament id column; pass --filament-col')

    # 1) Cross section points (plane ⟂ bundle)
    shortest_id, midpt = find_shortest_filament(df)
    dir_vec = calculate_normal_vector(df[df[fcol]==shortest_id][['rlnCoordinateX','rlnCoordinateY','rlnCoordinateZ']].values)
    cs = find_cross_section_points(df, dir_vec, midpt)

    # 2) Rotate cross section into xyz using per-row angles
    V = np.vstack([rotate_row_to_xyz(row) for _,row in cs.iterrows()])
    Xr, Yr, Zr = V[:,0], V[:,1], V[:,2]

    # 3) Ellipse fit and angle order
    e = fit_ellipse(Xr, Yr)
    center = (e['X0'], e['Y0']); axes = (e['a'], e['b']); phi = e['phi']
    thetas = angle_along_ellipse(center, axes, phi, list(zip(Xr, Yr)))
    order_idx = np.argsort(thetas)

    # make mapping filament-id -> order 1..N
    cs_ordered = cs.iloc[order_idx]
    mapping = {fid: i+1 for i, fid in enumerate(cs_ordered[fcol].tolist())}

    # 4) Map back to full df (unrotated)
    df_out = df.copy()
    df_out['rlnCrossSectionOrder'] = df_out[fcol].map(mapping).astype('Int64')

    # optional plot
    maybe_plot(Xr, Yr, [mapping[f] for f in cs[fcol]], args.plot)

    # write
    write_particles_table(tables, key, df_out, args.output)
    print(f"Wrote {args.output}\n - Kept original coords/angles\n - Added rlnCrossSectionOrder (ellipse-based) for {len(mapping)} filaments")

if __name__ == '__main__':
    main()
