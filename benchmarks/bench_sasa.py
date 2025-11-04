#!/usr/bin/env python3
"""
Quick SASA comparison: Our implementation vs MDTraj

Simple script for quick validation.
"""

import mdtraj as md
import numpy as np
import polars as pl
from rust_simulation_tools import calculate_sasa_trajectory, get_radii_array
import time

top = '../data/topology.pdb'
dcd = '../data/trajectory.dcd'

traj = md.load(dcd, top=top)

def rust(traj):    
    # Get coordinates and radii
    coords = traj.xyz.astype(np.float64) * 10.0  # nm to Angstrom
    
    # Simple uniform radii (for quick test)
    atoms = [atom.element.symbol for atom in traj.topology.atoms]
    resids = np.array([atom.residue.index for atom in traj.topology.atoms], dtype=np.uint64)
    
    n_points = 960
    
    radii = get_radii_array(atoms)
    our_sasa = calculate_sasa_trajectory(coords.reshape(coords.shape[0], coords.shape[1] * 3), radii, resids, probe_radius=1.4, 
                                         n_sphere_points=n_points)
    
def python(traj):
    n_points = 960
    # MDTraj implementation
    mdtraj_sasa = md.shrake_rupley(traj, probe_radius=0.14, 
                                    n_sphere_points=n_points)
    
    

if __name__ == '__main__':
    rust_times = []
    python_times = []
    for _ in range(100):
        start = time.perf_counter()
        rust(traj)
        rust_times.append(time.perf_counter() - start)
        
        start = time.perf_counter()
        python(traj)
        python_times.append(time.perf_counter() - start)

    df = pl.DataFrame({'rust': rust_times, 'python': python_times})
    print(df.describe())
