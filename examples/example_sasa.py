"""
Example usage with MDAnalysis

This example shows how to use sasa_calculator with MDAnalysis
to calculate SASA for protein structures and trajectories.
"""

import numpy as np
import MDAnalysis as mda
from sasa_calculator import calculate_sasa, calculate_residue_sasa, calculate_total_sasa

# Standard Van der Waals radii (Angstroms)
VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80,
    'F': 1.47, 'CL': 1.75, 'BR': 1.85, 'I': 1.98,
}

def get_radii_from_universe(universe, selection='protein'):
    """
    Extract atomic radii from MDAnalysis Universe based on element names.
    
    Parameters
    ----------
    universe : MDAnalysis.Universe
        The molecular system
    selection : str
        MDAnalysis selection string (default: 'protein')
    
    Returns
    -------
    np.ndarray
        Array of atomic radii
    """
    atoms = universe.select_atoms(selection)
    radii = np.array([VDW_RADII.get(atom.element, 1.70) for atom in atoms])
    return radii


def calculate_sasa_mdanalysis(universe, selection='protein', probe_radius=1.4, 
                               n_sphere_points=960):
    """
    Calculate SASA for a selection in MDAnalysis Universe.
    
    Parameters
    ----------
    universe : MDAnalysis.Universe
        The molecular system
    selection : str
        MDAnalysis selection string (default: 'protein')
    probe_radius : float
        Probe radius in Angstroms (default: 1.4 for water)
    n_sphere_points : int
        Number of sphere points for accuracy (default: 960)
    
    Returns
    -------
    dict
        Dictionary with 'per_atom', 'per_residue', and 'total' SASA values
    """
    atoms = universe.select_atoms(selection)
    
    # Get coordinates (shape: n_atoms x 3)
    coordinates = atoms.positions.astype(np.float64)
    
    # Get radii
    radii = get_radii_from_universe(universe, selection)
    
    # Get residue indices
    residue_indices = np.array([atom.resindex for atom in atoms], dtype=np.uintp)
    
    # Calculate SASA
    result = calculate_sasa(
        coordinates, 
        radii, 
        residue_indices,
        probe_radius=probe_radius,
        n_sphere_points=n_sphere_points
    )
    
    return result


def calculate_sasa_trajectory(universe, selection='protein', probe_radius=1.4,
                              n_sphere_points=960, stride=1):
    """
    Calculate SASA for each frame in a trajectory.
    
    Parameters
    ----------
    universe : MDAnalysis.Universe
        The molecular system with trajectory
    selection : str
        MDAnalysis selection string (default: 'protein')
    probe_radius : float
        Probe radius in Angstroms (default: 1.4)
    n_sphere_points : int
        Number of sphere points (default: 960)
    stride : int
        Analyze every nth frame (default: 1)
    
    Returns
    -------
    dict
        Dictionary with 'total_sasa' array and 'per_residue_sasa' dict of arrays
    """
    atoms = universe.select_atoms(selection)
    radii = get_radii_from_universe(universe, selection)
    residue_indices = np.array([atom.resindex for atom in atoms], dtype=np.uintp)
    
    n_frames = len(universe.trajectory[::stride])
    total_sasa_array = np.zeros(n_frames)
    
    # Get number of residues
    n_residues = len(set(residue_indices))
    per_residue_sasa_array = {i: np.zeros(n_frames) for i in range(n_residues)}
    
    for frame_idx, ts in enumerate(universe.trajectory[::stride]):
        coordinates = atoms.positions.astype(np.float64)
        
        result = calculate_sasa(
            coordinates,
            radii,
            residue_indices,
            probe_radius=probe_radius,
            n_sphere_points=n_sphere_points
        )
        
        total_sasa_array[frame_idx] = result['total']
        
        for res_idx, sasa in result['per_residue'].items():
            per_residue_sasa_array[res_idx][frame_idx] = sasa
    
    return {
        'total_sasa': total_sasa_array,
        'per_residue_sasa': per_residue_sasa_array,
        'time': universe.trajectory.time[::stride]
    }


# Example usage
if __name__ == '__main__':
    # Load a PDB file
    u = mda.Universe('protein.pdb')
    
    print("="*60)
    print("SASA Calculation with MDAnalysis")
    print("="*60)
    
    # Calculate SASA for entire protein
    result = calculate_sasa_mdanalysis(u, selection='protein')
    
    print(f"\nTotal protein SASA: {result['total']:.2f} Ų")
    print(f"Number of atoms: {len(result['per_atom'])}")
    print(f"Number of residues: {len(result['per_residue'])}")
    
    # Show per-residue SASA for first 10 residues
    print("\nPer-residue SASA (first 10 residues):")
    print("-" * 40)
    for res_idx in sorted(result['per_residue'].keys())[:10]:
        sasa = result['per_residue'][res_idx]
        residue = u.residues[res_idx]
        print(f"  {residue.resname:3s} {residue.resid:4d}: {sasa:6.2f} Ų")
    
    # Calculate SASA for specific selection
    print("\n" + "="*60)
    print("SASA for backbone atoms only")
    print("="*60)
    
    backbone_result = calculate_sasa_mdanalysis(u, selection='backbone')
    print(f"Backbone SASA: {backbone_result['total']:.2f} Ų")
    
    # If trajectory is available
    if len(u.trajectory) > 1:
        print("\n" + "="*60)
        print("SASA Analysis over Trajectory")
        print("="*60)
        
        traj_result = calculate_sasa_trajectory(u, stride=10)
        
        print(f"Analyzed {len(traj_result['total_sasa'])} frames")
        print(f"Mean SASA: {np.mean(traj_result['total_sasa']):.2f} Ų")
        print(f"Std SASA:  {np.std(traj_result['total_sasa']):.2f} Ų")
        print(f"Min SASA:  {np.min(traj_result['total_sasa']):.2f} Ų")
        print(f"Max SASA:  {np.max(traj_result['total_sasa']):.2f} Ų")
