import MDAnalysis as mda
import mdtraj as md
import numpy as np
from rust_simulation_tools import unwrap_system
import subprocess
import time

TOP = '../data/unwrapping.pdb'
DCD = '../data/unwrapping.dcd'

def stats(times: list[float], label: str) -> str:
    return f'{label}: {np.mean(times)} +/- {np.std(times)}'

def rust_mda(n_trials: int=100) -> str:
    times = []
    for _ in range(100):
        start_time = time.perf_counter()
        u = mda.Universe(TOP, DCD)

        assignments = np.zeros(len(u.atoms), dtype=np.int64)
        for frag_id, fragment in enumerate(u.atoms.fragments):
            assignments[fragment.indices] = frag_id
        
        trajectory = np.zeros((len(u.trajectory), len(u.atoms), 3))
        box = np.zeros((len(u.trajectory), 3))
        for i, ts in enumerate(u.trajectory):
            trajectory[i] = ts.positions
            box[i] = ts.dimensions[:3]

        unwrap_system(trajectory, box, assignments)

        end_time = time.perf_counter()
        times.append(end_time - start_time)

    return stats(times, 'Rust (MDA)')

def rust_mdt(n_trials: int=100) -> str:
    times = []
    for _ in range(n_trials):
        start_time = time.perf_counter()
        traj = md.load(DCD, top=TOP)
        
        trajectory = traj.xyz
        box = traj.unitcell_lengths

        assignments = np.array([atom.residue.index for atom in traj.top.atoms], dtype=np.int64)
        
        unwrap_system(trajectory, box, assignments)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    return stats(times, 'Rust (MDtraj)')

def ambertools(n_trials: int=100) -> str:
    cpptraj_recipe = [f'parm {TOP}', f'trajin {DCD}', 'autoimage', 'unwrap']
    with open('cpptraj.in', 'w') as f:
        f.write('\n'.join(cpptraj_recipe))

    cmd = [f'cpptraj -i cpptraj.in']

    times = []
    for _ in range(n_trials):
        start_time = time.perf_counter()
        subprocess.run(cmd, shell=True)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    return stats(times, 'Ambertools')

if __name__ == '__main__':
    results = []
    results.append(rust_mda())
    results.append(rust_mdt())
    results.append(ambertools())

    print('\n'.join(results))
