import warnings
warnings.filterwarnings('ignore')

import MDAnalysis as mda
from MDAnalysis.analysis.align import AlignTraj
from MDAnalysis.tests.datafiles import PSF, DCD
import mdtraj as md
import numpy as np
from rust_simulation_tools import kabsch_align
import time

u = mda.Universe(PSF, DCD)
selection_text = 'name CA'

def benchmark_mda_rust():
    times = []
    alg_times = []
    for _ in range(100):
        start_time = time.perf_counter()
        sel = u.select_atoms(selection_text).indices
        pos = u.atoms.positions
        
        traj = np.zeros((len(u.trajectory), len(u.atoms), 3), dtype=np.float32)
        for i, ts in enumerate(u.trajectory):
            traj[i] = u.atoms.positions
        
        alg_start = time.perf_counter()
        kabsch_align(traj, pos, sel)
        alg_end = time.perf_counter()

        times.append(alg_end - start_time)
        alg_times.append(alg_end - alg_start)
    
    ret_str = f'Pure algorithm: {np.mean(alg_times)} +/- {np.std(alg_times)}\n'
    ret_str += f'Rust w. MDA: {np.mean(times)} +/- {np.std(times)}'
    return ret_str

def benchmark_mda():
    times = []
    for _ in range(100):
        start_time = time.perf_counter()

        AlignTraj(u, u, select=selection_text).run()

        end_time = time.perf_counter()
        
        times.append(end_time - start_time)

    return f'MDA time: {np.mean(times)} +/- {np.std(times)}'

def benchmark_mdt_rust():
    traj = md.load(DCD, top=PSF)
    indices = traj.topology.select('name CA')

    times = []
    for _ in range(100):
        start_time = time.perf_counter()
        kabsch_align(traj.xyz, traj.xyz[0], indices)
        end_time = time.perf_counter()

        times.append(end_time - start_time)

    return f'Rust w. MDTraj: {np.mean(times)} +/- {np.std(times)}'

def benchmark_mdt():
    traj = md.load(DCD, top=PSF)
    indices = traj.topology.select('name CA')

    times = []
    for _ in range(100):
        start_time = time.perf_counter()
        traj.superpose(traj, atom_indices=indices)
        end_time = time.perf_counter()

        times.append(end_time - start_time)

    return f'MDtraj time: {np.mean(times)} +/- {np.std(times)}'

if __name__ == '__main__':
    statement = ['--------------',
                 'Mean +/- std.',
                 '--------------']
    statement.append(benchmark_mda_rust())
    statement.append(benchmark_mdt_rust())
    statement.append(benchmark_mda())
    statement.append(benchmark_mdt())

    print('\n'.join(statement))
