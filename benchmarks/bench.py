import MDAnalysis as mda
from MDAnalysis.analysis.align import AlignTraj
from MDAnalysis.tests.datafiles import PSF, DCD
import numpy as np
from rust_simulation_tools import kabsch_align
import time

u = mda.Universe(PSF, DCD)
selection_text = 'name CA'

# measure our implementation
start_time = time.perf_counter()
sel = u.select_atoms(selection_text).atoms.ix
pos = u.atoms.positions

traj = np.zeros((len(u.trajectory), len(u.atoms), 3))
for i, ts in enumerate(u.trajectory):
    traj[i] = u.atoms.positions

alg_start = time.perf_counter()
kabsch_align(traj, pos, sel)
alg_end = time.perf_counter()

print(f'Full scraping of coords + alg: {alg_end - start_time}')
print(f'Just algorithm: {alg_end - alg_start}')

# measure the MDA implementation
start_time = time.perf_counter()

AlignTraj(u, u, select=selection_text).run()

end_time = time.perf_counter()

print(f'MDAnalysis: {end_time - start_time}')
