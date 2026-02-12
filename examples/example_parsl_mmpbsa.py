from molecular_simulation.utils.parsl_settings import LocalCPUSettings
import parsl
from pathlib import Path
import polars as pl

@parsl.python_app
def run_mmpbsa(
    path: Path,
    n_target_residues: int = 307,
    salt_concentration: float = 0.15,
    grid_spacing: float = 0.5,
    solvent_dielectric: float = 80.0,
) -> dict:
    """Run a single MM-PBSA or MM-GBSA calculation and return results as a dict."""
    import dill as pickle
    from rust_simulation_tools import (
        read_prmtop,
        compute_binding_energy,
        PbParams,
        SaParams,
    )

    prmtop_path = path / 'system.prmtop'
    trajectory_path = path / 'prod.dcd'

    topo = read_prmtop(prmtop_path)
    receptor_residues = list(range(n_target_residues))
    n_protein_residues = topo.select('protein').n_residues
    binder_residues = list(range(n_target_residues, n_protein_residues))

    pb_params = PbParams(
        grid_spacing=grid_spacing,
        solvent_dielectric=solvent_dielectric,
        salt_concentration=salt_concentration,
    )
    sa_params = SaParams()

    result = compute_binding_energy(
        topo,
        trajectory_path=trajectory_path,
        receptor_residues=receptor_residues,
        ligand_residues=binder_residues,
        pb_params=pb_params,
        sa_params=sa_params,
        trajectory_format='dcd',
    )

    data = {
        "path": str(path),
        "n_frames": len(result.frames),
        "mean_delta_total": result.mean_delta_total,
        "std_delta_total": result.std_delta_total,
    }

    with open(path / 'deltaG.pkl', 'wb') as f:
        pickle.dump(data, f)

    return data

settings = {
    'nodes': 1,
    'max_workers_per_node': 100,
    'cores_per_worker': 2.0,
    'worker_init': 'export TMPDIR=/tmp',
}

config = LocalCPUSettings(**settings).config_factory(Path.cwd())
parsl.load(config)

paths = []
futures = [run_mmpbsa(path) for path in paths]
results = [future.result() for future in futures]

df = pl.DataFrame(results)
df.write_parquet('dG.parquet')
