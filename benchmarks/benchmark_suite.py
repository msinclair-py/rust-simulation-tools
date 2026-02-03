#!/usr/bin/env python3
"""
Comprehensive benchmark suite for rust-simulation-tools.

Uses proper statistical benchmarking with warmup, outlier detection,
and configurable parameters. Results are reported with confidence intervals.

Usage:
    python benchmark_suite.py                    # Run all benchmarks
    python benchmark_suite.py --filter kabsch   # Run only kabsch benchmarks
    python benchmark_suite.py --export results.json  # Export results
"""

import argparse
import gc
import json
import statistics
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

warnings.filterwarnings("ignore")


@dataclass
class BenchmarkResult:
    """Results from a single benchmark."""

    name: str
    mean_ns: float
    std_ns: float
    min_ns: float
    max_ns: float
    median_ns: float
    iterations: int
    samples: list[float] = field(default_factory=list)
    throughput: Optional[float] = None  # items/sec if applicable
    throughput_unit: Optional[str] = None

    def __str__(self) -> str:
        mean_str = format_time(self.mean_ns)
        std_str = format_time(self.std_ns)
        return f"{self.name}: {mean_str} +/- {std_str} (n={self.iterations})"


def format_time(ns: float) -> str:
    """Format nanoseconds to appropriate unit."""
    if ns < 1000:
        return f"{ns:.2f} ns"
    elif ns < 1_000_000:
        return f"{ns / 1000:.2f} us"
    elif ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    else:
        return f"{ns / 1_000_000_000:.2f} s"


def remove_outliers(data: list[float], m: float = 2.0) -> list[float]:
    """Remove outliers using median absolute deviation."""
    if len(data) < 4:
        return data
    d = np.abs(np.array(data) - np.median(data))
    mdev = np.median(d)
    if mdev == 0:
        return data
    s = d / mdev
    return [x for x, si in zip(data, s) if si < m]


class Benchmark:
    """Benchmark runner with proper warmup and statistics."""

    def __init__(
        self,
        warmup_iterations: int = 5,
        min_iterations: int = 10,
        max_iterations: int = 100,
        min_time_ns: float = 1_000_000_000,  # Run for at least 1 second
    ):
        self.warmup_iterations = warmup_iterations
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.min_time_ns = min_time_ns

    def run(
        self,
        name: str,
        func: Callable[[], None],
        setup: Optional[Callable[[], None]] = None,
        teardown: Optional[Callable[[], None]] = None,
        throughput_items: Optional[int] = None,
        throughput_unit: str = "items",
    ) -> BenchmarkResult:
        """Run a benchmark with warmup and adaptive iteration count."""
        # Warmup
        for _ in range(self.warmup_iterations):
            if setup:
                setup()
            func()
            if teardown:
                teardown()

        # Force garbage collection before timing
        gc.collect()
        gc.disable()

        samples = []
        total_time = 0
        iterations = 0

        try:
            while iterations < self.max_iterations and (
                iterations < self.min_iterations or total_time < self.min_time_ns
            ):
                if setup:
                    setup()

                start = time.perf_counter_ns()
                func()
                elapsed = time.perf_counter_ns() - start

                if teardown:
                    teardown()

                samples.append(elapsed)
                total_time += elapsed
                iterations += 1
        finally:
            gc.enable()

        # Remove outliers for statistics
        clean_samples = remove_outliers(samples)
        if len(clean_samples) < 3:
            clean_samples = samples

        result = BenchmarkResult(
            name=name,
            mean_ns=statistics.mean(clean_samples),
            std_ns=statistics.stdev(clean_samples) if len(clean_samples) > 1 else 0,
            min_ns=min(clean_samples),
            max_ns=max(clean_samples),
            median_ns=statistics.median(clean_samples),
            iterations=iterations,
            samples=samples,
        )

        if throughput_items is not None:
            # Calculate throughput in items/second
            result.throughput = throughput_items / (result.mean_ns / 1_000_000_000)
            result.throughput_unit = throughput_unit

        return result


class BenchmarkGroup:
    """A group of related benchmarks for comparison."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.results: list[BenchmarkResult] = []

    def add(self, result: BenchmarkResult):
        self.results.append(result)

    def print_report(self):
        """Print a formatted comparison report."""
        print(f"\n{'=' * 60}")
        print(f" {self.name}")
        if self.description:
            print(f" {self.description}")
        print("=" * 60)

        if not self.results:
            print("  No results")
            return

        # Find the fastest result for comparison
        fastest = min(self.results, key=lambda r: r.mean_ns)

        # Calculate max name length for alignment
        max_name_len = max(len(r.name) for r in self.results)

        for result in self.results:
            name_padded = result.name.ljust(max_name_len)
            mean_str = format_time(result.mean_ns).rjust(12)
            std_str = format_time(result.std_ns).rjust(10)

            # Calculate speedup relative to fastest
            if result == fastest:
                speedup_str = "(baseline)"
            else:
                slowdown = result.mean_ns / fastest.mean_ns
                speedup_str = f"({slowdown:.2f}x slower)"

            throughput_str = ""
            if result.throughput is not None:
                throughput_str = f" | {result.throughput:.0f} {result.throughput_unit}/s"

            print(f"  {name_padded}  {mean_str} +/- {std_str}  {speedup_str}{throughput_str}")


# ============================================================================
# BENCHMARK IMPLEMENTATIONS
# ============================================================================


def benchmark_kabsch(bench: Benchmark) -> BenchmarkGroup:
    """Benchmark Kabsch alignment implementations."""
    group = BenchmarkGroup(
        "Kabsch Alignment",
        "Trajectory alignment to a reference structure",
    )

    from rust_simulation_tools import kabsch_align

    # Check for local test data first
    top_path = Path(__file__).parent.parent / "data" / "topology.pdb"
    dcd_path = Path(__file__).parent.parent / "data" / "trajectory.dcd"

    if not top_path.exists():
        print("  Skipping kabsch benchmarks: test data not found")
        return group

    try:
        import mdtraj as md
    except ImportError as e:
        print(f"  Skipping kabsch benchmarks: {e}")
        return group

    # Load trajectory with MDTraj
    traj_mdt = md.load(str(dcd_path), top=str(top_path))
    n_frames = len(traj_mdt)
    n_atoms = traj_mdt.n_atoms

    # Select CA atoms for alignment
    try:
        indices = traj_mdt.topology.select("name CA")
        if len(indices) == 0:
            # Fall back to backbone if no CA
            indices = traj_mdt.topology.select("backbone")
        if len(indices) == 0:
            # Fall back to first 10% of atoms
            indices = np.arange(n_atoms // 10)
    except Exception:
        indices = np.arange(min(100, n_atoms))

    # Convert to Rust format (Angstroms, f64)
    coords = (traj_mdt.xyz * 10.0).astype(np.float64)  # nm to Angstrom
    ref_coords = coords[0].copy()
    indices_i64 = indices.astype(np.int64)

    # Rust implementation
    def rust_align():
        kabsch_align(coords.copy(), ref_coords, indices_i64)

    result = bench.run(
        f"Rust ({n_frames} frames, {n_atoms} atoms)",
        rust_align,
        throughput_items=n_frames,
        throughput_unit="frames",
    )
    group.add(result)

    # MDTraj
    def mdt_align():
        traj_mdt.superpose(traj_mdt, atom_indices=indices)

    result = bench.run(
        f"MDTraj ({n_frames} frames, {n_atoms} atoms)",
        mdt_align,
        throughput_items=n_frames,
        throughput_unit="frames",
    )
    group.add(result)

    # Try MDAnalysis if available
    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis.align import AlignTraj

        u = mda.Universe(str(top_path), str(dcd_path))
        selection_text = "name CA" if len(indices) > 0 else "all"

        def mda_align():
            AlignTraj(u, u, select=selection_text).run()

        result = bench.run(
            f"MDAnalysis ({n_frames} frames, {n_atoms} atoms)",
            mda_align,
            throughput_items=n_frames,
            throughput_unit="frames",
        )
        group.add(result)
    except Exception as e:
        print(f"  MDAnalysis benchmark skipped: {e}")

    return group


def benchmark_sasa(bench: Benchmark) -> BenchmarkGroup:
    """Benchmark SASA calculations."""
    try:
        import mdtraj as md
    except ImportError as e:
        print(f"  Skipping SASA benchmarks: {e}")
        return BenchmarkGroup("SASA Calculation", "Skipped - missing dependencies")

    from rust_simulation_tools import calculate_sasa, get_radii_array

    group = BenchmarkGroup(
        "SASA Calculation",
        "Shrake-Rupley solvent accessible surface area",
    )

    # Check if we have local test data
    top_path = Path(__file__).parent.parent / "data" / "topology.pdb"
    dcd_path = Path(__file__).parent.parent / "data" / "trajectory.dcd"

    if not top_path.exists():
        # Fall back to MDAnalysis test files
        from MDAnalysis.tests.datafiles import DCD, PSF

        top_path = PSF
        dcd_path = DCD

    traj = md.load(str(dcd_path), top=str(top_path))
    n_frames = len(traj)
    n_atoms = traj.n_atoms

    # Prepare data for Rust
    coords = traj.xyz.astype(np.float64) * 10.0  # nm to Angstrom
    atoms = [atom.element.symbol for atom in traj.topology.atoms]
    resids = np.array([atom.residue.index for atom in traj.topology.atoms])
    n_points = 960
    radii = get_radii_array(atoms)

    # Rust implementation (single frame)
    def rust_sasa_single():
        calculate_sasa(coords[0], radii, resids, probe_radius=1.4, n_sphere_points=n_points)

    result = bench.run(
        f"Rust (single frame, {n_atoms} atoms)",
        rust_sasa_single,
        throughput_items=n_atoms,
        throughput_unit="atoms",
    )
    group.add(result)

    # MDTraj (single frame)
    single_frame = traj[0]

    def mdt_sasa_single():
        md.shrake_rupley(single_frame, probe_radius=0.14, n_sphere_points=n_points)

    result = bench.run(
        f"MDTraj (single frame, {n_atoms} atoms)",
        mdt_sasa_single,
        throughput_items=n_atoms,
        throughput_unit="atoms",
    )
    group.add(result)

    return group


def benchmark_unwrap(bench: Benchmark) -> BenchmarkGroup:
    """Benchmark trajectory unwrapping."""
    from rust_simulation_tools import unwrap_system

    group = BenchmarkGroup(
        "Trajectory Unwrapping",
        "Remove periodic boundary artifacts",
    )

    # Check for test data
    top_path = Path(__file__).parent.parent / "data" / "unwrapping.pdb"
    dcd_path = Path(__file__).parent.parent / "data" / "unwrapping.dcd"

    if not top_path.exists():
        print("  Skipping unwrap benchmarks: test data not found")
        return group

    try:
        import MDAnalysis as mda
    except ImportError:
        print("  Skipping unwrap benchmarks: MDAnalysis not installed")
        return group

    u = mda.Universe(str(top_path), str(dcd_path))
    n_frames = len(u.trajectory)
    n_atoms = len(u.atoms)

    # Build fragment assignments
    assignments = np.zeros(n_atoms, dtype=np.int64)
    for frag_id, fragment in enumerate(u.atoms.fragments):
        assignments[fragment.indices] = frag_id

    # Pre-load trajectory
    trajectory = np.zeros((n_frames, n_atoms, 3), dtype=np.float64)
    box = np.zeros((n_frames, 3), dtype=np.float64)
    for i, ts in enumerate(u.trajectory):
        trajectory[i] = ts.positions
        box[i] = ts.dimensions[:3]

    def rust_unwrap():
        unwrap_system(trajectory.copy(), box, assignments)

    result = bench.run(
        f"Rust ({n_frames} frames, {n_atoms} atoms)",
        rust_unwrap,
        throughput_items=n_frames,
        throughput_unit="frames",
    )
    group.add(result)

    return group


def benchmark_fingerprint(bench: Benchmark) -> BenchmarkGroup:
    """Benchmark interaction fingerprint calculations."""
    group = BenchmarkGroup(
        "Interaction Fingerprints",
        "Protein-ligand interaction fingerprint calculation",
    )

    try:
        from rust_simulation_tools import FingerprintSession
    except ImportError:
        print("  Skipping fingerprint benchmarks: FingerprintSession not available")
        return group

    # Check for test data
    top_path = Path(__file__).parent.parent / "data" / "topology.pdb"
    dcd_path = Path(__file__).parent.parent / "data" / "trajectory.dcd"

    if not top_path.exists():
        print("  Skipping fingerprint benchmarks: test data not found")
        return group

    try:
        import mdtraj as md
    except ImportError:
        print("  Skipping fingerprint benchmarks: mdtraj not installed")
        return group

    traj = md.load(str(dcd_path), top=str(top_path))
    coords = traj.xyz  # Keep in nm for fingerprints

    # Get residue info (simplified - use last residue as ligand)
    n_residues = traj.n_residues
    n_atoms = traj.n_atoms

    # Build atom data
    atom_names = [a.name for a in traj.topology.atoms]
    atom_elements = [a.element.symbol if a.element else "C" for a in traj.topology.atoms]
    residue_indices = np.array([a.residue.index for a in traj.topology.atoms], dtype=np.int64)
    residue_names = [r.name for r in traj.topology.residues]
    charges = np.zeros(n_atoms, dtype=np.float64)  # Placeholder

    # Use first frame
    frame_coords = coords[0]

    # Create session and compute
    try:
        session = FingerprintSession(
            atom_names=atom_names,
            atom_elements=atom_elements,
            residue_indices=residue_indices,
            residue_names=residue_names,
            charges=charges,
            receptor_residues=list(range(n_residues - 1)),
            ligand_residues=[n_residues - 1],
        )

        def compute_fps():
            session.compute_frame(frame_coords)

        result = bench.run(
            f"Rust ({n_residues} residues, {n_atoms} atoms)",
            compute_fps,
            throughput_items=n_residues,
            throughput_unit="residues",
        )
        group.add(result)
    except Exception as e:
        print(f"  Skipping fingerprint benchmarks: {e}")

    return group


def benchmark_mmpbsa_components(bench: Benchmark) -> BenchmarkGroup:
    """Benchmark MM-PBSA energy components via Python bindings."""
    group = BenchmarkGroup(
        "MM-PBSA Energy Components",
        "Individual energy term calculations",
    )

    # Check for test data
    prmtop_path = Path(__file__).parent.parent / "data" / "mmpbsa.prmtop"
    inpcrd_path = Path(__file__).parent.parent / "data" / "mmpbsa.inpcrd"

    if not prmtop_path.exists():
        print("  Skipping MM-PBSA benchmarks: test data not found")
        return group

    try:
        from rust_simulation_tools import (
            compute_gb_energy,
            compute_mm_energy,
            compute_sa_energy,
            read_prmtop,
            read_inpcrd,
        )
    except ImportError as e:
        print(f"  Skipping MM-PBSA benchmarks: {e}")
        return group

    # Load topology and coordinates
    topo = read_prmtop(str(prmtop_path))
    positions, _box = read_inpcrd(str(inpcrd_path))
    coords = (positions * 10.0).astype(np.float64)  # nm to Angstrom

    n_atoms = len(coords)

    # MM Energy
    def mm_energy():
        compute_mm_energy(topo, coords)

    result = bench.run(f"MM Energy ({n_atoms} atoms)", mm_energy)
    group.add(result)

    # GB Energy
    def gb_energy():
        compute_gb_energy(topo, coords)

    result = bench.run(f"GB Energy ({n_atoms} atoms)", gb_energy)
    group.add(result)

    # SA Energy
    def sa_energy():
        compute_sa_energy(topo, coords)

    result = bench.run(f"SA Energy ({n_atoms} atoms)", sa_energy)
    group.add(result)

    return group


# ============================================================================
# MAIN
# ============================================================================


def run_all_benchmarks(filter_pattern: Optional[str] = None) -> list[BenchmarkGroup]:
    """Run all benchmarks and return results."""
    bench = Benchmark(
        warmup_iterations=3,
        min_iterations=10,
        max_iterations=50,
        min_time_ns=500_000_000,  # 0.5 seconds minimum
    )

    benchmark_funcs = [
        ("kabsch", benchmark_kabsch),
        ("sasa", benchmark_sasa),
        ("unwrap", benchmark_unwrap),
        ("fingerprint", benchmark_fingerprint),
        ("mmpbsa", benchmark_mmpbsa_components),
    ]

    results = []
    for name, func in benchmark_funcs:
        if filter_pattern and filter_pattern.lower() not in name.lower():
            continue
        print(f"\nRunning {name} benchmarks...")
        group = func(bench)
        group.print_report()
        results.append(group)

    return results


def export_results(groups: list[BenchmarkGroup], path: str):
    """Export benchmark results to JSON."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "groups": [],
    }

    for group in groups:
        group_data = {
            "name": group.name,
            "description": group.description,
            "results": [],
        }
        for result in group.results:
            group_data["results"].append(
                {
                    "name": result.name,
                    "mean_ns": result.mean_ns,
                    "std_ns": result.std_ns,
                    "min_ns": result.min_ns,
                    "max_ns": result.max_ns,
                    "median_ns": result.median_ns,
                    "iterations": result.iterations,
                    "throughput": result.throughput,
                    "throughput_unit": result.throughput_unit,
                }
            )
        data["groups"].append(group_data)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults exported to {path}")


def main():
    parser = argparse.ArgumentParser(description="Run rust-simulation-tools benchmarks")
    parser.add_argument(
        "--filter",
        type=str,
        help="Only run benchmarks matching this pattern",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export results to JSON file",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" rust-simulation-tools Benchmark Suite")
    print("=" * 60)

    results = run_all_benchmarks(args.filter)

    if args.export:
        export_results(results, args.export)

    print("\n" + "=" * 60)
    print(" Benchmark complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
