#!/usr/bin/env python3
"""
Standalone test for SASA calculation with KD-tree optimization
Run this to verify the implementation works correctly
"""

import numpy as np
import sys
import pytest

def test_imports():
    """Test that all functions can be imported"""
    print("Testing imports...")
    try:
        from rust_simulation_tools import (
            calculate_sasa,
            calculate_residue_sasa,
            calculate_total_sasa,
            calculate_sasa_trajectory,
            get_vdw_radius,
            get_radii_array,
        )
        print("  ✓ All functions imported successfully")
    except ImportError as e:
        pytest.fail(f"Import error: {e}")


def test_single_atom_sasa():
    """Test SASA of a single isolated atom"""
    print("\nTest 1: Single Atom SASA")
    print("-" * 40)
    
    from rust_simulation_tools import calculate_sasa, get_vdw_radius
    
    # Single carbon atom at origin
    coords = np.array([[0.0, 0.0, 0.0]])
    radii = np.array([get_vdw_radius('C')])  # 1.7 Å
    residue_indices = np.array([0])  # Must be unsigned
    
    result = calculate_sasa(coords, radii, residue_indices)
    
    # Analytical result: 4π(1.7 + 1.4)² = 4π(3.1)² ≈ 120.76 Ų
    expected = 4 * np.pi * (1.7 + 1.4)**2
    error = abs(result['total'] - expected) / expected * 100
    
    print(f"  Expected: {expected:.2f} Ų")
    print(f"  Calculated: {result['total']:.2f} Ų")
    print(f"  Error: {error:.2f}%")
    
    assert error < 5.0, f"Error too large: {error:.2f}%"
    print("  ✓ PASS: Within 5% of expected value")


def test_two_distant_atoms():
    """Test SASA of two widely separated atoms"""
    print("\nTest 2: Two Distant Atoms")
    print("-" * 40)
    
    from rust_simulation_tools import calculate_sasa
    
    # Two atoms 100 Å apart (no interaction)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0]
    ])
    radii = np.array([1.7, 1.7])
    residue_indices = np.array([0, 1])
    
    result = calculate_sasa(coords, radii, residue_indices)
    
    # Each atom should have full SASA
    single_atom_sasa = 4 * np.pi * (1.7 + 1.4)**2
    expected = 2 * single_atom_sasa
    error = abs(result['total'] - expected) / expected * 100
    
    print(f"  Expected: {expected:.2f} Ų")
    print(f"  Calculated: {result['total']:.2f} Ų")
    print(f"  Error: {error:.2f}%")
    print(f"  Atom 1 SASA: {result['per_atom'][0]:.2f} Ų")
    print(f"  Atom 2 SASA: {result['per_atom'][1]:.2f} Ų")
    
    assert error < 5.0, f"Error too large: {error:.2f}%"
    print("  ✓ PASS: Both atoms have full SASA")


def test_two_close_atoms():
    """Test SASA of two atoms close enough to occlude each other"""
    print("\nTest 3: Two Close Atoms (Occlusion)")
    print("-" * 40)
    
    from rust_simulation_tools import calculate_sasa
    
    # Two atoms 3 Å apart (partial occlusion)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0]
    ])
    radii = np.array([1.7, 1.7])
    residue_indices = np.array([0, 1])
    
    result = calculate_sasa(coords, radii, residue_indices)
    
    single_atom_sasa = 4 * np.pi * (1.7 + 1.4)**2
    max_total = 2 * single_atom_sasa
    
    print(f"  Maximum possible: {max_total:.2f} Ų")
    print(f"  Calculated: {result['total']:.2f} Ų")
    print(f"  Reduction: {(1 - result['total']/max_total)*100:.1f}%")
    print(f"  Atom 1 SASA: {result['per_atom'][0]:.2f} Ų")
    print(f"  Atom 2 SASA: {result['per_atom'][1]:.2f} Ų")
    
    # Should have reduced SASA due to occlusion
    assert result['total'] < max_total * 0.95, "No occlusion detected"
    print("  ✓ PASS: Occlusion detected")


def test_buried_atom():
    """Test SASA of a buried atom surrounded by others"""
    print("\nTest 4: Buried Atom")
    print("-" * 40)
    
    from rust_simulation_tools import calculate_sasa
    
    # Central atom surrounded by 6 neighbors
    coords = np.array([
        [0.0, 0.0, 0.0],    # Central atom
        [3.5, 0.0, 0.0],
        [-3.5, 0.0, 0.0],
        [0.0, 3.5, 0.0],
        [0.0, -3.5, 0.0],
        [0.0, 0.0, 3.5],
        [0.0, 0.0, -3.5],
    ])
    radii = np.full(7, 1.7)
    residue_indices = np.arange(7)
    
    result = calculate_sasa(coords, radii, residue_indices)
    
    central_sasa = result['per_atom'][0]
    peripheral_mean = result['per_atom'][1:].mean()
    
    print(f"  Central atom SASA: {central_sasa:.2f} Ų")
    print(f"  Peripheral mean SASA: {peripheral_mean:.2f} Ų")
    print(f"  Burial ratio: {central_sasa/peripheral_mean:.2f}")
    
    assert central_sasa < peripheral_mean, "Central atom should be more buried"
    print("  ✓ PASS: Central atom is more buried")


def test_vdw_radii():
    """Test VDW radius lookup"""
    print("\nTest 5: VDW Radii Lookup")
    print("-" * 40)
    
    from rust_simulation_tools import get_vdw_radius
    
    test_cases = [
        ('C', 1.70),
        ('N', 1.55),
        ('O', 1.52),
        ('H', 1.20),
        ('S', 1.80),
        ('P', 1.80),
    ]
    
    for element, expected in test_cases:
        radius = get_vdw_radius(element)
        status = "✓" if radius == expected else "✗"
        print(f"  {status} {element}: {radius:.2f} Å (expected {expected:.2f})")
        assert radius == expected, f"{element} radius mismatch: {radius} != {expected}"
    
    # Test case insensitivity
    assert get_vdw_radius('c') == get_vdw_radius('C'), "Case sensitivity issue"
    print("  ✓ Case insensitive")
    print("  ✓ PASS: All radii correct")


def test_per_residue():
    """Test per-residue SASA aggregation"""
    print("\nTest 6: Per-Residue Aggregation")
    print("-" * 40)
    
    from rust_simulation_tools import calculate_sasa
    
    # 3 atoms: 2 in residue 0, 1 in residue 1
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [100.0, 0.0, 0.0],
    ])
    radii = np.array([1.7, 1.5, 1.7])
    residue_indices = np.array([0, 0, 1])
    
    result = calculate_sasa(coords, radii, residue_indices)
    
    res0_sasa = result['per_residue'][0]
    res1_sasa = result['per_residue'][1]
    atom0_sasa = result['per_atom'][0]
    atom1_sasa = result['per_atom'][1]
    atom2_sasa = result['per_atom'][2]
    
    print(f"  Residue 0 SASA: {res0_sasa:.2f} Ų")
    print(f"    Atom 0: {atom0_sasa:.2f} Ų")
    print(f"    Atom 1: {atom1_sasa:.2f} Ų")
    print(f"    Sum: {atom0_sasa + atom1_sasa:.2f} Ų")
    print(f"  Residue 1 SASA: {res1_sasa:.2f} Ų")
    print(f"    Atom 2: {atom2_sasa:.2f} Ų")
    
    # Check that per-residue is sum of per-atom
    assert abs(res0_sasa - (atom0_sasa + atom1_sasa)) < 0.01, "Residue 0 aggregation mismatch"
    assert abs(res1_sasa - atom2_sasa) < 0.01, "Residue 1 aggregation mismatch"
    print("  ✓ PASS: Per-residue equals sum of per-atom")


def test_fast_functions():
    """Test fast calculation variants"""
    print("\nTest 7: Fast Calculation Functions")
    print("-" * 40)
    
    from rust_simulation_tools import (
        calculate_sasa,
        calculate_residue_sasa,
        calculate_total_sasa,
    )
    
    coords = np.array([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ])
    radii = np.array([1.7, 1.7])
    residue_indices = np.array([0, 1])
    
    # Full calculation
    full_result = calculate_sasa(coords, radii, residue_indices)
    
    # Fast residue-only
    residue_result = calculate_residue_sasa(coords, radii, residue_indices)
    
    # Fast total-only
    total_result = calculate_total_sasa(coords, radii)
    
    print(f"  Full result total: {full_result['total']:.2f} Ų")
    print(f"  Total-only result: {total_result:.2f} Ų")
    print(f"  Difference: {abs(full_result['total'] - total_result):.4f} Ų")
    
    # Check consistency
    assert abs(full_result['total'] - total_result) < 0.01, "Total mismatch"
    assert abs(full_result['per_residue'][0] - residue_result[0]) < 0.01, "Residue 0 mismatch"
    assert abs(full_result['per_residue'][1] - residue_result[1]) < 0.01, "Residue 1 mismatch"
    print("  ✓ PASS: All functions give consistent results")


def test_probe_radius():
    """Test different probe radii"""
    print("\nTest 8: Different Probe Radii")
    print("-" * 40)
    
    from rust_simulation_tools import calculate_total_sasa
    
    coords = np.array([[0.0, 0.0, 0.0]])
    radii = np.array([1.5])
    
    probe_radii = [0.0, 1.0, 1.4, 2.0]
    results = []
    
    for probe_r in probe_radii:
        sasa = calculate_total_sasa(coords, radii, probe_radius=probe_r)
        results.append(sasa)
        expected = 4 * np.pi * (1.5 + probe_r)**2
        print(f"  Probe {probe_r:.1f} Å: {sasa:.2f} Ų (expected {expected:.2f})")
    
    # SASA should increase with probe radius
    for i in range(len(results) - 1):
        assert results[i] < results[i+1], f"SASA didn't increase: {results[i]} >= {results[i+1]}"
    print("  ✓ PASS: SASA increases with probe radius")


def test_sphere_points():
    """Test different sphere point densities"""
    print("\nTest 9: Sphere Point Densities")
    print("-" * 40)
    
    import time
    from rust_simulation_tools import calculate_total_sasa
    
    coords = np.array([[0.0, 0.0, 0.0]])
    radii = np.array([1.5])
    
    point_counts = [92, 162, 242, 480, 960]
    
    print("  Points | SASA (Ų) | Time (ms)")
    print("  " + "-" * 36)
    
    results = []
    for n_points in point_counts:
        start = time.time()
        sasa = calculate_total_sasa(coords, radii, n_sphere_points=n_points)
        elapsed = (time.time() - start) * 1000
        results.append(sasa)
        print(f"  {n_points:5d}  | {sasa:8.2f}  | {elapsed:6.2f}")
    
    # Results should converge (std should be small)
    std = np.std(results)
    assert std < 2.0, f"Poor convergence: std = {std:.2f}"
    print(f"  ✓ PASS: Results converge (std = {std:.2f})")


def test_performance():
    """Test performance on larger system"""
    print("\nTest 10: Performance Benchmark")
    print("-" * 40)
    
    import time
    from rust_simulation_tools import calculate_sasa
    
    sizes = [100, 500, 1000]
    
    print("  Atoms | Time (ms) | SASA (Ų)")
    print("  " + "-" * 36)
    
    for n_atoms in sizes:
        # Generate random system
        coords = np.random.randn(n_atoms, 3) * 10.0
        radii = np.random.uniform(1.2, 1.9, n_atoms)
        residue_indices = np.arange(n_atoms)
        
        start = time.time()
        result = calculate_sasa(coords, radii, residue_indices, n_sphere_points=480)
        elapsed = (time.time() - start) * 1000
        
        print(f"  {n_atoms:5d} | {elapsed:9.2f} | {result['total']:8.1f}")
    
    print("  ✓ PASS: Performance test complete")


def run_all_tests():
    """Run all tests and report results"""
    print("=" * 60)
    print("SASA Implementation Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Single Atom", test_single_atom_sasa),
        ("Two Distant Atoms", test_two_distant_atoms),
        ("Two Close Atoms", test_two_close_atoms),
        ("Buried Atom", test_buried_atom),
        ("VDW Radii", test_vdw_radii),
        ("Per-Residue", test_per_residue),
        ("Fast Functions", test_fast_functions),
        ("Probe Radius", test_probe_radius),
        ("Sphere Points", test_sphere_points),
        ("Performance", test_performance),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True))
        except AssertionError as e:
            print(f"\n  ✗ FAIL: {e}")
            results.append((name, False))
        except Exception as e:
            print(f"\n  ✗ ERROR: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print("-" * 60)
    print(f"  Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
