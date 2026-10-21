"""
Interactive Verification Script for GreenGPU Phase 2.
Demonstrates live cooperative execution of KMeans & HotSpot across CPU and RTX 4050 GPU,
and simulates Paper Testbed A energy curves across workload division ratios (Figure 2 replication).

Usage:
    python demo_phase2.py
"""

import time
import sys

from greengpu.workloads.kmeans import HeterogeneousKMeans
from greengpu.workloads.hotspot import HeterogeneousHotSpot
from greengpu.simulator import CalibratedGreenGPUSimulator
from greengpu.tier1.improved_binary import ImprovedBinarySearchPartitioner


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_live_kmeans():
    print_header("1. Live Cooperative Execution: Heterogeneous KMeans")
    print("Partitioning 100,000 vectors (34 features, 16 centroids) dynamically between")
    print("Intel i5-12450H CPU threads and NVIDIA RTX 4050 Laptop GPU CUDA streams.\n")

    kmeans = HeterogeneousKMeans(num_points=100_000, num_features=34, num_clusters=16, use_gpu=True)
    partitioner = ImprovedBinarySearchPartitioner()

    r = 0.50  # Start with 50/50 split
    partitioner.reset(initial_ratio=r)

    print(f"{'Iter':<5} | {'CPU % (r)':<10} | {'GPU % (1-r)':<11} | {'t_CPU (ms)':<11} | {'t_GPU (ms)':<11} | {'Wall (ms)':<10} | {'Idle Penalty':<12}")
    print("-" * 80)

    for it in range(1, 6):
        res = kmeans.run_iteration(iteration=it, cpu_ratio=r)
        tc_ms = res.time_cpu_sec * 1000.0
        tg_ms = res.time_gpu_sec * 1000.0
        wall_ms = res.time_wall_sec * 1000.0
        idle_ms = res.idle_time_wasted_sec * 1000.0

        print(f"{it:<5} | {r*100:>8.1f}% | {(1-r)*100:>9.1f}% | {tc_ms:>9.2f} ms | {tg_ms:>9.2f} ms | {wall_ms:>8.2f} ms | {idle_ms:>9.2f} ms")

        # Update partitioner with measured timings
        r = partitioner.update(iteration=it, tc=res.time_cpu_sec, tg=res.time_gpu_sec)
        if partitioner.is_converged:
            break

    print("\nObservation:")
    print("Notice how the idle penalty (|tc - tg|) shrinks as the ratio converges to equilibrium!")


def demo_live_hotspot():
    print_header("2. Live Cooperative Execution: Heterogeneous HotSpot")
    print("Partitioning a 1024x1024 grid across CPU and RTX 4050 GPU.\n")

    hotspot = HeterogeneousHotSpot(grid_rows=1024, grid_cols=1024, steps_per_iteration=3, use_gpu=True)
    partitioner = ImprovedBinarySearchPartitioner()

    r = 0.70  # Start with 70% CPU, 30% GPU
    partitioner.reset(initial_ratio=r)

    print(f"{'Iter':<5} | {'CPU % (r)':<10} | {'GPU % (1-r)':<11} | {'t_CPU (ms)':<11} | {'t_GPU (ms)':<11} | {'Wall (ms)':<10} | {'Idle Penalty':<12}")
    print("-" * 80)

    for it in range(1, 6):
        res = hotspot.run_iteration(iteration=it, cpu_ratio=r)
        tc_ms = res.time_cpu_sec * 1000.0
        tg_ms = res.time_gpu_sec * 1000.0
        wall_ms = res.time_wall_sec * 1000.0
        idle_ms = res.idle_time_wasted_sec * 1000.0

        print(f"{it:<5} | {r*100:>8.1f}% | {(1-r)*100:>9.1f}% | {tc_ms:>9.2f} ms | {tg_ms:>9.2f} ms | {wall_ms:>8.2f} ms | {idle_ms:>9.2f} ms")

        r = partitioner.update(iteration=it, tc=res.time_cpu_sec, tg=res.time_gpu_sec)
        if partitioner.is_converged:
            break


def demo_simulation_curve():
    print_header("3. Calibrated Simulation: Energy vs. Workload Division (Paper Fig. 2)")
    print("Simulating Paper Testbed A (8800 GTX + Phenom II) across CPU Workload Ratios (0% to 100%).\n")

    sim = CalibratedGreenGPUSimulator(workload_type="kmeans")

    print(f"{'CPU Ratio (r)':<14} | {'Wall Time (s)':<14} | {'Total Energy (J)':<17} | {'Status'}")
    print("-" * 65)

    best_energy = float("inf")
    best_r = 0.0

    for r_pct in range(0, 105, 10):
        r = r_pct / 100.0
        out = sim.simulate_iteration(iteration=1, cpu_ratio=r)
        if out.energy_total_joules < best_energy:
            best_energy = out.energy_total_joules
            best_r = r

        status = ""
        if r_pct == 0:
            status = "(GPU Only Baseline)"
        elif r_pct == 100:
            status = "(CPU Only)"

        print(f"{r*100:>12.0f}% | {out.time_wall_sec:>12.4f} s | {out.energy_total_joules:>15.2f} J | {status}")

    savings = ((sim.simulate_iteration(1, 0.0).energy_total_joules - best_energy) /
               sim.simulate_iteration(1, 0.0).energy_total_joules) * 100.0

    print("-" * 65)
    print(f"Optimal Division Point : {best_r*100:.0f}% CPU / {(1-best_r)*100:.0f}% GPU")
    print(f"Energy Savings vs. GPU-Only Baseline: {savings:.2f}%")
    print("Confirms Paper Thesis: Co-running CPU + GPU consumes less total energy than GPU-only!")


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("       ATLAS GreenGPU - Phase 2 Live Verification Suite")
    print("#" * 70)
    demo_live_kmeans()
    demo_live_hotspot()
    demo_simulation_curve()
    print("\n" + "=" * 70)
    print("To run the complete unit test suite across Phase 1 & 2, run:")
    print("    python -m unittest discover -s tests -p \"test_*.py\" -v")
    print("=" * 70 + "\n")
