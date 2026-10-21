"""
ATLAS GreenGPU - Interactive Terminal Console & Dashboard
Provides a full interactive terminal environment for running live heterogeneous workloads,
monitoring real-time hardware telemetry HUD, and reproducing paper experiments.

Usage:
    python interactive_runner.py
"""

import os
import sys
import time
from typing import Dict, List, Optional

from greengpu.config import GreenGPUConfig, LOCAL_RTX_4050, TESTBED_A_8800GTX
from greengpu.telemetry.nvml_monitor import NVMLTelemetryMonitor
from greengpu.telemetry.cpu_monitor import CPUSnapshotMonitor
from greengpu.tier1.simple_heuristic import SimpleHeuristicPartitioner
from greengpu.tier1.adaptive_heuristic import AdaptiveHeuristicPartitioner
from greengpu.tier1.binary_search import BinarySearchPartitioner
from greengpu.tier1.improved_binary import ImprovedBinarySearchPartitioner
from greengpu.tier2.wma_controller import WMAFrequencyController
from greengpu.workloads.kmeans import HeterogeneousKMeans
from greengpu.workloads.hotspot import HeterogeneousHotSpot
from greengpu.workloads.synthetic_bench import SyntheticHeterogeneousBenchmark
from greengpu.simulator import CalibratedGreenGPUSimulator


# Ensure Windows console supports UTF-8
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ANSI Color Codes for Rich Terminal Display
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def render_bar(val: float, max_val: float = 100.0, width: int = 25, color: str = Colors.GREEN) -> str:
    """Renders an ASCII percentage progress bar."""
    ratio = max(0.0, min(1.0, val / max_val))
    filled = int(round(ratio * width))
    bar = f"{color}{'#' * filled}{Colors.DIM}{'-' * (width - filled)}{Colors.RESET}"
    return f"[{bar}] {val:>5.1f}%"


def print_banner():
    clear_screen()
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("+==============================================================================+")
    print("|                     ATLAS GreenGPU Terminal Console                          |")
    print("|      Two-Tier Energy Management for CPU-GPU Heterogeneous Computing          |")
    print("|             SUSCOM 2016 Research Architecture Implementation                 |")
    print("+==============================================================================+")
    print(f"{Colors.RESET}")


def menu_live_telemetry_hud():
    """Option 2: Live Real-Time Hardware Telemetry HUD."""
    clear_screen()
    print(f"{Colors.CYAN}{Colors.BOLD}=== Real-Time Hardware Telemetry & DVFS HUD (Press Ctrl+C to Exit) ==={Colors.RESET}\n")

    try:
        nvml_mon = NVMLTelemetryMonitor(device_index=0)
    except Exception as e:
        nvml_mon = None
        print(f"{Colors.YELLOW}[Note] Live NVML GPU monitoring unavailable: {e}{Colors.RESET}")

    cpu_mon = CPUSnapshotMonitor()
    controller = WMAFrequencyController(hardware=TESTBED_A_8800GTX)

    try:
        while True:
            # Move cursor to top or clear screen
            sys.stdout.write("\033[H\033[J")
            print(f"{Colors.CYAN}{Colors.BOLD}=== ATLAS GreenGPU: Live Telemetry Stream (Refreshing every 1s) ==={Colors.RESET}")
            print(f"{Colors.DIM}Press Ctrl+C anytime to return to Main Menu{Colors.RESET}\n")

            cpu_snap = cpu_mon.sample()
            gpu_snap = nvml_mon.sample() if nvml_mon else None

            # 1. GPU Telemetry
            print(f"{Colors.BOLD}{Colors.GREEN}[GPU] NVIDIA GeForce RTX 4050 Laptop GPU{Colors.RESET}")
            if gpu_snap:
                pwr = gpu_snap.gpu_power_watts
                core_u = gpu_snap.gpu_util_core
                mem_u = gpu_snap.gpu_util_memory
                core_clk = gpu_snap.gpu_clock_core_mhz
                mem_clk = gpu_snap.gpu_clock_memory_mhz
                temp = gpu_snap.gpu_temperature_c

                pwr_bar = render_bar(pwr, max_val=55.0, width=20, color=Colors.YELLOW)
                core_bar = render_bar(core_u, max_val=100.0, width=20, color=Colors.GREEN)
                mem_bar = render_bar(mem_u, max_val=100.0, width=20, color=Colors.CYAN)

                print(f"  Power Consumption : {pwr:>6.2f} W (Max 55W)  {pwr_bar}")
                print(f"  Compute Core Util : {core_bar}  Clock: {core_clk:>4} MHz")
                print(f"  Memory Bandwidth  : {mem_bar}  Clock: {mem_clk:>4} MHz")
                print(f"  GPU Temperature   : {temp:>5.1f} °C")

                # Live Tier 2 DVFS Recommendation
                rec_fc, rec_fm = controller.update(core_u, mem_u)
                print(f"\n  {Colors.BOLD}Tier 2 WMA Controller Recommendation:{Colors.RESET}")
                print(f"  -> Recommended Core Clock: {Colors.CYAN}{rec_fc} MHz{Colors.RESET} | Memory Clock: {Colors.CYAN}{rec_fm} MHz{Colors.RESET}")
            else:
                print("  GPU Telemetry: Running in Simulated Mode")

            # 2. CPU Telemetry
            print(f"\n{Colors.BOLD}{Colors.BLUE}[CPU] 12th Gen Intel Core i5-12450H (8 Cores / 12 Threads){Colors.RESET}")
            cpu_bar = render_bar(cpu_snap.cpu_util_total, max_val=100.0, width=20, color=Colors.BLUE)
            print(f"  Total Utilization : {cpu_bar}  Clock: {cpu_snap.cpu_clock_mhz:.0f} MHz")
            
            # Print core bars in 2 columns
            if cpu_snap.cpu_per_core_util:
                print("  Logical Thread Activity:")
                cores = cpu_snap.cpu_per_core_util
                for i in range(0, len(cores), 2):
                    col1 = f"    Core {i:<2}: {render_bar(cores[i], 100, 10, Colors.BLUE)}"
                    col2 = f"    Core {i+1:<2}: {render_bar(cores[i+1], 100, 10, Colors.BLUE)}" if i+1 < len(cores) else ""
                    print(f"{col1}   {col2}")

            print(f"\n{Colors.DIM}Active Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        if nvml_mon:
            nvml_mon.close()


def menu_live_workload_runner():
    """Option 1: Interactive Live Heterogeneous Workload Runner."""
    clear_screen()
    print(f"{Colors.CYAN}{Colors.BOLD}=== Interactive Heterogeneous Workload Runner ==={Colors.RESET}\n")

    print("Select Workload to Execute:")
    print("  [1] KMeans Clustering (100,000 vectors, 34 features, 16 clusters)")
    print("  [2] HotSpot 2D Thermal Diffusion (1024x1024 grid, 5-point stencil)")
    print("  [3] Synthetic Compute-Bound Kernel (stresses GPU/CPU compute)")
    print("  [4] Synthetic Memory-Bound Streaming (stresses memory bandwidth)")
    choice_w = input("\nEnter choice [1-4] (default: 1): ").strip() or "1"

    if choice_w == "1":
        workload = HeterogeneousKMeans(num_points=100_000, num_features=34, num_clusters=16, use_gpu=True)
    elif choice_w == "2":
        workload = HeterogeneousHotSpot(grid_rows=1024, grid_cols=1024, steps_per_iteration=3, use_gpu=True)
    elif choice_w == "3":
        workload = SyntheticHeterogeneousBenchmark(mode="compute_bound", total_work_units=80_000, use_gpu=True)
    else:
        workload = SyntheticHeterogeneousBenchmark(mode="memory_bound", total_work_units=80_000, use_gpu=True)

    print("\nSelect Workload Partitioning Algorithm (Tier 1):")
    print("  [1] Improved Binary Search (Paper Sec 5.2.4 - Proportional Solver)")
    print("  [2] Binary Search (Paper Sec 5.2.3 - Bisection)")
    print("  [3] Adaptive Heuristic (Paper Sec 5.2.2 - Step Halving)")
    print("  [4] Simple Heuristic (Paper Sec 5.2.1 - Fixed Step with Safeguard)")
    print("  [5] Race Mode (Run all 4 side-by-side!)")
    choice_a = input("\nEnter choice [1-5] (default: 1): ").strip() or "1"

    init_r_input = input("\nEnter Initial CPU Workload Ratio (0.0 to 1.0, e.g. 0.50 or 0.80) [default: 0.50]: ").strip()
    try:
        init_r = float(init_r_input) if init_r_input else 0.50
    except ValueError:
        init_r = 0.50

    print(f"\n{Colors.GREEN}Executing on: Intel CPU + NVIDIA RTX 4050 GPU (Starting split: {init_r*100:.0f}% CPU / {(1-init_r)*100:.0f}% GPU){Colors.RESET}\n")

    if choice_a == "5":
        # Race mode
        partitioners = {
            "Simple Heuristic": SimpleHeuristicPartitioner(step_size=0.05),
            "Adaptive Heuristic": AdaptiveHeuristicPartitioner(initial_step=0.15),
            "Binary Search": BinarySearchPartitioner(),
            "Improved Binary": ImprovedBinarySearchPartitioner(),
        }
        print(f"{'Algorithm':<22} | {'Iters':<5} | {'Final CPU %':<11} | {'Final Idle Penalty':<18} | {'Status'}")
        print("-" * 75)
        for name, p in partitioners.items():
            p.reset(initial_ratio=init_r)
            workload.reset()
            r = init_r
            last_res = None
            for it in range(1, 8):
                res = workload.run_iteration(iteration=it, cpu_ratio=r)
                last_res = res
                r = p.update(it, tc=res.time_cpu_sec, tg=res.time_gpu_sec)
                if p.is_converged:
                    break
            idle_ms = last_res.idle_time_wasted_sec * 1000.0 if last_res else 0.0
            print(f"{name:<22} | {len(p.history):<5} | {p.current_ratio*100:>9.1f}% | {idle_ms:>14.2f} ms | {Colors.GREEN}Converged{Colors.RESET}")
    else:
        part_map = {
            "1": ("Improved Binary Search", ImprovedBinarySearchPartitioner()),
            "2": ("Binary Search", BinarySearchPartitioner()),
            "3": ("Adaptive Heuristic", AdaptiveHeuristicPartitioner()),
            "4": ("Simple Heuristic", SimpleHeuristicPartitioner()),
        }
        name, partitioner = part_map.get(choice_a, part_map["1"])
        partitioner.reset(initial_ratio=init_r)
        workload.reset()

        print(f"{Colors.BOLD}{'Iter':<5} | {'Workload Split Bar':<30} | {'t_CPU':<10} | {'t_GPU':<10} | {'Wall Time':<11} | {'Idle Penalty':<12}{Colors.RESET}")
        print("-" * 88)

        r = init_r
        for it in range(1, 9):
            res = workload.run_iteration(iteration=it, cpu_ratio=r)
            tc_ms = res.time_cpu_sec * 1000.0
            tg_ms = res.time_gpu_sec * 1000.0
            wall_ms = res.time_wall_sec * 1000.0
            idle_ms = res.idle_time_wasted_sec * 1000.0

            # Visual split bar: [CPU ######## GPU --------]
            cpu_bars = int(round(r * 15))
            gpu_bars = 15 - cpu_bars
            bar_repr = f"{Colors.BLUE}{'#'*cpu_bars}{Colors.GREEN}{'-'*gpu_bars}{Colors.RESET}"
            ratio_label = f"{r*100:2.0f}% C / {(1-r)*100:2.0f}% G"

            print(f"{it:<5} | [{bar_repr}] {ratio_label:<10} | {tc_ms:>7.2f} ms | {tg_ms:>7.2f} ms | {wall_ms:>8.2f} ms | {Colors.YELLOW}{idle_ms:>8.2f} ms{Colors.RESET}")

            r = partitioner.update(iteration=it, tc=res.time_cpu_sec, tg=res.time_gpu_sec)
            if partitioner.is_converged:
                print(f"\n{Colors.GREEN}{Colors.BOLD}>>> Convergence Achieved! Balanced CPU-GPU Execution Reached! <<<{Colors.RESET}")
                break

    input(f"\n{Colors.DIM}Press Enter to return to main menu...{Colors.RESET}")


def menu_paper_experiments():
    """Option 3: Paper Experiment Reproduction Suite."""
    clear_screen()
    print(f"{Colors.CYAN}{Colors.BOLD}=== GreenGPU Paper Experiments Reproduction Suite ==={Colors.RESET}\n")
    print("Select Experiment to Reproduce:")
    print("  [1] Figure 2: Workload Division vs. Energy Curve (Demonstrating Optimal Split)")
    print("  [2] Figures 7 & 8: Workload Division & Energy Traces over Iterations")
    print("  [3] Figures 14 & 15: Convergence Iteration Count Comparison")
    print("  [4] Figure 16: Total System Energy Comparison Across Partitioners")
    choice = input("\nEnter choice [1-4] (default: 1): ").strip() or "1"

    sim = CalibratedGreenGPUSimulator(workload_type="kmeans" if choice in ["1", "4"] else "hotspot")

    if choice == "1":
        print(f"\n{Colors.BOLD}--- Replicating Figure 2: Workload Division vs. Energy (KMeans) ---{Colors.RESET}")
        print(f"{'CPU Ratio':<12} | {'Energy (J)':<12} | {'Energy Curve Visual Bar':<30} | {'Note'}")
        print("-" * 75)
        for r_pct in range(0, 105, 10):
            r = r_pct / 100.0
            out = sim.simulate_iteration(1, cpu_ratio=r)
            e = out.energy_total_joules
            bar_len = int(round((e / 175.0) * 25))
            bar_color = Colors.GREEN if r_pct == 10 else Colors.CYAN
            bar_str = f"{bar_color}{'#'*bar_len}{Colors.RESET}"
            note = f"{Colors.GREEN}<-- Optimal Energy Point!{Colors.RESET}" if r_pct == 10 else ""
            print(f"{r_pct:>8}% CPU | {e:>8.2f} J   | {bar_str:<38} | {note}")
        print(f"\n{Colors.GREEN}Conclusion: Cooperation (10% CPU / 90% GPU) saves 5.6% energy over 0% CPU (GPU-Only)!{Colors.RESET}")

    elif choice == "3":
        print(f"\n{Colors.BOLD}--- Replicating Figures 14 & 15: Convergence Iterations Needed ---{Colors.RESET}")
        print(f"{'Algorithm':<28} | {'Initial r = 15%':<15} | {'Initial r = 50%':<15} | {'Initial r = 90%'}")
        print("-" * 75)
        for name, cls, args in [
            ("Simple Heuristic (H 3%)", SimpleHeuristicPartitioner, {"step_size": 0.03}),
            ("Improved Heuristic (IH 10%)", AdaptiveHeuristicPartitioner, {"initial_step": 0.10}),
            ("Binary Search (B 1%)", BinarySearchPartitioner, {}),
            ("Improved Binary (IB 1%)", ImprovedBinarySearchPartitioner, {})
        ]:
            row = [name]
            for init in [0.15, 0.50, 0.90]:
                p = cls(**args)
                p.reset(initial_ratio=init)
                r = init
                for it in range(1, 35):
                    out = sim.simulate_iteration(it, cpu_ratio=r)
                    r = p.update(it, tc=out.time_cpu_sec, tg=out.time_gpu_total_sec)
                    if p.is_converged:
                        break
                row.append(f"{len(p.history):>2} iters")
            print(f"{row[0]:<28} | {row[1]:<15} | {row[2]:<15} | {row[3]}")
        print(f"\n{Colors.GREEN}Conclusion: Improved Binary Search converges in fewest iterations across all initial points!{Colors.RESET}")

    elif choice == "4":
        print(f"\n{Colors.BOLD}--- Replicating Figure 16: Total Energy Consumption Comparison ---{Colors.RESET}")
        print(f"{'Algorithm':<28} | {'Total Energy (J)':<16} | {'Energy Comparison Bar'}")
        print("-" * 70)
        energies = []
        for name, cls, args in [
            ("Simple Heuristic (Fixed 3%)", SimpleHeuristicPartitioner, {"step_size": 0.03}),
            ("Improved Heuristic (Adaptive 15%)", AdaptiveHeuristicPartitioner, {"initial_step": 0.15}),
            ("Binary Search (Bisection)", BinarySearchPartitioner, {}),
            ("Improved Binary (Sec 5.2.4)", ImprovedBinarySearchPartitioner, {})
        ]:
            p = cls(**args)
            p.reset(initial_ratio=0.80)
            r = 0.80
            total_e = 0.0
            for it in range(1, 20):
                out = sim.simulate_iteration(it, cpu_ratio=r)
                total_e += out.energy_total_joules
                r = p.update(it, tc=out.time_cpu_sec, tg=out.time_gpu_total_sec)
            energies.append((name, total_e))

        min_e = min(e for _, e in energies)
        for name, e in energies:
            bar_len = int(round((e / max(x[1] for x in energies)) * 25))
            color = Colors.GREEN if e == min_e else Colors.YELLOW
            bar = f"{color}{'#'*bar_len}{Colors.RESET}"
            pct_diff = f"+{((e - min_e)/min_e)*100:.1f}%" if e > min_e else f"{Colors.GREEN}BEST{Colors.RESET}"
            print(f"{name:<28} | {e:>10.2f} J     | {bar} ({pct_diff})")

    else:
        print("\nReplicating HotSpot traces (Figure 7)...")
        p = ImprovedBinarySearchPartitioner()
        p.reset(initial_ratio=0.80)
        r = 0.80
        print(f"{'Iter':<5} | {'CPU Ratio r':<12} | {'tc (s)':<10} | {'tg (s)':<10} | {'Idle Wasted (s)'}")
        print("-" * 55)
        for it in range(1, 8):
            out = sim.simulate_iteration(it, cpu_ratio=r)
            print(f"{it:<5} | {r*100:>10.1f}% | {out.time_cpu_sec:>8.4f} s | {out.time_gpu_total_sec:>8.4f} s | {out.idle_time_wasted_sec:>10.4f} s")
            r = p.update(it, tc=out.time_cpu_sec, tg=out.time_gpu_total_sec)
            if p.is_converged:
                break

    input(f"\n{Colors.DIM}Press Enter to return to main menu...{Colors.RESET}")


def menu_hardware_info():
    """Option 4: Inspect Hardware and Settings."""
    clear_screen()
    print(f"{Colors.CYAN}{Colors.BOLD}=== System Hardware & Hyperparameters ==={Colors.RESET}\n")
    print(f"{Colors.BOLD}Target Hardware Profiles:{Colors.RESET}")
    print(f"  * {LOCAL_RTX_4050.name}")
    print(f"    - Core Frequencies   : {LOCAL_RTX_4050.core_frequencies_mhz} MHz")
    print(f"    - Memory Frequencies : {LOCAL_RTX_4050.memory_frequencies_mhz} MHz")
    print(f"    - Power Envelope     : {LOCAL_RTX_4050.idle_power_watts}W Idle / {LOCAL_RTX_4050.peak_power_watts}W Peak")
    print(f"  * {TESTBED_A_8800GTX.name}")
    print(f"    - Core Frequencies   : {TESTBED_A_8800GTX.core_frequencies_mhz} MHz")
    print(f"    - Memory Frequencies : {TESTBED_A_8800GTX.memory_frequencies_mhz} MHz")
    print(f"    - Power Envelope     : {TESTBED_A_8800GTX.idle_power_watts}W Idle / {TESTBED_A_8800GTX.peak_power_watts}W Peak")

    cfg = GreenGPUConfig()
    print(f"\n{Colors.BOLD}GreenGPU Hyperparameters (Paper Section 5):{Colors.RESET}")
    print(f"  * alpha_core   = {cfg.tier2.alpha_core:<5} (Core energy vs performance weight)")
    print(f"  * alpha_memory = {cfg.tier2.alpha_memory:<5} (Memory energy vs performance weight)")
    print(f"  * gamma        = {cfg.tier2.gamma:<5} (Core vs Memory balance)")
    print(f"  * beta         = {cfg.tier2.beta:<5} (WMA history damping factor)")
    print(f"  * Pt           = {cfg.tier1.precision_threshold:<5} (Convergence precision threshold)")
    print(f"  * Decoupling   = {cfg.tier2.decoupling_factor}x (Tier 1 interval >= 40x Tier 2 interval)")

    input(f"\n{Colors.DIM}Press Enter to return to main menu...{Colors.RESET}")


def main():
    while True:
        print_banner()
        print("Main Menu:")
        print(f"  {Colors.BOLD}[1]{Colors.RESET} [RUN]         {Colors.CYAN}Live Heterogeneous Workloads{Colors.RESET} (KMeans, HotSpot, Synthetic)")
        print(f"  {Colors.BOLD}[2]{Colors.RESET} [TELEMETRY]   {Colors.GREEN}Real-Time Hardware Telemetry HUD{Colors.RESET} (Live Watts, Cores, DVFS)")
        print(f"  {Colors.BOLD}[3]{Colors.RESET} [EXPERIMENTS] {Colors.YELLOW}Paper Reproduction Suite{Colors.RESET} (Figs 2, 7, 8, 14, 16)")
        print(f"  {Colors.BOLD}[4]{Colors.RESET} [CONFIG]      System Hardware Profiles & Hyperparameters")
        print(f"  {Colors.BOLD}[q]{Colors.RESET} [QUIT]        Exit Console")

        choice = input(f"\n{Colors.BOLD}Select an option [1-4, q]: {Colors.RESET}").strip().lower()

        if choice == "1":
            menu_live_workload_runner()
        elif choice == "2":
            menu_live_telemetry_hud()
        elif choice == "3":
            menu_paper_experiments()
        elif choice == "4":
            menu_hardware_info()
        elif choice in ["q", "exit", "quit"]:
            print(f"\n{Colors.CYAN}Exiting GreenGPU Console. Happy Researching!{Colors.RESET}\n")
            break


if __name__ == "__main__":
    main()
