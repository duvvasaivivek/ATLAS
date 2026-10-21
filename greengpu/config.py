"""
Configuration parameters for GreenGPU Two-Tier Energy Management Framework.
Derived from the SUSCOM 2016 GreenGPU research paper and modern hardware specifications.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class HardwareProfile:
    """Hardware profile defining discrete frequency levels for GPU core and memory."""
    name: str
    core_frequencies_mhz: List[int]
    memory_frequencies_mhz: List[int]
    peak_power_watts: float
    idle_power_watts: float
    min_power_limit_watts: float = 5.0
    max_power_limit_watts: float = 55.0


# Paper Testbed A (GeForce 8800 GTX, 6 core levels, 6 memory levels)
TESTBED_A_8800GTX = HardwareProfile(
    name="GeForce 8800 GTX (Paper Testbed A)",
    core_frequencies_mhz=[297, 356, 411, 466, 513, 576],
    memory_frequencies_mhz=[500, 580, 660, 740, 820, 900],
    peak_power_watts=185.0,
    idle_power_watts=45.0,
)

# Modern Local Hardware (NVIDIA GeForce RTX 4050 Laptop GPU)
LOCAL_RTX_4050 = HardwareProfile(
    name="NVIDIA GeForce RTX 4050 Laptop GPU",
    core_frequencies_mhz=[1050, 1350, 1605, 1890, 2100, 2370],
    memory_frequencies_mhz=[5001, 5501, 6001, 6501, 7001, 8001],
    peak_power_watts=55.0,
    idle_power_watts=12.0,
    min_power_limit_watts=5.0,
    max_power_limit_watts=55.0,
)


@dataclass
class Tier1Config:
    """Hyperparameters for Tier 1: Dynamic Workload Division."""
    precision_threshold: float = 0.01          # Pt: stop tuning when step < 1%
    simple_step_size: float = 0.03             # Default fixed step size (3% or 5%)
    adaptive_initial_step: float = 0.15        # Initial step size for Improved Heuristic (15%)
    initial_cpu_ratio: float = 0.50            # Starting CPU work percentage r (e.g., 50%)
    min_cpu_ratio: float = 0.00                # Minimum allowed CPU workload ratio
    max_cpu_ratio: float = 1.00                # Maximum allowed CPU workload ratio
    lookahead_safeguard: bool = True           # Enable oscillation prevention safeguard


@dataclass
class Tier2Config:
    """Hyperparameters for Tier 2: Coordinated GPU Core/Memory DVFS (Section 5.1)."""
    alpha_core: float = 0.15                   # alpha_c: Core energy vs performance trade-off weight
    alpha_memory: float = 0.02                 # alpha_m: Memory energy vs performance trade-off weight
    gamma: float = 0.30                        # gamma: Balances impact of core vs memory on total loss
    beta: float = 0.20                         # beta: History damping factor in WMA weight update
    sample_interval_sec: float = 3.0           # Periodic sampling interval (3 seconds)
    decoupling_factor: int = 40                # Ratio of Tier 1 period to Tier 2 period (>= 40x)


@dataclass
class GreenGPUConfig:
    """Master configuration encompassing hardware profiles and tier configurations."""
    hardware: HardwareProfile = field(default_factory=lambda: LOCAL_RTX_4050)
    tier1: Tier1Config = field(default_factory=Tier1Config)
    tier2: Tier2Config = field(default_factory=Tier2Config)
    random_seed: int = 42
