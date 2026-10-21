"""
Online Weighted Majority Algorithm (WMA) Controller for Coordinated GPU DVFS (Section 5.1).
Maintains an N x M weight table and updates frequency levels dynamically based on utilization.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from greengpu.config import HardwareProfile, Tier2Config, TESTBED_A_8800GTX
from greengpu.tier2.loss_functions import (
    compute_component_loss,
    compute_total_joint_loss,
    generate_umean_levels,
)


@dataclass
class DVFSDecision:
    """Record of an individual Tier 2 DVFS decision."""
    interval: int
    core_utilization: float
    memory_utilization: float
    core_freq_index: int
    memory_freq_index: int
    selected_core_mhz: int
    selected_memory_mhz: int
    min_loss: float
    weight_table_snapshot: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))


class WMAFrequencyController:
    """
    Online learning controller using Weighted Majority Algorithm.
    Tracks runtime core/memory utilization and enforces the optimal (fc, fm) clock pair.
    """

    def __init__(
        self,
        hardware: Optional[HardwareProfile] = None,
        config: Optional[Tier2Config] = None
    ):
        self.hardware = hardware or TESTBED_A_8800GTX
        self.config = config or Tier2Config()

        self.core_freqs = np.array(self.hardware.core_frequencies_mhz, dtype=np.int32)
        self.mem_freqs = np.array(self.hardware.memory_frequencies_mhz, dtype=np.int32)

        self.N = len(self.core_freqs)
        self.M = len(self.mem_freqs)

        # Generate target umean vectors for core and memory
        self.uc_mean = generate_umean_levels(self.N)
        self.um_mean = generate_umean_levels(self.M)

        # Initialize N x M weight table to zeros (Section 5.1: equal initial preference)
        self.weight_table = np.zeros((self.N, self.M), dtype=np.float64)
        
        self.history: List[DVFSDecision] = []
        self.current_interval = 0

    def update(
        self,
        core_utilization: float,
        memory_utilization: float
    ) -> Tuple[int, int]:
        """
        Process utilization observations, update WMA weight table, and return selected clocks.
        Args:
            core_utilization: Core busy ratio [0.0 - 1.0] or [0.0 - 100.0]
            memory_utilization: Memory bandwidth ratio [0.0 - 1.0] or [0.0 - 100.0]
        Returns:
            (selected_core_mhz, selected_memory_mhz)
        """
        self.current_interval += 1

        # Normalize utilization to [0.0, 1.0]
        uc = core_utilization / 100.0 if core_utilization > 1.0 else core_utilization
        um = memory_utilization / 100.0 if memory_utilization > 1.0 else memory_utilization

        # 1. Compute asymmetric core and memory losses
        lc, _, _ = compute_component_loss(uc, self.uc_mean, self.config.alpha_core)
        lm, _, _ = compute_component_loss(um, self.um_mean, self.config.alpha_memory)

        # 2. Compute Total Joint Loss matrix of shape (N, M)
        total_loss = compute_total_joint_loss(lc, lm, self.config.gamma)

        # 3. Update WMA weight table:
        # weight(t+1)_ij = weight(t)_ij + [1 - (1 - beta) * TotalLoss_ij]
        reward = 1.0 - ((1.0 - self.config.beta) * total_loss)
        self.weight_table += reward

        # 4. Select frequency pair corresponding to highest weight
        best_flat_idx = int(np.argmax(self.weight_table))
        best_i, best_j = np.unravel_index(best_flat_idx, (self.N, self.M))

        selected_fc = int(self.core_freqs[best_i])
        selected_fm = int(self.mem_freqs[best_j])

        decision = DVFSDecision(
            interval=self.current_interval,
            core_utilization=uc,
            memory_utilization=um,
            core_freq_index=best_i,
            memory_freq_index=best_j,
            selected_core_mhz=selected_fc,
            selected_memory_mhz=selected_fm,
            min_loss=float(total_loss[best_i, best_j]),
            weight_table_snapshot=self.weight_table.copy()
        )
        self.history.append(decision)

        return selected_fc, selected_fm

    def reset(self) -> None:
        """Reset weight table and decision history."""
        self.weight_table.fill(0.0)
        self.history.clear()
        self.current_interval = 0
