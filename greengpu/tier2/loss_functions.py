"""
Asymmetric Piecewise Loss Functions for GPU Coordinated DVFS (Section 5.1).
Computes Energy Loss (le) vs. Performance Loss (lp) for discrete core and memory clock levels.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from greengpu.config import Tier2Config


def generate_umean_levels(num_levels: int) -> np.ndarray:
    """
    Linearly maps discrete frequency levels to ideal target utilization levels umean.
    Lowest frequency is ideal for utilization 0.0 (0%); peak frequency is ideal for 1.0 (100%).
    """
    if num_levels <= 1:
        return np.array([1.0], dtype=np.float64)
    return np.linspace(0.0, 1.0, num_levels, dtype=np.float64)


def compute_component_loss(
    utilization: float,
    umean_levels: np.ndarray,
    alpha: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates asymmetric component loss across all frequency levels.
    
    If u > umean[i]:
        le = 0.0
        lp = u - umean[i]
    If u < umean[i]:
        le = umean[i] - u
        lp = 0.0
    
    loss[i] = alpha * le + (1 - alpha) * lp
    
    Returns:
        (total_loss, energy_loss, performance_loss)
    """
    u = np.clip(utilization, 0.0, 1.0)
    
    diff = u - umean_levels
    lp = np.maximum(0.0, diff)          # Performance loss (running too slow for current load)
    le = np.maximum(0.0, -diff)         # Energy loss (wasting power running faster than needed)
    
    loss = (alpha * le) + ((1.0 - alpha) * lp)
    return loss, le, lp


def compute_total_joint_loss(
    core_loss: np.ndarray,
    mem_loss: np.ndarray,
    gamma: float = 0.30
) -> np.ndarray:
    """
    Combines core loss and memory loss into joint loss matrix TotalLoss[i, j].
    TotalLoss_ij = gamma * l_c[i] + (1 - gamma) * l_m[j]
    Args:
        core_loss: Vector of size N (N core frequency levels)
        mem_loss: Vector of size M (M memory frequency levels)
        gamma: Balancing factor (paper default gamma = 0.30)
    Returns:
        Joint loss matrix of shape (N, M) with values in [0.0, 1.0]
    """
    # Outer sum: shape (N, 1) + shape (1, M) -> shape (N, M)
    joint_loss = (gamma * core_loss[:, np.newaxis]) + ((1.0 - gamma) * mem_loss[np.newaxis, :])
    return joint_loss
