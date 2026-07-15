"""Project Kepler v3.2.2."""

from .environment import enforce_mps_environment

# Phase-4 requires these variables to be fixed before any project module imports
# torch. Importing the top-level package is therefore the environment boundary.
enforce_mps_environment()

from .config import ExperimentConfig, ModelFamilyConfig
from .topology import FiniteFieldRule, TOPOLOGIES, anchored_lags

__all__ = [
    "ExperimentConfig",
    "FiniteFieldRule",
    "ModelFamilyConfig",
    "TOPOLOGIES",
    "anchored_lags",
    "enforce_mps_environment",
]

__version__ = "3.2.2"
