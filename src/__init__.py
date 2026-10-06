"""
stroke_rehab_bci.src
=====================
Subject-independent motor imagery BCI pipeline for acute stroke
rehabilitation (Liu et al., 2024 dataset, via MOABB).

Modules
-------
data_loader        MOABB ingestion helpers (no fitting/transforming happens here)
feature_extraction Custom NumPy CSP implementation + spatial utilities
pipelines           scikit-learn / pyriemann pipeline builders + LOSO-CV driver
simulation_stream   Circular-buffer streaming simulator for the real-time demo
"""

__all__ = ["data_loader", "feature_extraction", "pipelines", "simulation_stream"]
