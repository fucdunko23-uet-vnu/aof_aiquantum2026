# -*- coding: utf-8 -*-
"""
Module to run classical solvers for benchmarking portfolio optimization results.
"""

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver
from typing import Dict, Any

def solve_portfolio_classically(qp: QuadraticProgram) -> Dict[str, Any]:
    """
    Solve the formulated portfolio QuadraticProgram classically using an exact solver (NumPyMinimumEigensolver).
    
    Args:
        qp (QuadraticProgram): The portfolio optimization problem.
        
    Returns:
        Dict[str, Any]: Dict containing the exact classical optimal solution.
    """
    # TODO: Set up NumPyMinimumEigensolver and MinimumEigenOptimizer to solve classically.
    pass
