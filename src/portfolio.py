# -*- coding: utf-8 -*-
"""
Module to formulate the Portfolio Optimization problem and map it to QUBO.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import DocplexTranslator

def build_portfolio_quadratic_program(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_factor: float,
    budget: int
) -> QuadraticProgram:
    """
    Formulate the mean-variance portfolio optimization problem as a Qiskit QuadraticProgram.
    
    Objective:
        minimize risk_factor * w^T * Cov * w - returns^T * w
        subject to: sum(w) = budget
        where w is a binary vector (1 = select stock, 0 = do not select).
        
    Args:
        expected_returns (pd.Series): Expected returns for each asset.
        covariance_matrix (pd.DataFrame): Covariance matrix of asset returns.
        risk_factor (float): Risk appetite parameter (lambda).
        budget (int): The number of assets to select in the portfolio.
        
    Returns:
        QuadraticProgram: The formulated optimization problem.
    """
    # TODO: Build and return the QuadraticProgram using Qiskit Optimization
    pass
