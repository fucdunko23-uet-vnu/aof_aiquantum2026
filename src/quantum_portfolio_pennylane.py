# -*- coding: utf-8 -*-
"""
Quantum Portfolio Optimization Module using PennyLane and PyTorch.
Implements VQC/QAOA architecture mapped as an Ising Hamiltonian/QUBO formulation.
"""

import sys
import os
from typing import Tuple, List, Optional, Callable, Awaitable, Dict, Any
import numpy as np
import torch
import pennylane as qml

# Configure PennyLane to use PyTorch interface
torch.set_default_dtype(torch.float32)


def get_quantum_device(n_qubits: int) -> qml.devices.Device:
    """
    Initializes and returns a PennyLane quantum device.
    Attempts to use GPU acceleration ('lightning.gpu') if available, 
    otherwise falls back to 'lightning.qubit' or 'default.qubit'.
    
    Args:
        n_qubits (int): Number of qubits (assets).
        
    Returns:
        qml.devices.Device: PennyLane quantum device.
    """
    # Attempt to load lightning.gpu for CUDA acceleration
    try:
        dev = qml.device("lightning.gpu", wires=n_qubits)
        return dev
    except Exception:
        pass
        
    # Attempt to load lightning.qubit for fast CPU simulation
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
        return dev
    except Exception:
        pass
        
    # Fall back to default CPU simulator
    return qml.device("default.qubit", wires=n_qubits)


def make_quantum_circuit(n_qubits: int, n_layers: int = 3) -> Tuple[Callable, qml.devices.Device]:
    """
    Creates a QNode representing the Variational Quantum Circuit (VQC).
    
    Args:
        n_qubits (int): Number of assets/qubits.
        n_layers (int): Number of variational ansatz layers.
        
    Returns:
        Tuple[Callable, qml.devices.Device]: The QNode and the underlying device.
    """
    dev = get_quantum_device(n_qubits)
    
    @qml.qnode(dev, interface="torch")
    def circuit(weights: torch.Tensor, returns: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """
        Quantum circuit for encoding asset metrics and finding optimal portfolios.
        
        Args:
            weights (torch.Tensor): Trainable ansatz weights of shape (n_layers, 2, n_qubits).
            returns (torch.Tensor): Expected asset returns (N,).
            cov (torch.Tensor): Covariance matrix of returns (N, N).
            
        Returns:
            torch.Tensor: Probability distribution over all 2**N basis states.
        """
        # ==========================================
        # 1. QUANTUM FEATURE MAP (Ising Mapping)
        # ==========================================
        # Dynamically scale features into [-pi, pi] to ensure maximum expressiveness in rotation gates
        max_ret = torch.max(torch.abs(returns))
        scaled_returns = returns * (torch.pi / (2.0 * max_ret + 1e-8))
        
        max_cov = torch.max(torch.abs(cov))
        scaled_cov = cov * (torch.pi / (2.0 * max_cov + 1e-8))
        
        # Encode returns (diagonal linear term) using Ry rotations
        for i in range(n_qubits):
            qml.RY(scaled_returns[i], wires=i)
            
        # Encode variance (diagonal quadratic term) using Rx rotations
        for i in range(n_qubits):
            qml.RX(scaled_cov[i, i], wires=i)
            
        # Encode cross-covariances (off-diagonal terms) using CNOT entanglers and Rz rotations
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CNOT(wires=[i, j])
                qml.RZ(scaled_cov[i, j], wires=j)
                qml.CNOT(wires=[i, j])
                
        # ==========================================
        # 2. VARIATIONAL ENTANGLEMENT LAYER
        # ==========================================
        for layer in range(n_layers):
            # Trainable single-qubit rotations
            for i in range(n_qubits):
                qml.RY(weights[layer, 0, i], wires=i)
                qml.RX(weights[layer, 1, i], wires=i)
                
            # Entanglement layer (1D ring topology CNOTs)
            if n_qubits > 1:
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
                    
        return qml.probs(wires=range(n_qubits))
        
    return circuit, dev


def generate_binary_matrix(n: int, device: torch.device) -> torch.Tensor:
    """
    Generates a binary matrix of shape (2**n, n) representing the computational basis states.
    
    Args:
        n (int): Number of variables/qubits.
        device (torch.device): PyTorch device.
        
    Returns:
        torch.Tensor: Matrix of binary state combinations.
    """
    indices = torch.arange(2**n, device=device)
    B = torch.zeros((2**n, n), device=device)
    for i in range(n):
        B[:, n - 1 - i] = (indices >> i) & 1
    return B


def compute_custom_loss(
    probs: torch.Tensor,
    returns: torch.Tensor,
    cov: torch.Tensor,
    risk_appetite: float,
    esg_scores: torch.Tensor,
    target_esg: float,
    binary_matrix: torch.Tensor,
    lambda_esg: float = 1000.0,
    lambda_card: float = 1000.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom loss function combining Mean-Variance optimization with ESG and Cardinality hard penalties.
    Uses PyTorch for automatic differentiation.
    
    Args:
        probs (torch.Tensor): Probabilities of all 2**N states from the QNode. Shape (2**N,).
        returns (torch.Tensor): Expected returns tensor. Shape (N,).
        cov (torch.Tensor): Covariance matrix tensor. Shape (N, N).
        risk_appetite (float): Risk appetite parameter (value in [0, 1]).
                               Objective = (1 - risk_appetite) * Variance - risk_appetite * Return.
        esg_scores (torch.Tensor): ESG scores of the assets. Shape (N,).
        target_esg (float): Minimum target ESG score for the portfolio.
        binary_matrix (torch.Tensor): Binary matrix of shape (2**N, N).
        lambda_esg (float): Hard penalty multiplier for violating ESG constraint.
        lambda_card (float): Hard penalty multiplier for selecting > 5 assets.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - loss: Expected loss value to minimize.
            - expected_return: Portfolio expected return under quantum state.
            - expected_variance: Portfolio variance under quantum state.
            - expected_esg: Portfolio ESG score under quantum state.
    """
    device = probs.device
    N = returns.shape[0]
    
    # S[k] is the number of assets selected in basis state k
    S = binary_matrix.sum(dim=1)
    
    # Prevent division by zero for the empty portfolio state
    S_safe = torch.where(S > 0, S, torch.ones_like(S))
    
    # Portfolio allocation weights for each basis state: W[k, i] = B[k, i] / S_safe[k]
    # If S[k] = 0, all weights are 0
    W = binary_matrix / S_safe.unsqueeze(1)
    W = W * (S > 0).unsqueeze(1).float()
    
    # Calculate portfolio metrics per computational basis state
    R_states = torch.matmul(W, returns)  # Returns per state: (2**N,)
    V_states = torch.sum(torch.matmul(W, cov) * W, dim=1)  # Variance per state: (2**N,)
    ESG_states = torch.matmul(W, esg_scores)  # ESG score per state: (2**N,)
    
    # Base Markowitz objective for each state: (1 - risk_appetite) * Variance - risk_appetite * Return
    objective_states = (1.0 - risk_appetite) * V_states - risk_appetite * R_states
    
    # ESG Constraint Penalty: Penalize if portfolio ESG < target_esg
    # Differentiable hard penalty: lambda_esg * max(0, target_esg - ESG_states)^2
    esg_penalty = torch.relu(target_esg - ESG_states) ** 2
    
    # Cardinality Constraint Penalty: Penalize if number of assets selected > 5
    # Differentiable hard penalty: lambda_card * max(0, S - 5)^2
    card_penalty = torch.relu(S.float() - 5.0) ** 2
    
    # Empty Portfolio Penalty: Heavily penalize selecting no assets
    empty_penalty = torch.where(S == 0, torch.tensor(100.0, device=device), torch.tensor(0.0, device=device))
    
    # Combined cost per state
    cost_states = objective_states + lambda_esg * esg_penalty + lambda_card * card_penalty + empty_penalty
    
    # Calculate expectation value of costs across the quantum probability distribution
    loss = torch.sum(probs * cost_states)
    
    # Average portfolio metrics over the state distribution
    expected_return = torch.sum(probs * R_states)
    expected_variance = torch.sum(probs * V_states)
    expected_esg = torch.sum(probs * ESG_states)
    
    return loss, expected_return, expected_variance, expected_esg


async def optimize_portfolio(
    returns_np: np.ndarray,
    cov_np: np.ndarray,
    risk_appetite: float,
    esg_scores_np: np.ndarray,
    target_esg: float = 70.0,
    epochs: int = 100,
    lr: float = 0.01,
    n_layers: int = 3,
    epoch_callback: Optional[Callable[[int, float, np.ndarray, float, float, float], Awaitable[None]]] = None
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Performs portfolio optimization using PennyLane and PyTorch.
    
    Args:
        returns_np (np.ndarray): 1D array of expected returns.
        cov_np (np.ndarray): 2D covariance matrix.
        risk_appetite (float): Risk appetite factor in [0, 1].
        esg_scores_np (np.ndarray): 1D array of ESG scores.
        target_esg (float): Minimum target ESG score.
        epochs (int): Number of training epochs.
        lr (float): Classical optimizer learning rate.
        n_layers (int): Number of VQC layers.
        epoch_callback (Optional[Callable]): Async callback for streaming metrics per epoch.
                                             Signature: (epoch, loss, weights_np, expected_return, expected_variance, expected_esg)
                                             
    Returns:
        Tuple[np.ndarray, List[Dict[str, Any]]]:
            - optimal_weights (np.ndarray): Portfolio weights (sums to 1.0).
            - history (List[Dict[str, Any]]): Training logs and history metrics.
    """
    n_qubits = len(returns_np)
    
    # Convert input NumPy arrays to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    returns = torch.tensor(returns_np, dtype=torch.float32, device=device)
    cov = torch.tensor(cov_np, dtype=torch.float32, device=device)
    esg_scores = torch.tensor(esg_scores_np, dtype=torch.float32, device=device)
    
    # Initialize the Variational Quantum Circuit QNode
    circuit, dev = make_quantum_circuit(n_qubits, n_layers=n_layers)
    
    # Pre-generate computational basis state matrix
    binary_matrix = generate_binary_matrix(n_qubits, device=device)
    
    # Initialize trainable circuit parameters randomly: shape (n_layers, 2, n_qubits)
    # Using uniform distribution in [-pi, pi]
    weights = torch.nn.Parameter(
        (2 * torch.rand((n_layers, 2, n_qubits), dtype=torch.float32, device=device) - 1) * torch.pi
    )
    
    # Set up PyTorch classical optimizer (Adam)
    optimizer = torch.optim.Adam([weights], lr=lr)
    
    history: List[Dict[str, Any]] = []
    
    # Pre-calculate marginal multiplication matrix for fast weight retrieval
    # marginals[i, k] is B[k, i]
    marginals = binary_matrix.t() # shape (N, 2**N)
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # 1. Run forward pass of the quantum circuit
        probs = circuit(weights, returns, cov).to(dtype=torch.float32)
        
        # 2. Compute custom loss with penalties
        loss, exp_ret, exp_var, exp_esg = compute_custom_loss(
            probs=probs,
            returns=returns,
            cov=cov,
            risk_appetite=risk_appetite,
            esg_scores=esg_scores,
            target_esg=target_esg,
            binary_matrix=binary_matrix,
            lambda_esg=1000.0,
            lambda_card=1000.0
        )
        
        # 3. Perform backpropagation
        loss.backward()
        optimizer.step()
        
        # 4. Extract current optimal weights from the quantum state
        # We calculate the probability of each asset being selected (marginal probability of being in |1>)
        # asset_probs[i] = Sum_{k} probs[k] * B[k, i]
        with torch.no_grad():
            asset_probs = torch.matmul(marginals, probs) # shape (N,)
            total_prob = torch.sum(asset_probs)
            # Normalize to sum to 1.0 (portfolio weights)
            current_weights = (asset_probs / (total_prob + 1e-8)).cpu().numpy()
            
            loss_val = loss.item()
            exp_ret_val = exp_ret.item()
            exp_var_val = exp_var.item()
            exp_esg_val = exp_esg.item()
            
        # Log epoch data
        epoch_data = {
            "epoch": epoch,
            "loss": loss_val,
            "expected_return": exp_ret_val,
            "expected_variance": exp_var_val,
            "expected_esg": exp_esg_val,
            "weights": current_weights.tolist()
        }
        history.append(epoch_data)
        
        # Stream results asynchronously if callback is provided
        if epoch_callback is not None:
            await epoch_callback(
                epoch, 
                loss_val, 
                current_weights, 
                exp_ret_val, 
                exp_var_val, 
                exp_esg_val
            )
            
    # Final weight calculation from the trained quantum state
    with torch.no_grad():
        final_probs = circuit(weights, returns, cov).to(dtype=torch.float32)
        final_asset_probs = torch.matmul(marginals, final_probs)
        final_total_prob = torch.sum(final_asset_probs)
        final_weights = (final_asset_probs / (final_total_prob + 1e-8)).cpu().numpy()
        
    return final_weights, history
