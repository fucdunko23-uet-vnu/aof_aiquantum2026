# -*- coding: utf-8 -*-
"""
Asynchronous FastAPI Server serving Quantum Portfolio Optimization models.
Supports real-time training progress streaming via WebSockets.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import quantum optimization core
from src.quantum_portfolio_pennylane import optimize_portfolio

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuantumServer")

app = FastAPI(
    title="AI-Quantum Portfolio Optimization Server",
    description="High-performance server serving VQC/QAOA portfolio optimizer with real-time updates.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Concurrency control lock to queue quantum calculations under high load
OPTIMIZATION_LOCK = asyncio.Lock()

# Default stock ESG scores (VN30 representation)
DEFAULT_ESG_SCORES: Dict[str, float] = {
    "FPT": 78.0,
    "HPG": 65.0,
    "VCB": 82.0,
    "VNM": 85.0,
    "MSN": 70.0,
    "VIC": 62.0,
    "TCB": 74.0,
    "MWG": 68.0,
}


class PortfolioRequest(BaseModel):
    asset_symbols: List[str] = Field(..., description="List of stock tickers.")
    historical_prices: List[List[float]] = Field(..., description="2D array of historical prices (Timesteps x Assets).")
    risk_profile: str = Field("medium", description="Risk profile: low, medium, or high.")
    target_esg: float = Field(70.0, description="Target ESG score of the portfolio.")
    esg_scores: Optional[Dict[str, float]] = Field(None, description="Optional map of tickers to ESG scores.")
    epochs: int = Field(100, ge=10, le=1000, description="Number of training epochs.")
    lr: float = Field(0.01, gt=0.0, description="Optimizer learning rate.")


def map_risk_profile_to_appetite(risk_profile: str) -> float:
    """
    Maps user risk profile string to risk appetite parameter in [0, 1].
    Low risk profile -> High risk aversion -> Low risk appetite (0.2).
    Medium risk profile -> Balanced -> Moderate risk appetite (0.5).
    High risk profile -> High risk tolerance -> High risk appetite (0.8).
    """
    profile = risk_profile.strip().lower()
    if profile == "low":
        return 0.2
    elif profile == "high":
        return 0.8
    else:
        return 0.5  # default 'medium'


def preprocess_financial_data(prices_2d: List[List[float]], asset_symbols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes daily returns, expected returns vector, and covariance matrix using vectorized NumPy operations.
    Runs at near C++ speed.
    """
    prices = np.array(prices_2d, dtype=np.float64)
    
    if prices.ndim != 2:
        raise ValueError(f"Historical prices must be 2D array, received {prices.ndim}D array.")
        
    T, N = prices.shape
    if N != len(asset_symbols):
        raise ValueError(f"Prices dimensions ({N}) do not match assets length ({len(asset_symbols)}).")
        
    if T < 2:
        raise ValueError(f"Need at least 2 historical price timesteps to calculate returns, received {T}.")
        
    # Calculate daily percent change returns: R_t = (P_t - P_{t-1}) / P_{t-1}
    # Using numpy slicing for speed
    returns = (prices[1:, :] - prices[:-1, :]) / prices[:-1, :]
    
    # Calculate expected returns vector (mean daily returns)
    expected_returns = np.mean(returns, axis=0)
    
    # Calculate covariance matrix
    # rowvar=False because columns represent the assets
    cov_matrix = np.cov(returns, rowvar=False)
    
    # Handle single asset or edge cases
    if N == 1:
        cov_matrix = np.array([[cov_matrix.item()]])
        
    return expected_returns, cov_matrix


def get_asset_esg_scores(asset_symbols: List[str], custom_scores: Optional[Dict[str, float]]) -> np.ndarray:
    """
    Retrieves ESG scores for assets. Fallback to default dictionary or computes deterministic score.
    """
    scores = []
    custom = custom_scores or {}
    
    for symbol in asset_symbols:
        s_upper = symbol.upper().strip()
        if s_upper in custom:
            scores.append(custom[s_upper])
        elif s_upper in DEFAULT_ESG_SCORES:
            scores.append(DEFAULT_ESG_SCORES[s_upper])
        else:
            # Deterministic generator based on symbol string representation
            # Generates a realistic ESG score in [60, 85]
            val = sum(ord(c) for c in s_upper)
            det_score = 60.0 + (val % 26)
            scores.append(det_score)
            
    return np.array(scores, dtype=np.float64)


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "gpu_available": str(torch.cuda.is_available()),
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }


@app.websocket("/ws/quantum-optimize")
async def websocket_quantum_optimize(websocket: WebSocket):
    """
    WebSocket endpoint serving real-time quantum portfolio optimization.
    Streams training epochs and final portfolio allocation metrics.
    """
    await websocket.accept()
    logger.info("New WebSocket client connected.")
    
    try:
        # 1. Receive JSON payload
        data = await websocket.receive_json()
        
        # 2. Validate payload using Pydantic
        try:
            req = PortfolioRequest(**data)
        except Exception as e:
            await websocket.send_json({
                "status": "error",
                "message": f"Validation Error: {str(e)}"
            })
            await websocket.close()
            return
            
        # 3. High-speed Preprocessing using NumPy
        try:
            expected_returns, cov_matrix = preprocess_financial_data(
                prices_2d=req.historical_prices,
                asset_symbols=req.asset_symbols
            )
            esg_scores = get_asset_esg_scores(
                asset_symbols=req.asset_symbols,
                custom_scores=req.esg_scores
            )
        except Exception as e:
            await websocket.send_json({
                "status": "error",
                "message": f"Pre-processing Error: {str(e)}"
            })
            await websocket.close()
            return
            
        risk_appetite = map_risk_profile_to_appetite(req.risk_profile)
        
        # 4. Stream status: start of optimization
        await websocket.send_json({
            "status": "started",
            "message": "Data preprocessing completed. Quantum hardware queue requested.",
            "metrics": {
                "num_assets": len(req.asset_symbols),
                "expected_returns": expected_returns.tolist(),
                "esg_scores": esg_scores.tolist()
            }
        })
        
        # 5. Define asynchronous callback to stream training updates
        async def on_epoch(
            epoch: int, 
            loss_val: float, 
            weights_np: np.ndarray, 
            exp_ret_val: float, 
            exp_var_val: float, 
            exp_esg_val: float
        ) -> None:
            try:
                await websocket.send_json({
                    "status": "training",
                    "epoch": epoch,
                    "loss": loss_val,
                    "weights": weights_np.tolist(),
                    "metrics": {
                        "expected_return": float(exp_ret_val),
                        "portfolio_variance": float(exp_var_val),
                        "esg_score": float(exp_esg_val)
                    }
                })
                # Add tiny sleep to allow the event loop to yield to other tasks
                await asyncio.sleep(0.001)
            except Exception:
                pass  # Client might have disconnected during training
                
        # 6. Execute core quantum optimization under concurrency lock
        # This protects from race conditions and GPU resource starvation
        logger.info(f"Queuing optimization job for {len(req.asset_symbols)} assets.")
        async with OPTIMIZATION_LOCK:
            logger.info("Executing optimization job.")
            final_weights, history = await optimize_portfolio(
                returns_np=expected_returns,
                cov_np=cov_matrix,
                risk_appetite=risk_appetite,
                esg_scores_np=esg_scores,
                target_esg=req.target_esg,
                epochs=req.epochs,
                lr=req.lr,
                epoch_callback=on_epoch
            )
            
        # 7. Optimization successfully completed, send final payload
        await websocket.send_json({
            "status": "completed",
            "weights": final_weights.tolist(),
            "metrics": {
                "expected_return": float(history[-1]["expected_return"]),
                "portfolio_variance": float(history[-1]["expected_variance"]),
                "esg_score": float(history[-1]["expected_esg"])
            }
        })
        logger.info("Optimization job successfully completed.")
        
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"Unexpected error in websocket connection: {str(e)}")
        try:
            await websocket.send_json({
                "status": "error",
                "message": f"Server Error: {str(e)}"
            })
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
