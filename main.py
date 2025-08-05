"""
MVST Prediction Algorithm
This file implements the Minimum Variance Self-Tuning (MVST) prediction algorithm
for time-series forecasting, as described in the paper. The algorithm uses recursive
least squares with error weighting for single- and multi-step predictions.

Dependencies: numpy

Functions:
- update_parameters: Updates model parameters using recursive least squares.
- ewma_error: Applies exponential weighted moving average to smooth errors.
- predict_and_validate: Main function for MVST prediction and validation.

Usage:
    predictions, mape = predict_and_validate(time_stamps, Y, U, d=1, ...)
"""

import numpy as np

def update_parameters(P, K, theta, e_weighted, phi, lambda_factor, correction_factor=1.0):
    """
    Update model parameters using recursive least squares.
    
    Args:
        P (ndarray): Covariance matrix.
        K (ndarray): Gain vector.
        theta (ndarray): Parameter vector.
        e_weighted (float): Weighted prediction error.
        phi (ndarray): Historical vector of past outputs and inputs.
        lambda_factor (float): Forgetting factor (0 < lambda_factor <= 1).
        correction_factor (float): Correction factor for parameter update.
    
    Returns:
        tuple: Updated P, theta, K.
    """
    K = np.dot(P, phi) / (lambda_factor + np.dot(np.dot(phi.T, P), phi))
    theta = theta + correction_factor * K * e_weighted
    P = (P - np.outer(K, np.dot(phi.T, P))) / lambda_factor
    return P, theta, K

def ewma_error(e, previous_error, alpha=1.0):
    """
    Apply exponential weighted moving average to smooth prediction error.
    
    Args:
        e (float): Current prediction error.
        previous_error (float): Previous smoothed error.
        alpha (float): Smoothing factor (0 <= alpha <= 1).
    
    Returns:
        float: Smoothed error.
    """
    return alpha * e + (1 - alpha) * previous_error

def predict_and_validate(time_stamps, Y, U, d=1, lambda_factor=0.99, correction_factor=1.0, D=0.5, amend_max=8, w_i=None):
    """
    Perform MVST prediction and validation for single- or multi-step forecasting.
    
    Args:
        time_stamps (ndarray): Array of timestamps.
        Y (ndarray): Target values (e.g., load or emissions).
        U (ndarray): Exogenous inputs (e.g., temperature).
        d (int): Prediction horizon (number of steps).
        lambda_factor (float): Forgetting factor for parameter update.
        correction_factor (float): Correction factor for parameter update.
        D (float): Error correction factor.
        amend_max (float): Maximum amplitude for error correction.
        w_i (ndarray, optional): Error weights for multi-step prediction (default: uniform weights).
    
    Returns:
        tuple: List of predictions and MAPE (Mean Absolute Percentage Error).
    """
    if w_i is None:
        w_i = np.ones(d)  # Default to uniform weights
    P = np.eye(2 * d)
    theta = 0.5 * np.ones(2 * d)
    K = np.zeros(2 * d)
    predictions = list(Y[:d])  # Initialize with actual values
    errors = []
    previous_error = 0
    cut = 0
    
    for k in range(d, len(time_stamps), d):
        phi = np.concatenate([Y[k-d:k][::-1], U[k-d:k][::-1]])
        multi_pred = []
        multi_err = []
        
        for step in range(min(d, len(time_stamps) - k)):
            D_amend = D * previous_error
            if np.abs(D_amend) > amend_max:
                D_amend = D_amend / np.abs(D_amend) * amend_max
                cut += 1
            y_pred = np.dot(phi, theta) + D_amend
            multi_pred.append(y_pred)
            
            e_step = Y[k + step] - y_pred
            multi_err.append(np.abs(e_step))
        
        predictions.extend(multi_pred)
        
        if multi_err:
            e_weighted = np.sum(w_i[:len(multi_err)] * multi_err)
            smoothed_error = ewma_error(e_weighted, previous_error, alpha=1)
            for idx, err in enumerate(multi_err):
                if Y[k + idx] != 0:
                    errors.append(err / Y[k + idx])
            previous_error = smoothed_error
            
            P, theta, K = update_parameters(P, K, theta, smoothed_error, phi, lambda_factor, correction_factor)
    
    MAPE = np.mean(errors) * 100 if errors else None  # Convert to percentage
    print(f"Over-limit corrections: {cut}")
    return predictions, MAPE
