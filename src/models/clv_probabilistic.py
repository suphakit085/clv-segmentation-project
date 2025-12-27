"""
CLV Probabilistic Models Module
===============================
Probabilistic models for Customer Lifetime Value prediction.
Includes BG/NBD and Pareto/NBD models.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1, betaln


class BGNBDModel:
    """
    BG/NBD (Beta Geometric/Negative Binomial Distribution) Model.
    
    This model predicts the number of transactions a customer will make
    in a future time period, accounting for customer "death" (churn).
    """
    
    def __init__(self):
        self.params = None
        self.fitted = False
    
    def _log_likelihood(self, params: np.ndarray, 
                        frequency: np.ndarray, 
                        recency: np.ndarray, 
                        T: np.ndarray) -> float:
        """
        Calculate the negative log-likelihood for the BG/NBD model.
        
        Args:
            params: Model parameters [r, alpha, a, b]
            frequency: Number of repeat transactions
            recency: Time of last transaction (in periods)
            T: Customer age (in periods)
            
        Returns:
            Negative log-likelihood value
        """
        r, alpha, a, b = params
        
        if r <= 0 or alpha <= 0 or a <= 0 or b <= 0:
            return 1e10
        
        try:
            # BG/NBD log-likelihood calculation
            ln_A_1 = gammaln(r + frequency) - gammaln(r) + r * np.log(alpha)
            ln_A_2 = gammaln(a + b) + gammaln(b + frequency) - gammaln(b) - gammaln(a + b + frequency)
            ln_A_3 = -(r + frequency) * np.log(alpha + T)
            ln_A_4 = np.log(a) - np.log(a + b + frequency - 1) - (r + frequency) * np.log(alpha + recency)
            
            # For customers with frequency > 0
            freq_mask = frequency > 0
            ll = np.zeros_like(frequency, dtype=float)
            
            ll[~freq_mask] = ln_A_1[~freq_mask] + ln_A_2[~freq_mask] + ln_A_3[~freq_mask]
            ll[freq_mask] = ln_A_1[freq_mask] + ln_A_2[freq_mask] + np.log(
                np.exp(ln_A_3[freq_mask]) + np.exp(ln_A_4[freq_mask])
            )
            
            return -np.sum(ll)
        except Exception:
            return 1e10
    
    def fit(self, frequency: np.ndarray, recency: np.ndarray, T: np.ndarray) -> 'BGNBDModel':
        """
        Fit the BG/NBD model to customer data.
        
        Args:
            frequency: Number of repeat transactions for each customer
            recency: Time of last transaction for each customer
            T: Customer age for each customer
            
        Returns:
            Self
        """
        initial_params = [1.0, 1.0, 1.0, 1.0]
        bounds = [(1e-4, None), (1e-4, None), (1e-4, None), (1e-4, None)]
        
        result = minimize(
            self._log_likelihood,
            initial_params,
            args=(frequency, recency, T),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        self.params = result.x
        self.fitted = True
        
        return self
    
    def predict_transactions(self, frequency: np.ndarray, 
                             recency: np.ndarray, 
                             T: np.ndarray,
                             t: float) -> np.ndarray:
        """
        Predict number of transactions in the next t periods.
        
        Args:
            frequency: Number of repeat transactions
            recency: Time of last transaction
            T: Customer age
            t: Prediction period length
            
        Returns:
            Expected number of transactions
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        r, alpha, a, b = self.params
        
        # Simplified expected transactions calculation
        expected = (a + b + frequency - 1) / (a - 1) * \
                   (1 - ((alpha + T) / (alpha + T + t)) ** (r + frequency))
        
        return expected
    
    def predict_alive_probability(self, frequency: np.ndarray,
                                   recency: np.ndarray,
                                   T: np.ndarray) -> np.ndarray:
        """
        Predict the probability that a customer is still "alive".
        
        Args:
            frequency: Number of repeat transactions
            recency: Time of last transaction
            T: Customer age
            
        Returns:
            Probability of being alive
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        r, alpha, a, b = self.params
        
        # Simplified probability calculation
        p_alive = 1 / (1 + (frequency > 0) * (a / (b + frequency - 1)) * \
                      ((alpha + T) / (alpha + recency)) ** (r + frequency))
        
        return p_alive


class GammaGammaModel:
    """
    Gamma-Gamma Model for estimating average transaction value.
    Used in conjunction with BG/NBD or Pareto/NBD for CLV calculation.
    """
    
    def __init__(self):
        self.params = None
        self.fitted = False
    
    def fit(self, frequency: np.ndarray, monetary_value: np.ndarray) -> 'GammaGammaModel':
        """
        Fit the Gamma-Gamma model.
        
        Args:
            frequency: Number of transactions (must be > 0)
            monetary_value: Average transaction value per customer
            
        Returns:
            Self
        """
        # Filter for customers with frequency > 0
        mask = frequency > 0
        x = frequency[mask]
        m = monetary_value[mask]
        
        # Simple parameter estimation using method of moments
        p = np.mean(m)
        q = np.var(m) / p if np.var(m) > 0 else 1.0
        gamma = p / q if q > 0 else 1.0
        
        self.params = {'p': p, 'q': max(q, 0.01), 'gamma': max(gamma, 0.01)}
        self.fitted = True
        
        return self
    
    def predict_monetary_value(self, frequency: np.ndarray, 
                               monetary_value: np.ndarray) -> np.ndarray:
        """
        Predict expected average transaction value.
        
        Args:
            frequency: Number of transactions
            monetary_value: Historical average transaction value
            
        Returns:
            Expected average transaction value
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        q = self.params['q']
        gamma = self.params['gamma']
        
        # Expected monetary value calculation
        expected_m = gamma * monetary_value / (gamma + frequency - 1)
        
        return expected_m


def calculate_clv(expected_transactions: np.ndarray,
                  expected_monetary_value: np.ndarray,
                  discount_rate: float = 0.1,
                  time_periods: int = 12) -> np.ndarray:
    """
    Calculate Customer Lifetime Value.
    
    Args:
        expected_transactions: Expected number of transactions per period
        expected_monetary_value: Expected average transaction value
        discount_rate: Discount rate per period
        time_periods: Number of periods to consider
        
    Returns:
        Estimated CLV for each customer
    """
    clv = expected_transactions * expected_monetary_value * (1 / (1 + discount_rate))
    
    return clv


if __name__ == "__main__":
    print("CLV Probabilistic Models Module")
