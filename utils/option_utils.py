import numpy as np
from scipy.stats import norm

class OptionUtils:
    @staticmethod
    def black_scholes_put(S, K, T, r, q, sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    @staticmethod
    def vega(S, K, T, r, q, sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return S * np.sqrt(T) * norm.pdf(d1) * np.exp(-q * T)

    @staticmethod
    def find_implied_volatility(params: dict, P_market, tol=1e-8, max_iterations=100):
        S = params['initial_stock_price']
        K = params['strike_price']
        T = params['time_to_maturity']
        r = params['risk_free_rate']
        q = params.get('dividend_yield', 0.0)

        sigma = 0.2  # initial guess
        for i in range(max_iterations):
            P_bs = OptionUtils.black_scholes_put(S, K, T, r, q, sigma)
            diff = P_market - P_bs  # f(sigma)
            if abs(diff) < tol:
                return sigma, i  # return implied volatility and number of iterations
            v = OptionUtils.vega(S, K, T, r, q, sigma)
            if v < 1e-8:
                break
            sigma = sigma + diff / v  # Newton's update
        return sigma, i  # return the last computed implied volatility and iterations
