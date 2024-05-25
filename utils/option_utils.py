import numpy as np
from scipy.stats import norm
from models.european.black_scholes import BlackScholesModel

class OptionUtils:
    @staticmethod
    def vega(params, sigma):
        S = params['initial_stock_price']
        K = params['strike_price']
        T = params['time_to_maturity']
        r = params['risk_free_rate']
        q = params.get('dividend_yield', 0.0)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return S * np.sqrt(T) * norm.pdf(d1) * np.exp(-q * T)

    @staticmethod
    def find_implied_volatility(params: dict, market_price, tol=1e-8, max_iterations=100, initial_guess=0.2, lower_bound=1e-6, upper_bound=5.0):

        pricer = BlackScholesModel()

        sigma = initial_guess  # initial guess
        for i in range(max_iterations):
            price = pricer.price({**params, "volatility": sigma})
            diff = market_price - price  # f(sigma)
            if abs(diff) < tol:
                return sigma, i, True  # return implied volatility, number of iterations, and convergence status
            v = OptionUtils.vega(params, sigma)
            if v < 1e-8:
                break

            sigma = sigma + diff / v  # Newton's update

            # Ensure sigma stays within bounds
            if sigma < lower_bound or sigma > upper_bound:
                break

        # If Newton-Raphson fails, use bisection method
        low, high = lower_bound, upper_bound
        for i in range(max_iterations):
            mid = (low + high) / 2.0
            price = pricer.price({**params, "volatility": mid})
            diff = market_price - price
            if abs(diff) < tol:
                return mid, i, True
            if price < market_price:
                low = mid
            else:
                high = mid

        return sigma, i, False  # return the last computed implied volatility, iterations, and convergence status
