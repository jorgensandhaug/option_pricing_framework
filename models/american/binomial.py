import numpy as np
from ..option_pricing_model import OptionPricingModel

class BinomialModel(OptionPricingModel):
    def __init__(self, steps=1000):
        self.steps = steps

    def risk_neutral_prob(self, r, delta, h, u, d):
        return (np.exp((r - delta) * h) - d) / (u - d)

    def cox_ross_rubinstein(self, sigma, h):
        u = np.exp(sigma * np.sqrt(h))
        d = 1 / u
        return u, d

    def jarrow_rudd(self, sigma, h, r, delta):
        x = np.exp((r - delta + sigma**2 / 2) * h)
        u = x * np.exp(sigma * np.sqrt(h))
        d = x * np.exp(-sigma * np.sqrt(h))
        return u, d

    def ud_binomial(self,sigma, h, r, delta):
        u = np.exp((r - delta) * h + sigma * np.sqrt(h))
        d = np.exp((r - delta) * h - sigma * np.sqrt(h))
        return u, d

    def price(self, params: dict):
        S0 = params['initial_stock_price']
        K = params['strike_price']
        T = params['time_to_maturity']
        r = params['risk_free_rate']
        sigma = params['volatility']
        delta = params.get('dividend_yield', 0.0)
        option_type = params['option_type']
        is_american = params.get('is_american', False)
        steps = params.get('steps', self.steps)

        h = T / steps
        u, d = self.ud_binomial(sigma, h, r, delta)
        p = self.risk_neutral_prob(r, delta, h, u, d)

        # Initialize the stock price grid
        stock_prices = np.zeros((steps + 1, steps + 1))  # A two-dimensional grid of zeros

        # Create an array of indices from 0 to N
        n = np.arange(steps + 1)

        # Create a 2D grid of indices
        N1, N2 = np.meshgrid(n, n)

        # Calculate stock prices efficiently
        stock_prices = S0 * np.power(u, N2) * np.power(d, (N1 - N2))

        # Initialize the option values grid to just the exercise values
        option_values = np.maximum(stock_prices - K, 0) if option_type == 'call' else np.maximum(K - stock_prices, 0)

        # Perform backward induction to calculate the option values at each node
        discount_factor = np.exp(-r * h)
        for n in reversed(range(steps)):
            option_values[:n+1, n] = discount_factor * (p * option_values[1:n+2, n+1] + (1 - p) * option_values[:n+1, n+1])
            if is_american:
                exercise_values = np.maximum(stock_prices[:n+1, n] - K, 0) if option_type == 'call' else np.maximum(K - stock_prices[:n+1, n], 0)
                option_values[:n+1, n] = np.maximum(option_values[:n+1, n], exercise_values)



        return option_values[0, 0]