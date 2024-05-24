import numpy as np
from .option_pricing_model import OptionPricingModel

class BinomialModel(OptionPricingModel):
    def __init__(self, steps=1000):
        self.steps = steps

    def risk_neutral_prob(self, r, delta, h, u, d):
        return (np.exp((r - delta) * h) - d) / (u - d)

    def ud_binomial(self, sigma, h, r, delta):
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

        # Initialize stock price and option value grids
        stock_prices = np.zeros((steps + 1, steps + 1))
        option_values = np.zeros((steps + 1, steps + 1))

        # Calculate stock prices
        for i in range(steps + 1):
            for j in range(i + 1):
                stock_prices[j, i] = S0 * (u ** (i - j)) * (d ** j)

        # Calculate option values at maturity
        if option_type == 'call':
            option_values[:, steps] = np.maximum(0, stock_prices[:, steps] - K)
        elif option_type == 'put':
            option_values[:, steps] = np.maximum(0, K - stock_prices[:, steps])
        else:
            raise ValueError("Invalid option type")

        # Backward induction
        discount_factor = np.exp(-r * h)
        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                option_values[j, i] = discount_factor * (p * option_values[j, i + 1] + (1 - p) * option_values[j + 1, i + 1])
                if is_american:
                    if option_type == 'call':
                        option_values[j, i] = np.maximum(option_values[j, i], stock_prices[j, i] - K)
                    elif option_type == 'put':
                        option_values[j, i] = np.maximum(option_values[j, i], K - stock_prices[j, i])

        return option_values[0, 0]
