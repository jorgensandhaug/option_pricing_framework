import numpy as np
from scipy.stats import norm
from .option_pricing_model import OptionPricingModel

class BlackScholesModel(OptionPricingModel):
    def d1(self, S0, K, T, r, sigma, delta):
        return (np.log(S0 / K) + (r - delta + sigma**2 / 2) * T) / (sigma * np.sqrt(T))

    def d2(self, d1, sigma, T):
        return d1 - sigma * np.sqrt(T)

    def price(self, params: dict):
        S0 = params['initial_stock_price']
        K = params['strike_price']
        T = params['time_to_maturity']
        r = params['risk_free_rate']
        sigma = params['volatility']
        delta = params.get('dividend_yield', 0.0)
        option_type = params['option_type']

        d1 = self.d1(S0, K, T, r, sigma, delta)
        d2 = self.d2(d1, sigma, T)
        if option_type == 'call':
            price = S0 * np.exp(-delta * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-delta * T) * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type")
        return price
