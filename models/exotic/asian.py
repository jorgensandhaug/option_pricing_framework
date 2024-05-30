from models.simulation_based_option_pricing import SimulationBasedOptionPricingModel
from models.option_pricing_model import OptionPricingModel
import numpy as np
from scipy.stats import norm

class AsianOptionSimulationModel(SimulationBasedOptionPricingModel):
    def price(self, params: dict, simulation_params: dict=None, simulated_prices: np.array=None):
        K = params['strike_price']
        T = params['time_to_maturity']
        r = params['risk_free_rate']
        option_type = params['option_type']
        asian_type = params['asian_type']
        average_type = params['average_type']

        if simulated_prices is None:
            simulated_prices = self.simulator.simulate(simulation_params)

        # Calculate the average price for each path
        if average_type == 'arithmetic':
            average_prices = np.mean(simulated_prices, axis=1)
        elif average_type == 'geometric':
            average_prices = np.exp(np.mean(np.log(simulated_prices), axis=1))
        else:
            raise ValueError("Invalid average type")

        if asian_type == 'price':
            S1 = average_prices
            S2 = K
        elif asian_type == 'strike':
            S1 = simulated_prices[:, -1]
            S2 = average_prices
        else:
            raise ValueError("Invalid Asian option type")
        
        # Calculate the payoff for each path
        if option_type == 'call':
            payoffs = np.maximum(S1 - S2, 0)
        elif option_type == 'put':
            payoffs = np.maximum(S2 - S1, 0)
        else:
            raise ValueError("Invalid option type")

        # Discount the payoffs back to the present value
        discounted_payoffs = np.exp(-r * T) * payoffs

        # Calculate the option price as the average discounted payoff
        option_price = np.mean(discounted_payoffs)
        return option_price

    def arithmetic_price_geometric_control_variate(self, params: dict, simulation_params: dict=None):
        K = params['strike_price']
        T = params['time_to_maturity']
        r = params['risk_free_rate']
        option_type = params['option_type']

        simulated_prices = self.simulator.simulate(simulation_params)

        # Calculate the average price for each path
        average_prices = np.mean(simulated_prices, axis=1)
        geometric_average_prices = np.exp(np.mean(np.log(simulated_prices), axis=1))

        # Calculate the payoff for each path
        if option_type == 'call':
            payoffs = np.maximum(average_prices - K, 0)
            geometric_payoffs = np.maximum(geometric_average_prices - K, 0)
        elif option_type == 'put':
            payoffs = np.maximum(K - average_prices, 0)
            geometric_payoffs = np.maximum(K - geometric_average_prices, 0)
        else:
            raise ValueError("Invalid option type")

        # Discount the payoffs back to the present value
        discounted_payoffs = np.exp(-r * T) * payoffs
        geometric_discounted_payoffs = np.exp(-r * T) * geometric_payoffs

        # Simulate some more paths to estimate the covariance
        additional_simulated_prices = self.simulator.simulate(simulation_params)
        additional_average_prices = np.mean(additional_simulated_prices, axis=1)
        additional_geometric_average_prices = np.exp(np.mean(np.log(additional_simulated_prices), axis=1))
        
        if option_type == 'call':
            additional_payoffs = np.maximum(additional_average_prices - K, 0)
            additional_geometric_payoffs = np.maximum(additional_geometric_average_prices - K, 0)
        elif option_type == 'put':
            additional_payoffs = np.maximum(K - additional_average_prices, 0)
            additional_geometric_payoffs = np.maximum(K - additional_geometric_average_prices, 0)
        else:
            raise ValueError("Invalid option type")

        additional_discounted_payoffs = np.exp(-r * T) * additional_payoffs
        additional_geometric_discounted_payoffs = np.exp(-r * T) * additional_geometric_payoffs
        
        # Calculate the covariance
        cov = np.cov(additional_discounted_payoffs, additional_geometric_discounted_payoffs)
        
        # Now back to the option price
        geometric_option_pricer = AnalyticalGeometricAsianOptionPricingModel()
        true_geometric_option_price = geometric_option_pricer.price({**simulation_params, **params})

        adjusted_arithmetic_option_prices = discounted_payoffs + cov[0, 1] / cov[1, 1] * (true_geometric_option_price - geometric_discounted_payoffs)

        # Calculate the option price as the average discounted payoff
        option_price = np.mean(adjusted_arithmetic_option_prices)
        
        return option_price

        
                                                     


class AnalyticalGeometricAsianOptionPricingModel(OptionPricingModel):
    def price(self, params: dict):
        S0 = params['initial_stock_price']
        K = params['strike_price']
        T = params['time_to_maturity']
        r = params['risk_free_rate']
        sigma = params['volatility']
        n = params['time_steps']
        asian_type = params['asian_type']

        if asian_type == 'price':
            mu = (r - 0.5 * sigma**2) * (n + 1) / (2 * n) + 0.5 * sigma**2 / n
            sigma_hat = sigma * np.sqrt((n + 1) * (2 * n + 1) / (6 * n**2))
            d1 = (np.log(S0 / K) + (mu + 0.5 * sigma_hat**2) * T) / (sigma_hat * np.sqrt(T))
            d2 = d1 - sigma_hat * np.sqrt(T)
            price = S0 * np.exp((mu - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif asian_type == 'strike':
            mu = (r - 0.5 * sigma**2) * (n - 1) / (2 * n) + 0.5 * sigma**2 / n
            sigma_hat = sigma * np.sqrt((n - 1) * (2 * n - 1) / (6 * n**2))
            d1 = (np.log(S0 / K) + (mu + 0.5 * sigma_hat**2) * T) / (sigma_hat * np.sqrt(T))
            d2 = d1 - sigma_hat * np.sqrt(T)
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp((mu - r) * T) * norm.cdf(-d1)
        else:
            raise ValueError("Invalid Asian option type")

        return price
            
             