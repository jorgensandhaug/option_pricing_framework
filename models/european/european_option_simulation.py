from ..simulation_based_option_pricing import SimulationBasedOptionPricingModel
import numpy as np

class EuropeanOptionSimulationModel(SimulationBasedOptionPricingModel):
    def price(self, params: dict, simulation_params: dict=None, simulated_prices: np.array=None):
        K = params['strike_price']
        T = params['time_to_maturity']
        r = params['risk_free_rate']
        option_type = params['option_type']

        if simulated_prices is None:
            simulated_prices = self.simulator.simulate(simulation_params)

        # Calculate the payoff for each path
        if option_type == 'call':
            payoffs = np.maximum(simulated_prices[:, -1] - K, 0)
        elif option_type == 'put':
            payoffs = np.maximum(K - simulated_prices[:, -1], 0)
        else:
            raise ValueError("Invalid option type")

        # Discount the payoffs back to the present value
        discounted_payoffs = np.exp(-r * T) * payoffs

        # Calculate the option price as the average discounted payoff
        option_price = np.mean(discounted_payoffs)
        return option_price
