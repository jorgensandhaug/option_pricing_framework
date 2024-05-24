from .simulation_based_option_pricing import SimulationBasedOptionPricingModel

class EuropeanOptionSimulationModel(SimulationBasedOptionPricingModel):
    def price(self, params: dict):
        S0 = params['initial_stock_price']
        K = params['strike_price']
        T = params['time_to_maturity']
        r = params['risk_free_rate']
        sigma = params['volatility']
        delta = params.get('dividend_yield', 0.0)
        option_type = params['option_type']
        steps = params.get('steps', 1000)
        simulations = params.get('simulation_paths', 10000)

        # Simulate price paths
        simulation_params = {
            'initial_stock_price': S0,
            'time_to_maturity': T,
            'risk_free_rate': r,
            'volatility': sigma,
            'dividend_yield': delta,
            'time_steps': steps,
            'simulation_paths': simulations
        }
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
