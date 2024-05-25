import numpy as np
from scipy.stats.qmc import Sobol
from scipy.stats import norm
from simulations.simulation_model import SimulationModel
from distributions.normal_distribution import NormalDistribution

class GeometricBrownianMotion(SimulationModel):
    def simulate(self, simulation_params:dict=None):
        if simulation_params is not None:
            self.simulation_params = simulation_params

        S0 = self.simulation_params['initial_stock_price']
        T = self.simulation_params['time_to_maturity']
        r = self.simulation_params['risk_free_rate']
        sigma = self.simulation_params['volatility']
        delta = self.simulation_params.get('dividend_yield', 0.0)
        steps = self.simulation_params.get('time_steps', 1000)
        simulations = self.simulation_params.get('simulation_paths', 10000)

        dt = T / steps
        drift = (r - delta - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        prices = np.zeros((simulations, steps + 1))
        prices[:, 0] = S0

        # Generate Sobol sequence
        sobol = Sobol(d=steps, scramble=True)
        m = int(np.ceil(np.log2(simulations)))
        sobol_samples = sobol.random_base2(m=m)
        
        # Ensure the number of samples matches the number of simulations
        if sobol_samples.shape[0] > simulations:
            sobol_samples = sobol_samples[:simulations, :]
        elif sobol_samples.shape[0] < simulations:
            raise ValueError("Number of Sobol samples is less than the number of simulations")

        # Transform Sobol samples to standard normal distribution
        z = self.simulation_params.get('distribution_model', NormalDistribution()).ppf(sobol_samples)

        prices[:, 1:] = S0 * np.exp(np.cumsum(drift + diffusion * z, axis=1))
        return prices
    


class GBMDiscreteStepVolatilities(SimulationModel):
    def simulate(self, simulation_params: dict = None):
        if simulation_params is not None:
            self.simulation_params = simulation_params

        S0 = self.simulation_params['initial_stock_price']
        T = self.simulation_params['time_to_maturity']
        r = self.simulation_params['risk_free_rate']
        delta = self.simulation_params.get('dividend_yield', 0.0)
        simulations = self.simulation_params.get('simulation_paths', 10000)

        volatilities = self.simulation_params.get('volatilities', None)
        if volatilities is None:
            raise ValueError("Volatilities must be provided")

        # Volatilities are assumed to be equally spaced
        steps = volatilities.shape[0]
        dt = T / steps
        prices = np.zeros((simulations, steps + 1))
        prices[:, 0] = S0

        sobol = Sobol(d=steps, scramble=True)
        m = int(np.ceil(np.log2(simulations)))
        sobol_samples = sobol.random_base2(m=m)

        # Ensure the number of samples matches the number of simulations
        if sobol_samples.shape[0] > simulations:
            sobol_samples = sobol_samples[:simulations, :]
        elif sobol_samples.shape[0] < simulations:
            raise ValueError("Number of Sobol samples is less than the number of simulations")

        z = self.simulation_params.get('distribution_model', NormalDistribution()).ppf(sobol_samples)

        drift = (r - delta - 0.5 * volatilities**2) * dt
        diffusion = volatilities * np.sqrt(dt)

        # Calculate the price paths
        for t in range(1, steps + 1):
            prices[:, t] = prices[:, t - 1] * np.exp(drift[t - 1] + diffusion[t - 1] * z[:, t - 1])

        return prices

