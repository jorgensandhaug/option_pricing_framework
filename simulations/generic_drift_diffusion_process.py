import numpy as np
from scipy.stats.qmc import Sobol
from scipy.stats import norm
from simulations.simulation_model import SimulationModel
from distributions.distribution_model import Distribution
from distributions.normal_distribution import NormalDistribution

class GenericDriftDiffusionProcess(SimulationModel):
    def __init__(self, simulation_params: dict, drift_function: callable, diffusion_function: callable):
        super().__init__(simulation_params)
        self.drift_function = drift_function
        self.diffusion_function = diffusion_function

    def simulate(self, simulation_params: dict):
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
        sqrt_dt = np.sqrt(dt)
        
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

        # Simulate paths
        for i in range(1, steps + 1):
            t = i * dt
            drift = self.drift_function(t, prices[:, i-1])
            diffusion = self.diffusion_function(t, prices[:, i-1])
            prices[:, i] = prices[:, i-1] + drift * dt + diffusion * sqrt_dt * z[:, i-1]

        return prices