import numpy as np
from scipy.stats.qmc import Sobol
from scipy.stats import norm
from simulations.simulation_model import SimulationModel
from distributions.normal_distribution import NormalDistribution

class HestonProcess(SimulationModel):

    def simulate(self, simulation_params: dict = None):
        prices, variances = self.simulate_prices_and_variances(simulation_params)
        return prices
    
    def simulate_prices_and_variances(self, simulation_params: dict = None):
        if simulation_params is not None:
            self.simulation_params = simulation_params

        S0 = self.simulation_params['initial_stock_price']
        T = self.simulation_params['time_to_maturity']
        r = self.simulation_params['risk_free_rate']
        v0 = self.simulation_params['initial_variance']
        steps = self.simulation_params['time_steps']
        simulations = self.simulation_params['simulation_paths']
        kappa = self.simulation_params['kappa']
        theta = self.simulation_params['theta']
        volvol = self.simulation_params['volvol']
        rho = self.simulation_params['rho']

        dt = T / steps
        sqrt_dt = np.sqrt(dt)

        prices = np.zeros((simulations, steps + 1))
        variances = np.zeros((simulations, steps + 1))
        prices[:, 0] = S0
        variances[:, 0] = v0

        # Generate Sobol sequence for 2*steps dimensions
        sobol = Sobol(d=2*steps, scramble=True)
        m = int(np.ceil(np.log2(simulations)))
        sobol_samples = sobol.random_base2(m=m)

        # Ensure the Sobol samples match the number of simulations
        if sobol_samples.shape[0] > simulations:
            sobol_samples = sobol_samples[:simulations, :]
        elif sobol_samples.shape[0] < simulations:
            raise ValueError("Number of Sobol samples is less than the number of simulations")

        # Transform Sobol samples to normal samples
        norm_samples = norm.ppf(sobol_samples)

        # Perform Cholesky decomposition for the correlation
        correlation_matrix = np.array([[1, rho], [rho, 1]])
        L = np.linalg.cholesky(correlation_matrix)

        # Reshape norm_samples to (simulations, steps, 2)
        norm_samples = norm_samples.reshape(simulations, steps, 2)
        
        # Generate correlated samples for each step
        correlated_samples = np.einsum('ij,klj->kli', L, norm_samples)

        # mu = np.array([0,0])
        # cov = np.array([[1,rho],[rho,1]])

        # correlated_samples = np.random.multivariate_normal(mu, cov, (simulations, steps))

        
        # Iterate over time steps to simulate the Heston model
        for t in range(1, steps + 1):
            Z1 = correlated_samples[:, t-1, 0]
            Z2 = correlated_samples[:, t-1, 1]

            # Update variance process
            variances[:, t] = np.maximum(variances[:, t-1] +
                               kappa * (theta - variances[:, t-1]) * dt +
                               volvol * np.sqrt(variances[:, t-1]) * sqrt_dt * Z1, 0)

            # Update stock price process
            prices[:, t] = (prices[:, t-1] *
                            np.exp((r - 0.5 * variances[:, t-1]) * dt +
                                   np.sqrt(variances[:, t-1]) * sqrt_dt * Z2))

        return prices, variances
