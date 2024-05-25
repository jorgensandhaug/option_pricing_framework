from models.european.black_scholes import BlackScholesModel
from models.american.binomial import BinomialModel
from models.european.european_option_simulation import EuropeanOptionSimulationModel
from models.american.least_squares_monte_carlo import LeastSquaresMonteCarloModel

from simulations.geometric_brownian_motion import GeometricBrownianMotion
from distributions.normal_distribution import NormalDistribution
from distributions.t_distribution import TDistribution

import numpy as np


# Example usage
if __name__ == "__main__":
    S0 = 200
    K = 210
    T = 3
    r = 0.05
    sigma = 0.2
    delta = 0.01
    option_type = 'put'
    is_american = True
    steps = 500

    option_params = {
        'initial_stock_price': S0,
        'strike_price': K,
        'time_to_maturity': T,
        'risk_free_rate': r,
        'volatility': sigma,
        'dividend_yield': delta,
        'option_type': option_type,
        'is_american': is_american,
    }

    simulation_params = {
        'initial_stock_price': S0,
        'time_to_maturity': T,
        'risk_free_rate': r,
        'volatility': sigma,
        'dividend_yield': delta,
        'time_steps': steps,
        'simulation_paths': 100000
    }

    # Using Black-Scholes Model
    bs_pricer = BlackScholesModel()
    print(f"Black-Scholes Call Price: {bs_pricer.price(option_params)}")

    # Using Normal Distribution for Geometric Brownian Motion Simulation
    normal_dist = NormalDistribution(mean=0, std=1)
    gbm_model = GeometricBrownianMotion(simulation_params)
    simulation_pricer = EuropeanOptionSimulationModel(gbm_model)
    print(f"European Call Price using Simulation (Normal Dist): {simulation_pricer.price(option_params)}")


    # Using Binomial Model
    binomial_pricer = BinomialModel(steps)
    print(f"Binomial Call Price: {binomial_pricer.price(option_params)}")

    # Using Least Squares Monte Carlo Model
    ls_pricer = LeastSquaresMonteCarloModel(gbm_model)
    print(f"Least Squares Monte Carlo Call Price: {ls_pricer.price(option_params)}")

