from models.black_scholes import BlackScholesModel
from models.binomial import BinomialModel
from models.european_option_simulation import EuropeanOptionSimulationModel
from simulations.geometric_brownian_motion import GeometricBrownianMotion
from distributions.normal_distribution import NormalDistribution
from distributions.t_distribution import TDistribution
from utils.historical_data import HistoricalData

class OptionPricer:
    def __init__(self, model):
        self.model = model

    def price(self, params):
        return self.model.price(params)

class Simulator:
    def __init__(self, model):
        self.model = model

    def simulate(self, params):
        return self.model.simulate(params)

# Example usage
if __name__ == "__main__":
    S0 = 200
    K = 210
    T = 3
    r = 0.05
    sigma = 0.2
    delta = 0.01
    option_type = 'call'
    is_american = True
    steps = 3000

    option_params = {
        'initial_stock_price': S0,
        'strike_price': K,
        'time_to_maturity': T,
        'risk_free_rate': r,
        'volatility': sigma,
        'dividend_yield': delta,
        'option_type': option_type,
        'is_american': is_american,
        'steps': steps
    }

    simulation_params = {
        'initial_stock_price': S0,
        'time_to_maturity': T,
        'risk_free_rate': r,
        'volatility': sigma,
        'dividend_yield': delta,
        'time_steps': steps,
        'simulation_paths': 10000
    }

    # Using Black-Scholes Model
    bs_model = BlackScholesModel()
    pricer = OptionPricer(bs_model)
    print(f"Black-Scholes Call Price: {pricer.price(option_params)}")

    # Using Binomial Model
    binomial_model = BinomialModel(steps)
    pricer = OptionPricer(binomial_model)
    print(f"Binomial Call Price: {pricer.price(option_params)}")

    # Using Normal Distribution for Geometric Brownian Motion Simulation
    normal_dist = NormalDistribution(mean=0, std=1)
    gbm_model = GeometricBrownianMotion(normal_dist)
    simulation_pricer = EuropeanOptionSimulationModel(gbm_model)
    pricer = OptionPricer(simulation_pricer)
    print(f"European Call Price using Simulation (Normal Dist): {pricer.price(option_params)}")

    # Using T-Distribution for Geometric Brownian Motion Simulation
    historical_prices = np.array([100, 102, 105, 107, 110, 108, 107])  # Example historical prices
    hist_data = HistoricalData(historical_prices)
    df, loc, scale = hist_data.fit_t_distribution()
    t_dist = TDistribution(df, loc, scale)
    gbm_model_t = GeometricBrownianMotion(t_dist)
    simulation_pricer_t = EuropeanOptionSimulationModel(gbm_model_t)
    pricer = OptionPricer(simulation_pricer_t)
    print(f"European Call Price using Simulation (T Dist): {pricer.price(option_params)}")
