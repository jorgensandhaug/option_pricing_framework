import numpy as np
import pytest
from simulations.geometric_brownian_motion import GeometricBrownianMotion
from simulations.generic_drift_diffusion_process import GenericDriftDiffusionProcess

# Define GBM drift and diffusion functions
def gbm_drift(t, S, r=0.05, delta=0.0):
    return (r - delta) * S

def gbm_diffusion(t, S, sigma=0.2):
    return sigma * S

@pytest.fixture
def simulation_params():
    return {
        'initial_stock_price': 100,
        'time_to_maturity': 1,
        'risk_free_rate': 0.05,
        'volatility': 0.2,
        'dividend_yield': 0.0,
        'time_steps': 1000,
        'simulation_paths': 10000
    }

def test_generic_drift_diffusion_process(simulation_params):
    # Create the simulation object
    simulation = GenericDriftDiffusionProcess(drift_function=gbm_drift, diffusion_function=gbm_diffusion)

    # Run the simulation
    simulated_prices = simulation.simulate(simulation_params)

    # Use the GeometricBrownianMotion to get the expected GBM results
    gbm_model = GeometricBrownianMotion(simulation_params)
    expected_prices = gbm_model.simulate()

    # Compare the results
    simulated_mean = np.mean(simulated_prices[:, -1])
    expected_mean = np.mean(expected_prices[:, -1])

    simulated_std = np.std(simulated_prices[:, -1])
    expected_std = np.std(expected_prices[:, -1])

    print(f"Simulated Mean: {simulated_mean}")
    print(f"Expected Mean: {expected_mean}")
    print(f"Simulated Std Dev: {simulated_std}")
    print(f"Expected Std Dev: {expected_std}")

    assert np.abs(simulated_mean - expected_mean) < 1.0, f"Mean difference too large: {np.abs(simulated_mean - expected_mean)}"
    assert np.abs(simulated_std - expected_std) < 1.0, f"Standard deviation difference too large: {np.abs(simulated_std - expected_std)}"
