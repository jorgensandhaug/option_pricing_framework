from sklearn.linear_model import LinearRegression
from typing import Optional
import numpy as np
from models.simulation_based_option_pricing import SimulationBasedOptionPricingModel

class LeastSquaresMonteCarloModel(SimulationBasedOptionPricingModel):

    def price(self, params: dict, simulation_params: Optional[dict] = None):
        price, boundary = self.price_and_boundary(params, simulation_params)
        return price

    def price_and_boundary(self, params: dict, simulation_params: Optional[dict] = None):
        if not params['is_american']:
            raise ValueError("Least squares Monte Carlo only meant for American options")
        
        prices = self.simulator.simulate(simulation_params)
        option_values = np.zeros_like(prices)
        if params['option_type'] == 'call':
            exercise_values = np.maximum(prices - params['strike_price'], 0)
        else:
            exercise_values = np.maximum(params['strike_price'] - prices, 0)

        option_values[:, -1] = exercise_values[:, -1]

        N = exercise_values.shape[1]

        dt = params['time_to_maturity'] / N
        # Pre-compute constants
        discount_factor = np.exp(-params['risk_free_rate'] * dt)


        did_exercise_matrix = np.zeros_like(exercise_values)
        # set last column of did_exercise_matrix to 1
        did_exercise_matrix[:, -1] = 1

        exercise_boundary = np.full((N, 2), np.nan)

        # Loop through each time step (backwards)
        for i in range(N-2, -1, -1):
            # Identify paths where option is in the money
            e = exercise_values[:, i] > 0


            # Proceed if there are paths where exercise is beneficial
            if np.any(e):
                # Y is the discounted option value at the next time step. The reason we do this instead of the discounted realized cash flows along each path, is because we make sure the option value for the next step always is updated with either the exercise value or the Y from the next step, a bit unlike how they do it in longstaff-schwartz
                Y = discount_factor * option_values[e, i+1]

                # Prepare the regressors, X, X^2 and constant
                X = prices[e, i].reshape(-1, 1)
                pows = np.arange(2, 3) # x^2
                X = np.concatenate([X] + [X ** i for i in pows], axis=1)

                # Linear regression
                model = LinearRegression().fit(X, Y)

                # Use the model to predict continuation values
                predicted_continuation_values = model.predict(X)

                # Check if exercising is beneficial compared to continuation value
                exercise_chosen = exercise_values[e, i] >= predicted_continuation_values

                # Store the decision to exercise or not
                did_exercise_matrix[e, i] = exercise_chosen.astype(int)

                # Update option values, based on the decision to exercise or not, if exercise is beneficial, set option value to exercise value, else set to Y, which is the discounted option value at the next time step for that path
                option_values[e, i] = np.where(exercise_chosen, exercise_values[e, i], Y)

                # Store the exercise boundary as the minimum/maximum price where exercise is chosen
                exercise_boundary[i, 0] = i*dt
                if np.any(exercise_chosen):
                    if params['option_type'] == 'call':
                        # Get lowest price where exercise is beneficial
                        exercise_boundary[i, 1] = np.min(prices[e, i]+1e6*(1-did_exercise_matrix[e, i])) 
                    elif params['option_type'] == 'put':
                        # Get highest price where exercise is beneficial
                        exercise_boundary[i, 1] = np.max(prices[e, i]-1e6*(1-did_exercise_matrix[e, i]))
                    else:
                        raise ValueError("Invalid option type")



            # Paths where exercise value is 0
            en = ~e

            # Update option values for paths where exercise is not beneficial
            option_values[en, i] = discount_factor * option_values[en, i+1]



        average_value = np.mean(option_values[:, 0])

        return average_value, exercise_boundary
