from simulations.generic_drift_diffusion_process import GenericDriftDiffusionProcess


class OrnsteinUhlenbeckProcess(GenericDriftDiffusionProcess):
    def drift_function(self, t, S):
        return self.kappa * (self.theta - S)

    def diffusion_function(self, t, S):
        return self.sigma


    def __init__(self, simulation_params, kappa, theta, sigma):
        super().__init__(simulation_params, self.drift_function, self.diffusion_function)
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

