import numpy as np
from scipy.stats import t, norm
import yfinance as yf
import pandas as pd
from scipy.stats import norm, t, skewnorm, gamma, expon, lognorm, beta, gumbel_r, weibull_min, pareto, genextreme, genpareto, gennorm, genhalflogistic
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


distributions = {
    'norm': norm,
    't': t,
    'skewnorm': skewnorm,
    'gamma': gamma,
    'expon': expon,
    'lognorm': lognorm,
    'beta': beta,
    'gumbel_r': gumbel_r,
    'weibull_min': weibull_min,
    'pareto': pareto,
    'genextreme': genextreme,
    'genpareto': genpareto,
    'gennorm': gennorm,
    'genhalflogistic': genhalflogistic
}

def fit_distribution(data: pd.Series, distribution: str):
    if distribution not in distributions:
        raise ValueError(f"Distribution '{distribution}' is not supported.")
    
    dist = distributions[distribution]
    params = dist.fit(data)

    return params




def fit_all_distributions(data: pd.Series):
    # Fit the distributions
    results = []

    for name, distribution in distributions.items():
        try:
            params = distribution.fit(data)
            results.append((name, distribution, params))
        except Exception as e:
            print(f"Error fitting {name}: {e}")

    # Sort the results by the negative log likelihood
    def dist_sort(x):
        try: 
            return x[1].nnlf(x[2], data)
        except Exception as e:
            name = x[0]
            print(f"Error calculating nnlf for {name}: {e}")
            return np.inf

    results.sort(key=lambda x: dist_sort(x))

    # Print the results
    for name, dist, params in results:
        print(f"{name}: {params}")

    # Big plot
    plt.figure(figsize=(12, 8))

    
    mean=data.mean()
    std=data.std()
    
    num_stds = 3
    # Min is mean - 10* std, max is mean + 10 * std
    min_x = mean - num_stds * std
    max_x = mean + num_stds * std

    # Define the bins for the histogram
    bins = np.linspace(min_x, max_x, 36)  # We use 36 points to get 35 bins

    # Plot the histogram of the returns
    hist = sns.histplot(data, kde=False, bins=bins, label='Data', stat='density')

    # Get the maximum height of the histogram
    max_height = max([h.get_height() for h in hist.patches])


    # Set the y-axis limits to not show anything more than 50% above the highest bin height
    plt.ylim(0, max_height * 1.5)

    # Set the x-axis limits to show the range of returns
    plt.xlim(min_x, max_x)



    # Plot the fitted distributions
    x = np.linspace(min_x, max_x, 1000)
    for name, distribution, params in results:
        try:
            plt.plot(x, distribution.pdf(x, *params), label=name)
        except Exception as e:
            print(f"Error plotting {name}: {e}")

    plt.title('Yearly Returns and Fitted Distributions')
    plt.xlabel('Return')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()

    print(f"Mean return: {mean}")
    print(f"Volatility: {std}")

    freq = data.index.to_series().diff().dropna().mode()[0].days 
    print(f"Frequency: {freq} (days)")
    print(f"Annualized Mean: {mean * freq * 252}")
    print(f"Annualized volatility: {std * np.sqrt(freq * 252)}")
    
    
    

