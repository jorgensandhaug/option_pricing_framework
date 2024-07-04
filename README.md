This options pricing framework is made for experimentation and learning.

# Option Pricing Framework
A comprehensive Python framework for pricing financial options using various models and simulation techniques.

## Features
### Multiple option pricing models:
- Black-Scholes
- Binomial
- Monte Carlo simulation
- Least Squares Monte Carlo for American options
- Analytical and simulated Asian options

### Flexible simulation models:
- Geometric Brownian Motion
- Generic Drift-Diffusion Process
- Heston Stochastic Volatility
- Ornstein-Uhlenbeck Process

### Support for different option types:
- European options
- American options
- Asian options

### Volatility models:
- GARCH

### Utility functions:
- Implied volatility calculation
- Historical data analysis
- Distribution fitting

## Installation
```
git clone https://github.com/yourusername/option_pricing_framework.git
cd option_pricing_framework
pip install -r requirements.txt
```

## Usage
Here's a basic example of pricing a European call option using the Black-Scholes model:
```python
from models.european.black_scholes import BlackScholesModel

params = {
    'initial_stock_price': 100,
    'strike_price': 100,
    'time_to_maturity': 1,
    'risk_free_rate': 0.05,
    'volatility': 0.2,
    'option_type': 'call'
}

bs_model = BlackScholesModel()
price = bs_model.price(params)
print(f"Option price: {price}")
```

## Project Structure
- `models/`: Contains various option pricing models
- `simulations/`: Implements different simulation techniques
- `distributions/`: Defines probability distributions used in simulations
- `utils/`: Utility functions for data analysis and calculations
- `experiments/`: Jupyter notebooks demonstrating various pricing scenarios
- `tests/`: Unit tests for the framework components

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License.

## Acknowledgements
This framework was inspired by and built upon various financial engineering concepts and techniques. Special thanks to all the researchers and practitioners in the field of quantitative finance.