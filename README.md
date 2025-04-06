# Bitcoin-Forecast-with-Bayesian
# Bitcoin Price Prediction with Bayesian State-Space Model

This project implements a Bayesian State-Space Model to predict Bitcoin (BTC) prices in USD using historical financial market data fetched from Yahoo Finance via the `yfinance` library. The model incorporates autoregressive dynamics, external features, and seasonality, with inference performed using Markov Chain Monte Carlo (MCMC) via NumPyro.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Required Libraries](#required-libraries)
3. [Data Collection](#data-collection)
4. [Dataset Description](#dataset-description)
5. [Visualization](#visualization)
6. [Model Details](#model-details)
7. [Inference](#inference)
8. [Prediction](#prediction)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Contributors](#contributors)
11. [License](#license)

---

## Project Overview
This repository contains a Python-based implementation for collecting, processing, and analyzing Bitcoin price data (`BTC-USD`) to build a predictive model. The core of the project is a Bayesian State-Space Model that combines autoregressive (AR) dynamics, external features (e.g., scaled historical data), and a seasonal component to forecast Bitcoin prices.

---

## Required Libraries
The following Python libraries and modules are used in this project:

- **numpy (`np`)**: For numerical operations and array handling.
- **pandas (`pd`)**: For data manipulation and analysis.
- **sklearn.preprocessing**: 
  - `StandardScaler`: Standardizes features by removing the mean and scaling to unit variance.
  - `PolynomialFeatures`: Generates polynomial and interaction features (optional).
  - `MinMaxScaler`: Scales features to a specified range (e.g., [0, 1]).
- **arviz (`az`)**: For Bayesian data analysis and visualization of posterior distributions.
- **matplotlib.pyplot (`plt`)**: For static plotting (optional).
- **datetime (`dt`)**: For handling date and time operations.
- **yfinance (`yf`)**: For fetching historical financial data from Yahoo Finance.
- **plotly**: 
  - `plotly.express (`px`)**: For interactive visualizations.
  - `plotly.graph_objects (`go`)**: For custom interactive plots.
  - `plotly.subplots`: For creating subplots.
- **itertools.cycle**: For cycling through colors or styles in visualizations.
- **sklearn.pipeline (`Pipeline`)**: For creating a pipeline of data transformations (optional).
- **sklearn.metrics**: 
  - `mean_squared_error`: Computes the mean squared error between predictions and actual values.
  - `mean_absolute_error`: Computes the mean absolute error between predictions and actual values.
  - `r2_score`: Computes the R² score (coefficient of determination, optional).
- **warnings**: To suppress warning messages during execution.
- **scipy.stats (`gaussian_kde`)**: For kernel density estimation in density plots.
- **numpyro**: For probabilistic programming and MCMC inference.
  - `dist`: Distributions (e.g., `HalfNormal`, `Laplace`, `Normal`, `StudentT`).
  - `sample`: Sampling from distributions.
- **jax**: 
  - `jnp`: JAX NumPy for array operations compatible with NumPyro.
  - `random.PRNGKey`: Random number generation for reproducibility.
- **numpyro.infer**: 
  - `NUTS`: No-U-Turn Sampler for MCMC.
  - `MCMC`: Markov Chain Monte Carlo inference.

---

## Data Collection
### Source of the Data
The data is collected using the `yfinance` library, which retrieves historical financial market data directly from Yahoo Finance.

### Dataset Description
- **Ticker Used**: `BTC-USD` (Bitcoin to US Dollar).
- **Data Source**: Yahoo Finance via the `yfinance` API.
- **Time Range**: From `2023-01-01` to `2025-01-25` (as specified in the code).
- **Columns**: 
  - `Open`: Opening price of Bitcoin in USD.
  - `High`: Highest price during the day.
  - `Low`: Lowest price during the day.
  - `Close`: Closing price of Bitcoin in USD (used as the target variable).
  - `Volume`: Trading volume in USD.

## Visualization
The project includes a function to generate interactive trace and density plots for posterior samples using Plotly:

### `plot_trace_and_density_plotly(samples, param_name)`
- **Purpose**: Creates side-by-side interactive plots showing the trace (time series of samples) and density (histogram with KDE) of model parameters (e.g., `beta`, `process_std`, `noise_std`).
- **Inputs**:
- `samples`: Array of sampled values from the posterior (e.g., `beta_samples`, `process_std_samples`).
- `param_name`: Name of the parameter being plotted (e.g., `beta[1]`, `Noise Std`).
- **Outputs**: 
- Left subplot: Trace plot of samples over iterations.
- Right subplot: Histogram of samples with a Kernel Density Estimate (KDE) overlay.
- **Libraries Used**: `plotly`, `scipy.stats.gaussian_kde`.
- **Example Usage**:
```python
plot_trace_and_density_plotly(posterior_samples['noise_std'], 'Noise Std')
``` 
# Model Details

# 🧠 Bayesian State-Space Model for Bitcoin Price Forecasting

This project implements a **Bayesian State-Space Model** using [NumPyro](https://num.pyro.ai/), combining:

- 📈 **AR(1) Latent Process**  
- 🧾 **External Features Integration**  
- 📆 **Quarterly Seasonality**  
- 🛡 **Robust Observations with Student’s t-distribution**

---

## 🔁 Model Overview

The model combines four key elements:
1. **Autoregressive (AR) Latent Process**: Captures persistence in the latent state.
2. **External Features**: Incorporates scaled historical data or other covariates.
3. **Quarterly Seasonality**: Models periodic effects with a 4-step cycle.
4. **Robust Observation Model**: Uses a Student's t-distribution to handle outliers.

---

### 🔮 Latent State Transition

The latent state evolves as:

$$
x_{t} = 0.8 \cdot x_{t-1} + \beta^{\top} X_{t-1} + s_{t \mod 4} + \epsilon_{t}
$$

### 🔑 Model Components

 #### Autoregressive Term :  
$$ 0.8 \cdot x_{t-1} $$  
- A fixed persistence coefficient \( 0.8 \) is applied to the previous latent state, capturing the inertia of the system over time.

 #### External Features:  
$$ \beta^{\top} X_{t-1} $$  
  - \( $\beta$\): Coefficient vector (length = number of external features).  
  - \( $X_{t-1}$ \): Feature vector at time \( $t-1$ \) (e.g., scaled historical data).

  #### Seasonality:  
$$ s_{t \mod 4} $$  
- The seasonal effect is modeled with a quarterly periodicity (i.e., a 4-step cycle), capturing repeating patterns within each year.

 #### Process Noise:  
$$ \epsilon_t \sim \mathcal{N}(0, \sigma_p) $$  
- Gaussian noise, added to the model, with standard deviation \( $\sigma_p$ \) representing random disturbances in the system.

---

### 👁️ Observation Model

The observed Bitcoin price is modeled using a **robust Student’s t-distribution**:

$$
y_t \sim \text{StudentT}(\nu=4, \mu=x_t, \sigma=\sigma_n)
$$

Where:

- \( $y_t$ \): Observed price at time \( $t$ \)  
- \( $x_t$ \): Latent state (mean of the observation)  
- \( $\sigma_n$ \): Observation noise standard deviation  

---

## 📏 Priors

| Parameter        | Distribution           | Description                            |
|------------------|------------------------|----------------------------------------|
| \( $\sigma_p$ \)   | HalfNormal(0.5)        | Std. dev. of process noise             |
| \( $\sigma_n$ \)   | HalfNormal(0.5)        | Std. dev. of observation noise         |
| \( $\beta$ \)      | Laplace(0, 0.1)        | Sparse prior for external features     |
| \( $s_k$ \)        | Normal(0, 1)           | Seasonal components (quarterly)        |
| \( $x_0$ \)        | Normal(0, 1)           | Prior on the initial latent state      |

---

# Inference

### MCMC with NUTS

Inference is performed using the **No-U-Turn Sampler (NUTS)**, a variant of Hamiltonian Monte Carlo, implemented in NumPyro. Here’s how the setup and execution are done:

#### Setup:
- **Random Seed**: `rng_key = random.PRNGKey(0)`
- **NUTS Kernel**: `nuts_kernel = NUTS(state_space_model_with_features, target_accept_prob=0.9)`
- **MCMC**: `mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)`

#### Execution:
```python
mcmc.run(rng_key, T=T_train, X=scaled_train_jax, obs=observations_train)
```
### Outputs: 
- Posterior samples for `process_std, noise_std, beta, seasonal, and latent states (x_t)`.

### Posterior Samples:
- Stored in `posterior_samples = mcmc.get_samples()`.
- Used for visualization and prediction.

# Prediction

### 🔮 Test Set Prediction
Predictions on the test set are generated using posterior samples:

**Inputs**:
- `T_test`: Length of the test set.
- `scaled_test_jax`: Scaled test features (as a JAX array).

**Process**:
1. Simulate latent states forward using sampled parameters:
   - \( $\beta$ \): Feature coefficients
   - \( $s$ \): Seasonal components
   - \( $\sigma_p$ \): Process noise standard deviation
2. Generate predictions via the observation model.

---

# Evaluation Metrics
Performance is measured using:

### Mean Squared Error (MSE) 
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

### Mean Absolute Error (MAE)
$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

Where:
- \( $y_i$ \): Actual Bitcoin price
- \( $\hat{y}_i$ \): Predicted price (mean of posterior predictive samples)
- \( $n$ \): Number of observations

---

### 🏆 Results

| Dataset    | MSE       | MAE       |
|------------|-----------|-----------|
| Training   | 0.0000    | 0.0026    |
| Test       | 0.0012    | 0.0305    |

**Interpretation**:  
The model demonstrates strong predictive accuracy on both training and test sets, with near-zero error values indicating close alignment between predictions and actual prices.


# Contributors:

Md Riajuliislam (Master of Science Student at TU Dortmund)

# License
This project is licensed under the MIT License.