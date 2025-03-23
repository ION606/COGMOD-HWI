import matplotlib.pyplot as plt
import numpy as np


# H Y P E R P A R A M E T E R S
mu_prior = 0      # prior mean
sigma2_prior = 2   # prior variance (omega_0^2)
sigma2_likelihood = 1  # likelihood variance (omega^2)
n_samples = 1000000      # number of Monte Carlo samples

# simulate θ ~ N(mu_0, omega_0^2) and y ~ N(θ, (omega)^2)
theta = np.random.normal(mu_prior, np.sqrt(sigma2_prior), n_samples)
y = np.random.normal(theta, np.sqrt(sigma2_likelihood))

# posterior params for each y
sigma2_posterior = 1 / (1 / sigma2_prior + 1 / sigma2_likelihood)
mu_posterior = (mu_prior / sigma2_prior + y / sigma2_likelihood) * \
    sigma2_posterior  # posterior mean

# E[Var[θ|y]]
expected_posterior_var = sigma2_posterior
var_posterior_mean = np.var(mu_posterior)  # var[𝔼[θ|y]]
prior_var = sigma2_prior                   # var[θ]

# verify identity
sum_terms = expected_posterior_var + var_posterior_mean

print(f"Prior Variance (Var[θ]): {prior_var:.4f}")
print(
    f"Expected Posterior Variance (𝔼[Var[θ|y]]): {expected_posterior_var:.4f}")
print(f"Variance of Posterior Mean (Var[𝔼[θ|y]]): {var_posterior_mean:.4f}")
print(f"Sum of Terms: {sum_terms:.4f}")
print(f"Identity Holds: {np.isclose(prior_var, sum_terms, atol=1e-3)}")

# Plot posterior means and variances
plt.figure(figsize=(10, 6))
plt.hist(mu_posterior, bins=50, density=True,
         alpha=0.6, label="Posterior Means")
plt.axvline(mu_prior, color='r', linestyle='--', label="Prior Mean")
plt.xlabel("Posterior Mean (𝔼[θ|y])")
plt.ylabel("Density")
plt.title("Distribution of Posterior Means vs. Prior")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('part3.png')